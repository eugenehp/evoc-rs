//! Label propagation initialization (`evoc.label_propagation`).

use crate::csr_matmul::scipy_csr_matmul;
use crate::embed::node_embedding;
use crate::numpy_rng::NumpyRandomState;
use crate::rng::{offset_state, tau_rand, tau_rand_int};
use faer::{Mat, Side};
use ndarray::{Array2, Axis};
use rustc_hash::FxHashMap;
use sprs::{CsMat, TriMat};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LabelPropError {
    #[error("base_init 'pca' requires data")]
    PcaRequiresData,
}

/// Base initialization for small graphs (Python `base_init` argument).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // `Random` / `Spectral` / `Mds` match Python API; default path uses `Pca`
pub enum BaseInit {
    Pca,
    Random,
    Spectral,
    Mds,
}

/// Upscaling strategy when merging partition embeddings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // `JitterExpander` / `NodeEmbedding` match Python; default uses `PartitionExpander`
pub enum Upscaling {
    PartitionExpander,
    JitterExpander,
    NodeEmbedding,
}

/// One label-propagation iteration (Python: `label_prop_iteration`).
pub fn label_prop_iteration(
    indptr: &[usize],
    indices: &[usize],
    data: &[f32],
    labels: &[i64],
    rng_state: &[i64; 3],
) -> Vec<i64> {
    let n_rows = indptr.len() - 1;
    let mut result = labels.to_vec();

    for i in 0..n_rows {
        let out = &mut result[i];
        let current_l = labels[i];
        if current_l >= 0 {
            continue;
        }
        let mut local_rng = offset_state(rng_state, i as i64);
        let mut votes: Vec<(i64, f64)> = Vec::new();
        for k in indptr[i]..indptr[i + 1] {
            let j = indices[k];
            let l = labels[j];
            let w = f64::from(data[k]);
            if let Some(entry) = votes.iter_mut().find(|(ll, _)| *ll == l) {
                entry.1 += w;
            } else {
                votes.push((l, w));
            }
        }

        let mut max_vote = 1.0f64;
        let mut tie_count = 1usize;
        *out = current_l;

        for &(l, v) in &votes {
            if l == -1 {
                continue;
            } else if v > max_vote {
                max_vote = v;
                *out = l;
                tie_count = 1;
            } else if v == max_vote {
                tie_count += 1;
                if current_l == -1 {
                    *out = l;
                } else if f64::from(tau_rand(&mut local_rng)) < 1.0 / tie_count as f64 {
                    *out = l;
                }
            }
        }
    }

    result
}

/// Assign labels to remaining outliers (Python: `label_outliers`).
pub fn label_outliers(
    indptr: &[usize],
    indices: &[usize],
    labels: &mut [i64],
    rng_state: &[i64; 3],
) {
    let _n_rows = indptr.len() - 1;
    let max_label = labels.iter().copied().max().unwrap_or(0).max(0);

    for i in 0..labels.len() {
        let mut local_rng = offset_state(rng_state, i as i64);
        if labels[i] >= 0 {
            continue;
        }
        let mut node_queue = vec![i];
        let mut unlabelled = true;
        let mut n_iter = 0usize;

        while unlabelled && n_iter < 64 && !node_queue.is_empty() {
            n_iter += 1;
            let current_node = node_queue.pop().unwrap();
            for k in indptr[current_node]..indptr[current_node + 1] {
                let j = indices[k];
                if labels[j] >= 0 {
                    labels[i] = labels[j];
                    unlabelled = false;
                    break;
                } else {
                    node_queue.push(j);
                }
            }
        }

        if n_iter >= 64 || unlabelled {
            let draw = tau_rand_int(&mut local_rng) as i64;
            labels[i] = draw.rem_euclid(max_label + 1);
        }
    }
}

/// Remap labels to contiguous integers (Python: `remap_labels`).
pub fn remap_labels(labels: &mut [i64]) {
    let mut unique: Vec<i64> = labels.iter().copied().collect();
    unique.sort_unstable();
    unique.dedup();
    if unique.first() == Some(&-1) {
        unique.remove(0);
    }

    let mut mapping: FxHashMap<i64, i64> = FxHashMap::default();
    for (i, &l) in unique.iter().enumerate() {
        mapping.insert(l, i as i64);
    }
    let mut next_label = unique.len() as i64;

    for label in labels.iter_mut() {
        if *label < 0 {
            *label = next_label;
            next_label += 1;
        } else {
            *label = mapping[label];
        }
    }
}

/// Run label propagation to convergence (Python: `label_prop_loop`).
pub fn label_prop_loop(
    indptr: &[usize],
    indices: &[usize],
    data: &[f32],
    labels: &mut [i64],
    rng: &mut NumpyRandomState,
    n_iter: usize,
    approx_n_parts: usize,
) -> Vec<i64> {
    let rng_state = rng.randint3_for_tau();

    let n = labels.len();
    for i in 0..approx_n_parts {
        let idx = rng.randint(0, n as i64) as usize;
        labels[idx] = i as i64;
    }

    let mut labels = labels.to_vec();
    for _ in 0..n_iter {
        labels = label_prop_iteration(indptr, indices, data, &labels, &rng_state);
    }

    label_outliers(indptr, indices, &mut labels, &rng_state);
    remap_labels(&mut labels);
    labels
}

#[doc(hidden)]
pub fn normalize_cols_l2(mat: &mut CsMat<f32>) {
    let ncols = mat.cols();
    let mut col_norms = vec![0.0f32; ncols];
    for (val, (_, c)) in mat.iter() {
        col_norms[c] += *val * *val;
    }
    for norm in &mut col_norms {
        *norm = norm.sqrt().max(1e-12);
    }
    let indices: Vec<usize> = mat.indices().to_vec();
    for (idx, val) in mat.data_mut().iter_mut().enumerate() {
        *val /= col_norms[indices[idx]];
    }
}

#[doc(hidden)]
pub fn normalize_rows_l1(mat: &mut CsMat<f32>) {
    let nrows = mat.rows();
    let mut row_sums = vec![0.0f32; nrows];
    for (val, (r, _)) in mat.iter() {
        row_sums[r] += val.abs();
    }
    let indptr: Vec<usize> = mat.indptr().raw_storage().to_vec();
    for r in 0..nrows {
        let s = row_sums[r].max(1e-12);
        for idx in indptr[r]..indptr[r + 1] {
            mat.data_mut()[idx] /= s;
        }
    }
}

#[doc(hidden)]
pub fn partition_reduction_map(partition: &[i64]) -> CsMat<f32> {
    let n = partition.len();
    let n_parts = partition.iter().copied().max().unwrap_or(0) as usize + 1;
    let mut tri = TriMat::new((n, n_parts.max(1)));
    for (i, &p) in partition.iter().enumerate() {
        if p >= 0 {
            tri.add_triplet(i, p as usize, 1.0);
        }
    }
    tri.to_csr()
}

#[doc(hidden)]
pub fn csr_matmul_dense(mat: &CsMat<f32>, dense: &Array2<f32>) -> Array2<f32> {
    let n = mat.rows();
    let k = dense.ncols();
    let mut out = Array2::zeros((n, k));
    for row in 0..n {
        let start = mat.indptr().raw_storage()[row];
        let end = mat.indptr().raw_storage()[row + 1];
        for idx in start..end {
            let col = mat.indices()[idx] as usize;
            let w = f64::from(mat.data()[idx]);
            for c in 0..k {
                out[[row, c]] += (w * f64::from(dense[[col, c]])) as f32;
            }
        }
    }
    out
}

/// Match Python small-graph init post-processing (`result -= result.mean()` then
/// divide by global `(max - min) / 2`, not per-column centering).
fn normalize_small_graph_init_like_python(mut result: Array2<f32>) -> Array2<f32> {
    let mean: f64 = result.iter().map(|&v| f64::from(v)).sum::<f64>() / result.len() as f64;
    result.mapv_inplace(|v| (f64::from(v) - mean) as f32);
    let min = result.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = result.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let scale = (max - min) / 2.0;
    if scale > 0.0 {
        result.mapv_inplace(|v| v / scale);
    }
    result
}

fn pca_init(data: &Array2<f32>, n_components: usize) -> Array2<f32> {
    let (n, d) = data.dim();
    let mean = data.mean_axis(Axis(0)).unwrap();
    let mat = Mat::from_fn(n, d, |i, j| f64::from(data[[i, j]]) - f64::from(mean[j]));
    let svd = mat.thin_svd();
    let mut u = Array2::<f64>::zeros((n, svd.u().ncols()));
    for i in 0..n {
        for j in 0..u.ncols() {
            u[[i, j]] = svd.u().read(i, j);
        }
    }
    let mut vt = Array2::<f64>::zeros((svd.v().ncols(), d));
    for i in 0..d {
        for j in 0..vt.nrows() {
            vt[[j, i]] = svd.v().read(i, j);
        }
    }
    svd_flip_vt_f64(&mut vt, &mut u);
    let s = svd.s_diagonal();
    let k = n_components.min(u.ncols()).min(s.nrows());

    let mut result = Array2::<f32>::zeros((n, n_components));
    for i in 0..n {
        for j in 0..k {
            result[[i, j]] = (u[[i, j]] * s.read(j)) as f32;
        }
    }
    normalize_small_graph_init_like_python(result)
}

fn svd_flip_vt_f64(vt: &mut Array2<f64>, u: &mut Array2<f64>) {
    let k = vt.nrows();
    for j in 0..k {
        let row = vt.row(j);
        let (max_idx, &max_val) = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .unwrap();
        let sign = if max_val >= 0.0 { 1.0 } else { -1.0 };
        vt.row_mut(j).mapv_inplace(|v| v * sign);
        u.column_mut(j).mapv_inplace(|v| v * sign);
        let _ = max_idx;
    }
}

fn random_unit_init(n: usize, n_components: usize, rng: &mut NumpyRandomState) -> Array2<f32> {
    let mut result = Array2::from_shape_fn((n, n_components), |_| rng.normal_scaled(1.0) as f32);
    for mut row in result.rows_mut() {
        let norm = row.dot(&row).sqrt();
        if norm > 0.0 {
            row /= norm;
        }
    }
    result
}

fn pairwise_sq_distances(data: &Array2<f32>) -> Array2<f32> {
    let n = data.nrows();
    let mut d = Array2::zeros((n, n));
    for i in 0..n {
        for j in i..n {
            let mut sq = 0.0f32;
            for k in 0..data.ncols() {
                let diff = data[[i, k]] - data[[j, k]];
                sq += diff * diff;
            }
            d[[i, j]] = sq;
            d[[j, i]] = sq;
        }
    }
    d
}

fn mds_init(data: &Array2<f32>, n_components: usize) -> Array2<f32> {
    let n = data.nrows();
    let dist_sq = pairwise_sq_distances(data);
    let mut h = Array2::<f64>::eye(n);
    for i in 0..n {
        for j in 0..n {
            h[[i, j]] -= 1.0 / n as f64;
        }
    }
    let dist64 = dist_sq.mapv(f64::from);
    let b = -0.5 * h.dot(&dist64).dot(&h);
    let mat = Mat::from_fn(n, n, |i, j| b[[i, j]] as f32);
    let evd = mat.selfadjoint_eigendecomposition(Side::Lower);
    let u = evd.u();
    let mut result = Array2::<f32>::zeros((n, n_components));
    let k = n_components.min(n);
    for j in 0..k {
        let eig = evd.s().column_vector().read(j).max(0.0).sqrt();
        for i in 0..n {
            result[[i, j]] = u.read(i, j) * eig;
        }
    }
    normalize_small_graph_init_like_python(result)
}

fn spectral_init(
    data: &Array2<f32>,
    n_components: usize,
    _rng: &mut NumpyRandomState,
) -> Array2<f32> {
    let n = data.nrows();
    let dist_sq = pairwise_sq_distances(data);
    let mut flat: Vec<f32> = dist_sq.iter().copied().filter(|&v| v > 0.0).collect();
    flat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let sigma = flat
        .get(flat.len() / 2)
        .copied()
        .unwrap_or(1.0)
        .sqrt()
        .max(1e-8);

    let mut tri = TriMat::new((n, n));
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let w = (-dist_sq[[i, j]] / (2.0 * sigma * sigma)).exp();
            if w > 1e-12 {
                tri.add_triplet(i, j, w);
            }
        }
    }
    let w: CsMat<f32> = tri.to_csr();
    let mut degrees = vec![0.0f32; n];
    for (val, (row, _)) in w.iter() {
        degrees[row] += *val;
    }
    let mut lap = vec![0.0f32; n * n];
    for i in 0..n {
        lap[i * n + i] = 1.0;
    }
    for (val, (row, col)) in w.iter() {
        let d_inv_sqrt = (degrees[row] * degrees[col]).sqrt().max(1e-12);
        lap[row * n + col] -= *val / d_inv_sqrt;
    }
    let mat = Mat::from_fn(n, n, |i, j| lap[i * n + j]);
    let evd = mat.selfadjoint_eigendecomposition(Side::Lower);
    let u = evd.u();

    let mut idx: Vec<usize> = (0..n).collect();
    let evals = evd.s().column_vector();
    idx.sort_by(|&a, &b| evals.read(a).partial_cmp(&evals.read(b)).unwrap());

    let mut result = Array2::<f32>::zeros((n, n_components));
    for (c, &ei) in idx.iter().take(n_components + 1).skip(1).enumerate() {
        if c >= n_components {
            break;
        }
        for i in 0..n {
            result[[i, c]] = u.read(i, ei);
        }
    }
    normalize_small_graph_init_like_python(result)
}

fn small_graph_init(
    data: Option<&Array2<f32>>,
    n_vertices: usize,
    n_components: usize,
    base_init: BaseInit,
    rng: &mut NumpyRandomState,
) -> Result<Array2<f32>, LabelPropError> {
    match base_init {
        BaseInit::Random => Ok(random_unit_init(n_vertices, n_components, rng)),
        BaseInit::Pca => {
            let data = data.ok_or(LabelPropError::PcaRequiresData)?;
            Ok(pca_init(data, n_components))
        }
        BaseInit::Spectral => {
            let data = data.ok_or(LabelPropError::PcaRequiresData)?;
            Ok(spectral_init(data, n_components, rng))
        }
        BaseInit::Mds => {
            let data = data.ok_or(LabelPropError::PcaRequiresData)?;
            Ok(mds_init(data, n_components))
        }
    }
}

#[doc(hidden)]
pub fn hadamard_symmetric(graph: &CsMat<f32>) -> CsMat<f32> {
    let gt = graph.transpose_view().to_csr();
    let mut tri = TriMat::with_capacity((graph.rows(), graph.cols()), graph.nnz());
    for (val, (r, c)) in graph.iter() {
        let v = gt.get(r, c).copied().unwrap_or(0.0);
        if v != 0.0 {
            tri.add_triplet(r, c, *val * v);
        }
    }
    tri.to_csr()
}

/// Initialize embeddings via label propagation (Python: `label_propagation_init`).
#[allow(clippy::too_many_arguments)]
pub fn label_propagation_init(
    graph: &CsMat<f32>,
    n_label_prop_iter: usize,
    n_embedding_epochs: usize,
    approx_n_parts: usize,
    n_components: usize,
    scaling: f32,
    random_scale: f32,
    noise_level: f32,
    rng: &mut NumpyRandomState,
    data: Option<&Array2<f32>>,
) -> Array2<f32> {
    label_propagation_init_with_options(
        graph,
        n_label_prop_iter,
        n_embedding_epochs,
        approx_n_parts,
        n_components,
        scaling,
        random_scale,
        noise_level,
        rng,
        data,
        true,
        BaseInit::Pca,
        64,
        Upscaling::PartitionExpander,
    )
    .expect("label_propagation_init")
}

/// Extended initializer matching all Python keyword arguments.
#[allow(clippy::too_many_arguments)]
pub fn label_propagation_init_with_options(
    graph: &CsMat<f32>,
    n_label_prop_iter: usize,
    n_embedding_epochs: usize,
    approx_n_parts: usize,
    n_components: usize,
    scaling: f32,
    random_scale: f32,
    noise_level: f32,
    rng: &mut NumpyRandomState,
    data: Option<&Array2<f32>>,
    recursive_init: bool,
    base_init: BaseInit,
    base_init_threshold: usize,
    upscaling: Upscaling,
) -> Result<Array2<f32>, LabelPropError> {
    let graph = graph.to_csr();
    let n_vertices = graph.rows();

    if n_vertices < base_init_threshold {
        return small_graph_init(data, n_vertices, n_components, base_init, rng);
    }

    let mut labels = vec![-1i64; n_vertices];
    let indptr: Vec<usize> = graph.indptr().raw_storage().iter().map(|&p| p).collect();
    let indices: Vec<usize> = graph.indices().iter().map(|&i| i).collect();
    let edge_data: Vec<f32> = graph.data().to_vec();

    let partition = label_prop_loop(
        &indptr,
        &indices,
        &edge_data,
        &mut labels,
        rng,
        n_label_prop_iter,
        approx_n_parts,
    );

    // Python keeps both `base_reduction_map` (unnormalized) and
    // `normalized_reduction_map` (L2 column-normalized); reduced_graph uses
    // `normalized_reduction_map.T @ graph @ base_reduction_map`.
    let base_reduction_map = partition_reduction_map(&partition);
    let mut normalized_reduction_map = base_reduction_map.clone();
    normalize_cols_l2(&mut normalized_reduction_map);

    // sklearn `normalize(X, norm="l1")` defaults to axis=1 (row L1).
    let mut data_reducer = normalized_reduction_map.transpose_view().to_csr();
    normalize_rows_l1(&mut data_reducer);

    let reduced_data = data.as_ref().map(|d| csr_matmul_dense(&data_reducer, d));

    let norm_t = normalized_reduction_map.transpose_view().to_csr();
    let temp = scipy_csr_matmul(&norm_t, &graph);
    let mut reduced_graph = scipy_csr_matmul(&temp, &base_reduction_map);
    for v in reduced_graph.data_mut() {
        *v = v.clamp(0.0, 1.0);
    }

    let reduced_init = if recursive_init {
        Some(label_propagation_init_with_options(
            &reduced_graph,
            n_label_prop_iter,
            n_embedding_epochs.min(255),
            approx_n_parts / 4,
            n_components,
            scaling,
            random_scale,
            noise_level,
            rng,
            reduced_data.as_ref(),
            true,
            base_init,
            base_init_threshold,
            upscaling,
        )?)
    } else {
        None
    };

    let reduced_layout = node_embedding(
        &reduced_graph,
        n_components,
        n_embedding_epochs,
        reduced_init,
        0.001 * n_embedding_epochs as f32,
        1.0,
        noise_level,
        rng,
        true,
    );

    // Reuse the same `normalized_reduction_map` as Python (do not rebuild after recursion).
    let result = match upscaling {
        Upscaling::PartitionExpander => {
            let sym = hadamard_symmetric(&graph);
            let mut data_expander = scipy_csr_matmul(&sym, &normalized_reduction_map);
            normalize_rows_l1(&mut data_expander);
            let mut part_norm = normalized_reduction_map.clone();
            normalize_rows_l1(&mut part_norm);
            let a = csr_matmul_dense(&data_expander, &reduced_layout);
            let b = csr_matmul_dense(&part_norm, &reduced_layout);
            let mut out = Array2::zeros((n_vertices, n_components));
            for i in 0..n_vertices {
                for c in 0..n_components {
                    out[[i, c]] = ((f64::from(a[[i, c]]) + f64::from(b[[i, c]])) * 0.5) as f32;
                }
            }
            out
        }
        Upscaling::JitterExpander => {
            let sym = hadamard_symmetric(&graph);
            let mut data_expander = scipy_csr_matmul(&sym, &normalized_reduction_map);
            normalize_rows_l1(&mut data_expander);
            let mut part_norm = normalized_reduction_map.clone();
            normalize_rows_l1(&mut part_norm);
            let expanded = {
                let a = csr_matmul_dense(&data_expander, &reduced_layout);
                let b = csr_matmul_dense(&part_norm, &reduced_layout);
                let mut e = Array2::zeros((n_vertices, n_components));
                for i in 0..n_vertices {
                    for c in 0..n_components {
                        e[[i, c]] = ((f64::from(a[[i, c]]) + f64::from(b[[i, c]])) * 0.5) as f32;
                    }
                }
                e
            };
            let mut jittered = Array2::zeros((n_vertices, n_components));
            for (i, &p) in partition.iter().enumerate() {
                if p >= 0 {
                    for c in 0..n_components {
                        jittered[[i, c]] = reduced_layout[[p as usize, c]];
                    }
                }
            }
            for v in jittered.iter_mut() {
                *v += rng.normal_scaled(f64::from(random_scale / 4.0)) as f32;
            }
            let mut combined = Array2::zeros((n_vertices, n_components));
            for i in 0..n_vertices {
                for c in 0..n_components {
                    combined[[i, c]] =
                        ((f64::from(expanded[[i, c]]) + f64::from(jittered[[i, c]])) * 0.5) as f32;
                }
            }
            combined
        }
        Upscaling::NodeEmbedding => {
            let mut result = Array2::zeros((n_vertices, n_components));
            for (i, &p) in partition.iter().enumerate() {
                if p >= 0 {
                    for c in 0..n_components {
                        result[[i, c]] = reduced_layout[[p as usize, c]];
                    }
                }
            }
            for v in result.iter_mut() {
                *v += rng.normal_scaled(f64::from(random_scale)) as f32;
            }
            result
        }
    };

    // Match NumPy: `(scaling * (result - result.mean(axis=0))).astype(np.float32)`
    let mut mean = vec![0.0f64; n_components];
    for c in 0..n_components {
        for r in 0..n_vertices {
            mean[c] += f64::from(result[[r, c]]);
        }
        mean[c] /= n_vertices as f64;
    }
    let mut out = Array2::<f32>::zeros(result.dim());
    for r in 0..n_vertices {
        for c in 0..n_components {
            out[[r, c]] = (f64::from(scaling) * (f64::from(result[[r, c]]) - mean[c])) as f32;
        }
    }
    Ok(out)
}
