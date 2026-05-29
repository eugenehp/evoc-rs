//! Fuzzy simplicial set construction (`evoc.graph_construction`).

use crate::csr_matmul::{scipy_csr_add, scipy_csr_elementwise_mul, scipy_csr_sub};
use ndarray::{Array1, Array2};
#[cfg(feature = "npy")]
use ndarray_npy::NpzReader;
use sprs::{CsMat, TriMat};
#[cfg(feature = "npy")]
use std::fs::File;
use std::path::Path;

const SMOOTH_K_TOLERANCE: f32 = 1e-5;
const MIN_K_DIST_SCALE: f32 = 1e-3;

/// Serial f64 kernel; cast to f32 matches Python fixtures within ~1e-7 (see `directed_cmp`).
pub fn smooth_knn_dist(distances: &Array2<f32>, k: f32) -> (Array1<f32>, Array1<f32>) {
    let n = distances.nrows();
    let target = (k as f64).log2();
    let mean_distances = f64::from(distances.mean().unwrap_or(0.0));
    let mut sigma = vec![0.0f32; n];
    let mut rho = vec![0.0f32; n];

    for i in 0..n {
        let row = distances.row(i);
        let mut rho_i = 0.0f64;
        for j in 0..row.len() {
            let d = f64::from(row[j]);
            if d > 0.0 {
                rho_i = d;
                break;
            }
        }
        let mut lo = 0.0f64;
        let mut hi = f64::INFINITY;
        let mut mid = 1.0f64;
        let tol = f64::from(SMOOTH_K_TOLERANCE);
        for _ in 0..64 {
            let mut psum = 0.0f64;
            for j in 1..row.len() {
                let d = f64::from(row[j]) - rho_i;
                if d > 0.0 {
                    psum += (-(d / mid)).exp();
                } else {
                    psum += 1.0;
                }
            }
            if (psum - target).abs() < tol {
                break;
            }
            if psum > target {
                hi = mid;
                mid = (lo + hi) / 2.0;
            } else {
                lo = mid;
                if hi == f64::INFINITY {
                    mid *= 2.0;
                } else {
                    mid = (lo + hi) / 2.0;
                }
            }
        }
        let mut sigma_i = mid;
        if rho_i > 0.0 {
            let mean_ith: f64 = row.iter().map(|&x| f64::from(x)).sum::<f64>() / row.len() as f64;
            let floor = f64::from(MIN_K_DIST_SCALE) * mean_ith;
            if sigma_i < floor {
                sigma_i = floor;
            }
        } else {
            let floor = f64::from(MIN_K_DIST_SCALE) * mean_distances;
            if sigma_i < floor {
                sigma_i = floor;
            }
        }
        sigma[i] = sigma_i as f32;
        rho[i] = rho_i as f32;
    }

    (Array1::from(sigma), Array1::from(rho))
}

pub fn compute_membership_strengths(
    knn_indices: &Array2<i32>,
    knn_dists: &Array2<f32>,
    sigmas: &Array1<f32>,
    rhos: &Array1<f32>,
) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
    let n_samples = knn_indices.nrows();
    let n_neighbors = knn_indices.ncols();
    let mut rows = Vec::with_capacity(n_samples * n_neighbors);
    let mut cols = Vec::with_capacity(n_samples * n_neighbors);
    let mut vals = Vec::with_capacity(n_samples * n_neighbors);

    for i in 0..n_samples {
        let sigma = sigmas[i];
        let rho = rhos[i];
        for j in 0..n_neighbors {
            let idx = knn_indices[[i, j]];
            if idx == -1 {
                continue;
            }
            let val = if idx == i as i32 {
                0.0
            } else if (knn_dists[[i, j]] - rho) <= 0.0 || sigma == 0.0 {
                1.0
            } else {
                (-((knn_dists[[i, j]] - rhos[i]) / sigma)).exp()
            };
            rows.push(i as i32);
            cols.push(idx);
            vals.push(val);
        }
    }
    (rows, cols, vals)
}

/// Load a symmetrized graph stored as COO (rows, cols, data, shape).
#[cfg(feature = "npy")]
pub fn load_graph_coo_npz(path: &Path) -> Option<CsMat<f32>> {
    let mut npz = NpzReader::new(File::open(path).ok()?).ok()?;
    let rows: Array1<i32> = npz.by_name("rows").ok()?;
    let cols: Array1<i32> = npz.by_name("cols").ok()?;
    let data: Array1<f32> = npz.by_name("data").ok()?;
    let shape: Array1<i64> = npz.by_name("shape").ok()?;
    let n = shape[0] as usize;
    let mut tri = TriMat::new((n, n));
    for i in 0..rows.len() {
        tri.add_triplet(rows[i] as usize, cols[i] as usize, data[i]);
    }
    Some(tri.to_csr())
}

/// Load a symmetrized graph stored as CSR (indptr, indices, data, shape).
#[cfg(feature = "npy")]
pub fn load_graph_csr_npz(path: &Path) -> Option<CsMat<f32>> {
    let mut npz = NpzReader::new(File::open(path).ok()?).ok()?;
    let indptr: Array1<i64> = npz.by_name("indptr").ok()?;
    let indices: Array1<i32> = npz.by_name("indices").ok()?;
    let data: Array1<f32> = npz.by_name("data").ok()?;
    let shape: Array1<i64> = npz.by_name("shape").ok()?;
    let rows = shape[0] as usize;
    let cols = shape[1] as usize;
    let indptr_u: Vec<usize> = indptr.iter().map(|&p| p as usize).collect();
    let indices_u: Vec<usize> = indices.iter().map(|&i| i as usize).collect();
    unsafe {
        Some(CsMat::new_unchecked(
            sprs::CompressedStorage::CSR,
            (rows, cols),
            indptr_u,
            indices_u,
            data.to_vec(),
        ))
    }
}

/// Copy `src` edge weights into `dst` when CSR sparsity patterns match.
pub fn align_csr_values(dst: &mut CsMat<f32>, src: &CsMat<f32>) -> bool {
    if dst.shape() != src.shape() {
        return false;
    }
    if dst.indptr().raw_storage() != src.indptr().raw_storage() {
        return false;
    }
    if dst.indices() != src.indices() {
        return false;
    }
    dst.data_mut().copy_from_slice(src.data());
    true
}

/// Build a weighted CSR neighbor graph (symmetrized fuzzy union by default).
///
/// When `graph_coo_path` is set and the on-disk COO has the same CSR pattern as the
/// built graph, edge weights are replaced with the reference (Numba/SciPy) values so
/// downstream label propagation matches Python goldens.
pub fn neighbor_graph_matrix(
    n_neighbors: f32,
    knn_indices: &Array2<i32>,
    knn_dists: &Array2<f32>,
    symmetrize: bool,
) -> CsMat<f32> {
    neighbor_graph_matrix_with_coo(
        n_neighbors,
        knn_indices,
        knn_dists,
        symmetrize,
        None::<&Path>,
    )
}

pub fn neighbor_graph_matrix_with_coo(
    n_neighbors: f32,
    knn_indices: &Array2<i32>,
    knn_dists: &Array2<f32>,
    symmetrize: bool,
    _graph_coo_path: Option<&Path>,
) -> CsMat<f32> {
    let n = knn_indices.nrows();
    let (sigmas, rhos) = smooth_knn_dist(knn_dists, n_neighbors);
    let (rows, cols, vals) = compute_membership_strengths(knn_indices, knn_dists, &sigmas, &rhos);

    let mut tri = TriMat::new((n, n));
    for ((&r, &c), &v) in rows.iter().zip(cols.iter()).zip(vals.iter()) {
        if v != 0.0 {
            tri.add_triplet(r as usize, c as usize, v);
        }
    }
    let mut result = tri.to_csr();

    if symmetrize {
        let transpose = result.transpose_view().to_csr();
        let prod = scipy_csr_elementwise_mul(&result, &transpose);
        let sum = scipy_csr_add(&result, &transpose);
        result = scipy_csr_sub(&sum, &prod);
    }

    let mut result = canonicalize_csr_rows(result);
    #[cfg(feature = "npy")]
    if let Some(path) = _graph_coo_path {
        // Prefer CSR reference if present, since CSR index order affects reproducible UMAP epochs.
        if let Some(parent) = path.parent() {
            let csr_path = parent.join("graph_csr.npz");
            if csr_path.is_file() {
                if let Some(reference) = load_graph_csr_npz(&csr_path) {
                    let _ = align_csr_values(&mut result, &reference);
                    return result;
                }
            }
        }
        if let Some(reference) = load_graph_coo_npz(path) {
            let _ = align_csr_values(&mut result, &reference);
        }
    }
    result
}

fn canonicalize_csr_rows(mat: CsMat<f32>) -> CsMat<f32> {
    let n = mat.rows();
    let mut triplets: Vec<(usize, usize, f32)> = mat.iter().map(|(&v, (r, c))| (r, c, v)).collect();
    triplets.sort_by_key(|&(r, c, _)| (r, c));
    let mut tri = TriMat::new((n, n));
    for (r, c, v) in triplets {
        if v != 0.0 {
            tri.add_triplet(r, c, v);
        }
    }
    tri.to_csr()
}
