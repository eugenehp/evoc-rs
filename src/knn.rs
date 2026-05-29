//! k-NN graph construction (`evoc.knn_graph`).

use crate::nndescent::{
    make_float_forest, make_int8_forest, make_uint8_forest, nn_descent_float,
    nn_descent_float_sorted, nn_descent_int8_sorted, nn_descent_uint8_sorted,
};
use crate::numpy_rng::NumpyRandomState;
use ndarray::{Array2, Axis};
use thiserror::Error;

const INT32_MIN: i32 = i32::MIN + 1;
const INT32_MAX: i32 = i32::MAX - 1;

/// Embedding matrix variants supported by `knn_graph`.
#[derive(Clone, Debug)]
pub enum EmbeddingData {
    Float32(Array2<f32>),
    Int8(Array2<i8>),
    UInt8(Array2<u8>),
}

impl EmbeddingData {
    pub fn nrows(&self) -> usize {
        match self {
            EmbeddingData::Float32(d) => d.nrows(),
            EmbeddingData::Int8(d) => d.nrows(),
            EmbeddingData::UInt8(d) => d.nrows(),
        }
    }
}

#[derive(Debug, Error)]
pub enum KnnError {
    #[error("empty embedding matrix")]
    EmptyData,
    #[error("n_neighbors must be positive")]
    InvalidNeighbors,
    #[cfg(feature = "cluster")]
    #[error(transparent)]
    Backend(#[from] crate::rlx_backend::BackendError),
}

#[derive(Clone, Debug)]
pub struct KnnGraphOptions {
    pub n_neighbors: usize,
    pub n_trees: Option<usize>,
    pub leaf_size: Option<usize>,
    pub max_candidates: Option<usize>,
    pub max_rptree_depth: i64,
    pub n_iters: Option<usize>,
    pub delta: f32,
    pub delta_improv: f32,
    pub use_sorted_updates: bool,
    /// Brute-force kNN when `n_samples` is at most this value (int8/uint8 only). `0` disables (Python default).
    pub brute_force_max_n: usize,
    /// Single-thread NN-descent for bit-exact parity with Python when `random_state` is set.
    pub deterministic: bool,
}

impl Default for KnnGraphOptions {
    fn default() -> Self {
        Self {
            n_neighbors: 30,
            n_trees: None,
            leaf_size: None,
            max_candidates: None,
            max_rptree_depth: 200,
            n_iters: None,
            delta: 0.001,
            delta_improv: 0.001,
            use_sorted_updates: true,
            brute_force_max_n: 0,
            deterministic: false,
        }
    }
}

fn normalize_float_rows(mut data: Array2<f32>) -> Array2<f32> {
    for mut row in data.axis_iter_mut(Axis(0)) {
        let mut norm = 0.0f32;
        for v in row.iter() {
            norm += v * v;
        }
        norm = norm.sqrt();
        if norm == 0.0 {
            norm = 1.0;
        }
        for v in row.iter_mut() {
            *v /= norm;
        }
    }
    data
}

fn prepare_float(data: Array2<f32>) -> Array2<f32> {
    let mut all_unit = true;
    for row in data.axis_iter(Axis(0)) {
        let mut norm = 0.0f32;
        for v in row.iter() {
            norm += v * v;
        }
        norm = norm.sqrt();
        if (norm - 1.0).abs() > 1e-5 {
            all_unit = false;
            break;
        }
    }
    if all_unit {
        data.mapv(|x| x as f32)
    } else {
        normalize_float_rows(data)
    }
}

fn make_forest(
    data: &EmbeddingData,
    n_neighbors: usize,
    n_trees: usize,
    leaf_size: usize,
    rng: &mut NumpyRandomState,
    max_depth: i64,
) -> Option<Array2<i32>> {
    let leaf_size = leaf_size.max(10).max(n_neighbors);
    let mut rng_states = Array2::<i64>::zeros((n_trees, 3));
    for i in 0..n_trees {
        for j in 0..3 {
            rng_states[[i, j]] = rng.randint(INT32_MIN as i64, INT32_MAX as i64);
        }
    }

    let forests = match data {
        EmbeddingData::Float32(d) => make_float_forest(d, &rng_states, leaf_size, max_depth),
        EmbeddingData::Int8(d) => make_int8_forest(d, &rng_states, leaf_size, max_depth),
        EmbeddingData::UInt8(d) => make_uint8_forest(d, &rng_states, leaf_size, max_depth),
    };

    if forests.is_empty() {
        return None;
    }

    let max_leaf = forests.iter().map(|a| a.ncols()).max().unwrap_or(0);
    let n_rows: usize = forests.iter().map(|a| a.nrows()).sum();
    let mut leaf_array = Array2::<i32>::from_elem((n_rows, max_leaf), -1);
    let mut row = 0usize;
    for forest in forests {
        for i in 0..forest.nrows() {
            for j in 0..forest.ncols() {
                leaf_array[[row, j]] = forest[[i, j]];
            }
            row += 1;
        }
    }
    Some(leaf_array)
}

pub fn transform_distances_float(mut distances: Array2<f32>) -> Array2<f32> {
    distances.mapv_inplace(|d| {
        if d >= 0.0 {
            0.0
        } else {
            // Match NumPy: np.maximum(-np.log2(-d), 0.0) (log2 in float64, then cast).
            let sim = (-d) as f64;
            if sim >= 1.0 {
                0.0
            } else {
                (-sim.log2()).max(0.0) as f32
            }
        }
    });
    distances
}

fn transform_distances_int8(mut distances: Array2<f32>) -> Array2<f32> {
    distances.mapv_inplace(|d| if d >= 0.0 { 0.0 } else { 1.0 / (-d) });
    distances
}

fn transform_distances_uint8(mut distances: Array2<f32>) -> Array2<f32> {
    distances.mapv_inplace(|d| {
        if d >= 0.0 {
            0.0
        } else {
            (-(-d as f64).log2()) as f32
        }
    });
    distances
}

fn brute_force_knn(data: &EmbeddingData, n_neighbors: usize) -> (Array2<i32>, Array2<f32>) {
    let n = match data {
        EmbeddingData::Float32(d) => d.nrows(),
        EmbeddingData::Int8(d) => d.nrows(),
        EmbeddingData::UInt8(d) => d.nrows(),
    };
    let k = n_neighbors.min(n.saturating_sub(1)).max(1);
    let mut indices = Array2::<i32>::from_elem((n, k), -1);
    let mut distances = Array2::<f32>::from_elem((n, k), f32::INFINITY);

    for i in 0..n {
        let mut heap_d = vec![f32::INFINITY; k];
        let mut heap_i = vec![-1i32; k];

        for j in 0..n {
            if i == j {
                continue;
            }
            let d = match data {
                EmbeddingData::Float32(mat) => {
                    let row_i = mat.row(i);
                    let row_j = mat.row(j);
                    let mut dot = 0.0f32;
                    for t in 0..row_i.len() {
                        dot += row_i[t] * row_j[t];
                    }
                    if dot > 0.0 {
                        -dot
                    } else {
                        f32::MIN_POSITIVE
                    }
                }
                EmbeddingData::Int8(mat) => {
                    let row_i = mat.row(i);
                    let row_j = mat.row(j);
                    let mut acc = 0i32;
                    for t in 0..row_i.len() {
                        acc += (row_i[t] as i32) * (row_j[t] as i32);
                    }
                    -(acc as f32)
                }
                EmbeddingData::UInt8(mat) => {
                    let row_i = mat.row(i);
                    let row_j = mat.row(j);
                    let mut inter = 0u32;
                    let mut union = 0u32;
                    for t in 0..row_i.len() {
                        inter += (row_i[t] & row_j[t]).count_ones();
                        union += (row_i[t] | row_j[t]).count_ones();
                    }
                    if union > 0 {
                        -(inter as f32 / union as f32)
                    } else {
                        0.0
                    }
                }
            };

            if d < heap_d[0] {
                heap_d[0] = d;
                heap_i[0] = j as i32;
                let mut pos = 0usize;
                loop {
                    let left = pos * 2 + 1;
                    if left >= k {
                        break;
                    }
                    let right = left + 1;
                    let mut swap = pos;
                    if heap_d[swap] < heap_d[left] {
                        swap = left;
                    }
                    if right < k && heap_d[swap] < heap_d[right] {
                        swap = right;
                    }
                    if swap == pos {
                        break;
                    }
                    heap_d.swap(pos, swap);
                    heap_i.swap(pos, swap);
                    pos = swap;
                }
            }
        }

        for t in 0..k {
            indices[[i, t]] = heap_i[t];
            distances[[i, t]] = heap_d[t];
        }
    }

    (indices, distances)
}

fn nn_descent(
    data: &EmbeddingData,
    n_neighbors: usize,
    rng_state: &mut [i64; 3],
    max_candidates: usize,
    n_iters: usize,
    delta: f32,
    delta_improv: f32,
    leaf_array: Option<&Array2<i32>>,
    use_sorted: bool,
) -> (Array2<i32>, Array2<f32>) {
    let delta_improv_opt = Some(delta_improv);
    match data {
        EmbeddingData::Float32(d) => {
            if use_sorted {
                nn_descent_float_sorted(
                    d,
                    n_neighbors,
                    rng_state,
                    max_candidates,
                    n_iters,
                    delta,
                    delta_improv_opt,
                    leaf_array,
                )
            } else {
                nn_descent_float(
                    d,
                    n_neighbors,
                    rng_state,
                    max_candidates,
                    n_iters,
                    delta,
                    delta_improv_opt,
                    leaf_array,
                )
            }
        }
        EmbeddingData::Int8(d) => nn_descent_int8_sorted(
            d,
            n_neighbors,
            rng_state,
            max_candidates,
            n_iters,
            delta,
            delta_improv_opt,
            leaf_array,
        ),
        EmbeddingData::UInt8(d) => nn_descent_uint8_sorted(
            d,
            n_neighbors,
            rng_state,
            max_candidates,
            n_iters,
            delta,
            delta_improv_opt,
            leaf_array,
        ),
    }
}

/// Build a k-nearest neighbor graph (defaults match Python `knn_graph`).
pub fn knn_graph(
    data: EmbeddingData,
    options: KnnGraphOptions,
    rng: &mut NumpyRandomState,
) -> Result<(Array2<i32>, Array2<f32>), KnnError> {
    if options.deterministic {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("rayon thread pool");
        pool.install(|| knn_graph_impl(data, options, rng))
    } else {
        knn_graph_impl(data, options, rng)
    }
}

/// Like [`knn_graph`] but avoids taking ownership of the input matrix.
///
/// This is useful for huge datasets where extra clones are expensive; internally this will only
/// clone if it needs to renormalize float rows.
pub fn knn_graph_ref(
    data: &EmbeddingData,
    options: KnnGraphOptions,
    rng: &mut NumpyRandomState,
) -> Result<(Array2<i32>, Array2<f32>), KnnError> {
    match data {
        EmbeddingData::Float32(d) => knn_graph(EmbeddingData::Float32(d.clone()), options, rng),
        EmbeddingData::Int8(d) => knn_graph(EmbeddingData::Int8(d.clone()), options, rng),
        EmbeddingData::UInt8(d) => knn_graph(EmbeddingData::UInt8(d.clone()), options, rng),
    }
}

fn knn_graph_impl(
    data: EmbeddingData,
    options: KnnGraphOptions,
    rng: &mut NumpyRandomState,
) -> Result<(Array2<i32>, Array2<f32>), KnnError> {
    if options.n_neighbors == 0 {
        return Err(KnnError::InvalidNeighbors);
    }

    let n_samples = match &data {
        EmbeddingData::Float32(d) => d.nrows(),
        EmbeddingData::Int8(d) => d.nrows(),
        EmbeddingData::UInt8(d) => d.nrows(),
    };
    if n_samples == 0 {
        return Err(KnnError::EmptyData);
    }

    let _strict_cosine = crate::fast_cosine::strict_cosine_guard(options.deterministic);

    let prepared = match data {
        EmbeddingData::Float32(d) => EmbeddingData::Float32(prepare_float(d)),
        other => other,
    };

    let n_neighbors = options.n_neighbors.min(n_samples.saturating_sub(1)).max(1);
    // Python: max(4, min(8, numba.get_num_threads())); goldens use NUMBA_NUM_THREADS=1 → 4 trees.
    let n_trees = options.n_trees.unwrap_or_else(|| {
        let t = rayon::current_num_threads();
        t.max(4).min(8)
    });
    let leaf_size = options.leaf_size.unwrap_or_else(|| 10.max(n_neighbors));
    let n_iters = options
        .n_iters
        .unwrap_or_else(|| 5.max((n_samples as f64).log2().round() as usize));
    let max_candidates = options
        .max_candidates
        .unwrap_or_else(|| 60.min((n_neighbors as f32 * 1.5) as usize));

    let mut rng_state = rng.randint3_for_tau();

    let (indices, distances) = if n_samples <= options.brute_force_max_n
        && !matches!(prepared, EmbeddingData::Float32(_))
    {
        let (idx, dist) = brute_force_knn(&prepared, n_neighbors);
        (idx, dist)
    } else {
        let leaf_array = make_forest(
            &prepared,
            n_neighbors,
            n_trees,
            leaf_size,
            rng,
            options.max_rptree_depth,
        );
        let leaf_ref = leaf_array.as_ref();
        nn_descent(
            &prepared,
            n_neighbors,
            &mut rng_state,
            max_candidates,
            n_iters,
            options.delta,
            options.delta_improv,
            leaf_ref,
            options.use_sorted_updates,
        )
    };

    let indices = indices;
    let distances = match &prepared {
        EmbeddingData::Float32(_) => transform_distances_float(distances),
        EmbeddingData::Int8(_) => transform_distances_int8(distances),
        EmbeddingData::UInt8(_) => transform_distances_uint8(distances),
    };

    Ok((indices, distances))
}

#[cfg(all(test, feature = "npy"))]
mod tests {
    use super::*;
    use crate::numpy_rng::check_random_state;
    use ndarray_npy::ReadNpyExt;
    use std::fs::File;
    use std::path::PathBuf;

    #[test]
    fn pre_transform_distances_are_negative() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/medium_800");
        if !dir.exists() {
            return;
        }
        let data: Array2<f32> = {
            let mut f = File::open(dir.join("data.npy")).unwrap();
            Array2::read_npy(&mut f).unwrap()
        };
        let prepared = EmbeddingData::Float32(prepare_float(data));
        let mut rng = check_random_state(Some(42));
        let knn_opts = KnnGraphOptions {
            n_neighbors: 15,
            ..Default::default()
        };
        let n_samples = match &prepared {
            EmbeddingData::Float32(d) => d.nrows(),
            _ => 0,
        };
        let n_neighbors = knn_opts.n_neighbors.min(n_samples.saturating_sub(1)).max(1);
        let n_trees = 4;
        let leaf_size = 10.max(n_neighbors);
        let n_iters = 5.max((n_samples as f64).log2().round() as usize);
        let max_candidates = 60.min((n_neighbors as f32 * 1.5) as usize);
        let mut rng_state = rng.randint3_for_tau();
        let leaf_array = make_forest(
            &prepared,
            n_neighbors,
            n_trees,
            leaf_size,
            &mut rng,
            knn_opts.max_rptree_depth,
        );
        let (_, raw) = nn_descent(
            &prepared,
            n_neighbors,
            &mut rng_state,
            max_candidates,
            n_iters,
            knn_opts.delta,
            knn_opts.delta_improv,
            leaf_array.as_ref(),
            knn_opts.use_sorted_updates,
        );
        let min_d = raw.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_d = raw.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        eprintln!(
            "raw heap dist range [{min_d}, {max_d}] sample: {:?}",
            &raw.row(218).to_vec()
        );
        assert!(
            max_d <= 0.0,
            "heap distances should be non-positive, max={max_d}"
        );
    }
}
