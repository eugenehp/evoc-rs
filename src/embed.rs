//! UMAP-like node embedding (`evoc.node_embedding`).

use crate::numpy_rng::NumpyRandomState;
use ndarray::Array2;
use sprs::CsMat;

const INT32_MAX: i32 = i32::MAX - 1;

/// Per-edge training frequency from graph weights (Python: `make_epochs_per_sample`).
pub fn make_epochs_per_sample(weights: &[f32], n_epochs: usize) -> Vec<f32> {
    let n_epochs_f = n_epochs as f32;
    let max_w = weights.iter().copied().fold(0.0f32, f32::max);
    if max_w == 0.0 {
        return vec![n_epochs_f; weights.len()];
    }
    weights
        .iter()
        .map(|&w| {
            let n_samples = (n_epochs_f * (w / max_w)).max(1.0);
            n_epochs_f / n_samples
        })
        .collect()
}

#[inline]
pub fn rdist(x: &[f32], y: &[f32]) -> f32 {
    let mut result = 0.0f32;
    for i in 0..x.len().min(y.len()) {
        let diff = x[i] - y[i];
        result += diff * diff;
    }
    result
}

/// Match Python `int((n - epoch_neg) / rate)` used as `range(n_neg_samples)` bound.
#[inline]
fn n_neg_samples_count(n: f32, epoch_neg: f32, rate: f32) -> usize {
    let v = (n - epoch_neg) / rate;
    if v <= 0.0 {
        0
    } else {
        v as i32 as usize
    }
}

#[inline]
fn clip(val: f32, lo: f32, hi: f32) -> f32 {
    if val > hi {
        hi
    } else if val < lo {
        lo
    } else {
        val
    }
}

/// Fast non-reproducible SGD epoch (Python: `node_embedding_epoch`).
pub fn node_embedding_epoch(
    embedding: &mut Array2<f32>,
    head: &[u32],
    tail: &[u32],
    n_vertices: u32,
    epochs_per_sample: &[f32],
    rng_state: u32,
    dim: usize,
    alpha: f32,
    epochs_per_negative_sample: &[f32],
    epoch_of_next_negative_sample: &mut [f32],
    epoch_of_next_sample: &mut [f32],
    n: u8,
    noise_level: f32,
) {
    let n_edges = epochs_per_sample.len();
    let nrows = embedding.nrows();

    for i in 0..n_edges {
        if epoch_of_next_sample[i] > f32::from(n) {
            continue;
        }
        let j = head[i] as usize;
        let k = tail[i] as usize;
        if j >= nrows || k >= nrows {
            continue;
        }

        let mut dist_squared = 0.0f32;
        for d in 0..dim {
            let diff = embedding[[j, d]] - embedding[[k, d]];
            dist_squared += diff * diff;
        }
        if dist_squared > 0.0 {
            let dist = dist_squared.sqrt();
            let grad_coeff =
                (-2.0 * noise_level * dist - 2.0) / (2.0 * dist_squared - 0.5 * dist + 1.0);
            for d in 0..dim {
                let grad_d = grad_coeff * (embedding[[j, d]] - embedding[[k, d]]);
                embedding[[j, d]] += grad_d * alpha;
                embedding[[k, d]] -= grad_d * alpha;
            }
        }

        epoch_of_next_sample[i] += epochs_per_sample[i];

        let n_neg_samples = n_neg_samples_count(
            f32::from(n),
            epoch_of_next_negative_sample[i],
            epochs_per_negative_sample[i],
        );

        for p in 0..n_neg_samples {
            let neg_k =
                ((usize::from(n) + usize::from(p)) * i * rng_state as usize) % n_vertices as usize;
            if neg_k >= nrows {
                continue;
            }
            let mut ds = 0.0f32;
            for d in 0..dim {
                let diff = embedding[[j, d]] - embedding[[neg_k, d]];
                ds += diff * diff;
            }
            if ds > 1e-2 {
                let grad_coeff = 4.0 / ((1.0 + 0.25 * ds) * ds);
                for d in 0..dim {
                    let grad_d = clip(
                        grad_coeff * (embedding[[j, d]] - embedding[[neg_k, d]]),
                        -4.0,
                        4.0,
                    );
                    embedding[[j, d]] += grad_d * alpha;
                }
            }
        }
        epoch_of_next_negative_sample[i] += (n_neg_samples as f32) * epochs_per_negative_sample[i];
    }
}

/// Reproducible CSR-block SGD epoch (Python: `node_embedding_epoch_repr`).
pub fn node_embedding_epoch_repr(
    embedding: &mut Array2<f32>,
    csr_indptr: &[u32],
    csr_indices: &[u32],
    n_vertices: u32,
    epochs_per_sample: &[f32],
    rng_state: u32,
    dim: usize,
    alpha: f32,
    epochs_per_negative_sample: &[f32],
    epoch_of_next_negative_sample: &mut [f32],
    epoch_of_next_sample: &mut [f32],
    n: u8,
    noise_level: f32,
    gamma: f32,
    updates: &mut Array2<f32>,
    node_order: &mut [u32],
    block_size: u32,
) {
    let n_vertices = n_vertices as usize;
    let block_size = block_size as usize;
    let n_f = f32::from(n);
    let mut current = vec![0.0f32; dim];

    let mut block_start = 0usize;
    while block_start < n_vertices {
        let block_end = (block_start + block_size).min(n_vertices);

        for node_idx in block_start..block_end {
            let from_node = node_order[node_idx] as usize;
            if from_node >= n_vertices {
                continue;
            }
            // Numba snapshots the source row once per node (`current = embedding[from_node]`).
            for d in 0..dim {
                current[d] = embedding[[from_node, d]];
            }

            let row_start = csr_indptr[from_node] as usize;
            let row_end = csr_indptr[from_node + 1] as usize;

            for raw_index in row_start..row_end {
                if epoch_of_next_sample[raw_index] > n_f {
                    continue;
                }
                let to_node = csr_indices[raw_index] as usize;
                if to_node >= n_vertices {
                    continue;
                }

                let dist_squared = rdist(&current, embedding.row(to_node).as_slice().unwrap());
                if dist_squared > 0.0 {
                    let dist = dist_squared.sqrt();
                    let grad_coeff =
                        (-2.0 * noise_level * dist - 2.0) / (2.0 * dist_squared - 0.5 * dist + 1.0);
                    for d in 0..dim {
                        let grad_d = grad_coeff * (current[d] - embedding[[to_node, d]]);
                        updates[[from_node, d]] += grad_d * alpha;
                    }
                }

                epoch_of_next_sample[raw_index] += epochs_per_sample[raw_index];

                let n_neg_samples = n_neg_samples_count(
                    n_f,
                    epoch_of_next_negative_sample[raw_index],
                    epochs_per_negative_sample[raw_index],
                );

                for p in 0..n_neg_samples {
                    let neg_idx =
                        (raw_index * (usize::from(n) + usize::from(p) + 1) * rng_state as usize)
                            % n_vertices;
                    let to_neg = node_order[neg_idx] as usize;
                    if to_neg >= n_vertices {
                        continue;
                    }
                    let dist_squared = rdist(&current, embedding.row(to_neg).as_slice().unwrap());
                    if dist_squared > 1e-2 {
                        let grad_coeff = gamma * 4.0 / ((1.0 + 0.25 * dist_squared) * dist_squared);
                        if grad_coeff > 0.0 {
                            for d in 0..dim {
                                let grad_d = clip(
                                    grad_coeff * (current[d] - embedding[[to_neg, d]]),
                                    -4.0,
                                    4.0,
                                );
                                updates[[from_node, d]] += grad_d * alpha;
                            }
                        }
                    }
                }
                epoch_of_next_negative_sample[raw_index] +=
                    (n_neg_samples as f32) * epochs_per_negative_sample[raw_index];
            }
        }

        for node_idx in block_start..block_end {
            let from_node = node_order[node_idx] as usize;
            if from_node >= n_vertices {
                continue;
            }
            for d in 0..dim {
                embedding[[from_node, d]] += updates[[from_node, d]];
            }
        }

        block_start = block_end;
    }
}

/// Learn a low-dimensional graph embedding (Python: `node_embedding`).
#[allow(clippy::too_many_arguments)]
pub fn node_embedding(
    graph: &CsMat<f32>,
    n_components: usize,
    n_epochs: usize,
    initial_embedding: Option<Array2<f32>>,
    initial_alpha: f32,
    negative_sample_rate: f32,
    noise_level: f32,
    rng: &mut NumpyRandomState,
    reproducible_flag: bool,
) -> Array2<f32> {
    let graph = graph.to_csr();
    let n_vertices = graph.rows();

    let mut embedding = match initial_embedding {
        Some(e) => e,
        None => Array2::from_shape_fn((n_vertices, n_components), |_| {
            rng.normal_scaled(0.25) as f32
        }),
    };

    let weights: Vec<f32> = graph.data().to_vec();
    let epochs_per_sample = make_epochs_per_sample(&weights, n_epochs);
    let mut epochs_per_negative_sample: Vec<f32> = epochs_per_sample
        .iter()
        .map(|&e| e / negative_sample_rate)
        .collect();
    if reproducible_flag {
        for e in &mut epochs_per_negative_sample {
            *e *= 1.5;
        }
    }
    let mut epoch_of_next_negative_sample = epochs_per_negative_sample.clone();
    let mut epoch_of_next_sample = epochs_per_sample.clone();

    let mut head = Vec::with_capacity(graph.nnz());
    let mut tail = Vec::with_capacity(graph.nnz());
    for (_, (r, c)) in graph.iter() {
        head.push(r as u32);
        tail.push(c as u32);
    }

    let csr_indptr: Vec<u32> = graph
        .indptr()
        .raw_storage()
        .iter()
        .map(|&p| p as u32)
        .collect();
    let csr_indices: Vec<u32> = graph.indices().iter().map(|&i| i as u32).collect();

    let mut updates = Array2::zeros((n_vertices, n_components));
    let mut node_order: Vec<u32> = (0..n_vertices as u32).collect();
    let gamma_schedule: Vec<f32> = (0..n_epochs)
        .map(|n| {
            if n_epochs <= 1 {
                1.0
            } else {
                0.5 + (n as f32 / (n_epochs - 1) as f32) * 1.0
            }
        })
        .collect();

    let n_vertices_u32 = n_vertices as u32;
    let block_size = (1024usize).max(n_vertices / 8) as u32;
    let dim = n_components.min(255);
    let mut alpha = initial_alpha;

    // Python batches all epoch seeds before the loop, then shuffles after each epoch.
    let rng_vals: Vec<u32> = (0..n_epochs)
        .map(|_| rng.randint_high(INT32_MAX as i64) as u32)
        .collect();

    for n in 0..n_epochs {
        let rng_val = rng_vals[n];
        if !reproducible_flag {
            node_embedding_epoch(
                &mut embedding,
                &head,
                &tail,
                n_vertices_u32,
                &epochs_per_sample,
                rng_val,
                dim,
                alpha,
                &epochs_per_negative_sample,
                &mut epoch_of_next_negative_sample,
                &mut epoch_of_next_sample,
                n as u8,
                noise_level,
            );
        } else {
            node_embedding_epoch_repr(
                &mut embedding,
                &csr_indptr,
                &csr_indices,
                n_vertices_u32,
                &epochs_per_sample,
                rng_val,
                dim,
                alpha,
                &epochs_per_negative_sample,
                &mut epoch_of_next_negative_sample,
                &mut epoch_of_next_sample,
                n as u8,
                noise_level,
                gamma_schedule[n],
                &mut updates,
                &mut node_order,
                block_size,
            );
            let decay = ((1.0 - f64::from(alpha)).powi(2) * 0.5) as f32;
            updates.mapv_inplace(|v| v * decay);
            rng.shuffle(&mut node_order);
        }
        alpha = initial_alpha * (1.0 - (n as f32 / n_epochs as f32));
    }

    embedding
}
