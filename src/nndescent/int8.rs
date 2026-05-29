//! Int8 NN-descent (port of `evoc.int8_nndescent`, sorted update path).

use crate::heap::{
    apply_sorted_graph_updates, build_candidates, deheap_sort, flagged_heap_push, make_heap,
    GraphHeap, SendMutPtr, INF,
};
use crate::rng::{offset_state, tau_rand_int, tau_rand_mod};
use ndarray::parallel::prelude::*;
use ndarray::{s, Array2, Array3, ArrayView1, Axis};

const EPS: f32 = 1e-8;

/// Cosine similarity for NN-descent heap (negative dot product, or `NEG_INF_SENTINEL`).
#[inline]
pub fn fast_int_inner_product_dissimilarity(x: ArrayView1<i8>, y: ArrayView1<i8>) -> f32 {
    let mut result = 0i32;
    for i in 0..x.len() {
        result += (x[i] as i32) * (y[i] as i32);
    }
    -(result as f32)
}

/// Split indices by a random hyperplane (quantized cosine margin).
pub fn int8_random_projection_split(
    data: &Array2<i8>,
    indices: &[i32],
    rng_state: &mut [i64; 3],
) -> (Vec<i32>, Vec<i32>) {
    let dim = data.ncols();
    let indices_size = indices.len() as u32;

    let isize = indices_size as i32;
    let left_index = tau_rand_mod(rng_state, isize) as u32;
    let mut right_index = tau_rand_mod(rng_state, isize) as u32;
    right_index += (left_index == right_index) as u32;
    right_index %= indices_size;

    let left = indices[left_index as usize] as usize;
    let right = indices[right_index as usize] as usize;
    let left_data = data.row(left);
    let right_data = data.row(right);

    let mut left_norm = 0.0f32;
    let mut right_norm = 0.0f32;
    for d in 0..dim {
        let lv = left_data[d] as f32;
        let rv = right_data[d] as f32;
        left_norm += lv * lv;
        right_norm += rv * rv;
    }
    left_norm = left_norm.sqrt();
    right_norm = right_norm.sqrt();

    let mut hyperplane_vector = vec![0.0f32; dim];
    let mut hyperplane_norm = 0.0f32;
    for d in 0..dim {
        let lv = if left_norm > 0.0 {
            left_data[d] as f32 / left_norm
        } else {
            0.0
        };
        let rv = if right_norm > 0.0 {
            right_data[d] as f32 / right_norm
        } else {
            0.0
        };
        hyperplane_vector[d] = lv - rv;
        hyperplane_norm += hyperplane_vector[d] * hyperplane_vector[d];
    }
    hyperplane_norm = hyperplane_norm.sqrt();
    if hyperplane_norm.abs() < EPS {
        hyperplane_norm = 1.0;
    }
    for d in 0..dim {
        hyperplane_vector[d] /= hyperplane_norm;
    }

    let max_size = indices.len();
    let mut temp_left = vec![0i32; max_size];
    let mut temp_right = vec![0i32; max_size];
    let mut n_left = 0usize;
    let mut n_right = 0usize;

    for (idx, &point_idx) in indices.iter().enumerate() {
        let mut local_rng = offset_state(rng_state, idx as i64);
        let test_data = data.row(point_idx as usize);
        let mut margin = 0.0f32;
        for d in 0..dim {
            margin += hyperplane_vector[d] * (test_data[d] as f32);
        }
        let classification = if margin.abs() < EPS {
            tau_rand_mod(&mut local_rng, 2) as u8
        } else if margin > 0.0 {
            0
        } else {
            1
        };
        if classification == 0 {
            temp_left[n_left] = point_idx;
            n_left += 1;
        } else {
            temp_right[n_right] = point_idx;
            n_right += 1;
        }
    }

    if n_left == 0 || n_right == 0 {
        n_left = 0;
        n_right = 0;
        for &point_idx in indices {
            let classification = tau_rand_mod(rng_state, 2) as u8;
            if classification == 0 {
                temp_left[n_left] = point_idx;
                n_left += 1;
            } else {
                temp_right[n_right] = point_idx;
                n_right += 1;
            }
        }
    }

    (temp_left[..n_left].to_vec(), temp_right[..n_right].to_vec())
}

pub fn make_int8_tree(
    data: &Array2<i8>,
    indices: &[i32],
    point_indices: &mut Vec<Vec<i32>>,
    rng_state: &mut [i64; 3],
    leaf_size: usize,
    max_depth: i64,
) {
    if indices.len() > leaf_size && max_depth > 0 {
        let (left_indices, right_indices) = int8_random_projection_split(data, indices, rng_state);
        make_int8_tree(
            data,
            &left_indices,
            point_indices,
            rng_state,
            leaf_size,
            max_depth - 1,
        );
        make_int8_tree(
            data,
            &right_indices,
            point_indices,
            rng_state,
            leaf_size,
            max_depth - 1,
        );
    } else {
        point_indices.push(indices.to_vec());
    }
}

fn make_int8_leaf_array_serial(
    data: &Array2<i8>,
    rng_state: &mut [i64; 3],
    leaf_size: usize,
    max_depth: i64,
) -> Array2<i32> {
    let n_points = data.nrows();
    let indices: Vec<i32> = (0..n_points as i32).collect();
    let mut point_indices: Vec<Vec<i32>> = Vec::new();

    make_int8_tree(
        data,
        &indices,
        &mut point_indices,
        rng_state,
        leaf_size,
        max_depth,
    );

    let n_leaves = point_indices.len();
    let mut max_leaf_size = leaf_size as i32;
    for points in &point_indices {
        max_leaf_size = max_leaf_size.max(points.len() as i32);
    }

    let mut result = Array2::from_elem((n_leaves, max_leaf_size as usize), -1);
    for (i, points) in point_indices.iter().enumerate() {
        for (j, &p) in points.iter().enumerate() {
            result[[i, j]] = p;
        }
    }
    result
}

/// One RP-tree leaf array per RNG state (serial tree build per thread).
pub fn make_int8_forest(
    data: &Array2<i8>,
    rng_states: &Array2<i64>,
    leaf_size: usize,
    max_depth: i64,
) -> Vec<Array2<i32>> {
    (0..rng_states.nrows())
        .into_par_iter()
        .map(|i| {
            let mut state = [rng_states[[i, 0]], rng_states[[i, 1]], rng_states[[i, 2]]];
            make_int8_leaf_array_serial(data, &mut state, leaf_size, max_depth)
        })
        .collect()
}

fn generate_leaf_updates_int8(
    updates: &mut Array3<f32>,
    n_updates_per_thread: &mut [i32],
    leaf_block: &Array2<i32>,
    dist_thresholds: &[f32],
    data: &Array2<i8>,
    n_threads: usize,
) {
    let block_size = leaf_block.nrows();
    let rows_per_thread = block_size / n_threads + 1;
    let max_leaf = leaf_block.ncols();

    let updates_ptr = SendMutPtr::new(updates.as_mut_ptr());
    let n_updates_ptr = SendMutPtr::new(n_updates_per_thread.as_mut_ptr());
    let updates_stride = updates.len_of(Axis(1)) * 3;

    (0..n_threads).into_par_iter().for_each(|t| {
        let mut idx = 0usize;
        for r in 0..rows_per_thread {
            let n = t * rows_per_thread + r;
            if n >= block_size {
                break;
            }

            for i in 0..max_leaf {
                let p = leaf_block[[n, i]];
                if p < 0 {
                    break;
                }
                let data_p = data.row(p as usize);
                unsafe {
                    let base = updates_ptr.as_ptr().add(t * updates_stride + idx * 3);
                    *base = p as f32;
                    *base.add(1) = p as f32;
                    *base.add(2) = -1.0;
                }
                idx += 1;

                for j in (i + 1)..max_leaf {
                    let q = leaf_block[[n, j]];
                    if q < 0 {
                        break;
                    }
                    let d = fast_int_inner_product_dissimilarity(data_p, data.row(q as usize));
                    let max_threshold =
                        dist_thresholds[p as usize].max(dist_thresholds[q as usize]);
                    if d < max_threshold {
                        unsafe {
                            let base = updates_ptr.as_ptr().add(t * updates_stride + idx * 3);
                            *base = p as f32;
                            *base.add(1) = q as f32;
                            *base.add(2) = d;
                        }
                        idx += 1;
                    }
                }
            }
        }
        unsafe {
            *n_updates_ptr.as_ptr().add(t) = idx as i32;
        }
    });
}

pub fn init_rp_tree_int8(
    data: &Array2<i8>,
    current_graph: &mut GraphHeap,
    leaf_array: &Array2<i32>,
    n_threads: usize,
) {
    let n_leaves = leaf_array.nrows();
    let block_size = n_threads * 64;
    let n_blocks = n_leaves / block_size;
    let max_leaf_size = leaf_array.ncols();
    let updates_per_thread =
        (block_size * max_leaf_size * (max_leaf_size.saturating_sub(1)) / (2 * n_threads)) + 1;

    let mut updates = Array3::zeros((n_threads, updates_per_thread, 3));
    let mut n_updates_per_thread = vec![0i32; n_threads];
    let n_vertices = current_graph.0.nrows();
    let vertex_block_size = n_vertices / n_threads + 1;
    let ncols = current_graph.0.ncols();

    let indices_ptr = SendMutPtr::new(current_graph.0.as_mut_ptr());
    let distances_ptr = SendMutPtr::new(current_graph.1.as_mut_ptr());
    let flags_ptr = SendMutPtr::new(current_graph.2.as_mut_ptr());

    for i in 0..=n_blocks {
        let block_start = i * block_size;
        let block_end = ((i + 1) * block_size).min(n_leaves);

        if block_start >= block_end {
            continue;
        }

        let leaf_block = leaf_array.slice(s![block_start..block_end, ..]);
        let dist_thresholds: Vec<f32> = (0..n_vertices).map(|v| current_graph.1[[v, 0]]).collect();

        generate_leaf_updates_int8(
            &mut updates,
            &mut n_updates_per_thread,
            &leaf_block.to_owned(),
            &dist_thresholds,
            data,
            n_threads,
        );

        let updates_stride = updates.len_of(Axis(1)) * 3;
        let updates_ptr = SendMutPtr::new(updates.as_ptr() as *mut f32);

        (0..n_threads).into_par_iter().for_each(|t| {
            let vb_start = t * vertex_block_size;
            let vb_end = (vb_start + vertex_block_size).min(n_vertices);

            for j in 0..n_threads {
                let count = n_updates_per_thread[j] as usize;
                for k in 0..count {
                    unsafe {
                        let base = updates_ptr.as_ptr().add(j * updates_stride + k * 3);
                        let p = *base as i32;
                        if p == -1 {
                            continue;
                        }
                        let q = *base.add(1) as i32;
                        let d = *base.add(2);

                        if (p as usize) >= vb_start && (p as usize) < vb_end {
                            flagged_heap_push(
                                std::slice::from_raw_parts_mut(
                                    distances_ptr.as_ptr().add(p as usize * ncols),
                                    ncols,
                                ),
                                std::slice::from_raw_parts_mut(
                                    indices_ptr.as_ptr().add(p as usize * ncols),
                                    ncols,
                                ),
                                std::slice::from_raw_parts_mut(
                                    flags_ptr.as_ptr().add(p as usize * ncols),
                                    ncols,
                                ),
                                d,
                                q,
                            );
                        }
                        if (q as usize) >= vb_start && (q as usize) < vb_end {
                            flagged_heap_push(
                                std::slice::from_raw_parts_mut(
                                    distances_ptr.as_ptr().add(q as usize * ncols),
                                    ncols,
                                ),
                                std::slice::from_raw_parts_mut(
                                    indices_ptr.as_ptr().add(q as usize * ncols),
                                    ncols,
                                ),
                                std::slice::from_raw_parts_mut(
                                    flags_ptr.as_ptr().add(q as usize * ncols),
                                    ncols,
                                ),
                                d,
                                p,
                            );
                        }
                    }
                }
            }
        });

        n_updates_per_thread.fill(0);
    }
}

pub fn init_random_int8(
    n_neighbors: usize,
    data: &Array2<i8>,
    heap: &mut GraphHeap,
    rng_state: &[i64; 3],
) {
    let n_points = data.nrows();
    let ncols = heap.0.ncols();

    let indices_ptr = SendMutPtr::new(heap.0.as_mut_ptr());
    let distances_ptr = SendMutPtr::new(heap.1.as_mut_ptr());
    let flags_ptr = SendMutPtr::new(heap.2.as_mut_ptr());

    (0..n_points).into_par_iter().for_each(|i| {
        let mut local_rng = offset_state(rng_state, i as i64);
        unsafe {
            let idx_row = std::slice::from_raw_parts(indices_ptr.as_ptr().add(i * ncols), ncols);
            if idx_row[0] >= 0 {
                return;
            }
            let filled = idx_row.iter().filter(|&&x| x >= 0).count();
            let to_fill = n_neighbors.saturating_sub(filled);

            for _ in 0..to_fill {
                let idx = (tau_rand_int(&mut local_rng).abs() as usize) % n_points;
                let idx_i32 = idx as i32;
                if idx_row.contains(&idx_i32) {
                    continue;
                }
                let d = fast_int_inner_product_dissimilarity(data.row(idx), data.row(i));
                flagged_heap_push(
                    std::slice::from_raw_parts_mut(distances_ptr.as_ptr().add(i * ncols), ncols),
                    std::slice::from_raw_parts_mut(indices_ptr.as_ptr().add(i * ncols), ncols),
                    std::slice::from_raw_parts_mut(flags_ptr.as_ptr().add(i * ncols), ncols),
                    d,
                    idx_i32,
                );
            }
        }
    });
}

/// Generate graph updates bucketed by target vertex block.
pub fn generate_sorted_graph_update_array_int8(
    update_array: &mut Array3<f32>,
    n_updates_per_block: &mut Array2<i32>,
    new_candidate_block: &Array2<i32>,
    old_candidate_block: &Array2<i32>,
    dist_thresholds: &[f32],
    data: &Array2<i8>,
    n_threads: usize,
) {
    let block_size_candidates = new_candidate_block.nrows();
    let max_new_candidates = new_candidate_block.ncols();
    let max_old_candidates = old_candidate_block.ncols();
    let rows_per_thread = block_size_candidates / n_threads + 1;

    let n_vertices = data.nrows();
    let vertex_block_size = n_vertices / n_threads + 1;
    let max_updates = update_array.len_of(Axis(1));
    let max_updates_per_src_thread = max_updates / n_threads;

    n_updates_per_block.fill(0);

    let update_ptr = SendMutPtr::new(update_array.as_mut_ptr());
    let counts_ptr = SendMutPtr::new(n_updates_per_block.as_mut_ptr());
    let update_stride = update_array.len_of(Axis(1)) * 3;
    let counts_ncols = n_updates_per_block.ncols();

    (0..n_threads).into_par_iter().for_each(|t| {
        let mut local_counts = vec![0i32; n_threads];

        for r in 0..rows_per_thread {
            let i = t * rows_per_thread + r;
            if i >= block_size_candidates {
                break;
            }

            for j in 0..max_new_candidates {
                let p = new_candidate_block[[i, j]];
                if p < 0 {
                    continue;
                }

                let data_p = data.row(p as usize);
                let dist_thresh_p = dist_thresholds[p as usize];
                let mut p_block = (p as usize) / vertex_block_size;
                if p_block >= n_threads {
                    p_block = n_threads - 1;
                }

                for k in (j + 1)..max_new_candidates {
                    let q = new_candidate_block[[i, k]];
                    if q < 0 {
                        continue;
                    }

                    let d = fast_int_inner_product_dissimilarity(data_p, data.row(q as usize));
                    let dist_thresh_q = dist_thresholds[q as usize];
                    let max_threshold = dist_thresh_p.max(dist_thresh_q);

                    if d <= max_threshold {
                        let mut q_block = (q as usize) / vertex_block_size;
                        if q_block >= n_threads {
                            q_block = n_threads - 1;
                        }

                        let bucket_idx = local_counts[p_block] as usize;
                        let write_idx = t * max_updates_per_src_thread + bucket_idx;
                        if write_idx < max_updates {
                            unsafe {
                                let base = update_ptr
                                    .as_ptr()
                                    .add(p_block * update_stride + write_idx * 3);
                                *base = p as f32;
                                *base.add(1) = q as f32;
                                *base.add(2) = d;
                            }
                            local_counts[p_block] += 1;
                        }

                        if q_block != p_block {
                            let bucket_idx = local_counts[q_block] as usize;
                            let write_idx = t * max_updates_per_src_thread + bucket_idx;
                            if write_idx < max_updates {
                                unsafe {
                                    let base = update_ptr
                                        .as_ptr()
                                        .add(q_block * update_stride + write_idx * 3);
                                    *base = p as f32;
                                    *base.add(1) = q as f32;
                                    *base.add(2) = d;
                                }
                                local_counts[q_block] += 1;
                            }
                        }
                    }
                }

                for k in 0..max_old_candidates {
                    let q = old_candidate_block[[i, k]];
                    if q < 0 {
                        continue;
                    }

                    let d = fast_int_inner_product_dissimilarity(data_p, data.row(q as usize));
                    let dist_thresh_q = dist_thresholds[q as usize];
                    let max_threshold = dist_thresh_p.max(dist_thresh_q);

                    if d <= max_threshold {
                        let mut q_block = (q as usize) / vertex_block_size;
                        if q_block >= n_threads {
                            q_block = n_threads - 1;
                        }

                        let bucket_idx = local_counts[p_block] as usize;
                        let write_idx = t * max_updates_per_src_thread + bucket_idx;
                        if write_idx < max_updates {
                            unsafe {
                                let base = update_ptr
                                    .as_ptr()
                                    .add(p_block * update_stride + write_idx * 3);
                                *base = p as f32;
                                *base.add(1) = q as f32;
                                *base.add(2) = d;
                            }
                            local_counts[p_block] += 1;
                        }

                        if q_block != p_block {
                            let bucket_idx = local_counts[q_block] as usize;
                            let write_idx = t * max_updates_per_src_thread + bucket_idx;
                            if write_idx < max_updates {
                                unsafe {
                                    let base = update_ptr
                                        .as_ptr()
                                        .add(q_block * update_stride + write_idx * 3);
                                    *base = p as f32;
                                    *base.add(1) = q as f32;
                                    *base.add(2) = d;
                                }
                                local_counts[q_block] += 1;
                            }
                        }
                    }
                }
            }
        }

        for b in 0..n_threads {
            unsafe {
                *counts_ptr.as_ptr().add(b * counts_ncols + t + 1) = local_counts[b];
            }
        }
    });
}

/// Approximate k-NN via NN-descent (sorted updates). Returns heap distances (negative cosine).
pub fn nn_descent_int8_sorted(
    data: &Array2<i8>,
    n_neighbors: usize,
    rng_state: &mut [i64; 3],
    max_candidates: usize,
    n_iters: usize,
    delta: f32,
    delta_improv: Option<f32>,
    leaf_array: Option<&Array2<i32>>,
) -> (Array2<i32>, Array2<f32>) {
    let n_threads = rayon::current_num_threads();
    let mut current_graph = make_heap(data.nrows(), n_neighbors);

    if let Some(leaf) = leaf_array {
        init_rp_tree_int8(data, &mut current_graph, leaf, n_threads);
    }
    init_random_int8(n_neighbors, data, &mut current_graph, rng_state);

    let n_vertices = data.nrows();
    let mut block_size = 65536 / n_threads;
    let mut n_blocks = n_vertices / block_size;

    let max_updates_per_thread = ((max_candidates * max_candidates
        + max_candidates * (max_candidates.saturating_sub(1)) / 2)
        * block_size) as usize;

    let mut sorted_update_array = Array3::zeros((n_threads, max_updates_per_thread, 3));

    let mut n_updates_per_block = Array2::<i32>::zeros((n_threads, n_threads + 1));
    let mut prev_sum_dist: Option<f64> = None;

    for _n in 0..n_iters {
        let (new_candidate_neighbors, old_candidate_neighbors) =
            build_candidates(&mut current_graph, max_candidates, rng_state, n_threads);

        let mut c = 0u32;
        let n_vertices = new_candidate_neighbors.nrows();
        for i in 0..=n_blocks {
            let block_start = i * block_size;
            let block_end = ((i + 1) * block_size).min(n_vertices);

            if block_start >= block_end {
                continue;
            }

            let new_block = new_candidate_neighbors.slice(s![block_start..block_end, ..]);
            let old_block = old_candidate_neighbors.slice(s![block_start..block_end, ..]);

            let dist_thresholds: Vec<f32> =
                (0..data.nrows()).map(|v| current_graph.1[[v, 0]]).collect();

            n_updates_per_block.fill(0);

            generate_sorted_graph_update_array_int8(
                &mut sorted_update_array,
                &mut n_updates_per_block,
                &new_block.to_owned(),
                &old_block.to_owned(),
                &dist_thresholds,
                data,
                n_threads,
            );

            c += apply_sorted_graph_updates(
                &mut current_graph,
                &sorted_update_array,
                &n_updates_per_block,
                n_threads,
            );
        }

        if c <= (delta * n_neighbors as f32 * data.nrows() as f32) as u32 {
            let (mut indices, mut distances) = (current_graph.0, current_graph.1);
            deheap_sort(&mut indices, &mut distances);
            return (indices, distances);
        }

        if let Some(delta_improv) = delta_improv {
            let sum_dist: f64 = current_graph
                .1
                .iter()
                .filter(|&&d| d < INF)
                .map(|&d| d as f64)
                .sum();

            if let Some(prev) = prev_sum_dist {
                let rel_improv = (sum_dist - prev).abs() / prev.abs();
                if rel_improv < delta_improv as f64 {
                    let (mut indices, mut distances) = (current_graph.0, current_graph.1);
                    deheap_sort(&mut indices, &mut distances);
                    return (indices, distances);
                }
            }
            prev_sum_dist = Some(sum_dist);
        }

        block_size = block_size.min(n_vertices).saturating_mul(2).max(1);
        n_blocks = n_vertices / block_size;
    }

    let (mut indices, mut distances) = (current_graph.0, current_graph.1);
    deheap_sort(&mut indices, &mut distances);
    (indices, distances)
}
