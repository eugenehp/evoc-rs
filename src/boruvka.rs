//! Borůvka MST via KD-tree component-aware queries (port of `boruvka.py`).

use crate::disjoint_set::{ds_find, ds_rank_create, ds_union_by_rank, RankDisjointSet};
use crate::kdtree::{parallel_tree_query, point_to_node_lower_bound_rdist, rdist, KdTree};
use ndarray::parallel::prelude::*;
use ndarray::{Array1, Array2, ArrayView1, Axis};
use rustc_hash::FxHashMap;
use std::sync::atomic::{AtomicU32, Ordering};

/// Raw pointer wrapper for parallel union-find updates (path compression races, as in Numba).
struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    #[inline]
    unsafe fn as_mut(&self) -> &mut T {
        &mut *self.0
    }
}

#[inline]
fn atomic_min_f32(slot: &AtomicU32, value: f32) {
    let value_bits = value.to_bits();
    let mut current = slot.load(Ordering::Relaxed);
    while value_bits < current {
        match slot.compare_exchange_weak(current, value_bits, Ordering::Relaxed, Ordering::Relaxed)
        {
            Ok(_) => break,
            Err(c) => current = c,
        }
    }
}

/// Find the minimum outgoing edge per component and union disjoint components.
pub fn merge_components(
    disjoint_set: &mut RankDisjointSet,
    candidate_neighbors: &Array1<i32>,
    candidate_neighbor_distances: &Array1<f32>,
    point_components: &Array1<i64>,
) -> Array2<f32> {
    let mut component_edges: FxHashMap<i64, (i64, i64, f32)> = FxHashMap::default();
    let mut component_order: Vec<i64> = Vec::new();

    for i in 0..candidate_neighbors.len() {
        let from_component = point_components[i];
        let dist = candidate_neighbor_distances[i];
        let neighbor = candidate_neighbors[i] as i64;

        match component_edges.get_mut(&from_component) {
            Some(edge) if dist < edge.2 => {
                *edge = (i as i64, neighbor, dist);
            }
            None => {
                component_edges.insert(from_component, (i as i64, neighbor, dist));
                // Match Python dict insertion order: first time we see a component wins.
                component_order.push(from_component);
            }
            _ => {}
        }
    }

    let mut result = Vec::with_capacity(component_edges.len());

    // Preserve insertion order of components (Python `dict.values()`).
    for &comp in component_order.iter() {
        let edge = match component_edges.get(&comp) {
            Some(e) => e,
            None => continue,
        };
        let from_component = ds_find(disjoint_set, edge.0 as i32);
        let to_component = ds_find(disjoint_set, edge.1 as i32);
        if from_component != to_component {
            result.push([edge.0 as f32, edge.1 as f32, edge.2]);
            ds_union_by_rank(disjoint_set, from_component, to_component);
        }
    }

    if result.is_empty() {
        Array2::zeros((0, 3))
    } else {
        Array2::from_shape_vec((result.len(), 3), result.into_iter().flatten().collect()).unwrap()
    }
}

/// Refresh `point_components` from the disjoint set, then propagate component ids up the tree.
pub fn update_component_vectors(
    tree: &KdTree,
    disjoint_set: &mut RankDisjointSet,
    node_components: &mut Array1<i64>,
    point_components: &mut Array1<i64>,
) {
    let ds_ptr = SendPtr(disjoint_set as *mut RankDisjointSet);
    point_components
        .as_slice_mut()
        .unwrap()
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, pc)| unsafe {
            *pc = ds_find(ds_ptr.as_mut(), i as i32) as i64;
        });

    for i in (0..tree.idx_start.len()).rev() {
        let is_leaf = tree.is_leaf[i];
        let idx_start = tree.idx_start[i] as usize;
        let idx_end = tree.idx_end[i] as usize;

        if is_leaf {
            let candidate_component = point_components[tree.idx_array[idx_start] as usize];
            let mut uniform = true;
            for j in (idx_start + 1)..idx_end {
                let idx = tree.idx_array[j] as usize;
                if point_components[idx] != candidate_component {
                    uniform = false;
                    break;
                }
            }
            if uniform {
                node_components[i] = candidate_component;
            }
        } else {
            let left = 2 * i + 1;
            let right = left + 1;
            if node_components[left] == node_components[right] {
                node_components[i] = node_components[left];
            }
        }
    }
}

/// Component-pruned KD-tree query for a single query point (1-element heaps).
#[allow(clippy::too_many_arguments)]
pub fn component_aware_query_recursion(
    tree: &KdTree,
    node: i32,
    point: ArrayView1<f32>,
    heap_p: &mut [f32],
    heap_i: &mut [i32],
    current_core_distance: f32,
    core_distances: ArrayView1<f32>,
    current_component: i64,
    node_components: &Array1<i64>,
    point_components: &Array1<i64>,
    dist_lower_bound: f32,
    component_nearest_neighbor_dist: &mut [f32],
) {
    let node = node as usize;
    let is_leaf = tree.is_leaf[node];
    let idx_start = tree.idx_start[node] as usize;
    let idx_end = tree.idx_end[node] as usize;

    if dist_lower_bound > heap_p[0] {
        return;
    }

    if dist_lower_bound > component_nearest_neighbor_dist[0]
        || current_core_distance > component_nearest_neighbor_dist[0]
    {
        return;
    }

    if node_components[node] == current_component {
        return;
    }

    if is_leaf {
        for i in idx_start..idx_end {
            let idx = tree.idx_array[i] as usize;
            if point_components[idx] != current_component
                && core_distances[idx] < component_nearest_neighbor_dist[0]
            {
                let d = rdist(point, tree.data.row(idx))
                    .max(current_core_distance)
                    .max(core_distances[idx]);
                if d < heap_p[0] {
                    heap_p[0] = d;
                    heap_i[0] = idx as i32;
                    if d < component_nearest_neighbor_dist[0] {
                        component_nearest_neighbor_dist[0] = d;
                    }
                }
            }
        }
    } else {
        let left = (2 * node + 1) as i32;
        let right = left + 1;
        let left_u = left as usize;
        let right_u = right as usize;
        let dist_lower_bound_left = point_to_node_lower_bound_rdist(
            tree.node_bounds.slice(ndarray::s![0, left_u, ..]),
            tree.node_bounds.slice(ndarray::s![1, left_u, ..]),
            point,
        );
        let dist_lower_bound_right = point_to_node_lower_bound_rdist(
            tree.node_bounds.slice(ndarray::s![0, right_u, ..]),
            tree.node_bounds.slice(ndarray::s![1, right_u, ..]),
            point,
        );

        if dist_lower_bound_left <= dist_lower_bound_right {
            component_aware_query_recursion(
                tree,
                left,
                point,
                heap_p,
                heap_i,
                current_core_distance,
                core_distances,
                current_component,
                node_components,
                point_components,
                dist_lower_bound_left,
                component_nearest_neighbor_dist,
            );
            component_aware_query_recursion(
                tree,
                right,
                point,
                heap_p,
                heap_i,
                current_core_distance,
                core_distances,
                current_component,
                node_components,
                point_components,
                dist_lower_bound_right,
                component_nearest_neighbor_dist,
            );
        } else {
            component_aware_query_recursion(
                tree,
                right,
                point,
                heap_p,
                heap_i,
                current_core_distance,
                core_distances,
                current_component,
                node_components,
                point_components,
                dist_lower_bound_right,
                component_nearest_neighbor_dist,
            );
            component_aware_query_recursion(
                tree,
                left,
                point,
                heap_p,
                heap_i,
                current_core_distance,
                core_distances,
                current_component,
                node_components,
                point_components,
                dist_lower_bound_left,
                component_nearest_neighbor_dist,
            );
        }
    }
}

/// Parallel component-aware nearest-neighbor search for Borůvka (non-reproducible; may race on component bounds).
pub fn boruvka_tree_query(
    tree: &KdTree,
    node_components: &Array1<i64>,
    point_components: &Array1<i64>,
    core_distances: &Array1<f32>,
) -> (Array1<f32>, Array1<i32>) {
    let n = tree.data.nrows();
    let n_features = tree.data.ncols();
    let mut candidate_distances = Array1::from_elem(n, f32::INFINITY);
    let mut candidate_indices = Array1::from_elem(n, -1i32);
    let component_bounds: Vec<AtomicU32> = (0..n)
        .map(|_| AtomicU32::new(f32::INFINITY.to_bits()))
        .collect();

    let root_lower = tree.node_bounds.slice(ndarray::s![0, 0, ..]).to_owned();
    let root_upper = tree.node_bounds.slice(ndarray::s![1, 0, ..]).to_owned();
    let core = core_distances.to_owned();

    candidate_distances
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(candidate_indices.axis_iter_mut(Axis(0)).into_par_iter())
        .enumerate()
        .for_each(|(i, (mut heap_p, mut heap_i))| {
            let mut point_buf = vec![0.0f32; n_features];
            point_buf.copy_from_slice(tree.data.row(i).as_slice().unwrap());
            let point = ArrayView1::from(&point_buf);

            let distance_lower_bound =
                point_to_node_lower_bound_rdist(root_lower.view(), root_upper.view(), point);

            let current_component = point_components[i];
            let comp_idx = current_component as usize;
            let mut local_bound = [f32::from_bits(
                component_bounds[comp_idx].load(Ordering::Relaxed),
            )];

            component_aware_query_recursion(
                tree,
                0,
                point,
                heap_p.as_slice_mut().unwrap(),
                heap_i.as_slice_mut().unwrap(),
                core[i],
                core.view(),
                current_component,
                node_components,
                point_components,
                distance_lower_bound,
                &mut local_bound,
            );

            atomic_min_f32(&component_bounds[comp_idx], local_bound[0]);
        });

    (candidate_distances, candidate_indices)
}

fn calculate_block_size(n_components: usize, n_points: usize, num_threads: usize) -> usize {
    let points_per_component = if n_components == 0 {
        n_points as f64
    } else {
        n_points as f64 / n_components as f64
    };

    let block_size = if points_per_component < 10.0 {
        num_threads * 512
    } else if points_per_component < 100.0 {
        num_threads * 128
    } else if points_per_component < 1000.0 {
        num_threads * 32
    } else {
        num_threads * 8
    };

    block_size.max(num_threads).min(n_points / 4 + 1)
}

fn update_component_bounds_from_block(
    component_nearest_neighbor_dist: &mut Array1<f32>,
    block_component_bounds: &[f32],
    point_components: &Array1<i64>,
    block_start: usize,
    block_end: usize,
) {
    for i in block_start..block_end {
        let component = point_components[i] as usize;
        let block_bound = block_component_bounds[i - block_start];
        if block_bound < component_nearest_neighbor_dist[component] {
            component_nearest_neighbor_dist[component] = block_bound;
        }
    }
}

/// Block-sequential Borůvka tree query for reproducible component-bound updates.
pub fn boruvka_tree_query_reproducible(
    tree: &KdTree,
    node_components: &Array1<i64>,
    point_components: &Array1<i64>,
    core_distances: &Array1<f32>,
    block_size: usize,
) -> (Array1<f32>, Array1<i32>) {
    let n = tree.data.nrows();
    let mut candidate_distances = Array1::from_elem(n, f32::INFINITY);
    let mut candidate_indices = Array1::from_elem(n, -1i32);
    let mut component_nearest_neighbor_dist = Array1::from_elem(n, f32::INFINITY);
    let mut max_block_component_bounds = Array1::from_elem(block_size, f32::INFINITY);

    let mut block_start = 0usize;
    while block_start < n {
        let block_end = (block_start + block_size).min(n);
        let block_size_actual = block_end - block_start;

        for v in max_block_component_bounds
            .as_slice_mut()
            .unwrap()
            .iter_mut()
            .take(block_size_actual)
        {
            *v = f32::INFINITY;
        }

        let _n_features = tree.data.ncols();
        let root_lower = tree.node_bounds.slice(ndarray::s![0, 0, ..]).to_owned();
        let root_upper = tree.node_bounds.slice(ndarray::s![1, 0, ..]).to_owned();

        // Sequential per-point processing (matches Numba `prange` with NUMBA_NUM_THREADS=1).
        for local_i in 0..block_size_actual {
            let i = block_start + local_i;
            let point = tree.data.row(i);

            let distance_lower_bound =
                point_to_node_lower_bound_rdist(root_lower.view(), root_upper.view(), point);

            let current_component = point_components[i];
            let mut local_bound = [component_nearest_neighbor_dist[current_component as usize]];
            let mut heap_p = [f32::INFINITY];
            let mut heap_i = [-1i32];

            component_aware_query_recursion(
                tree,
                0,
                point,
                &mut heap_p,
                &mut heap_i,
                core_distances[i],
                core_distances.view(),
                current_component,
                node_components,
                point_components,
                distance_lower_bound,
                &mut local_bound,
            );

            candidate_distances[i] = heap_p[0];
            candidate_indices[i] = heap_i[0];
            max_block_component_bounds[local_i] = local_bound[0];
        }

        update_component_bounds_from_block(
            &mut component_nearest_neighbor_dist,
            max_block_component_bounds.as_slice().unwrap(),
            point_components,
            block_start,
            block_end,
        );

        block_start = block_end;
    }

    (candidate_distances, candidate_indices)
}

/// Seed the Borůvka MST from a k-NN graph and core distances.
pub fn initialize_boruvka_from_knn(
    knn_indices: &Array2<i32>,
    knn_distances: &Array2<f32>,
    core_distances: &Array1<f32>,
    disjoint_set: &mut RankDisjointSet,
) -> Array2<f32> {
    let n = knn_indices.nrows();
    let k = knn_indices.ncols();
    let mut component_edges = vec![[-1.0f64; 3]; n];

    component_edges
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, edge)| {
            for j in 1..k {
                let neighbor = knn_indices[[i, j]] as i32;
                let k_usize = neighbor as usize;
                if core_distances[i] >= core_distances[k_usize] {
                    let edge_weight = core_distances[i].max(knn_distances[[i, j]]);
                    *edge = [i as f64, neighbor as f64, edge_weight as f64];
                    break;
                }
            }
        });

    let mut result = Vec::new();
    for edge in component_edges {
        if edge[0] < 0.0 {
            continue;
        }
        let from_component = ds_find(disjoint_set, edge[0] as i32);
        let to_component = ds_find(disjoint_set, edge[1] as i32);
        if from_component != to_component {
            result.push([edge[0] as f32, edge[1] as f32, edge[2] as f32]);
            ds_union_by_rank(disjoint_set, from_component, to_component);
        }
    }

    if result.is_empty() {
        Array2::zeros((0, 3))
    } else {
        Array2::from_shape_vec((result.len(), 3), result.into_iter().flatten().collect()).unwrap()
    }
}

fn count_unique_components(point_components: &Array1<i64>) -> usize {
    let mut sorted: Vec<i64> = point_components.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    sorted.len()
}

/// Compute a minimum spanning tree with Borůvka's algorithm on `tree.data`.
///
/// Returns edge list with shape `(n_samples - 1, 3)`; column 2 holds Euclidean distances (sqrt of squared distances used internally).
pub fn parallel_boruvka(
    tree: &KdTree,
    min_samples: i64,
    reproducible: bool,
    n_threads: usize,
) -> Array2<f32> {
    let n = tree.data.nrows();
    let n_threads = n_threads.max(1);

    let mut components_disjoint_set = ds_rank_create(n);
    let mut point_components: Array1<i64> = Array1::from_iter(0i64..n as i64);
    let mut node_components = Array1::from_elem(tree.idx_start.len(), -1i64);

    let (core_distances, initial_edges) = if min_samples > 1 {
        let (distances, neighbors) = parallel_tree_query(tree, &tree.data, min_samples + 1, true);
        let core_distances = distances.column(distances.ncols() - 1).to_owned();
        let edges = initialize_boruvka_from_knn(
            &neighbors,
            &distances,
            &core_distances,
            &mut components_disjoint_set,
        );
        update_component_vectors(
            tree,
            &mut components_disjoint_set,
            &mut node_components,
            &mut point_components,
        );
        (core_distances, edges)
    } else {
        let core_distances = Array1::zeros(n);
        let (distances, neighbors) = parallel_tree_query(tree, &tree.data, 2, true);
        let edges = initialize_boruvka_from_knn(
            &neighbors,
            &distances,
            &core_distances,
            &mut components_disjoint_set,
        );
        update_component_vectors(
            tree,
            &mut components_disjoint_set,
            &mut node_components,
            &mut point_components,
        );
        (core_distances, edges)
    };

    let mut n_components = count_unique_components(&point_components);
    let max_edges = n - 1;
    let mut all_edges = Array2::zeros((max_edges, 3));
    let mut n_edges = initial_edges.nrows();
    if n_edges > 0 {
        all_edges
            .slice_mut(ndarray::s![..n_edges, ..])
            .assign(&initial_edges);
    }

    while n_components > 1 {
        let (candidate_distances, candidate_indices) = if reproducible {
            let block_size = calculate_block_size(n_components, n, n_threads);
            boruvka_tree_query_reproducible(
                tree,
                &node_components,
                &point_components,
                &core_distances,
                block_size,
            )
        } else {
            boruvka_tree_query(tree, &node_components, &point_components, &core_distances)
        };

        let new_edges = merge_components(
            &mut components_disjoint_set,
            &candidate_indices,
            &candidate_distances,
            &point_components,
        );

        n_components = n_components.saturating_sub(new_edges.nrows());

        update_component_vectors(
            tree,
            &mut components_disjoint_set,
            &mut node_components,
            &mut point_components,
        );

        if new_edges.nrows() > 0 {
            let new_n = new_edges.nrows();
            all_edges
                .slice_mut(ndarray::s![n_edges..n_edges + new_n, ..])
                .assign(&new_edges);
            n_edges += new_n;
        }
    }

    for v in all_edges.column_mut(2) {
        *v = v.sqrt();
    }

    all_edges
}
