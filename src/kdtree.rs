//! KD-tree construction and parallel k-nearest-neighbor queries (port of `numba_kdtree.py`).

use ndarray::{s, Array1, Array2, Array3, ArrayView1, Axis, Zip};
use rayon::prelude::*;

/// KD-tree arrays matching the Python `NumbaKDTree` namedtuple layout.
#[derive(Clone, Debug)]
pub struct KdTree {
    pub data: Array2<f32>,
    pub idx_array: Array1<i64>,
    pub idx_start: Array1<i64>,
    pub idx_end: Array1<i64>,
    #[allow(dead_code)]
    pub radius: Array1<f32>,
    pub is_leaf: Array1<bool>,
    /// Shape `(2, n_nodes, n_features)`: `[0]` lower bounds, `[1]` upper bounds per node.
    pub node_bounds: Array3<f32>,
}

fn compare_indices(data: &Array2<f32>, axis: i64, idx1: i64, idx2: i64) -> i8 {
    let val1 = data[[idx1 as usize, axis as usize]];
    let val2 = data[[idx2 as usize, axis as usize]];
    if val1 < val2 {
        -1
    } else if val1 > val2 {
        1
    } else if idx1 < idx2 {
        -1
    } else if idx1 > idx2 {
        1
    } else {
        0
    }
}

fn insertion_sort_indices(
    data: &Array2<f32>,
    idx_array: &mut [i64],
    axis: i64,
    left: usize,
    right: usize,
) {
    for i in (left + 1)..right {
        let key_idx = idx_array[i];
        let mut j = i;
        while j > left && compare_indices(data, axis, idx_array[j - 1], key_idx) > 0 {
            idx_array[j] = idx_array[j - 1];
            j -= 1;
        }
        idx_array[j] = key_idx;
    }
}

fn sift_down_indices(
    data: &Array2<f32>,
    idx_array: &mut [i64],
    axis: i64,
    offset: usize,
    start: usize,
    end: usize,
) {
    let mut root = start;
    while root * 2 + 1 < end {
        let child = root * 2 + 1;
        let mut swap = root;

        if compare_indices(
            data,
            axis,
            idx_array[offset + swap],
            idx_array[offset + child],
        ) < 0
        {
            swap = child;
        }

        if child + 1 < end
            && compare_indices(
                data,
                axis,
                idx_array[offset + swap],
                idx_array[offset + child + 1],
            ) < 0
        {
            swap = child + 1;
        }

        if swap == root {
            return;
        }

        idx_array.swap(offset + root, offset + swap);
        root = swap;
    }
}

fn heapsort_indices(
    data: &Array2<f32>,
    idx_array: &mut [i64],
    axis: i64,
    left: usize,
    right: usize,
) {
    let size = right - left;
    if size <= 1 {
        return;
    }

    for i in (0..=(size / 2).saturating_sub(1)).rev() {
        sift_down_indices(data, idx_array, axis, left, i, size);
    }

    for i in (1..size).rev() {
        idx_array.swap(left, left + i);
        sift_down_indices(data, idx_array, axis, left, 0, i);
    }
}

fn median_of_three_pivot(
    data: &Array2<f32>,
    idx_array: &mut [i64],
    axis: i64,
    left: usize,
    right: usize,
) -> usize {
    let mid = (left + right - 1) / 2;

    let mut idx_left = idx_array[left];
    let mut idx_mid = idx_array[mid];
    let mut idx_right = idx_array[right - 1];

    if compare_indices(data, axis, idx_left, idx_mid) > 0 {
        idx_array.swap(left, mid);
        std::mem::swap(&mut idx_left, &mut idx_mid);
    }

    if compare_indices(data, axis, idx_mid, idx_right) > 0 {
        idx_array.swap(mid, right - 1);
        std::mem::swap(&mut idx_mid, &mut idx_right);

        if compare_indices(data, axis, idx_left, idx_mid) > 0 {
            idx_array.swap(left, mid);
        }
    }

    mid
}

fn partition_indices(
    data: &Array2<f32>,
    idx_array: &mut [i64],
    axis: i64,
    left: usize,
    right: usize,
    pivot_idx: usize,
) -> usize {
    idx_array.swap(pivot_idx, right - 1);
    let pivot_original_idx = idx_array[right - 1];

    let mut i = left as isize;
    let mut j = (right - 2) as isize;

    loop {
        while i <= j && compare_indices(data, axis, idx_array[i as usize], pivot_original_idx) < 0 {
            i += 1;
        }
        while i <= j && compare_indices(data, axis, idx_array[j as usize], pivot_original_idx) >= 0
        {
            j -= 1;
        }
        if i >= j {
            break;
        }
        idx_array.swap(i as usize, j as usize);
        i += 1;
        j -= 1;
    }

    idx_array.swap(i as usize, right - 1);
    i as usize
}

fn introselect_impl(
    data: &Array2<f32>,
    idx_array: &mut [i64],
    axis: i64,
    mut left: usize,
    mut right: usize,
    nth: usize,
    mut depth_limit: i64,
) {
    while right - left > 16 {
        if depth_limit == 0 {
            heapsort_indices(data, idx_array, axis, left, right);
            return;
        }
        depth_limit -= 1;

        let pivot_idx = median_of_three_pivot(data, idx_array, axis, left, right);
        let pivot_pos = partition_indices(data, idx_array, axis, left, right, pivot_idx);

        if nth < pivot_pos {
            right = pivot_pos;
        } else if nth > pivot_pos {
            left = pivot_pos + 1;
        } else {
            return;
        }
    }

    insertion_sort_indices(data, idx_array, axis, left, right);
}

fn introselect(
    data: &Array2<f32>,
    idx_array: &mut [i64],
    axis: i64,
    left: usize,
    right: usize,
    nth: usize,
) {
    let size = right - left;
    if size <= 16 {
        insertion_sort_indices(data, idx_array, axis, left, right);
        return;
    }

    let max_depth = (2.0 * (size as f64).log2()) as i64;
    introselect_impl(data, idx_array, axis, left, right, nth, max_depth);
}

fn find_node_split_dim(
    data: &Array2<f32>,
    idx_array: &[i64],
    idx_start: usize,
    idx_end: usize,
) -> i64 {
    let n_features = data.ncols();
    let mut result = 0i64;
    let mut max_spread = 0.0f32;

    for j in 0..n_features {
        let mut max_val = data[[idx_array[idx_start] as usize, j]];
        let mut min_val = max_val;
        for i in (idx_start + 1)..idx_end {
            let val = data[[idx_array[i] as usize, j]];
            max_val = max_val.max(val);
            min_val = min_val.min(val);
        }
        let spread = max_val - min_val;
        if spread > max_spread {
            max_spread = spread;
            result = j as i64;
        }
    }
    result
}

fn init_node(
    data: &Array2<f32>,
    node_bounds: &mut Array3<f32>,
    idx_array: &[i64],
    idx_start_array: &mut Array1<i64>,
    idx_end_array: &mut Array1<i64>,
    radius_array: &mut Array1<f32>,
    node: usize,
    idx_start: usize,
    idx_end: usize,
) {
    let n_features = data.ncols();

    for j in 0..n_features {
        node_bounds[[0, node, j]] = f32::INFINITY;
        node_bounds[[1, node, j]] = f32::NEG_INFINITY;
    }

    for i in idx_start..idx_end {
        let row = data.row(idx_array[i] as usize);
        for j in 0..n_features {
            let v = row[j];
            node_bounds[[0, node, j]] = node_bounds[[0, node, j]].min(v);
            node_bounds[[1, node, j]] = node_bounds[[1, node, j]].max(v);
        }
    }

    let mut radius = 0.0f32;
    for j in 0..n_features {
        let diff = (node_bounds[[1, node, j]] - node_bounds[[0, node, j]]).abs() * 0.5;
        radius += diff * diff;
    }

    idx_start_array[node] = idx_start as i64;
    idx_end_array[node] = idx_end as i64;
    radius_array[node] = radius.sqrt();
}

fn recursive_build_tree(
    data: &Array2<f32>,
    idx_array: &mut Array1<i64>,
    idx_start_array: &mut Array1<i64>,
    idx_end_array: &mut Array1<i64>,
    radius_array: &mut Array1<f32>,
    is_leaf_array: &mut Array1<bool>,
    node_bounds: &mut Array3<f32>,
    idx_start: usize,
    idx_end: usize,
    node: usize,
) {
    let n_points = idx_end - idx_start;
    let n_mid = n_points / 2;

    init_node(
        data,
        node_bounds,
        idx_array.as_slice().unwrap(),
        idx_start_array,
        idx_end_array,
        radius_array,
        node,
        idx_start,
        idx_end,
    );

    if 2 * node + 1 >= is_leaf_array.len() {
        is_leaf_array[node] = true;
    } else if idx_end - idx_start < 2 {
        is_leaf_array[node] = true;
    } else {
        is_leaf_array[node] = false;
        let axis = find_node_split_dim(data, idx_array.as_slice().unwrap(), idx_start, idx_end);
        introselect(
            data,
            idx_array.as_slice_mut().unwrap(),
            axis,
            idx_start,
            idx_end,
            idx_start + n_mid,
        );
        recursive_build_tree(
            data,
            idx_array,
            idx_start_array,
            idx_end_array,
            radius_array,
            is_leaf_array,
            node_bounds,
            idx_start,
            idx_start + n_mid,
            2 * node + 1,
        );
        recursive_build_tree(
            data,
            idx_array,
            idx_start_array,
            idx_end_array,
            radius_array,
            is_leaf_array,
            node_bounds,
            idx_start + n_mid,
            idx_end,
            2 * node + 2,
        );
    }
}

/// Build a balanced KD-tree over `data` with leaf nodes holding between `leaf_size` and `2 * leaf_size` points.
pub fn build_kdtree(data: Array2<f32>, leaf_size: usize) -> KdTree {
    assert!(
        leaf_size >= 1,
        "leaf_size must be greater than or equal to 1"
    );

    let n_samples = data.nrows();
    let n_features = data.ncols();

    let n_levels = ((n_samples.saturating_sub(1)).max(1) as f64 / leaf_size as f64)
        .log2()
        .max(0.0)
        .floor() as i32
        + 1;
    let n_nodes = (2i32.pow(n_levels as u32) - 1) as usize;

    let mut idx_array = Array1::from_iter(0i64..n_samples as i64);
    let mut idx_start_array = Array1::zeros(n_nodes);
    let mut idx_end_array = Array1::zeros(n_nodes);
    let mut radius_array = Array1::zeros(n_nodes);
    let mut is_leaf_array = Array1::from_elem(n_nodes, false);
    let mut node_bounds = Array3::zeros((2, n_nodes, n_features));

    recursive_build_tree(
        &data,
        &mut idx_array,
        &mut idx_start_array,
        &mut idx_end_array,
        &mut radius_array,
        &mut is_leaf_array,
        &mut node_bounds,
        0,
        n_samples,
        0,
    );

    KdTree {
        data,
        idx_array,
        idx_start: idx_start_array,
        idx_end: idx_end_array,
        radius: radius_array,
        is_leaf: is_leaf_array,
        node_bounds,
    }
}

/// Squared Euclidean distance between two points (matches Numba `rdist` with `fastmath`).
pub fn rdist(x: ArrayView1<f32>, y: ArrayView1<f32>) -> f32 {
    let mut result = 0.0f32;
    for i in 0..x.len() {
        let diff = x[i] - y[i];
        result += diff * diff;
    }
    result
}

/// Lower bound on squared distance from `pt` to the axis-aligned box (`upper`, `lower` parameter order matches Python).
pub fn point_to_node_lower_bound_rdist(
    upper: ArrayView1<f32>,
    lower: ArrayView1<f32>,
    pt: ArrayView1<f32>,
) -> f32 {
    let mut result = 0.0f32;
    for i in 0..pt.len() {
        let d_lo = if upper[i] > pt[i] {
            upper[i] - pt[i]
        } else {
            0.0
        };
        let d_hi = if pt[i] > lower[i] {
            pt[i] - lower[i]
        } else {
            0.0
        };
        let d = d_lo + d_hi;
        result += d * d;
    }
    result
}

/// Push into a max-heap of size `priorities.len()`; returns 1 if inserted, 0 if rejected.
pub fn simple_heap_push(priorities: &mut [f32], indices: &mut [i32], p: f32, n: i32) -> i32 {
    if p >= priorities[0] {
        return 0;
    }

    let size = priorities.len();
    priorities[0] = p;
    indices[0] = n;

    let mut i = 0usize;
    loop {
        let ic1 = 2 * i + 1;
        let ic2 = ic1 + 1;

        if ic1 >= size {
            break;
        }

        let i_swap = if ic2 >= size {
            if priorities[ic1] > p {
                ic1
            } else {
                break;
            }
        } else if priorities[ic1] >= priorities[ic2] {
            if p < priorities[ic1] {
                ic1
            } else {
                break;
            }
        } else if p < priorities[ic2] {
            ic2
        } else {
            break;
        };

        priorities[i] = priorities[i_swap];
        indices[i] = indices[i_swap];
        i = i_swap;
    }

    priorities[i] = p;
    indices[i] = n;
    1
}

fn siftdown(heap1: &mut [f32], heap2: &mut [i32], mut elt: usize) {
    while elt * 2 + 1 < heap1.len() {
        let left_child = elt * 2 + 1;
        let right_child = left_child + 1;
        let mut swap = elt;

        if heap1[swap] < heap1[left_child] {
            swap = left_child;
        }

        if right_child < heap1.len() && heap1[swap] < heap1[right_child] {
            swap = right_child;
        }

        if swap == elt {
            break;
        }

        heap1.swap(elt, swap);
        heap2.swap(elt, swap);
        elt = swap;
    }
}

/// Sort each row's k-NN heap (max-heap to ascending order), in parallel over rows.
pub fn deheap_sort(distances: Array2<f32>, indices: Array2<i32>) -> (Array2<f32>, Array2<i32>) {
    let k = distances.ncols();
    let mut distances = distances;
    let mut indices = indices;

    distances
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(indices.axis_iter_mut(Axis(0)).into_par_iter())
        .for_each(|(mut dist_row, mut idx_row)| {
            let dist_slice = dist_row.as_slice_mut().unwrap();
            let idx_slice = idx_row.as_slice_mut().unwrap();
            for j in (1..k).rev() {
                dist_slice.swap(0, j);
                idx_slice.swap(0, j);
                siftdown(&mut dist_slice[..j], &mut idx_slice[..j], 0);
            }
        });

    (distances, indices)
}

/// Recursive k-NN query from `node` for `point` into max-heaps `heap_p` / `heap_i`.
pub fn tree_query_recursion(
    tree: &KdTree,
    node: usize,
    point: ArrayView1<f32>,
    heap_p: &mut [f32],
    heap_i: &mut [i32],
    dist_lower_bound: f32,
) {
    let idx_start = tree.idx_start[node] as usize;
    let idx_end = tree.idx_end[node] as usize;
    let is_leaf = tree.is_leaf[node];

    if dist_lower_bound > heap_p[0] {
        return;
    }

    if is_leaf {
        for i in idx_start..idx_end {
            let idx = tree.idx_array[i] as usize;
            let d = rdist(point, tree.data.row(idx));
            if d < heap_p[0] {
                simple_heap_push(heap_p, heap_i, d, idx as i32);
            }
        }
    } else {
        let left = 2 * node + 1;
        let right = left + 1;
        let dist_lower_bound_left = point_to_node_lower_bound_rdist(
            tree.node_bounds.slice(s![0, left, ..]),
            tree.node_bounds.slice(s![1, left, ..]),
            point,
        );
        let dist_lower_bound_right = point_to_node_lower_bound_rdist(
            tree.node_bounds.slice(s![0, right, ..]),
            tree.node_bounds.slice(s![1, right, ..]),
            point,
        );

        if dist_lower_bound_left <= dist_lower_bound_right {
            tree_query_recursion(tree, left, point, heap_p, heap_i, dist_lower_bound_left);
            tree_query_recursion(tree, right, point, heap_p, heap_i, dist_lower_bound_right);
        } else {
            tree_query_recursion(tree, right, point, heap_p, heap_i, dist_lower_bound_right);
            tree_query_recursion(tree, left, point, heap_p, heap_i, dist_lower_bound_left);
        }
    }
}

/// Parallel k-NN query for each row of `data`. When `output_rdist` is false, distances are square-rooted before sorting.
pub fn parallel_tree_query(
    tree: &KdTree,
    data: &Array2<f32>,
    k: i64,
    output_rdist: bool,
) -> (Array2<f32>, Array2<i32>) {
    let n_queries = data.nrows();
    let k = k as usize;
    let mut distances = Array2::from_elem((n_queries, k), f32::INFINITY);
    let mut indices = Array2::from_elem((n_queries, k), -1i32);

    Zip::from(data.rows())
        .and(distances.rows_mut())
        .and(indices.rows_mut())
        .into_par_iter()
        .for_each(|(point, mut heap_p, mut heap_i)| {
            let distance_lower_bound = point_to_node_lower_bound_rdist(
                tree.node_bounds.slice(s![0, 0, ..]),
                tree.node_bounds.slice(s![1, 0, ..]),
                point.view(),
            );
            tree_query_recursion(
                tree,
                0,
                point.view(),
                heap_p.as_slice_mut().unwrap(),
                heap_i.as_slice_mut().unwrap(),
                distance_lower_bound,
            );
        });

    if output_rdist {
        deheap_sort(distances, indices)
    } else {
        distances.mapv_inplace(|x| x.sqrt());
        deheap_sort(distances, indices)
    }
}
