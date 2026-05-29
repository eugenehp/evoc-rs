//! Cluster selection utilities (port of `clustering_utilities.py`).

use crate::boruvka::parallel_boruvka;
use crate::cluster_trees::{
    condense_tree, extract_leaves, get_cluster_label_vector, get_point_membership_strength_vector,
    mst_to_linkage_tree, CondensedTree,
};
use crate::kdtree::build_kdtree;
use ndarray::{Array1, Array2, ArrayView1, Axis};
use rustc_hash::FxHashSet;
use std::collections::{HashMap, HashSet};

/// Local maxima indices in a 1-D signal (scipy-style peak finding).
pub fn find_peaks(x: ArrayView1<f32>) -> Array1<i64> {
    let n = x.len();
    if n < 3 {
        return Array1::zeros(0);
    }

    let mut midpoints: Vec<i64> = Vec::with_capacity(n / 2);
    let mut left_edges = Vec::with_capacity(n / 2);
    let mut right_edges = Vec::with_capacity(n / 2);
    let mut m = 0usize;

    let mut i = 1usize;
    let i_max = n - 1;
    while i < i_max {
        if x[i - 1] < x[i] {
            let mut i_ahead = i + 1;
            while i_ahead < i_max && x[i_ahead] == x[i] {
                i_ahead += 1;
            }
            if x[i_ahead] < x[i] {
                left_edges.push(i);
                right_edges.push(i_ahead - 1);
                midpoints.push(((left_edges[m] + right_edges[m]) / 2) as i64);
                m += 1;
                i = i_ahead;
            }
        }
        i += 1;
    }

    Array1::from(midpoints)
}

/// Persistence barcode arrays for a cluster tree at minimum size `min_size`.
pub fn min_cluster_size_barcode(
    cluster_tree: &CondensedTree,
    n_points: i64,
    min_size: f32,
) -> (Array1<f32>, Array1<f32>, Array1<i32>, Array1<f32>) {
    let last_child = cluster_tree.child[cluster_tree.child.len() - 1];
    let n_nodes = (last_child - n_points + 1) as usize;

    let mut parents = vec![0i32; n_nodes];
    let mut lambda_deaths = vec![0.0f32; n_nodes];
    let mut size_deaths = vec![0.0f32; n_nodes];
    let mut size_births = vec![min_size; n_nodes];
    lambda_deaths[0] = 0.0;
    size_deaths[0] = n_points as f32;
    parents[0] = n_points as i32;

    let n_rows = cluster_tree.child.len();
    for idx in (1..n_rows).rev().step_by(2) {
        let out_idx = (cluster_tree.child[idx] - n_points) as usize;
        let parent_val = cluster_tree.parent[idx] as i32;
        let lambda_death = (-1.0 / cluster_tree.lambda_val[idx] as f64).exp() as f32;

        let death_size = cluster_tree.child_size[idx - 1].min(cluster_tree.child_size[idx]) as f32;

        for slot in &mut parents[out_idx - 1..=out_idx] {
            *slot = parent_val;
        }
        for slot in &mut lambda_deaths[out_idx - 1..=out_idx] {
            *slot = lambda_death;
        }
        for slot in &mut size_deaths[out_idx - 1..=out_idx] {
            *slot = death_size;
        }

        let parent_out = (cluster_tree.parent[idx] - n_points) as usize;
        size_births[parent_out] = size_births[out_idx - 1]
            .max(size_births[out_idx])
            .max(death_size);
    }

    (
        Array1::from(size_births),
        Array1::from(size_deaths),
        Array1::from(parents),
        Array1::from(lambda_deaths),
    )
}

/// Integrate persistence over size scales (left-open `(birth, death]` intervals).
pub fn compute_total_persistence(
    births: &Array1<f32>,
    deaths: &Array1<f32>,
    lambda_deaths: &Array1<f32>,
) -> (Array1<f32>, Array1<f32>) {
    let mut sizes: Vec<f32> = births.to_vec();
    sizes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sizes.dedup_by(|a, b| a.to_bits() == b.to_bits());

    let mut total_persistence = vec![0.0f32; sizes.len()];

    for i in 1..births.len() {
        let birth = births[i];
        let death = deaths[i];
        let lambda_death = lambda_deaths[i];

        if death <= birth {
            continue;
        }

        let birth_idx = sizes.iter().position(|&s| s >= birth).unwrap_or(0);
        let death_idx = sizes
            .iter()
            .position(|&s| s >= death)
            .unwrap_or(sizes.len());

        for k in birth_idx..death_idx {
            total_persistence[k] += (death - birth) * lambda_death;
        }
    }

    (Array1::from(sizes), Array1::from(total_persistence))
}

/// Label and strength vectors for explicitly selected cluster ids.
pub fn extract_clusters_by_id(
    condensed_tree: &CondensedTree,
    selected_ids: &[i64],
) -> (Array1<i64>, Array1<f32>) {
    let n_samples = condensed_tree.parent[0] as usize;
    let labels = get_cluster_label_vector(condensed_tree, selected_ids, 0.0_f64, n_samples);
    let strengths = get_point_membership_strength_vector(condensed_tree, selected_ids, &labels);
    (labels, strengths)
}

/// Jaccard index of two integer sets given as slices.
pub fn jaccard_similarity(set_a: &[i64], set_b: &[i64]) -> f64 {
    let mut union_set: FxHashSet<i64> = set_a.iter().copied().collect();
    let mut intersection_count = 0usize;

    for &item in set_b {
        if union_set.contains(&item) {
            intersection_count += 1;
        } else {
            union_set.insert(item);
        }
    }

    let union_count = union_set.len();
    if union_count > 0 {
        intersection_count as f64 / union_count as f64
    } else {
        0.0
    }
}

/// Jaccard similarity of clusters active at two birth sizes.
pub fn estimate_cluster_similarity(
    births: &Array1<f32>,
    deaths: &Array1<f32>,
    birth_a: f32,
    birth_b: f32,
) -> f64 {
    let mut clusters_a = Vec::new();
    for i in 0..births.len() {
        if births[i] <= birth_a && deaths[i] > birth_a {
            clusters_a.push(i as i64);
        }
    }

    let mut clusters_b = Vec::new();
    for i in 0..births.len() {
        if births[i] <= birth_b && deaths[i] > birth_b {
            clusters_b.push(i as i64);
        }
    }

    jaccard_similarity(&clusters_a, &clusters_b)
}

/// Greedy peak selection by persistence with Jaccard diversity on birth sizes.
pub fn select_diverse_peaks(
    peaks: &[i64],
    total_persistence: &Array1<f32>,
    sizes: &Array1<f32>,
    births: &Array1<f32>,
    deaths: &Array1<f32>,
    min_similarity_threshold: f64,
    max_layers: usize,
) -> Array1<i64> {
    if peaks.is_empty() {
        return Array1::zeros(0);
    }

    let mut sorted_indices: Vec<usize> = (0..peaks.len()).collect();
    sorted_indices.sort_by(|&a, &b| {
        total_persistence[peaks[b] as usize]
            .partial_cmp(&total_persistence[peaks[a] as usize])
            .unwrap()
    });

    let mut selected_peaks = Vec::with_capacity(max_layers);
    let mut selected_births = Vec::with_capacity(max_layers);

    for &i in &sorted_indices {
        if selected_peaks.len() >= max_layers {
            break;
        }

        let peak = peaks[i];
        let birth_size = sizes[peak as usize];

        let mut is_diverse = true;
        for &selected_birth in &selected_births {
            let similarity =
                estimate_cluster_similarity(births, deaths, birth_size, selected_birth);
            if similarity > min_similarity_threshold {
                is_diverse = false;
                break;
            }
        }

        if is_diverse {
            selected_peaks.push(peak);
            selected_births.push(birth_size);
        }
    }

    Array1::from(selected_peaks)
}

pub(crate) fn binary_search_for_n_clusters_inner(
    uncondensed_tree: &crate::cluster_trees::Linkage,
    approx_n_clusters: usize,
    n_samples: usize,
) -> (Array1<i64>, Array1<i64>, Array1<f32>) {
    let mut lower_bound_min_cluster_size = 2i64;
    let mut upper_bound_min_cluster_size = (n_samples / 2) as i64;

    let upper_tree = condense_tree(uncondensed_tree, upper_bound_min_cluster_size);
    let upper_leaves = extract_leaves(&upper_tree, true);
    let mut upper_n_clusters = upper_leaves.len();

    let lower_tree = condense_tree(uncondensed_tree, lower_bound_min_cluster_size);
    let lower_leaves = extract_leaves(&lower_tree, true);
    let mut lower_n_clusters = lower_leaves.len();

    while upper_bound_min_cluster_size - lower_bound_min_cluster_size > 1 {
        let mid_min_cluster_size =
            ((lower_bound_min_cluster_size + upper_bound_min_cluster_size) as f64 / 2.0).round()
                as i64;

        if mid_min_cluster_size == lower_bound_min_cluster_size
            || mid_min_cluster_size == upper_bound_min_cluster_size
        {
            break;
        }

        let mid_tree = condense_tree(uncondensed_tree, mid_min_cluster_size);
        let mid_leaves = extract_leaves(&mid_tree, true);
        let mid_n_clusters = mid_leaves.len();

        if mid_n_clusters < approx_n_clusters {
            upper_bound_min_cluster_size = mid_min_cluster_size;
            upper_n_clusters = mid_n_clusters;
        } else {
            lower_bound_min_cluster_size = mid_min_cluster_size;
            lower_n_clusters = mid_n_clusters;
        }
    }

    let lower_dist = (lower_n_clusters as i64 - approx_n_clusters as i64).unsigned_abs();
    let upper_dist = (upper_n_clusters as i64 - approx_n_clusters as i64).unsigned_abs();

    if lower_dist < upper_dist {
        let lower_tree = condense_tree(uncondensed_tree, lower_bound_min_cluster_size);
        let leaves = extract_leaves(&lower_tree, true);
        let leaf_ids: Vec<i64> = leaves.to_vec();
        let clusters = get_cluster_label_vector(&lower_tree, &leaf_ids, 0.0_f64, n_samples);
        let strengths = get_point_membership_strength_vector(&lower_tree, &leaf_ids, &clusters);
        (leaves, clusters, strengths)
    } else if lower_dist > upper_dist {
        let upper_tree = condense_tree(uncondensed_tree, upper_bound_min_cluster_size);
        let leaves = extract_leaves(&upper_tree, true);
        let leaf_ids: Vec<i64> = leaves.to_vec();
        let clusters = get_cluster_label_vector(&upper_tree, &leaf_ids, 0.0_f64, n_samples);
        let strengths = get_point_membership_strength_vector(&upper_tree, &leaf_ids, &clusters);
        (leaves, clusters, strengths)
    } else {
        let lower_tree = condense_tree(uncondensed_tree, lower_bound_min_cluster_size);
        let lower_leaves = extract_leaves(&lower_tree, true);
        let lower_leaf_ids: Vec<i64> = lower_leaves.to_vec();
        let lower_clusters =
            get_cluster_label_vector(&lower_tree, &lower_leaf_ids, 0.0_f64, n_samples);

        let upper_tree = condense_tree(uncondensed_tree, upper_bound_min_cluster_size);
        let upper_leaves = extract_leaves(&upper_tree, true);
        let upper_leaf_ids: Vec<i64> = upper_leaves.to_vec();
        let upper_clusters =
            get_cluster_label_vector(&upper_tree, &upper_leaf_ids, 0.0_f64, n_samples);

        let lower_labeled = lower_clusters.iter().filter(|&&l| l >= 0).count();
        let upper_labeled = upper_clusters.iter().filter(|&&l| l >= 0).count();

        if lower_labeled > upper_labeled {
            let strengths =
                get_point_membership_strength_vector(&lower_tree, &lower_leaf_ids, &lower_clusters);
            (lower_leaves, lower_clusters, strengths)
        } else {
            let strengths =
                get_point_membership_strength_vector(&upper_tree, &upper_leaf_ids, &upper_clusters);
            (upper_leaves, upper_clusters, strengths)
        }
    }
}

/// Build MST linkage, binary-search `min_cluster_size`, and return labels and strengths.
pub fn binary_search_for_n_clusters(
    data: &Array2<f32>,
    approx_n_clusters: usize,
    min_samples: i64,
) -> (Array1<i64>, Array1<f32>) {
    let tree = build_kdtree(data.clone(), 40);
    let edges = parallel_boruvka(&tree, min_samples, false, rayon::current_num_threads());

    let mut sort_order: Vec<usize> = (0..edges.nrows()).collect();
    sort_order.sort_by(|&a, &b| edges[[a, 2]].partial_cmp(&edges[[b, 2]]).unwrap());

    let sorted_mst = edges.select(Axis(0), &sort_order);
    let uncondensed_tree = mst_to_linkage_tree(sorted_mst.view());
    let n_samples = data.nrows();

    let (_leaves, clusters, strengths) =
        binary_search_for_n_clusters_inner(&uncondensed_tree, approx_n_clusters, n_samples);

    (clusters, strengths)
}

fn build_cluster_tree_mapping(labels: &[Array1<i64>]) -> Vec<(usize, i64, usize, i64)> {
    let n_layers = labels.len();
    let mut mapping: Vec<(usize, i64, usize, i64)> = Vec::new();
    let mut found: Vec<FxHashSet<i64>> = (0..n_layers).map(|_| FxHashSet::default()).collect();

    for upper_layer in 1..n_layers {
        let upper_labels = &labels[upper_layer];
        let mut upper_unique: Vec<i64> = upper_labels.to_vec();
        upper_unique.sort_unstable();
        upper_unique.dedup();

        for lower_layer in (0..upper_layer).rev() {
            let lower_labels = &labels[lower_layer];

            let mut order: Vec<usize> = (0..upper_labels.len()).collect();
            order.sort_by_key(|&i| upper_labels[i]);

            let max_shifted = upper_labels
                .iter()
                .map(|&l| l + 1)
                .max()
                .unwrap_or(0)
                .max(0) as usize;
            let mut counts = vec![0usize; max_shifted + 1];
            for &l in upper_labels.iter() {
                counts[(l + 1) as usize] += 1;
            }

            let mut split_positions = Vec::new();
            let mut cum = 0usize;
            for i in 0..counts.len() {
                cum += counts[i];
                if i + 1 < counts.len() {
                    split_positions.push(cum);
                }
            }

            let sorted_lower: Vec<i64> = order.iter().map(|&i| lower_labels[i]).collect();
            let mut groups: Vec<Vec<i64>> = Vec::new();
            let mut start = 0usize;
            for &end in &split_positions {
                groups.push(sorted_lower[start..end].to_vec());
                start = end;
            }
            groups.push(sorted_lower[start..].to_vec());

            for (i, &label) in upper_unique.iter().enumerate() {
                if label >= 0 {
                    for &child in &groups[i] {
                        if child >= 0 && !found[lower_layer].contains(&child) {
                            mapping.push((upper_layer, label, lower_layer, child));
                            found[lower_layer].insert(child);
                        }
                    }
                }
            }
        }
    }

    for lower_layer in (0..n_layers).rev() {
        let max_label = labels[lower_layer].iter().copied().max().unwrap_or(-1);
        for child in 0..=max_label {
            if child >= 0 && !found[lower_layer].contains(&child) {
                mapping.push((n_layers, 0, lower_layer, child));
            }
        }
    }

    mapping
}

/// Parent/child adjacency between layers and cluster ids.
pub fn build_cluster_tree(labels: &[Array1<i64>]) -> HashMap<(usize, i64), Vec<(usize, i64)>> {
    let mut result: HashMap<(usize, i64), Vec<(usize, i64)>> = HashMap::new();
    for (parent_layer, parent_cluster, child_layer, child_cluster) in
        build_cluster_tree_mapping(labels)
    {
        result
            .entry((parent_layer, parent_cluster))
            .or_default()
            .push((child_layer, child_cluster));
    }
    result
}

/// Pairs of duplicate point indices from k-NN structure.
pub fn find_duplicates(knn_inds: &Array2<i32>, knn_dists: &Array2<f32>) -> HashSet<(usize, usize)> {
    let duplicate_distance = knn_dists.column(0).iter().copied().fold(f32::NAN, f32::max);
    let mut duplicates = HashSet::new();

    for i in 0..knn_inds.nrows() {
        for j in 0..knn_inds.ncols() {
            if knn_dists[[i, j]] <= duplicate_distance {
                let k = knn_inds[[i, j]] as usize;
                if i < k {
                    duplicates.insert((i, k));
                } else if k < i {
                    duplicates.insert((k, i));
                }
            }
        }
    }

    duplicates
}
