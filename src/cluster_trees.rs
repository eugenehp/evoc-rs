//! Condensed cluster trees and linkage conversions (port of `cluster_trees.py`).

use crate::disjoint_set::{ds_find, ds_rank_create, ds_union_by_rank};
use ndarray::{Array1, Array2, ArrayView2};
use rustc_hash::{FxHashMap, FxHashSet};

/// SciPy-style linkage matrix with shape `(n - 1, 4)`.
pub type Linkage = Array2<f64>;
/// Condensed hierarchical clustering tree.
#[derive(Clone, Debug)]
pub struct CondensedTree {
    pub parent: Array1<i64>,
    pub child: Array1<i64>,
    pub lambda_val: Array1<f32>,
    pub child_size: Array1<i64>,
}

impl CondensedTree {
    pub fn len(&self) -> usize {
        self.parent.len()
    }

    pub fn is_empty(&self) -> bool {
        self.parent.is_empty()
    }
}

struct LinkageMergeData {
    parent: Vec<i64>,
    size: Vec<i64>,
    next: i64,
}

fn create_linkage_merge_data(base_size: usize) -> LinkageMergeData {
    let total = 2 * base_size - 1;
    let mut size = vec![1i64; base_size];
    size.resize(total, 0);
    LinkageMergeData {
        parent: vec![-1i64; total],
        size,
        next: base_size as i64,
    }
}

fn linkage_merge_find(linkage_merge: &mut LinkageMergeData, mut node: i64) -> i64 {
    let mut relabel = node;
    while linkage_merge.parent[node as usize] != -1 && linkage_merge.parent[node as usize] != node {
        node = linkage_merge.parent[node as usize];
    }
    linkage_merge.parent[node as usize] = node;

    while linkage_merge.parent[relabel as usize] != node {
        let next_relabel = linkage_merge.parent[relabel as usize];
        linkage_merge.parent[relabel as usize] = node;
        relabel = next_relabel;
    }
    node
}

fn linkage_merge_join(linkage_merge: &mut LinkageMergeData, left: i64, right: i64) {
    let next = linkage_merge.next as usize;
    linkage_merge.size[next] =
        linkage_merge.size[left as usize] + linkage_merge.size[right as usize];
    linkage_merge.parent[left as usize] = linkage_merge.next;
    linkage_merge.parent[right as usize] = linkage_merge.next;
    linkage_merge.next += 1;
}

/// Convert a distance-sorted MST to scipy-style linkage, shape `(n - 1, 4)`.
pub fn mst_to_linkage_tree(sorted_mst: ArrayView2<f32>) -> Linkage {
    let n_edges = sorted_mst.nrows();
    let n_samples = n_edges + 1;
    let mut result = Array2::<f64>::zeros((n_edges, 4));
    let mut linkage_merge = create_linkage_merge_data(n_samples);

    for index in 0..n_edges {
        let left = sorted_mst[[index, 0]] as i64;
        let right = sorted_mst[[index, 1]] as i64;
        let delta = sorted_mst[[index, 2]] as f64;

        let left_component = linkage_merge_find(&mut linkage_merge, left);
        let right_component = linkage_merge_find(&mut linkage_merge, right);

        if left_component > right_component {
            result[[index, 0]] = left_component as f64;
            result[[index, 1]] = right_component as f64;
        } else {
            result[[index, 1]] = left_component as f64;
            result[[index, 0]] = right_component as f64;
        }

        result[[index, 2]] = delta;
        result[[index, 3]] = (linkage_merge.size[left_component as usize]
            + linkage_merge.size[right_component as usize]) as f64;

        linkage_merge_join(&mut linkage_merge, left_component, right_component);
    }

    result
}

fn bfs_from_hierarchy(hierarchy: &Linkage, bfs_root: i64, num_points: i64) -> Vec<i64> {
    let mut to_process = vec![bfs_root];
    let mut result = Vec::new();

    while !to_process.is_empty() {
        result.extend(to_process.iter().copied());
        let mut next_to_process = Vec::new();
        for &n in &to_process {
            if n >= num_points {
                let i = (n - num_points) as usize;
                next_to_process.push(hierarchy[[i, 0]] as i64);
                next_to_process.push(hierarchy[[i, 1]] as i64);
            }
        }
        to_process = next_to_process;
    }

    result
}

fn eliminate_branch(
    branch_node: i64,
    parent_node: i64,
    lambda_value: f32,
    parents: &mut [i64],
    children: &mut [i64],
    lambdas: &mut [f32],
    mut idx: usize,
    ignore: &mut [bool],
    hierarchy: &Linkage,
    num_points: i64,
) -> usize {
    if branch_node < num_points {
        parents[idx] = parent_node;
        children[idx] = branch_node;
        lambdas[idx] = lambda_value;
        idx += 1;
    } else {
        for sub_node in bfs_from_hierarchy(hierarchy, branch_node, num_points) {
            if sub_node < num_points {
                children[idx] = sub_node;
                parents[idx] = parent_node;
                lambdas[idx] = lambda_value;
                idx += 1;
            } else {
                ignore[sub_node as usize] = true;
            }
        }
    }
    idx
}

/// Build a condensed tree from a linkage matrix, filtering splits below `min_cluster_size`.
pub fn condense_tree(hierarchy: &Linkage, min_cluster_size: i64) -> CondensedTree {
    let root = 2 * hierarchy.nrows() as i64;
    let num_points = hierarchy.nrows() as i64 + 1;
    let mut next_label = num_points + 1;

    let node_list = bfs_from_hierarchy(hierarchy, root, num_points);

    let mut relabel = vec![0i64; (root + 1) as usize];
    relabel[root as usize] = num_points;

    let root_usize = root as usize;
    let mut parents = vec![1i64; root_usize];
    let mut children = vec![0i64; root_usize];
    let mut lambdas = vec![0.0f32; root_usize];
    let mut sizes = vec![1i64; root_usize];
    let mut ignore = vec![false; (root + 1) as usize];

    let mut idx = 0usize;

    for node in node_list {
        if ignore[node as usize] || node < num_points {
            continue;
        }

        let hrow = (node - num_points) as usize;
        let left = hierarchy[[hrow, 0]] as i64;
        let right = hierarchy[[hrow, 1]] as i64;
        let d = hierarchy[[hrow, 2]];
        let lambda_value = if d > 0.0 {
            (1.0 / d) as f32
        } else {
            f32::INFINITY
        };

        let parent_node = relabel[node as usize];

        let left_count = if left >= num_points {
            hierarchy[[(left - num_points) as usize, 3]] as i64
        } else {
            1
        };
        let right_count = if right >= num_points {
            hierarchy[[(right - num_points) as usize, 3]] as i64
        } else {
            1
        };

        if left < num_points && right_count >= min_cluster_size {
            relabel[right as usize] = parent_node;
            parents[idx] = parent_node;
            children[idx] = left;
            lambdas[idx] = lambda_value;
            idx += 1;
        } else if left_count < min_cluster_size && right_count >= min_cluster_size {
            relabel[right as usize] = parent_node;
            idx = eliminate_branch(
                left,
                parent_node,
                lambda_value,
                &mut parents,
                &mut children,
                &mut lambdas,
                idx,
                &mut ignore,
                hierarchy,
                num_points,
            );
        } else if left_count >= min_cluster_size && right_count < min_cluster_size {
            relabel[left as usize] = parent_node;
            idx = eliminate_branch(
                right,
                parent_node,
                lambda_value,
                &mut parents,
                &mut children,
                &mut lambdas,
                idx,
                &mut ignore,
                hierarchy,
                num_points,
            );
        } else if left_count < min_cluster_size && right_count < min_cluster_size {
            idx = eliminate_branch(
                left,
                parent_node,
                lambda_value,
                &mut parents,
                &mut children,
                &mut lambdas,
                idx,
                &mut ignore,
                hierarchy,
                num_points,
            );
            idx = eliminate_branch(
                right,
                parent_node,
                lambda_value,
                &mut parents,
                &mut children,
                &mut lambdas,
                idx,
                &mut ignore,
                hierarchy,
                num_points,
            );
        } else {
            relabel[left as usize] = next_label;
            parents[idx] = parent_node;
            children[idx] = next_label;
            lambdas[idx] = lambda_value;
            sizes[idx] = left_count;
            next_label += 1;
            idx += 1;

            relabel[right as usize] = next_label;
            parents[idx] = parent_node;
            children[idx] = next_label;
            lambdas[idx] = lambda_value;
            sizes[idx] = right_count;
            next_label += 1;
            idx += 1;
        }
    }

    CondensedTree {
        parent: Array1::from(parents[..idx].to_vec()),
        child: Array1::from(children[..idx].to_vec()),
        lambda_val: Array1::from(lambdas[..idx].to_vec()),
        child_size: Array1::from(sizes[..idx].to_vec()),
    }
}

/// Return cluster node ids that are leaves in the condensed tree.
pub fn extract_leaves(condensed_tree: &CondensedTree, _allow_single_cluster: bool) -> Array1<i64> {
    if condensed_tree.is_empty() {
        return Array1::zeros(0);
    }

    let n_nodes = condensed_tree.parent.iter().copied().max().unwrap_or(0) + 1;
    let n_points = condensed_tree.parent.iter().copied().min().unwrap_or(0);

    let mut leaf_indicator = vec![true; n_nodes as usize];
    for i in 0..n_points as usize {
        leaf_indicator[i] = false;
    }

    for (&parent, &child_size) in condensed_tree
        .parent
        .iter()
        .zip(condensed_tree.child_size.iter())
    {
        if child_size > 1 {
            leaf_indicator[parent as usize] = false;
        }
    }

    let leaves: Vec<i64> = leaf_indicator
        .iter()
        .enumerate()
        .filter_map(|(i, &is_leaf)| is_leaf.then_some(i as i64))
        .collect();

    Array1::from(leaves)
}

/// Subset a condensed tree by boolean mask (one entry per row).
pub fn mask_condensed_tree(condensed_tree: &CondensedTree, mask: &[bool]) -> CondensedTree {
    assert_eq!(mask.len(), condensed_tree.len());

    let mut parent = Vec::new();
    let mut child = Vec::new();
    let mut lambda_val = Vec::new();
    let mut child_size = Vec::new();

    for i in 0..condensed_tree.len() {
        if mask[i] {
            parent.push(condensed_tree.parent[i]);
            child.push(condensed_tree.child[i]);
            lambda_val.push(condensed_tree.lambda_val[i]);
            child_size.push(condensed_tree.child_size[i]);
        }
    }

    CondensedTree {
        parent: Array1::from(parent),
        child: Array1::from(child),
        lambda_val: Array1::from(lambda_val),
        child_size: Array1::from(child_size),
    }
}

fn max_lambdas(tree: &CondensedTree, clusters: &FxHashSet<i64>) -> FxHashMap<i64, f32> {
    let mut result: FxHashMap<i64, f32> = clusters.iter().map(|&c| (c, 0.0f32)).collect();

    for i in 0..tree.parent.len() {
        let cluster = tree.parent[i];
        if clusters.contains(&cluster) && tree.child_size[i] == 1 {
            let entry = result.entry(cluster).or_insert(0.0);
            *entry = entry.max(tree.lambda_val[i]);
        }
    }

    result
}

/// Label each sample with a cluster index, or `-1` for noise.
pub fn get_cluster_label_vector(
    tree: &CondensedTree,
    clusters: &[i64],
    cluster_selection_epsilon: f64,
    n_samples: usize,
) -> Array1<i64> {
    if clusters.len() == 1 {
        return get_single_cluster_label_vector(
            tree,
            clusters[0],
            cluster_selection_epsilon,
            n_samples,
        );
    }

    if tree.is_empty() {
        return Array1::from_elem(n_samples, -1);
    }

    let root_cluster = tree.parent.iter().copied().min().unwrap();
    let mut result = Array1::from_elem(n_samples, -1);

    let mut sorted_clusters: Vec<i64> = clusters.to_vec();
    sorted_clusters.sort_unstable();
    let cluster_label_map: FxHashMap<i64, i64> = sorted_clusters
        .iter()
        .enumerate()
        .map(|(n, &c)| (c, n as i64))
        .collect();

    let max_node = tree
        .parent
        .iter()
        .chain(tree.child.iter())
        .copied()
        .max()
        .unwrap_or(0)
        + 1;
    let mut disjoint_set = ds_rank_create(max_node as usize);
    let clusters_set: FxHashSet<i64> = clusters.iter().copied().collect();

    for i in 0..tree.parent.len() {
        let child = tree.child[i];
        let parent = tree.parent[i];
        if !clusters_set.contains(&child) {
            ds_union_by_rank(&mut disjoint_set, parent as i32, child as i32);
        }
    }

    for n in 0..n_samples {
        let cluster = ds_find(&mut disjoint_set, n as i32) as i64;
        if cluster <= root_cluster {
            result[n] = -1;
        } else {
            result[n] = cluster_label_map[&cluster];
        }
    }

    result
}

/// Label samples belonging to a single selected cluster.
pub fn get_single_cluster_label_vector(
    tree: &CondensedTree,
    cluster: i64,
    cluster_selection_epsilon: f64,
    n_samples: usize,
) -> Array1<i64> {
    if tree.is_empty() {
        return Array1::from_elem(n_samples, -1);
    }

    let mut result = Array1::from_elem(n_samples, -1);
    let max_lambda = tree
        .lambda_val
        .iter()
        .enumerate()
        .filter(|&(i, _)| tree.parent[i] == cluster)
        .map(|(_, &v)| v)
        .fold(f32::NEG_INFINITY, f32::max);

    for i in 0..tree.child.len() {
        let n = tree.child[i] as usize;
        if n >= n_samples {
            continue;
        }
        let cur_lambda = tree.lambda_val[i];
        if cluster_selection_epsilon > 0.0 {
            if cur_lambda >= (1.0 / cluster_selection_epsilon) as f32 {
                result[n] = 0;
            } else {
                result[n] = -1;
            }
        } else if cur_lambda >= max_lambda {
            result[n] = 0;
        }
    }

    result
}

/// Per-point membership strength in `[0, 1]` for the assigned cluster label.
pub fn get_point_membership_strength_vector(
    tree: &CondensedTree,
    clusters: &[i64],
    labels: &Array1<i64>,
) -> Array1<f32> {
    let mut result = Array1::<f32>::zeros(labels.len());
    let clusters_set: FxHashSet<i64> = clusters.iter().copied().collect();
    let deaths = max_lambdas(tree, &clusters_set);
    let root_cluster = tree.parent.iter().copied().min().unwrap();

    let mut sorted_clusters: Vec<i64> = clusters.to_vec();
    sorted_clusters.sort_unstable();
    let cluster_index_map: FxHashMap<i64, i64> = sorted_clusters
        .iter()
        .enumerate()
        .map(|(n, &c)| (n as i64, c))
        .collect();

    for i in 0..tree.child.len() {
        let point = tree.child[i];
        if point >= root_cluster || labels[point as usize] < 0 {
            continue;
        }

        let cluster = cluster_index_map[&labels[point as usize]];
        let max_lambda = deaths[&cluster];
        if max_lambda == 0.0 || !tree.lambda_val[i].is_finite() {
            result[point as usize] = 1.0;
        } else {
            let lambda_val = tree.lambda_val[i].min(max_lambda);
            result[point as usize] = lambda_val / max_lambda;
        }
    }

    result
}
