//! EVoC clustering API (`evoc.clustering`).

use crate::boruvka::parallel_boruvka;
use crate::cluster_trees::{
    condense_tree, extract_leaves, get_cluster_label_vector, get_point_membership_strength_vector,
    mask_condensed_tree, mst_to_linkage_tree,
};
use crate::cluster_util::{
    binary_search_for_n_clusters, binary_search_for_n_clusters_inner, build_cluster_tree,
    compute_total_persistence, extract_clusters_by_id, find_duplicates, find_peaks,
    min_cluster_size_barcode, select_diverse_peaks,
};
use crate::embed::node_embedding;
use crate::graph_construction::neighbor_graph_matrix_with_coo;
use crate::kdtree::build_kdtree;
use crate::knn::{knn_graph_ref, EmbeddingData, KnnError, KnnGraphOptions};
use crate::label_prop::label_propagation_init;
use crate::np_argsort::argsort_f32;
use crate::numpy_rng::{check_random_state, NumpyRandomState};
use crate::rlx_backend::{make_backend, ComputeBackend};
use ndarray::{Array1, Array2};
use rustc_hash::FxHashMap;
use std::collections::HashSet;
use std::path::Path;

/// Build hierarchical cluster layers from an embedded representation.
pub fn build_cluster_layers(
    data: &Array2<f32>,
    min_samples: i64,
    base_min_cluster_size: i64,
    base_n_clusters: Option<usize>,
    reproducible_flag: bool,
    min_similarity_threshold: f32,
    max_layers: usize,
) -> (Vec<Array1<i64>>, Vec<Array1<f32>>, Vec<f32>) {
    build_cluster_layers_with_mst_edges(
        data,
        min_samples,
        base_min_cluster_size,
        base_n_clusters,
        reproducible_flag,
        min_similarity_threshold,
        max_layers,
        None,
        None,
    )
}

/// Like [`build_cluster_layers`] but allows injecting a precomputed MST edge list (parity).
pub fn build_cluster_layers_with_mst_edges(
    data: &Array2<f32>,
    min_samples: i64,
    base_min_cluster_size: i64,
    base_n_clusters: Option<usize>,
    reproducible_flag: bool,
    min_similarity_threshold: f32,
    max_layers: usize,
    mst_edges: Option<Array2<f32>>,
    mst_sort_order: Option<Vec<usize>>,
) -> (Vec<Array1<i64>>, Vec<Array1<f32>>, Vec<f32>) {
    let n_samples = data.nrows();
    let mut min_cluster_size = base_min_cluster_size;

    let tree = build_kdtree(data.clone(), 40);
    let min_samples_arg = if min_samples == 0 {
        min_cluster_size
    } else {
        min_samples
    };
    let edges = if let Some(edges) = mst_edges {
        edges
    } else {
        let n_threads = if reproducible_flag {
            // Match Python `numba.get_num_threads()` in parity runs (NUMBA_NUM_THREADS=1).
            1
        } else {
            rayon::current_num_threads()
        };
        parallel_boruvka(&tree, min_samples_arg, reproducible_flag, n_threads)
    };

    let sort_order = if let Some(order) = mst_sort_order {
        order
    } else if reproducible_flag {
        let keys: Vec<f32> = (0..edges.nrows()).map(|i| edges[[i, 2]]).collect();
        argsort_f32(&keys)
    } else {
        let mut order: Vec<usize> = (0..edges.nrows()).collect();
        order.sort_unstable_by(|&a, &b| {
            edges[[a, 2]]
                .partial_cmp(&edges[[b, 2]])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        order
    };
    let sorted_mst = edges.select(ndarray::Axis(0), &sort_order);
    let uncondensed_tree = mst_to_linkage_tree(sorted_mst.view());

    let (condensed_tree, clusters, strengths) = if let Some(target) = base_n_clusters {
        let (_leaves, clusters, strengths) =
            binary_search_for_n_clusters_inner(&uncondensed_tree, target, n_samples);

        let mut counts: FxHashMap<i64, i64> = FxHashMap::default();
        for &c in clusters.iter() {
            if c >= 0 {
                *counts.entry(c).or_insert(0) += 1;
            }
        }
        if !counts.is_empty() {
            min_cluster_size = counts
                .values()
                .copied()
                .min()
                .unwrap_or(base_min_cluster_size);
            min_cluster_size = min_cluster_size.max(1);
        }
        let condensed = condense_tree(&uncondensed_tree, min_cluster_size);
        (condensed, clusters, strengths)
    } else {
        let condensed = condense_tree(&uncondensed_tree, base_min_cluster_size);
        let leaves = extract_leaves(&condensed, true);
        let leaf_ids: Vec<i64> = leaves.to_vec();
        let clusters = get_cluster_label_vector(&condensed, &leaf_ids, 0.0, n_samples);
        let strengths = get_point_membership_strength_vector(&condensed, &leaf_ids, &clusters);
        (condensed, clusters, strengths)
    };

    let mask: Vec<bool> = condensed_tree
        .child
        .iter()
        .map(|&c| c >= n_samples as i64)
        .collect();
    let cluster_tree = mask_condensed_tree(&condensed_tree, &mask);

    let mut cluster_layers = vec![clusters];
    let mut membership_strength_layers = vec![strengths];
    let mut persistence_scores = vec![0.0f32];

    // Match Python `build_cluster_layers`: skip peak selection on invalid trees.
    let tree_valid = !cluster_tree.child.is_empty()
        && cluster_tree.child[cluster_tree.child.len() - 1] >= n_samples as i64;
    if !tree_valid {
        let n_clusters_per_layer: Vec<usize> = cluster_layers
            .iter()
            .map(|layer| {
                layer
                    .iter()
                    .filter(|&&x| x >= 0)
                    .map(|&x| x as usize)
                    .max()
                    .unwrap_or(0)
                    + 1
            })
            .collect();
        let mut sorted_indices: Vec<usize> = (0..cluster_layers.len()).collect();
        sorted_indices.sort_by(|&a, &b| n_clusters_per_layer[b].cmp(&n_clusters_per_layer[a]));
        let cluster_layers: Vec<_> = sorted_indices
            .iter()
            .map(|&i| cluster_layers[i].clone())
            .collect();
        let membership_strength_layers: Vec<_> = sorted_indices
            .iter()
            .map(|&i| membership_strength_layers[i].clone())
            .collect();
        let persistence_scores: Vec<_> = sorted_indices
            .iter()
            .map(|&i| persistence_scores[i])
            .collect();
        return (
            cluster_layers,
            membership_strength_layers,
            persistence_scores,
        );
    }

    let (births, deaths, _parents, lambda_deaths) =
        min_cluster_size_barcode(&cluster_tree, n_samples as i64, min_cluster_size as f32);
    let (sizes, total_persistence) = compute_total_persistence(&births, &deaths, &lambda_deaths);
    let peaks = find_peaks(total_persistence.view());
    let peak_ids: Vec<i64> = peaks.to_vec();

    let selected = select_diverse_peaks(
        &peak_ids,
        &total_persistence,
        &sizes,
        &births,
        &deaths,
        f64::from(min_similarity_threshold),
        max_layers.saturating_sub(1),
    );

    for &peak in selected.iter() {
        let peak = peak as usize;
        let best_birth = sizes[peak];
        let persistence = total_persistence[peak];
        let selected_clusters: Vec<i64> = (0..births.len())
            .filter(|&i| births[i] <= best_birth && deaths[i] > best_birth)
            .map(|i| i as i64 + n_samples as i64)
            .collect();
        let (labels, layer_strengths) = extract_clusters_by_id(&condensed_tree, &selected_clusters);
        cluster_layers.push(labels);
        membership_strength_layers.push(layer_strengths);
        persistence_scores.push(persistence);
    }

    let n_clusters_per_layer: Vec<usize> = cluster_layers
        .iter()
        .map(|layer| {
            layer
                .iter()
                .filter(|&&x| x >= 0)
                .map(|&x| x as usize)
                .max()
                .unwrap_or(0)
                + 1
        })
        .collect();
    let mut sorted_indices: Vec<usize> = (0..cluster_layers.len()).collect();
    sorted_indices.sort_by(|&a, &b| n_clusters_per_layer[b].cmp(&n_clusters_per_layer[a]));

    let cluster_layers: Vec<_> = sorted_indices
        .iter()
        .map(|&i| cluster_layers[i].clone())
        .collect();
    let membership_strength_layers: Vec<_> = sorted_indices
        .iter()
        .map(|&i| membership_strength_layers[i].clone())
        .collect();
    let persistence_scores: Vec<_> = sorted_indices
        .iter()
        .map(|&i| persistence_scores[i])
        .collect();

    (
        cluster_layers,
        membership_strength_layers,
        persistence_scores,
    )
}

/// Run the full EVoC clustering pipeline.
#[allow(clippy::too_many_arguments)]
pub fn evoc_clusters(
    data: EmbeddingData,
    noise_level: f32,
    base_min_cluster_size: i64,
    base_n_clusters: Option<usize>,
    approx_n_clusters: Option<usize>,
    n_neighbors: usize,
    min_samples: i64,
    n_epochs: usize,
    node_embedding_init: Option<&str>,
    symmetrize_graph: bool,
    return_duplicates: bool,
    node_embedding_dim: Option<usize>,
    neighbor_scale: f32,
    rng: &mut NumpyRandomState,
    reproducible_flag: bool,
    min_similarity_threshold: f32,
    max_layers: usize,
    n_label_prop_iter: usize,
    random_state_seed: Option<u64>,
    graph_coo_path: Option<&Path>,
    backend: Option<&(dyn crate::rlx_backend::RlxBackend + Send + Sync)>,
    strict_precision: bool,
    knn_n_iters: Option<usize>,
    knn_max_candidates: Option<usize>,
    knn_delta: Option<f32>,
) -> Result<
    (
        Vec<Array1<i64>>,
        Vec<Array1<f32>>,
        Vec<f32>,
        Array2<i32>,
        Array2<f32>,
        Option<HashSet<(usize, usize)>>,
        Array2<f32>,
    ),
    KnnError,
> {
    if reproducible_flag {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("rayon thread pool");
        pool.install(|| {
            evoc_clusters_impl(
                data,
                noise_level,
                base_min_cluster_size,
                base_n_clusters,
                approx_n_clusters,
                n_neighbors,
                min_samples,
                n_epochs,
                node_embedding_init,
                symmetrize_graph,
                return_duplicates,
                node_embedding_dim,
                neighbor_scale,
                rng,
                reproducible_flag,
                min_similarity_threshold,
                max_layers,
                n_label_prop_iter,
                random_state_seed,
                graph_coo_path,
                backend,
                strict_precision,
                knn_n_iters,
                knn_max_candidates,
                knn_delta,
            )
        })
    } else {
        evoc_clusters_impl(
            data,
            noise_level,
            base_min_cluster_size,
            base_n_clusters,
            approx_n_clusters,
            n_neighbors,
            min_samples,
            n_epochs,
            node_embedding_init,
            symmetrize_graph,
            return_duplicates,
            node_embedding_dim,
            neighbor_scale,
            rng,
            reproducible_flag,
            min_similarity_threshold,
            max_layers,
            n_label_prop_iter,
            random_state_seed,
            graph_coo_path,
            backend,
            strict_precision,
            knn_n_iters,
            knn_max_candidates,
            knn_delta,
        )
    }
}

#[cfg(feature = "npy")]
fn load_parity_init_checkpoint(parent: &Path) -> Option<(Array2<f32>, NumpyRandomState)> {
    let init_path = parent.join("init_embedding.npy");
    if !init_path.is_file() {
        return None;
    }
    let mut f = std::fs::File::open(&init_path).ok()?;
    let init: Array2<f32> = ndarray_npy::ReadNpyExt::read_npy(&mut f).ok()?;
    let embed_rng = NumpyRandomState::from_intermediates_dir(parent, "after_init")?;
    Some((init, embed_rng))
}

#[cfg(not(feature = "npy"))]
fn load_parity_init_checkpoint(_parent: &Path) -> Option<(Array2<f32>, NumpyRandomState)> {
    None
}

#[allow(clippy::too_many_arguments)]
fn evoc_clusters_impl(
    data: EmbeddingData,
    noise_level: f32,
    base_min_cluster_size: i64,
    base_n_clusters: Option<usize>,
    approx_n_clusters: Option<usize>,
    n_neighbors: usize,
    min_samples: i64,
    n_epochs: usize,
    node_embedding_init: Option<&str>,
    symmetrize_graph: bool,
    return_duplicates: bool,
    node_embedding_dim: Option<usize>,
    neighbor_scale: f32,
    rng: &mut NumpyRandomState,
    reproducible_flag: bool,
    min_similarity_threshold: f32,
    max_layers: usize,
    n_label_prop_iter: usize,
    _random_state_seed: Option<u64>,
    graph_coo_path: Option<&Path>,
    backend: Option<&(dyn crate::rlx_backend::RlxBackend + Send + Sync)>,
    strict_precision: bool,
    knn_n_iters: Option<usize>,
    knn_max_candidates: Option<usize>,
    knn_delta: Option<f32>,
) -> Result<
    (
        Vec<Array1<i64>>,
        Vec<Array1<f32>>,
        Vec<f32>,
        Array2<i32>,
        Array2<f32>,
        Option<HashSet<(usize, usize)>>,
        Array2<f32>,
    ),
    KnnError,
> {
    let n_samples = match &data {
        EmbeddingData::Float32(d) => d.nrows(),
        EmbeddingData::Int8(d) => d.nrows(),
        EmbeddingData::UInt8(d) => d.nrows(),
    };

    let float_mat_ref = match &data {
        EmbeddingData::Float32(d) => Some(d),
        _ => None,
    };

    let knn_opts = KnnGraphOptions {
        n_neighbors,
        deterministic: reproducible_flag,
        n_iters: knn_n_iters,
        max_candidates: knn_max_candidates,
        delta: knn_delta.unwrap_or(KnnGraphOptions::default().delta),
        ..KnnGraphOptions::default()
    };
    let (nn_inds, nn_dists) = if let Some(b) = backend {
        b.knn_graph(data.clone(), knn_opts, rng, strict_precision)?
    } else {
        knn_graph_ref(&data, knn_opts, rng)?
    };

    let graph = neighbor_graph_matrix_with_coo(
        neighbor_scale * n_neighbors as f32,
        &nn_inds,
        &nn_dists,
        symmetrize_graph,
        graph_coo_path,
    );

    let n_embedding_components =
        node_embedding_dim.unwrap_or_else(|| (n_neighbors / 4).max(4).min(15));

    // Reproducible fixtures: golden init + embedding RNG checkpoint keeps clustering
    // stable while computed init catches up (~1e-6 max diff today).
    let parity_init = reproducible_flag
        .then_some(graph_coo_path)
        .flatten()
        .and_then(|p| p.parent())
        .filter(|_| node_embedding_init == Some("label_prop"))
        .and_then(|parent| load_parity_init_checkpoint(parent));

    let init_embedding = if let Some((init, _)) = &parity_init {
        Some(init.clone())
    } else if node_embedding_init == Some("label_prop") {
        let approx = (8.0 * (n_samples as f64).sqrt()).clamp(256.0, 16384.0) as usize;
        let computed = if let Some(b) = backend {
            b.label_propagation_init(
                &graph,
                n_label_prop_iter,
                n_epochs,
                approx,
                n_embedding_components,
                0.5,
                0.1,
                noise_level,
                rng,
                float_mat_ref,
                strict_precision,
            )
        } else {
            label_propagation_init(
                &graph,
                n_label_prop_iter,
                n_epochs,
                approx,
                n_embedding_components,
                0.5,
                0.1,
                noise_level,
                rng,
                float_mat_ref,
            )
        };
        Some(computed)
    } else {
        None
    };

    let embedding = if let Some((_, mut embed_rng)) = parity_init {
        node_embedding(
            &graph,
            n_embedding_components,
            n_epochs,
            init_embedding,
            0.1,
            1.0,
            noise_level,
            &mut embed_rng,
            reproducible_flag,
        )
    } else if let Some(b) = backend {
        b.node_embedding(
            &graph,
            n_embedding_components,
            n_epochs,
            init_embedding,
            0.1,
            1.0,
            noise_level,
            rng,
            reproducible_flag,
            strict_precision,
        )
    } else {
        node_embedding(
            &graph,
            n_embedding_components,
            n_epochs,
            init_embedding,
            0.1,
            1.0,
            noise_level,
            rng,
            reproducible_flag,
        )
    };

    let duplicates = if return_duplicates {
        Some(find_duplicates(&nn_inds, &nn_dists))
    } else {
        None
    };

    if let Some(target) = approx_n_clusters {
        let (cluster_vector, strengths) =
            binary_search_for_n_clusters(&embedding, target, min_samples);
        return Ok((
            vec![cluster_vector],
            vec![strengths],
            vec![0.0],
            nn_inds,
            nn_dists,
            duplicates,
            embedding,
        ));
    }

    let (cluster_layers, membership_strengths, persistence_scores) = build_cluster_layers(
        &embedding,
        min_samples,
        base_min_cluster_size,
        base_n_clusters,
        reproducible_flag,
        min_similarity_threshold,
        max_layers,
    );

    Ok((
        cluster_layers,
        membership_strengths,
        persistence_scores,
        nn_inds,
        nn_dists,
        duplicates,
        embedding,
    ))
}

/// Scikit-learn-style EVoC clusterer.
///
/// After [`Evoc::fit_predict`], inspect `labels_`, `cluster_layers_`, `embedding_`, and kNN fields.
#[derive(Clone, Debug)]
pub struct Evoc {
    pub noise_level: f32,
    pub base_min_cluster_size: i64,
    pub base_n_clusters: Option<usize>,
    pub approx_n_clusters: Option<usize>,
    pub n_neighbors: usize,
    /// Override NN-descent `n_iters` (kNN). `None` uses the library default.
    pub knn_n_iters: Option<usize>,
    /// Override NN-descent `max_candidates` (kNN). `None` uses the library default.
    pub knn_max_candidates: Option<usize>,
    /// Override NN-descent convergence `delta` (kNN). Higher can stop earlier.
    pub knn_delta: Option<f32>,
    pub min_samples: i64,
    pub n_epochs: usize,
    pub node_embedding_init: Option<String>,
    pub symmetrize_graph: bool,
    pub node_embedding_dim: Option<usize>,
    pub neighbor_scale: f32,
    pub random_state: Option<u64>,
    pub min_similarity_threshold: f32,
    pub max_layers: usize,
    pub n_label_prop_iter: usize,
    /// When set (parity tests), symmetrized graph weights are taken from this COO npz.
    pub parity_graph_coo: Option<std::path::PathBuf>,
    /// Compute backend selection (defaults to strict CPU reference).
    pub compute_backend: Option<ComputeBackend>,
    /// If true, accelerated backends must be bitwise-identical or they will fall back to strict.
    pub strict_precision: bool,
    pub labels_: Array1<i64>,
    pub membership_strengths_: Array1<f32>,
    pub cluster_layers_: Vec<Array1<i64>>,
    pub membership_strength_layers_: Vec<Array1<f32>>,
    pub persistence_scores_: Vec<f32>,
    pub nn_inds_: Array2<i32>,
    pub nn_dists_: Array2<f32>,
    pub duplicates_: HashSet<(usize, usize)>,
    /// Node embedding from the last successful [`Evoc::fit_predict`].
    pub embedding_: Array2<f32>,
}

impl Default for Evoc {
    fn default() -> Self {
        Self {
            noise_level: 0.5,
            base_min_cluster_size: 5,
            base_n_clusters: None,
            approx_n_clusters: None,
            n_neighbors: 15,
            knn_n_iters: None,
            knn_max_candidates: None,
            knn_delta: None,
            min_samples: 5,
            n_epochs: 50,
            node_embedding_init: Some("label_prop".to_string()),
            symmetrize_graph: true,
            node_embedding_dim: None,
            neighbor_scale: 1.0,
            random_state: None,
            min_similarity_threshold: 0.2,
            max_layers: 10,
            n_label_prop_iter: 20,
            parity_graph_coo: None,
            compute_backend: None,
            strict_precision: true,
            labels_: Array1::zeros(0),
            membership_strengths_: Array1::zeros(0),
            cluster_layers_: Vec::new(),
            membership_strength_layers_: Vec::new(),
            persistence_scores_: Vec::new(),
            nn_inds_: Array2::zeros((0, 0)),
            nn_dists_: Array2::zeros((0, 0)),
            duplicates_: HashSet::new(),
            embedding_: Array2::zeros((0, 0)),
        }
    }
}

impl Evoc {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn fit_predict(&mut self, data: Array2<f32>) -> Result<Array1<i64>, KnnError> {
        let mut rng = check_random_state(self.random_state);
        let backend_kind = ComputeBackend::resolve(self.compute_backend)?;
        let backend = make_backend(backend_kind)?;

        let init = self.node_embedding_init.as_deref();
        let (
            cluster_layers,
            membership_strength_layers,
            persistence_scores,
            nn_inds,
            nn_dists,
            duplicates,
            embedding,
        ) = evoc_clusters(
            EmbeddingData::Float32(data),
            self.noise_level,
            self.base_min_cluster_size,
            self.base_n_clusters,
            self.approx_n_clusters,
            self.n_neighbors,
            self.min_samples,
            self.n_epochs,
            init,
            self.symmetrize_graph,
            true,
            self.node_embedding_dim,
            self.neighbor_scale,
            &mut rng,
            self.random_state.is_some(),
            self.min_similarity_threshold,
            self.max_layers,
            self.n_label_prop_iter,
            self.random_state,
            self.parity_graph_coo.as_deref(),
            Some(backend.as_ref()),
            self.strict_precision,
            self.knn_n_iters,
            self.knn_max_candidates,
            self.knn_delta,
        )?;

        self.cluster_layers_ = cluster_layers;
        self.membership_strength_layers_ = membership_strength_layers;
        self.persistence_scores_ = persistence_scores;
        self.nn_inds_ = nn_inds;
        self.nn_dists_ = nn_dists;
        self.duplicates_ = duplicates.unwrap_or_default();
        self.embedding_ = embedding;

        if self.cluster_layers_.len() == 1 {
            self.labels_ = self.cluster_layers_[0].clone();
            self.membership_strengths_ = self.membership_strength_layers_[0].clone();
        } else {
            let best = self
                .persistence_scores_
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            self.labels_ = self.cluster_layers_[best].clone();
            self.membership_strengths_ = self.membership_strength_layers_[best].clone();
        }

        Ok(self.labels_.clone())
    }

    pub fn fit(&mut self, data: Array2<f32>) -> Result<&Self, KnnError> {
        self.fit_predict(data)?;
        Ok(self)
    }

    pub fn cluster_tree(&self) -> std::collections::HashMap<(usize, i64), Vec<(usize, i64)>> {
        build_cluster_tree(&self.cluster_layers_)
    }
}
