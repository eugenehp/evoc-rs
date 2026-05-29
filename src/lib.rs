//! # evoc — Embedding Vector Oriented Clustering (Rust)
//!
//! Rust port of [EVōC](https://github.com/TutteInstitute/evoc) for clustering
//! high-dimensional embedding vectors: kNN graph → label-propagation init →
//! UMAP-like node embedding → Borůvka MST / HDBSCAN-style multi-layer clustering.
//!
//! ## Cargo features
//!
//! | Feature | Description |
//! |---------|-------------|
//! | `full` *(default)* | `cluster` + `npy` + `datasets` |
//! | `cluster` | End-to-end [`Evoc`] API (implies `embed` → `init` → `graph` → `knn`) |
//! | `knn` | kNN graph construction (`rlx-cpu` fast path; C when deterministic) |
//! | `graph` | Fuzzy neighbor graph from kNN |
//! | `init` | Label-propagation initialization |
//! | `embed` | UMAP-style node embedding |
//! | `npy` | `.npy` / `.npz` load helpers for parity and benchmarks |
//! | `datasets` | MNIST, Fashion-MNIST, BBC News, 20 Newsgroups loaders |
//! | `rlx-cpu` … `rlx-wgpu` | Optional RLX compute backends (enable only what you need) |
//! | `rlx-all` | Convenience: enables all `rlx-*` backends |
//!
//! Minimal build: `evoc = { version = "0.0.1", default-features = false, features = ["cluster"] }`
//!
//! ## Quick example
//!
//! ```no_run
//! use evoc::Evoc;
//! use ndarray::Array2;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let data: Array2<f32> = Array2::zeros((100, 32)); // use L2-normalized rows in practice
//! let mut clusterer = Evoc {
//!     random_state: Some(42),
//!     ..Evoc::default()
//! };
//! let labels = clusterer.fit_predict(data)?;
//! let _embedding = clusterer.embedding_.clone();
//! # Ok(())
//! # }
//! ```
//!
//! ## Documentation
//!
//! - Repository README: pipeline overview, examples, parity
//! - [Architecture](https://github.com/eugenehp/evoc-rs/blob/main/ARCHITECTURE.md): module map and data flow
//! - [Citation](https://github.com/eugenehp/evoc-rs/blob/main/CITATION.md): PLSCAN,
//!   EVōC (Python), and **evoc-rs** (Rust; BibTeX key `evoc_rs2026`)
//!
//! ## License
//!
//! BSD-2-Clause; see LICENSE file in the repository root.

#![warn(unused_imports, dead_code)]

mod numpy_rng;

#[cfg(feature = "knn")]
mod fast_cosine;
#[cfg(feature = "knn")]
mod heap;
#[cfg(feature = "knn")]
mod kdtree;
#[cfg(feature = "knn")]
mod knn;
#[cfg(feature = "knn")]
mod nndescent;
#[cfg(feature = "knn")]
mod np_argsort;
#[cfg(feature = "cluster")]
mod rlx_backend;
#[cfg(feature = "knn")]
mod rng;

#[cfg(feature = "graph")]
mod csr_matmul;
#[cfg(feature = "graph")]
mod graph_construction;

#[cfg(feature = "init")]
mod label_prop;

#[cfg(feature = "embed")]
mod embed;

#[cfg(feature = "cluster")]
mod boruvka;
#[cfg(feature = "cluster")]
mod cluster_trees;
#[cfg(feature = "cluster")]
mod cluster_util;
#[cfg(feature = "cluster")]
mod clustering;
#[cfg(feature = "cluster")]
mod disjoint_set;

#[cfg(feature = "datasets")]
pub mod bbc_news_data;
#[cfg(feature = "datasets")]
mod dataset_util;
#[cfg(feature = "datasets")]
pub mod fashion_mnist_data;
#[cfg(feature = "datasets")]
pub mod idx_digits;
#[cfg(feature = "datasets")]
pub mod mnist_data;
#[cfg(feature = "datasets")]
pub mod news20_data;
#[cfg(feature = "datasets")]
mod text_bow;

#[cfg(feature = "cluster")]
pub use clustering::{
    build_cluster_layers, build_cluster_layers_with_mst_edges, evoc_clusters, Evoc,
};

#[cfg(feature = "embed")]
pub use embed::node_embedding;

#[cfg(feature = "graph")]
pub use graph_construction::{
    align_csr_values, neighbor_graph_matrix, neighbor_graph_matrix_with_coo,
};

#[cfg(all(feature = "graph", feature = "npy"))]
pub use graph_construction::{load_graph_coo_npz, load_graph_csr_npz};

#[cfg(feature = "knn")]
pub use knn::knn_graph_ref;
#[cfg(feature = "knn")]
pub use knn::{knn_graph, transform_distances_float, EmbeddingData, KnnError, KnnGraphOptions};

#[cfg(feature = "init")]
pub use label_prop::{label_prop_loop, label_propagation_init};

pub use numpy_rng::{check_random_state, NumpyRandomState};

#[cfg(feature = "cluster")]
pub use rlx_backend::{BackendError, ComputeBackend};

/// Helpers used by the golden-fixture parity tests.
///
/// Not part of the stable public API.
#[cfg(all(feature = "graph", feature = "init"))]
pub mod parity {
    pub use crate::csr_matmul::scipy_csr_matmul;
    pub use crate::graph_construction::smooth_knn_dist;
    pub use crate::label_prop::{
        csr_matmul_dense, normalize_cols_l2, normalize_rows_l1, partition_reduction_map,
    };
}

/// Low-level embedding kernels (unstable; diagnostics / bench tooling).
#[cfg(feature = "embed")]
pub mod embed_kernels {
    pub use crate::embed::{make_epochs_per_sample, node_embedding_epoch_repr};
}
