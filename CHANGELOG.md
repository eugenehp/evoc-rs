# Changelog

All notable changes to this project are documented here.

## 0.0.1 — 2026-05-28

First public release ([crates.io](https://crates.io/crates/evoc), [GitHub](https://github.com/eugenehp/evoc-rs)).

### Added
- Full EVōC pipeline in Rust: kNN, fuzzy graph, label-propagation init, UMAP-style embedding, Borůvka MST / multi-layer clustering
- `Evoc::fit_predict`, `embedding_`, optional `parity_graph_coo` for golden parity
- Dataset helpers and examples: MNIST, Fashion-MNIST, BBC News, 20 Newsgroups; in-memory / `.npy` examples
- Binaries: `mnist_fetch`, `fashion_mnist_fetch`, `mnist_labels`, `bench`, `emb_epoch_diff`
- Release docs: [ARCHITECTURE.md](ARCHITECTURE.md), [AUTHORS.md](AUTHORS.md), [CITATION.md](CITATION.md), [CONTRIBUTING.md](CONTRIBUTING.md), [NOTICE.md](NOTICE.md), [LICENSE](LICENSE)

### Changed
- f32 kNN: **rlx-cpu** NEON dot on the default fast path; **C `fast_cosine.c`** (`-ffast-math`) when `deterministic` for Python golden parity
- Label parity: 0 mismatches on `small_200`, `medium_800`, `large_2000` when using fixture checkpoints
- Major subsystems behind Cargo features (`cluster`, `knn`, `npy`, `datasets`, …); default `full`
- RLX compute backends: `rlx-cpu`, `rlx-cuda`, `rlx-mlx`, `rlx-rocm`, `rlx-wgpu` — each Cargo feature is independent; `EVOC_BACKEND` / `backend_smoke` run one backend at a time (delegate to strict until GPU kernels land)

## Pre-release development

### 0.3.2 — 2026-05-28

### Added
- BBC News dataset (`evoc::bbc_news_data`, `bbc_news_clustering` example, 5 topics)
- Shared text BoW helpers (`text_bow`)
- README cluster figures (`examples/images/`, `examples/render_readme_figures.py`)
- Fashion-MNIST download (`evoc::fashion_mnist_data`, `fashion_mnist_fetch`, `fashion_mnist_clustering` example)
- 20 Newsgroups download + hashed BoW (`evoc::news20_data`, `news_clustering` example)
- Rust MNIST download (`evoc::mnist_data`, `mnist_fetch` binary): CVDF IDX cache, subsample, L2-normalize
- `mnist_labels --mnist` runs EVoC without a pre-built `.npy` file
- MNIST clustering example (`examples/mnist_clusters.py`) with aligned Python/Rust figures
- `mnist_labels` binary: export labels and embedding from `.npy` data; optional parity directory
- `embedding_` field on `Evoc`, populated after `fit_predict`
- `graph_csr.npz` in parity fixtures for reproducible UMAP edge order
- `emb_epoch_diff` diagnostic binary and per-epoch embedding dumps
- `bench_mnist.py` benchmark script

### Fixed
- Label parity on golden fixtures (`small_200`, `medium_800`, `large_2000`) at 0 mismatches
- SciPy-compatible CSR matmul for label propagation expander
- UMAP reproducible epoch kernel (`node_embedding_epoch_repr`) drift reduced
- Missing `epoch_of_next_sample` update in non-reproducible embedding epoch

### Changed
- Parity tests use golden init checkpoint when `parity_graph_coo` is set (stable clustering)
- Pure Rust `rdist` for embedding (C `-ffast-math` rdist breaks label stability)
- `n_neg_samples` uses Python `int()` semantics via `usize`

### Known limitations
- Without parity intermediates, Rust and Python may diverge on large ad-hoc datasets (init/embedding drift)
- RLX GPU/CPU backends are scaffolding only

## 0.3.1

Initial public Rust port of EVoC with kNN, graph construction, label propagation, UMAP embedding, and HDBSCAN-style clustering.
