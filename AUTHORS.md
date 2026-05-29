# Authors and attribution

## Upstream EVōC (Python)

**EVōC** — Embedding Vector Oriented Clustering — is developed by the
[Tutte Institute for Mathematics and Computing](https://github.com/TutteInstitute).

| Role | Name | Contact |
|------|------|---------|
| Author & maintainer | Leland McInnes | leland.mcinnes@gmail.com |
| Copyright holder | Tutte Institute for Mathematics and Computing | |

- Repository: https://github.com/TutteInstitute/evoc  
- Documentation: https://evoc.readthedocs.io/en/latest/

## evoc-rs (this repository)

| Role | Name |
|------|------|
| Author & maintainer (Rust port) | Eugene Hauptmann |

- Repository: https://github.com/eugenehp/evoc-rs

**evoc-rs** is a Rust reimplementation of the EVōC pipeline, maintained for:

- Native Rust embedding / clustering workflows (no Python runtime required)
- Golden-fixture parity against upstream on reference datasets
- Optional dataset loaders and examples (MNIST, Fashion-MNIST, BBC News, 20 Newsgroups)

The port includes kNN (NN-descent), fuzzy graph construction, label-propagation
initialization, UMAP-style node embedding, Borůvka MST, and HDBSCAN-style
multi-layer extraction.

When publishing work that uses **this crate**, please cite:

1. The **PLSCAN** paper (cluster extraction algorithm used by EVōC) — see [CITATION.md](CITATION.md)
2. The **EVōC** software (Python reference implementation)
3. **evoc-rs** (Rust implementation) — BibTeX key `evoc_rs2026` in [CITATION.bib](CITATION.bib)

## Third-party Rust dependencies

See [NOTICE.md](NOTICE.md) for licenses of bundled and runtime dependencies
(`ndarray`, `rayon`, `sprs`, `faer`, etc.).
