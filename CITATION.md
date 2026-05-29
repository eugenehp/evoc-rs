# Citation

## Algorithm and upstream software

EVōC implements density-based clustering on kNN graphs with multi-resolution layer
extraction. The cluster-tree extraction builds on **persistent multiscale density-based
clustering (PLSCAN)**.

If you use EVōC or this Rust port in academic work, cite:

### PLSCAN (recommended primary reference)

> D.M. Bot, L. McInnes, J. Aerts.  
> **Persistent Multiscale Density-based Clustering.**  
> arXiv preprint arXiv:2512.16558, 2025.  
> https://arxiv.org/abs/2512.16558

### Python EVōC (reference implementation)

> McInnes, L., Tutte Institute for Mathematics and Computing.  
> **EVōC: Embedding Vector Oriented Clustering** (version 0.3.x).  
> https://github.com/TutteInstitute/evoc  
> https://evoc.readthedocs.io/en/latest/

### evoc-rs (Rust implementation)

Cite this entry when your experiments use the **Rust crate** (not only the Python
reference). Replace the version with the tag or crate version you ran.

> Hauptmann, E.  
> **evoc: Embedding Vector Oriented Clustering (Rust implementation).**  
> Version 0.0.1, 2026.  
> https://github.com/eugenehp/evoc-rs  
> https://docs.rs/evoc/0.0.1/evoc  
> Rust port of [EVōC](https://github.com/TutteInstitute/evoc) with golden-fixture
> parity tests against the Python reference.

**Short form (in-text):**  
*evoc* 0.0.1 (Rust; Hauptmann, 2026)

**Version pin:** `Cargo.toml` dependency, `git describe`, or `cargo pkgid evoc`.

## BibTeX

Full file: [CITATION.bib](CITATION.bib). Rust-only entry:

```bibtex
@software{evoc_rs2026,
  title        = {{evoc}: Embedding Vector Oriented Clustering (Rust implementation)},
  author       = {Hauptmann, Eugene},
  year         = {2026},
  version      = {0.0.1},
  url          = {https://github.com/eugenehp/evoc-rs},
  note         = {Rust software. Port of EVōC. Documentation: https://docs.rs/evoc}
}
```

## Citation File Format

Machine-readable metadata: [CITATION.cff](CITATION.cff) (supported by GitHub and Zenodo).
