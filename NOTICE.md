# Third-party notices

evoc-rs depends on the following open-source crates (non-exhaustive; see
`Cargo.lock` for the full dependency tree). Each is used under its own license.

| Crate | Typical license | Role |
|-------|-----------------|------|
| [ndarray](https://github.com/rust-ndarray/ndarray) | Apache-2.0 OR MIT | Dense arrays |
| [rayon](https://github.com/rayon-rs/rayon) | Apache-2.0 OR MIT | Data parallelism |
| [sprs](https://github.com/vbarrielle/rust-sprs) | Apache-2.0 OR MIT | Sparse CSR matrices |
| [faer](https://github.com/sarah-quinones/faer-rs) | MIT | SVD / linear algebra (label propagation) |
| [rand](https://github.com/rust-random/rand) | Apache-2.0 OR MIT | RNG (NumPy-compatible streams) |
| [thiserror](https://github.com/dtolnay/thiserror) | Apache-2.0 OR MIT | Error derives |
| [ndarray-npy](https://github.com/potatotoby/ndarray-npy) | MIT OR Apache-2.0 | `.npy` I/O |
| [ureq](https://github.com/algesten/ureq) | Apache-2.0 OR MIT | HTTP downloads (examples) |
| [flate2](https://github.com/rust-lang/flate2-rs) | Apache-2.0 OR MIT | gzip decompression |
| [tar](https://github.com/alexcrichton/tar-rs) | Apache-2.0 OR MIT | tar archives (20 Newsgroups) |
| [zip](https://github.com/zip-rs/zip2) | MIT | zip archives (BBC News) |
| [rlx-cpu](https://github.com/MIT-RLX/rlx) (`knn` feature) | GPL-3.0-only | f32 cosine dot (fast path, NEON on aarch64) |
| [rlx](https://github.com/MIT-RLX/rlx) (`rlx-cuda`, `rlx-mlx`, `rlx-rocm`, `rlx-wgpu`) | GPL-3.0-only | Optional RLX GPU backends |

f32 kNN: RLX dot product by default; deterministic parity uses `native/fast_cosine.c`
(`-ffast-math`, matches Numba goldens). Embedding distance uses pure Rust `rdist`.

Upstream **Python EVōC** may additionally depend on NumPy, Numba, and scikit-learn when
used for parity scripts — those are **not** runtime dependencies of the Rust crate.
