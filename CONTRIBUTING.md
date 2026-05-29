# Contributing

Thank you for your interest in **evoc-rs**. This project tracks the Python
[EVōC](https://github.com/TutteInstitute/evoc) reference; changes that affect
numerics or labels should preserve or update golden parity fixtures.

## Development setup

**Requirements**

- Rust 1.70+ (2021 edition)
- For parity scripts: Python 3.10+, upstream EVoC checkout, NumPy

```bash
git clone https://github.com/eugenehp/evoc-rs.git
cd evoc-rs
cargo build --release
cargo test --release
```

### Python parity environment (optional)

```bash
python3 -m venv .venv-parity
.venv-parity/bin/pip install numpy scipy scikit-learn
export EVOC_ROOT=/path/to/python/evoc   # TutteInstitute/evoc clone
.venv-parity/bin/python3 scripts/generate_parity_fixtures.py
```

Regenerate fixtures only when intentionally changing algorithms; commit updated
`tests/fixtures/**` with a clear CHANGELOG entry. Fixtures are **git-only** (excluded
from the published crate via `Cargo.toml` `exclude`).

## Running tests

```bash
# Full suite
cargo test --release

# Parity-focused
cargo test --release parity

# Single-threaded (matches Numba `NUMBA_NUM_THREADS=1` in fixtures)
RAYON_NUM_THREADS=1 cargo test --release parity -- --nocapture
```

| Test area | Files |
|-----------|--------|
| Smoke | `tests/smoke.rs` |
| Graph | `tests/graph_parity.rs` |
| kNN / labels | `tests/parity.rs` |
| Staged pipeline | `tests/parity_intermediates.rs` |

## Code style

- `cargo fmt` before submitting
- Match existing module boundaries (see [ARCHITECTURE.md](ARCHITECTURE.md))
- Prefer focused diffs; avoid drive-by refactors
- Document non-obvious numeric parity constraints in code comments

## Adding examples or datasets

1. Add a loader under `src/` (or extend `idx_digits` / `text_bow`)
2. Add `examples/your_example.rs` with `cargo run --release --example …`
3. Update `examples/README.md` and optional `render_readme_figures.py`
4. Note download URL, cache env var, and license in README

## Release checklist

- [ ] `cargo test --release`
- [ ] CHANGELOG.md updated
- [ ] Version bumped in `Cargo.toml`, `CITATION.cff`
- [ ] README / ARCHITECTURE accurate for new APIs
- [ ] LICENSE / NOTICE unchanged unless new bundled code

## Conduct

Be respectful and constructive. For upstream algorithm questions, consider opening
issues on [TutteInstitute/evoc](https://github.com/TutteInstitute/evoc) as well.
