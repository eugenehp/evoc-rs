# Benchmarks

Criterion sources live here (`evoc_bench.rs`). Wall-time comparison uses the `bench_backends` binary and [`scripts/bench_backends.sh`](../scripts/bench_backends.sh).

## Per-backend comparison (recommended)

Times `fit_predict` for **strict** and every compiled RLX backend on the same fixture:

```bash
./scripts/bench_backends.sh large_2000
# or
cargo run --release --bin bench_backends --features "cluster,npy,bench-json,rlx-all" -- large_2000 --runs 5
```

Enable only the backends you need:

```bash
BENCH_FEATURES="cluster,npy,bench-json,rlx-mlx" ./scripts/bench_backends.sh small_200
EVOC_BACKEND=cuda BENCH_FEATURES="cluster,npy,bench-json,rlx-cuda" ./scripts/bench_backends.sh
```

JSON for scripts / history:

```bash
cargo run --release --bin bench_backends --features "cluster,npy,bench-json,rlx-all" -- large_2000 --json
```

## Criterion (`cargo bench`)

Statistical results under `target/criterion/`:

```bash
cargo bench --features "cluster,npy,rlx-all" --bench evoc_bench
EVOC_BACKEND=mlx cargo bench --features "cluster,npy,rlx-mlx" --bench evoc_bench
```

Run Criterion once per backend:

```bash
BENCH_CRITERION=1 ./scripts/bench_backends.sh
```

## History snapshots

```bash
.venv-parity/bin/python3 scripts/record_benchmarks.py
```

Writes Python wall time, `bench_backends` JSON (per backend), `bench_huge`, and Criterion output into `benches/history/`.
