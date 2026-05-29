#!/usr/bin/env bash
# Benchmark each compiled EVOC compute backend (strict + enabled rlx-*).
#
# Usage:
#   ./scripts/bench_backends.sh [FIXTURE]              # default: large_2000
#   BENCH_RUNS=10 ./scripts/bench_backends.sh medium_800
#   BENCH_CRITERION=1 ./scripts/bench_backends.sh      # also run Criterion per backend
#   BENCH_FEATURES="cluster,npy,rlx-cuda" EVOC_BACKEND=cuda ./scripts/bench_backends.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

FIXTURE="${1:-large_2000}"
FEATURES="${BENCH_FEATURES:-cluster,npy,bench-json,rlx-all}"
RUNS="${BENCH_RUNS:-5}"
WARMUP="${BENCH_WARMUP:-1}"

echo "==> bench_backends (features=${FEATURES}, fixture=${FIXTURE}, runs=${RUNS})"
cargo run --release --bin bench_backends --features "${FEATURES}" -- "${FIXTURE}" --runs "${RUNS}" --warmup "${WARMUP}"

if [[ "${BENCH_CRITERION:-0}" == "1" ]]; then
  echo "==> Criterion (one invocation per backend via EVOC_BACKEND)"
  for backend in strict cpu cuda mlx metal rocm wgpu gpu; do
    echo "--- ${backend} ---"
    if ! EVOC_BACKEND="${backend}" cargo bench --features "${FEATURES}" --bench evoc_bench -- --sample-size 10; then
      echo "(skip ${backend})"
    fi
  done
fi
