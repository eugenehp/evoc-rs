#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from parity_import import PROJECT_ROOT, prepend_evoc_package


def make_data(n: int, d: int, centers: int, seed: int) -> np.ndarray:
    from sklearn.datasets import make_blobs

    data, _ = make_blobs(n_samples=n, n_features=d, centers=centers, random_state=seed)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    return (data / np.maximum(norms, 1e-12)).astype(np.float32)


def bench_python(data: np.ndarray, seed: int) -> float:
    prepend_evoc_package()
    import evoc  # type: ignore

    t0 = time.perf_counter()
    clusterer = evoc.EVoC(random_state=seed, n_neighbors=15)
    _ = clusterer.fit_predict(data)
    return time.perf_counter() - t0


def bench_rust(data_path: Path, seed: int, rayon_threads: str | None) -> tuple[float, str]:
    env = os.environ.copy()
    if rayon_threads is None:
        env.pop("RAYON_NUM_THREADS", None)
    else:
        env["RAYON_NUM_THREADS"] = rayon_threads

    cmd = ["cargo", "run", "--release", "--bin", "bench", "--", str(data_path), str(seed)]
    t0 = time.perf_counter()
    out = subprocess.check_output(cmd, cwd=PROJECT_ROOT, env=env, text=True)
    dt = time.perf_counter() - t0
    return dt, out.strip()


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    d = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    centers = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42

    data = make_data(n, d, centers, seed)
    tmp = PROJECT_ROOT / "tests" / "fixtures" / "_bench_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    data_path = tmp / f"data_{n}_{d}_{centers}_{seed}.npy"
    np.save(data_path, data)

    # Python baseline (Numba threads controlled by env; parity_import defaults to 1)
    py_s = bench_python(data, seed)
    print(f"python seconds {py_s:.6f} (NUMBA_NUM_THREADS={os.environ.get('NUMBA_NUM_THREADS')})")

    # Rust: sweep Rayon thread counts
    thread_sweep = [None, "1", "2", "4", "8", "16"]
    for t in thread_sweep:
        dt, line = bench_rust(data_path, seed, t)
        label = "default" if t is None else t
        print(f"rust wall {dt:.6f} rayon={label} :: {line}")


if __name__ == "__main__":
    main()

