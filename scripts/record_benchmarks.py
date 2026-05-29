#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from parity_import import PROJECT_ROOT, prepend_evoc_package


def now_tag() -> str:
    # filesystem-friendly timestamp
    return time.strftime("%Y%m%d-%H%M%S")


def git_rev() -> str:
    try:
        # Repo may have no commits yet; fall back to "unborn".
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--verify", "HEAD"], cwd=PROJECT_ROOT, text=True
            ).strip()
        )
    except Exception:
        return "unborn"


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


def main() -> None:
    out_root = PROJECT_ROOT / "benches" / "history"
    out_root.mkdir(parents=True, exist_ok=True)
    tag = now_tag()
    out_dir = out_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "tag": tag,
        "git_rev": git_rev(),
        "platform": platform.platform(),
        "python": shutil.which("python3"),
        "rayon_threads": os.environ.get("RAYON_NUM_THREADS"),
        "numba_threads": os.environ.get("NUMBA_NUM_THREADS"),
        "timestamp": time.time(),
    }

    # Python benchmark on a stable generated dataset (kept here, independent of fixtures).
    n, d, centers, seed = 2000, 64, 10, 42
    data = make_data(n, d, centers, seed)
    np.save(out_dir / f"py_data_{n}_{d}_{centers}_{seed}.npy", data)
    py_s = bench_python(data, seed)
    meta["python_fit_predict_seconds"] = py_s

    # Rust stage-timing benchmark for larger synthetic sizes (no Python equivalent here).
    huge_cases = [
        (50_000, 128, 15, 200, 42),
        (100_000, 128, 15, 400, 42),
        (50_000, 768, 15, 200, 42),
    ]
    huge_out = []
    for n, d, k, centers, seed in huge_cases:
        cmd = [
            "cargo",
            "run",
            "--release",
            "--bin",
            "bench_huge",
            "--",
            str(n),
            str(d),
            str(k),
            str(centers),
            str(seed),
        ]
        p = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        huge_out.append(
            {
                "n": n,
                "d": d,
                "k": k,
                "centers": centers,
                "seed": seed,
                "exit_code": p.returncode,
                "output": p.stdout.strip(),
            }
        )
    meta["rust_bench_huge"] = huge_out

    # Per-backend fit_predict wall times (strict + compiled rlx-* backends).
    backend_cases = ["small_200", "large_2000"]
    backend_out = []
    for fixture in backend_cases:
        cmd = [
            "cargo",
            "run",
            "--release",
            "--bin",
            "bench_backends",
            "--features",
            "cluster,npy,bench-json,rlx-all",
            "--",
            fixture,
            "--runs",
            "3",
            "--warmup",
            "1",
            "--json",
        ]
        p = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        payload = None
        if p.returncode == 0 and p.stdout.strip():
            try:
                payload = json.loads(p.stdout)
            except json.JSONDecodeError:
                payload = {"raw": p.stdout.strip()}
        backend_out.append(
            {
                "fixture": fixture,
                "exit_code": p.returncode,
                "output": payload if payload is not None else p.stdout.strip(),
            }
        )
    meta["rust_bench_backends"] = backend_out

    # Rust criterion benchmark
    # Note: criterion writes to target/criterion; we snapshot it after the run.
    t0 = time.perf_counter()
    proc = subprocess.run(
        ["cargo", "bench", "--features", "cluster,npy,rlx-all", "--bench", "evoc_bench"],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    meta["rust_cargo_bench_wall_seconds"] = time.perf_counter() - t0
    meta["rust_cargo_bench_exit_code"] = proc.returncode
    (out_dir / "rust_cargo_bench_output.txt").write_text(proc.stdout)

    crit = PROJECT_ROOT / "target" / "criterion"
    if crit.exists():
        # Copy the whole criterion tree for repeatability.
        shutil.make_archive(str(out_dir / "criterion"), "zip", root_dir=crit)
        meta["criterion_zip"] = "criterion.zip"

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    print(str(out_dir))


if __name__ == "__main__":
    main()

