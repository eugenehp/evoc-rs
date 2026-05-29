#!/usr/bin/env python3
"""
Benchmark EVoC on MNIST with the same stage breakdown as bench_huge / record_benchmarks.

Data: MNIST pixels flattened to 784-d vectors, L2-normalized (cosine kNN).

Usage:
  .venv-parity/bin/python3 scripts/bench_mnist.py [n_samples] [seed]
  .venv-parity/bin/python3 scripts/bench_mnist.py sweep
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from parity_import import PROJECT_ROOT, prepend_evoc_package


def load_mnist(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Load n MNIST train samples (784-d), L2-normalized. Returns (data, labels)."""
    from sklearn.datasets import fetch_openml

    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    x = mnist.data.astype(np.float32)
    y = mnist.target.astype(np.int64)

    rng = np.random.RandomState(seed)
    if n <= len(x):
        idx = rng.choice(len(x), size=n, replace=False)
    else:
        # Same sizes as huge bench may exceed 60k; sample with replacement.
        idx = rng.choice(len(x), size=n, replace=True)

    data = x[idx]
    labels = y[idx]
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    data = (data / np.maximum(norms, 1e-12)).astype(np.float32)
    return data, labels


def cluster_distribution(cluster_labels: np.ndarray) -> dict:
    """Summary stats for cluster label histogram (noise = -1)."""
    labels = cluster_labels.astype(np.int64)
    n_noise = int(np.sum(labels < 0))
    valid = labels[labels >= 0]
    if valid.size == 0:
        return {
            "n_clusters": 0,
            "n_noise": n_noise,
            "largest": 0,
            "median": 0,
            "smallest": 0,
        }
    _, counts = np.unique(valid, return_counts=True)
    return {
        "n_clusters": int(len(counts)),
        "n_noise": n_noise,
        "largest": int(counts.max()),
        "median": int(np.median(counts)),
        "smallest": int(counts.min()),
    }


def bench_python_stages(data: np.ndarray, seed: int) -> dict:
    prepend_evoc_package()
    from evoc.graph_construction import neighbor_graph_matrix
    from evoc.knn_graph import knn_graph
    from evoc.label_propagation import label_propagation_init
    from evoc.node_embedding import node_embedding

    k = 15
    n = data.shape[0]
    n_comp = max(4, min(15, k // 4))
    approx = int(np.clip(8 * np.sqrt(n), 256, 16384))

    rng = np.random.RandomState(seed)
    t0 = time.perf_counter()
    nn_inds, nn_dists = knn_graph(data, n_neighbors=k, random_state=rng)
    t_knn = time.perf_counter() - t0

    t1 = time.perf_counter()
    graph = neighbor_graph_matrix(float(k), nn_inds, nn_dists, symmetrize=True)
    t_graph = time.perf_counter() - t1

    t2 = time.perf_counter()
    init = label_propagation_init(
        graph,
        n_label_prop_iter=20,
        n_embedding_epochs=50,
        approx_n_parts=approx,
        n_components=n_comp,
        scaling=0.5,
        random_scale=0.1,
        noise_level=0.5,
        random_state=rng,
        data=data,
    )
    t_init = time.perf_counter() - t2

    t3 = time.perf_counter()
    emb = node_embedding(
        graph,
        n_comp,
        50,
        initial_embedding=init.astype(np.float32, order="C"),
        initial_alpha=0.1,
        noise_level=0.5,
        random_state=rng,
        reproducible_flag=True,
    )
    t_emb = time.perf_counter() - t3

    t4 = time.perf_counter()
    clusterer = __import__("evoc").EVoC(random_state=seed, n_neighbors=k)
    fit_labels = clusterer.fit_predict(data)
    t_fit = time.perf_counter() - t4

    return {
        "knn_s": t_knn,
        "graph_s": t_graph,
        "init_s": t_init,
        "emb_s": t_emb,
        "fit_predict_s": t_fit,
        "labels": fit_labels,
        "n_layers": len(clusterer.cluster_layers_),
    }


def bench_rust_fit(data_path: Path, seed: int) -> tuple[float, str]:
    cmd = ["cargo", "run", "--release", "--bin", "bench", "--", str(data_path), str(seed)]
    t0 = time.perf_counter()
    out = subprocess.check_output(cmd, cwd=PROJECT_ROOT, text=True, stderr=subprocess.STDOUT)
    return time.perf_counter() - t0, out.strip()


def run_case(n: int, seed: int = 42) -> None:
    print(f"\n=== MNIST n={n} d=784 seed={seed} ===")
    data, true_labels = load_mnist(n, seed)
    out_dir = PROJECT_ROOT / "tests" / "fixtures" / "_bench_mnist"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / f"mnist_{n}_{seed}.npy"
    np.save(data_path, data)

    py = bench_python_stages(data, seed)
    dist = cluster_distribution(py["labels"])
    print(
        "python stages  "
        f"knn={py['knn_s']:.3f}s graph={py['graph_s']:.3f}s "
        f"init={py['init_s']:.3f}s emb={py['emb_s']:.3f}s "
        f"fit={py['fit_predict_s']:.3f}s layers={py['n_layers']}"
    )
    print(
        "cluster dist   "
        f"clusters={dist['n_clusters']} noise={dist['n_noise']} "
        f"size(largest/median/smallest)={dist['largest']}/{dist['median']}/{dist['smallest']}"
    )

    # Optional: overlap with digit labels (not ground-truth metric, just diagnostic)
    valid = py["labels"] >= 0
    if valid.any():
        # mean purity proxy: for each cluster, max digit fraction
        purities = []
        for c in np.unique(py["labels"][valid]):
            mask = py["labels"] == c
            digits = true_labels[mask]
            _, cnt = np.unique(digits, return_counts=True)
            purities.append(cnt.max() / cnt.sum())
        print(f"mean cluster digit-purity {float(np.mean(purities)):.3f}")

    try:
        wall, line = bench_rust_fit(data_path, seed)
        print(f"rust fit_predict wall={wall:.3f}s :: {line}")
    except subprocess.CalledProcessError as e:
        print(f"rust bench failed:\n{e.output}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        # Same order-of-magnitude sizes as record_benchmarks huge cases
        for n in [10_000, 50_000, 60_000]:
            run_case(n, seed=42)
        return

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    run_case(n, seed)


if __name__ == "__main__":
    main()
