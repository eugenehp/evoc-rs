#!/usr/bin/env python3
"""Generate golden outputs from Python EVoC for Rust parity tests."""

from __future__ import annotations

import importlib
import json
import time
from pathlib import Path

import numpy as np

from parity_import import EVOC_ROOT, PROJECT_ROOT, prepend_evoc_package

FIXTURES = PROJECT_ROOT / "tests" / "fixtures"


def make_data(n: int, d: int, centers: int, seed: int) -> np.ndarray:
    from sklearn.datasets import make_blobs

    data, _ = make_blobs(
        n_samples=n, n_features=d, centers=centers, random_state=seed
    )
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    return (data / np.maximum(norms, 1e-12)).astype(np.float32)


def import_python_evoc():
    prepend_evoc_package()
    return importlib.import_module("evoc")


def run_case(name: str, n: int, d: int, centers: int, seed: int) -> dict:
    evoc = import_python_evoc()
    data = make_data(n, d, centers, seed)

    t0 = time.perf_counter()
    clusterer = evoc.EVoC(random_state=seed, n_neighbors=15)
    labels_fit = clusterer.fit_predict(data)
    py_s = time.perf_counter() - t0

    out_dir = FIXTURES / name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "data.npy", data)
    np.save(out_dir / "nn_inds.npy", clusterer.nn_inds_.astype(np.int32))
    np.save(out_dir / "nn_dists.npy", clusterer.nn_dists_.astype(np.float32))

    from evoc.graph_construction import neighbor_graph_matrix

    inter = out_dir / "intermediates"
    inter.mkdir(exist_ok=True)
    graph = neighbor_graph_matrix(
        15.0, clusterer.nn_inds_, clusterer.nn_dists_, symmetrize=True
    )
    coo = graph.tocoo()
    np.savez(
        inter / "graph_coo.npz",
        rows=coo.row.astype(np.int32),
        cols=coo.col.astype(np.int32),
        data=coo.data.astype(np.float32),
        shape=np.array(graph.shape, dtype=np.int64),
    )
    graph.sort_indices()
    np.savez(
        inter / "graph_csr.npz",
        indptr=graph.indptr.astype(np.int64),
        indices=graph.indices.astype(np.int32),
        data=graph.data.astype(np.float32),
        shape=np.array(graph.shape, dtype=np.int64),
    )

    from evoc.knn_graph import knn_graph
    from evoc.label_propagation import label_propagation_init
    from evoc.node_embedding import node_embedding

    rng = np.random.RandomState(seed)
    nn_inds, nn_dists = knn_graph(data, n_neighbors=15, random_state=rng)

    def save_rng(tag: str) -> None:
        st = rng.get_state()
        np.save(inter / f"rng_{tag}_key.npy", st[1])
        np.savez(
            inter / f"rng_{tag}_meta.npz",
            pos=np.int32(st[2]),
            has_gauss=np.int32(st[3]),
            gauss=np.float64(st[4]),
        )

    save_rng("after_knn")
    graph2 = neighbor_graph_matrix(15.0, nn_inds, nn_dists, symmetrize=True)
    n_neighbors = 15
    n_components = min(max(n_neighbors // 4, 4), 15)
    init = label_propagation_init(
        graph2,
        n_label_prop_iter=20,
        n_embedding_epochs=50,
        approx_n_parts=int(np.clip(8 * np.sqrt(data.shape[0]), 256, 16384)),
        n_components=n_components,
        scaling=0.5,
        random_scale=0.1,
        noise_level=0.5,
        random_state=rng,
        data=data,
    )
    save_rng("after_init")
    init_for_emb = init.astype(np.float32, order="C")
    np.save(inter / "init_embedding.npy", init_for_emb.copy())
    emb = node_embedding(
        graph2,
        n_components,
        50,
        initial_embedding=init_for_emb,
        initial_alpha=0.1,
        noise_level=0.5,
        random_state=rng,
        reproducible_flag=True,
    )
    save_rng("after_emb")
    np.save(inter / "embedding.npy", emb.astype(np.float32))

    from evoc.clustering import build_cluster_layers

    layers, _, persist = build_cluster_layers(
        emb,
        min_samples=5,
        base_min_cluster_size=5,
        base_n_clusters=None,
        reproducible_flag=True,
        min_similarity_threshold=0.2,
        max_layers=10,
    )
    labels = labels_fit
    np.save(out_dir / "labels.npy", labels.astype(np.int64))

    if len(clusterer.persistence_scores_):
        np.save(
            out_dir / "persistence_scores.npy",
            np.asarray(clusterer.persistence_scores_, dtype=np.float32),
        )

    meta = {
        "n": n,
        "d": d,
        "centers": centers,
        "seed": seed,
        "n_clusters": int(len(set(labels[labels >= 0]))),
        "noise": int((labels == -1).sum()),
        "python_seconds": py_s,
        "n_layers": len(clusterer.cluster_layers_),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> int:
    cases = [
        ("small_200", 200, 32, 10, 42),
        ("medium_800", 800, 32, 20, 42),
        ("large_2000", 2000, 64, 40, 7),
    ]
    FIXTURES.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, n, d, c, seed in cases:
        print(f"Generating {name} ...")
        summary[name] = run_case(name, n, d, c, seed)
        print(f"  {summary[name]}")
    (FIXTURES / "summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
