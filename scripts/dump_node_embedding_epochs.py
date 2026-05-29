#!/usr/bin/env python3
"""
Dump per-epoch embeddings for a fixture using upstream Python EVōC kernels.

Writes:
  tests/fixtures/<fixture>/intermediates/emb_epoch_000.npy, ...

This is used to diagnose tiny embedding drift between Rust and Python.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from parity_import import PROJECT_ROOT, prepend_evoc_package


def load_rng(inter: Path, tag: str) -> np.random.RandomState:
    key = np.load(inter / f"rng_{tag}_key.npy").astype(np.uint32)
    meta = np.load(inter / f"rng_{tag}_meta.npz")
    pos = int(meta["pos"])
    has_gauss = int(meta["has_gauss"])
    gauss = float(meta["gauss"])
    rng = np.random.RandomState()
    rng.set_state(("MT19937", key, pos, has_gauss, gauss))
    return rng


def main() -> None:
    fixture = sys.argv[1] if len(sys.argv) > 1 else "large_2000"
    out_every = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    n_epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 50

    prepend_evoc_package()
    from evoc.node_embedding import (  # type: ignore
        INT32_MAX,
        make_epochs_per_sample,
        node_embedding_epoch_repr,
    )
    from evoc.graph_construction import neighbor_graph_matrix  # type: ignore

    base = PROJECT_ROOT / "tests" / "fixtures" / fixture / "intermediates"
    # Build the same fuzzy graph used during fixture generation.
    nn_inds = np.load(base.parent / "nn_inds.npy")
    nn_dists = np.load(base.parent / "nn_dists.npy")
    graph = neighbor_graph_matrix(15.0, nn_inds, nn_dists, symmetrize=True)
    graph.sort_indices()

    # Save the exact CSR used for epoch dumps so Rust can load it.
    np.savez(
        base / "graph_csr_ref.npz",
        indptr=graph.indptr.astype(np.int64),
        indices=graph.indices.astype(np.int32),
        data=graph.data.astype(np.float32),
        shape=np.array(graph.shape, dtype=np.int64),
    )

    indptr = graph.indptr.astype(np.uint32, copy=False)
    indices = graph.indices.astype(np.uint32, copy=False)
    data = graph.data.astype(np.float32, copy=False)
    n_vertices = np.uint32(graph.shape[0])

    embedding = np.load(base / "init_embedding.npy").astype(np.float32, order="C", copy=True)
    epochs_per_sample = make_epochs_per_sample(data, n_epochs).astype(np.float32, order="C")
    epochs_per_negative_sample = epochs_per_sample.astype(np.float32, copy=True)
    epochs_per_negative_sample /= np.float32(1.0)
    epochs_per_negative_sample *= np.float32(1.5)
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    epoch_of_next_sample = epochs_per_sample.copy()

    updates = np.zeros_like(embedding)
    node_order = np.arange(int(n_vertices), dtype=np.uint32)
    gamma_schedule = np.linspace(0.5, 1.5, n_epochs)
    block_size = np.uint32(max(1024, int(n_vertices) // 8))
    dim = np.uint8(int(embedding.shape[1]))
    alpha0 = np.float32(0.1)
    noise_level = np.float32(0.5)

    rng = load_rng(base, "after_init")
    rng_val = rng.randint(INT32_MAX, size=n_epochs)

    out_dir = base / "emb_epochs"
    out_dir.mkdir(exist_ok=True)

    alpha = np.float32(alpha0)
    for n in range(n_epochs):
        node_embedding_epoch_repr(
            embedding,
            indptr,
            indices,
            n_vertices,
            epochs_per_sample,
            np.uint32(rng_val[n]),
            dim,
            alpha,
            epochs_per_negative_sample,
            epoch_of_next_negative_sample,
            epoch_of_next_sample,
            np.uint8(n),
            noise_level,
            gamma_schedule[n],
            updates,
            node_order,
            block_size,
        )
        updates *= (1.0 - float(alpha)) ** 2 * 0.5
        rng.shuffle(node_order)
        alpha = np.float32(alpha0 * (1.0 - (float(n) / float(n_epochs))))

        if (n % out_every) == 0:
            np.save(out_dir / f"emb_epoch_{n:03d}.npy", embedding.astype(np.float32, copy=False))

    print(str(out_dir))


if __name__ == "__main__":
    main()

