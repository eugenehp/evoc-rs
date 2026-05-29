#!/usr/bin/env python3
"""
MNIST clustering example — Python vs Rust EVoC with aligned clusters.

Exports Python parity intermediates (graph + init + RNG), then runs Rust with the
same golden path used in tests so both backends produce matching cluster labels.

Outputs:
  examples/output/mnist_clusters_python.png
  examples/output/mnist_clusters_rust.png

Usage:
  .venv-parity/bin/python3 examples/mnist_clusters.py
  .venv-parity/bin/python3 examples/mnist_clusters.py 3000 42
"""

from __future__ import annotations

import argparse
import colorsys
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EXAMPLES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLES_DIR.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
OUTPUT_DIR = EXAMPLES_DIR / "output"
sys.path.insert(0, str(SCRIPTS_DIR))

from parity_import import PROJECT_ROOT as REPO_ROOT, prepend_evoc_package  # noqa: E402

GOLDEN_RATIO = 0.618033988749895
NOISE_COLOR = "#9aa0a6"


@dataclass
class ClusterRun:
    backend: str
    labels: np.ndarray
    embedding: np.ndarray
    n_clusters: int
    n_noise: int
    mean_purity: float


def load_mnist(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.datasets import fetch_openml

    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    x = mnist.data.astype(np.float32)
    y = mnist.target.astype(np.int64)

    rng = np.random.RandomState(seed)
    idx = rng.choice(len(x), size=n, replace=False) if n <= len(x) else rng.choice(len(x), size=n, replace=True)

    flat = x[idx]
    digits = y[idx]
    pixels = flat.reshape(-1, 28, 28)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    data = (flat / np.maximum(norms, 1e-12)).astype(np.float32)
    return data, digits, pixels


def export_parity_intermediates(data: np.ndarray, seed: int, inter: Path) -> None:
    """Write graph + init + RNG checkpoints (same layout as generate_parity_fixtures.py)."""
    prepend_evoc_package()
    from evoc.graph_construction import neighbor_graph_matrix
    from evoc.knn_graph import knn_graph
    from evoc.label_propagation import label_propagation_init

    inter.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    nn_inds, nn_dists = knn_graph(data, n_neighbors=15, random_state=rng)
    graph = neighbor_graph_matrix(15.0, nn_inds, nn_dists, symmetrize=True)

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
    n_components = min(max(15 // 4, 4), 15)
    approx = int(np.clip(8 * np.sqrt(data.shape[0]), 256, 16384))
    init = label_propagation_init(
        graph,
        n_label_prop_iter=20,
        n_embedding_epochs=50,
        approx_n_parts=approx,
        n_components=n_components,
        scaling=0.5,
        random_scale=0.1,
        noise_level=0.5,
        random_state=rng,
        data=data,
    )
    save_rng("after_init")
    np.save(inter / "init_embedding.npy", init.astype(np.float32, order="C"))


def run_python(data: np.ndarray, seed: int, inter: Path) -> ClusterRun:
    prepend_evoc_package()
    from evoc import EVoC

    clusterer = EVoC(random_state=seed, n_neighbors=15)
    labels = clusterer.fit_predict(data).astype(np.int64)
    embedding = run_python_embedding(data, seed, inter)
    return ClusterRun("python", labels, embedding, 0, 0, float("nan"))


def run_python_embedding(data: np.ndarray, seed: int, inter: Path) -> np.ndarray:
    prepend_evoc_package()
    from evoc.graph_construction import neighbor_graph_matrix
    from evoc.knn_graph import knn_graph
    from evoc.node_embedding import node_embedding

    rng = np.random.RandomState(seed)
    nn_inds, nn_dists = knn_graph(data, n_neighbors=15, random_state=rng)
    graph = neighbor_graph_matrix(15.0, nn_inds, nn_dists, symmetrize=True)
    init = np.load(inter / "init_embedding.npy")
    n_components = min(max(15 // 4, 4), 15)
    # Restore RNG to after_init checkpoint for embedding
    key = np.load(inter / "rng_after_init_key.npy")
    meta = np.load(inter / "rng_after_init_meta.npz")
    rng.set_state(
        ("MT19937", key, int(meta["pos"]), int(meta["has_gauss"]), float(meta["gauss"]))
    )
    emb = node_embedding(
        graph,
        n_components,
        50,
        initial_embedding=init.astype(np.float32, order="C"),
        initial_alpha=0.1,
        noise_level=0.5,
        random_state=rng,
        reproducible_flag=True,
    )
    np.save(inter / "embedding.npy", emb.astype(np.float32))
    return emb.astype(np.float32)


def run_rust(
    data: np.ndarray, seed: int, cache: Path, inter: Path
) -> tuple[np.ndarray, np.ndarray]:
    cache.mkdir(parents=True, exist_ok=True)
    stem = f"mnist_{data.shape[0]}_{seed}"
    data_path = cache / f"{stem}.npy"
    labels_path = cache / f"{stem}_rust_labels.npy"
    emb_path = cache / f"{stem}_rust_embedding.npy"
    np.save(data_path, data)

    subprocess.check_call(
        [
            "cargo",
            "run",
            "--release",
            "--bin",
            "mnist_labels",
            "--",
            str(data_path),
            str(seed),
            str(labels_path),
            str(emb_path),
            str(inter),
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return (
        np.load(labels_path).astype(np.int64),
        np.load(emb_path).astype(np.float32),
    )


def hue_color(index: int) -> str:
    hue = (index * GOLDEN_RATIO) % 1.0
    sat = 0.85 if index % 2 == 0 else 0.72
    val = 0.95 if index % 3 else 0.82
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def dominant_digit(digits: np.ndarray, mask: np.ndarray) -> int:
    vals, counts = np.unique(digits[mask], return_counts=True)
    return int(vals[int(np.argmax(counts))])


def summarize(labels: np.ndarray, digits: np.ndarray) -> tuple[int, int, float]:
    valid = labels >= 0
    n_noise = int(np.sum(~valid))
    if not valid.any():
        return 0, n_noise, float("nan")
    purities = []
    for cid in np.unique(labels[valid]):
        mask = labels == cid
        _, cnt = np.unique(digits[mask], return_counts=True)
        purities.append(cnt.max() / cnt.sum())
    return len(set(labels[valid])), n_noise, float(np.mean(purities))


def shared_cluster_colors(
    py_labels: np.ndarray,
    rust_labels: np.ndarray,
    digits: np.ndarray,
) -> tuple[dict[int, str], dict[int, str]]:
    """Same color for cluster pairs with high point overlap; C{id} uses same hue when ids match."""
    py_ids = sorted(int(c) for c in np.unique(py_labels) if c >= 0)
    rust_ids = sorted(int(c) for c in np.unique(rust_labels) if c >= 0)

    def iou(a: np.ndarray, ca: int, b: np.ndarray, cb: int) -> float:
        ma, mb = a == ca, b == cb
        u = int(np.sum(ma | mb))
        return int(np.sum(ma & mb)) / u if u else 0.0

    pairs: list[tuple[float, int, int]] = []
    for pid in py_ids:
        for rid in rust_ids:
            pairs.append((iou(py_labels, pid, rust_labels, rid), pid, rid))
    pairs.sort(reverse=True)

    py_colors: dict[int, str] = {}
    rust_colors: dict[int, str] = {}
    used_py: set[int] = set()
    used_rust: set[int] = set()
    palette_idx = 0

    for score, pid, rid in pairs:
        if pid in used_py or rid in used_rust or score < 0.5:
            continue
        color = hue_color(palette_idx)
        palette_idx += 1
        py_colors[pid] = color
        rust_colors[rid] = color
        used_py.add(pid)
        used_rust.add(rid)

    for cid in py_ids:
        if cid not in py_colors:
            py_colors[cid] = hue_color(palette_idx)
            palette_idx += 1
    for cid in rust_ids:
        if cid not in rust_colors:
            rust_colors[cid] = hue_color(palette_idx)
            palette_idx += 1

    return py_colors, rust_colors


def validate_run(run: ClusterRun, digits: np.ndarray) -> None:
    if run.labels.shape[0] != digits.shape[0]:
        raise RuntimeError(f"{run.backend}: label length mismatch")
    if run.n_clusters < 2:
        raise RuntimeError(f"{run.backend}: expected >=2 clusters, got {run.n_clusters}")
    if run.mean_purity < 0.5:
        raise RuntimeError(f"{run.backend}: mean digit purity {run.mean_purity:.3f} too low")
    print(
        f"  {run.backend}: OK — {run.n_clusters} clusters, {run.n_noise} noise, "
        f"purity={run.mean_purity:.3f}"
    )


def embedding_xy(embedding: np.ndarray) -> np.ndarray:
    from sklearn.decomposition import PCA

    return PCA(n_components=2, random_state=0).fit_transform(embedding)


def render_backend_figure(
    run: ClusterRun,
    pixels: np.ndarray,
    digits: np.ndarray,
    colors: dict[int, str],
    out_path: Path,
    *,
    max_clusters: int,
    samples_per_cluster: int,
    seed: int,
    subtitle: str,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    labels = run.labels
    xy = embedding_xy(run.embedding)
    valid = labels >= 0
    cluster_ids = sorted(int(c) for c in np.unique(labels[valid]))
    cluster_ids_display = cluster_ids[:max_clusters]

    fig = plt.figure(figsize=(15, 11), facecolor="white")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 1.35], hspace=0.28)
    ax = fig.add_subplot(gs[0])
    legend_handles: list = []

    noise = labels < 0
    if noise.any():
        ax.scatter(
            xy[noise, 0], xy[noise, 1], c=NOISE_COLOR, s=10, alpha=0.45, linewidths=0, zorder=1
        )
        legend_handles.append(
            Patch(facecolor=NOISE_COLOR, edgecolor="none", label=f"noise (n={int(noise.sum())})")
        )

    for cid in cluster_ids:
        mask = labels == cid
        dom = dominant_digit(digits, mask)
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            c=colors[cid],
            s=16,
            alpha=0.9,
            linewidths=0.3,
            edgecolors="white",
            zorder=2,
        )
        legend_handles.append(
            Line2D(
                [0], [0], marker="o", color="w", markerfacecolor=colors[cid],
                markeredgecolor="white", markersize=8,
                label=f"C{cid} · digit {dom} · n={int(mask.sum())}",
            )
        )

    ax.set_title(
        f"{run.backend.upper()} EVoC — {run.n_clusters} clusters, {run.n_noise} noise, "
        f"purity {run.mean_purity:.2f}\n{subtitle}",
        fontsize=11,
    )
    ax.set_xlabel("PCA axis 1")
    ax.set_ylabel("PCA axis 2")
    ax.set_facecolor("#fafafa")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(
        handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
        fontsize=7.5, frameon=True, title="Clusters",
    )

    montage_gs = gs[1].subgridspec(
        len(cluster_ids_display), 1 + samples_per_cluster, wspace=0.08, hspace=0.35
    )
    rng = np.random.default_rng(seed)
    for row, cid in enumerate(cluster_ids_display):
        mask = np.flatnonzero(labels == cid)
        dom = dominant_digit(digits, mask)
        ax_mean = fig.add_subplot(montage_gs[row, 0])
        ax_mean.imshow(pixels[mask].mean(axis=0), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax_mean.set_title(f"C{cid}\n{dom}", fontsize=8, color=colors[cid])
        ax_mean.set_xticks([])
        ax_mean.set_yticks([])
        for spine in ax_mean.spines.values():
            spine.set_edgecolor(colors[cid])
            spine.set_linewidth(2)
        pick = mask if mask.size <= samples_per_cluster else rng.choice(mask, size=samples_per_cluster, replace=False)
        for col, idx in enumerate(pick[:samples_per_cluster], start=1):
            ax_s = fig.add_subplot(montage_gs[row, col])
            ax_s.imshow(pixels[idx], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax_s.set_xticks([])
            ax_s.set_yticks([])
            ax_s.set_title(str(int(digits[idx])), fontsize=7)

    fig.suptitle(f"MNIST · {run.backend.upper()} · n={len(labels)} · seed={seed}", fontsize=14, y=0.99)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MNIST EVoC — aligned Python/Rust figures.")
    parser.add_argument("n_samples", nargs="?", type=int, default=3000)
    parser.add_argument("seed", nargs="?", type=int, default=42)
    parser.add_argument("--max-clusters", type=int, default=10)
    parser.add_argument("--samples", type=int, default=6)
    args = parser.parse_args()

    try:
        import matplotlib  # noqa: F401
    except ImportError as e:
        raise SystemExit("matplotlib required: pip install matplotlib") from e

    print(f"Loading MNIST n={args.n_samples} seed={args.seed} ...")
    data, digits, pixels = load_mnist(args.n_samples, args.seed)

    cache = REPO_ROOT / "tests" / "fixtures" / "_bench_mnist"
    inter = cache / f"parity_{args.n_samples}_{args.seed}"

    print("Exporting Python parity intermediates ...")
    export_parity_intermediates(data, args.seed, inter)

    print("Running Python EVoC (fit_predict) ...")
    py_run = run_python(data, args.seed, inter)
    py_run.n_clusters, py_run.n_noise, py_run.mean_purity = summarize(py_run.labels, digits)
    validate_run(py_run, digits)

    print("Running Rust EVoC (parity intermediates) ...")
    rust_labels, rust_emb = run_rust(data, args.seed, cache, inter)
    rust_run = ClusterRun("rust", rust_labels, rust_emb, 0, 0, float("nan"))
    rust_run.n_clusters, rust_run.n_noise, rust_run.mean_purity = summarize(rust_labels, digits)
    validate_run(rust_run, digits)

    mismatches = int(np.sum(py_run.labels != rust_run.labels))
    print(f"Python vs Rust label mismatches: {mismatches}/{len(py_run.labels)}")
    if mismatches > 0:
        print("  warning: parity path still differs — check init/embedding tolerance on this size")
        py_colors, rust_colors = shared_cluster_colors(py_run.labels, rust_run.labels, digits)
    else:
        cluster_ids = sorted(
            int(c) for c in np.unique(py_run.labels) if c >= 0
        )
        palette = {cid: hue_color(i) for i, cid in enumerate(cluster_ids)}
        py_colors = rust_colors = palette
        print("  clusters aligned — identical labels and matching colors per cluster id")

    render_backend_figure(
        py_run, pixels, digits, py_colors, OUTPUT_DIR / "mnist_clusters_python.png",
        max_clusters=args.max_clusters, samples_per_cluster=args.samples, seed=args.seed,
        subtitle="canonical Python pipeline",
    )
    render_backend_figure(
        rust_run, pixels, digits, rust_colors, OUTPUT_DIR / "mnist_clusters_rust.png",
        max_clusters=args.max_clusters, samples_per_cluster=args.samples, seed=args.seed + 1000,
        subtitle="Rust (Python parity cache) · labels match Python",
    )


if __name__ == "__main__":
    main()
