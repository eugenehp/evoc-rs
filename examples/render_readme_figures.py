#!/usr/bin/env python3
"""Generate README figures (Rust EVoC + matplotlib). Writes to examples/images/."""

from __future__ import annotations

import argparse
import colorsys
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
IMAGES = REPO / "examples" / "images"
TMP = REPO / "examples" / "output" / "_readme_figures"

GOLDEN_RATIO = 0.618033988749895
NOISE_COLOR = "#9aa0a6"

FASHION_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

BBC_CATEGORIES = ["business", "entertainment", "politics", "sport", "tech"]

NEWS20_CATEGORIES = [
    "alt.atheism",
    "comp.graphics",
    "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware",
    "comp.windows.x",
    "misc.forsale",
    "rec.autos",
    "rec.motorcycles",
    "rec.sport.baseball",
    "rec.sport.hockey",
    "sci.crypt",
    "sci.electronics",
    "sci.med",
    "sci.space",
    "soc.religion.christian",
    "talk.politics.guns",
    "talk.politics.mideast",
    "talk.politics.misc",
    "talk.religion.misc",
]


def hue_color(index: int) -> str:
    hue = (index * GOLDEN_RATIO) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.8 if index % 2 == 0 else 0.65, 0.92)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def pca2(embedding: np.ndarray) -> np.ndarray:
    from sklearn.decomposition import PCA

    return PCA(n_components=2, random_state=0).fit_transform(embedding)


def cluster_colors(labels: np.ndarray) -> dict[int, str]:
    ids = sorted(int(c) for c in np.unique(labels) if c >= 0)
    return {cid: hue_color(i) for i, cid in enumerate(ids)}


def cargo(*args: str) -> None:
    print("  $", "cargo", "run", "--release", *args)
    subprocess.check_call(["cargo", "run", "--release", *args], cwd=REPO)


def subsample_openml(name: str, n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.datasets import fetch_openml

    ds = fetch_openml(name, version=1, as_frame=False, parser="liac-arff")
    x = ds.data.astype(np.float32)
    y = ds.target.astype(np.int64)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(x), size=min(n, len(x)), replace=False)
    flat = x[idx]
    truth = y[idx]
    pixels = flat.reshape(-1, 28, 28)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    data = (flat / np.maximum(norms, 1e-12)).astype(np.float32)
    return data, truth, pixels


def run_rust_mnist(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels_path = TMP / f"mnist_{n}_{seed}_labels.npy"
    emb_path = TMP / f"mnist_{n}_{seed}_emb.npy"
    truth_path = TMP / f"mnist_{n}_{seed}_truth.npy"
    pixels_path = TMP / f"mnist_{n}_{seed}_pixels.npy"
    cargo(
        "--bin",
        "mnist_labels",
        "--",
        "--mnist",
        str(n),
        str(seed),
        str(labels_path),
        str(emb_path),
        str(truth_path),
        str(pixels_path),
    )
    pixels = np.load(pixels_path).reshape(-1, 28, 28)
    return (
        np.load(labels_path).astype(np.int64),
        np.load(emb_path).astype(np.float32),
        np.load(truth_path).astype(np.int64),
        pixels,
    )


def run_rust_fashion(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels_path = TMP / f"fashion_{n}_{seed}_labels.npy"
    emb_path = TMP / f"fashion_{n}_{seed}_emb.npy"
    truth_path = TMP / f"fashion_{n}_{seed}_truth.npy"
    pixels_path = TMP / f"fashion_{n}_{seed}_pixels.npy"
    cargo(
        "--example",
        "fashion_mnist_clustering",
        "--",
        str(n),
        str(seed),
        str(labels_path),
        str(emb_path),
        str(truth_path),
        str(pixels_path),
    )
    pixels = np.load(pixels_path).reshape(-1, 28, 28)
    return (
        np.load(labels_path).astype(np.int64),
        np.load(emb_path).astype(np.float32),
        np.load(truth_path).astype(np.int64),
        pixels,
    )


def run_rust_news(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels_path = TMP / f"news_{n}_{seed}_labels.npy"
    emb_path = TMP / f"news_{n}_{seed}_emb.npy"
    truth_path = TMP / f"news_{n}_{seed}_truth.npy"
    cargo(
        "--example",
        "news_clustering",
        "--",
        str(n),
        str(seed),
        str(labels_path),
        str(emb_path),
        str(truth_path),
    )
    return (
        np.load(labels_path).astype(np.int64),
        np.load(emb_path).astype(np.float32),
        np.load(truth_path).astype(np.int64),
    )


def dominant_class(mask: np.ndarray, truth: np.ndarray, names: list[str]) -> str:
    vals, counts = np.unique(truth[mask], return_counts=True)
    label = int(vals[int(np.argmax(counts))])
    return names[label] if label < len(names) else str(label)


def render_digit_figure(
    title: str,
    labels: np.ndarray,
    embedding: np.ndarray,
    truth: np.ndarray,
    pixels: np.ndarray,
    class_names: list[str],
    out_path: Path,
    *,
    max_clusters: int = 8,
    samples_per_cluster: int = 5,
    seed: int = 0,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    colors = cluster_colors(labels)
    xy = pca2(embedding)
    valid = labels >= 0
    cluster_ids = sorted(int(c) for c in np.unique(labels[valid]))[:max_clusters]

    fig = plt.figure(figsize=(14, 10), facecolor="white")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.2], hspace=0.3)
    ax = fig.add_subplot(gs[0])
    legend_handles: list = []

    noise = labels < 0
    if noise.any():
        ax.scatter(xy[noise, 0], xy[noise, 1], c=NOISE_COLOR, s=8, alpha=0.4, linewidths=0)
        legend_handles.append(Patch(facecolor=NOISE_COLOR, label=f"noise (n={int(noise.sum())})"))

    for cid in sorted(int(c) for c in np.unique(labels) if c >= 0):
        mask = labels == cid
        dom = dominant_class(mask, truth, class_names)
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            c=colors[cid],
            s=14,
            alpha=0.85,
            linewidths=0.2,
            edgecolors="white",
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=colors[cid],
                markersize=7,
                label=f"C{cid} · {dom[:14]} · n={int(mask.sum())}",
            )
        )

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_facecolor("#fafafa")
    ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)

    montage_gs = gs[1].subgridspec(len(cluster_ids), 1 + samples_per_cluster, wspace=0.06, hspace=0.3)
    rng = np.random.default_rng(seed)
    for row, cid in enumerate(cluster_ids):
        mask = np.flatnonzero(labels == cid)
        dom = dominant_class(mask, truth, class_names)
        ax_m = fig.add_subplot(montage_gs[row, 0])
        ax_m.imshow(pixels[mask].mean(axis=0), cmap="gray", vmin=0, vmax=255)
        ax_m.set_title(f"C{cid}\n{dom[:12]}", fontsize=7, color=colors[cid])
        ax_m.set_xticks([])
        ax_m.set_yticks([])
        pick = mask if mask.size <= samples_per_cluster else rng.choice(mask, samples_per_cluster, replace=False)
        for col, idx in enumerate(pick[:samples_per_cluster], start=1):
            ax_s = fig.add_subplot(montage_gs[row, col])
            ax_s.imshow(pixels[idx], cmap="gray", vmin=0, vmax=255)
            ax_s.set_xticks([])
            ax_s.set_yticks([])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def topic_short(name: str) -> str:
    return name.split(".")[-1].replace("_", " ")[:22]


def run_rust_bbc(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels_path = TMP / f"bbc_{n}_{seed}_labels.npy"
    emb_path = TMP / f"bbc_{n}_{seed}_emb.npy"
    truth_path = TMP / f"bbc_{n}_{seed}_truth.npy"
    cargo(
        "--example",
        "bbc_news_clustering",
        "--",
        str(n),
        str(seed),
        str(labels_path),
        str(emb_path),
        str(truth_path),
    )
    return (
        np.load(labels_path).astype(np.int64),
        np.load(emb_path).astype(np.float32),
        np.load(truth_path).astype(np.int64),
    )


def render_news_figure(
    labels: np.ndarray,
    embedding: np.ndarray,
    truth: np.ndarray,
    category_names: list[str],
    out_path: Path,
    *,
    n_samples: int,
    dataset_name: str = "20 Newsgroups",
    truth_title: str = "Ground truth · 20 newsgroups",
) -> None:
    """Side-by-side: ground-truth topics (left) vs EVoC clusters (right)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    xy = pca2(embedding)
    cmap = plt.get_cmap("tab20")

    fig, (ax_truth, ax_clust) = plt.subplots(
        1, 2, figsize=(16, 7), facecolor="white", constrained_layout=True
    )

    # Left: known newsgroup (always 20 clear groups).
    for label in sorted(np.unique(truth)):
        mask = truth == label
        color = cmap(int(label) % 20)
        ax_truth.scatter(
            xy[mask, 0],
            xy[mask, 1],
            c=[color],
            s=14,
            alpha=0.75,
            linewidths=0,
        )
    truth_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=cmap(i % 20),
            markersize=6,
            label=topic_short(category_names[i]),
        )
        for i in sorted(np.unique(truth).astype(int))
    ]
    ax_truth.set_title(truth_title, fontsize=12)
    ax_truth.set_xlabel("PCA 1")
    ax_truth.set_ylabel("PCA 2")
    ax_truth.set_facecolor("#fafafa")
    ax_truth.legend(
        handles=truth_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=6.5,
        ncol=1,
        title="Topic",
    )

    # Right: EVoC clusters (noise de-emphasized).
    cluster_colors_map = cluster_colors(labels)
    noise = labels < 0
    if noise.any():
        ax_clust.scatter(
            xy[noise, 0],
            xy[noise, 1],
            c=NOISE_COLOR,
            s=6,
            alpha=0.25,
            linewidths=0,
            zorder=1,
        )
    clust_handles = []
    if noise.any():
        clust_handles.append(Patch(facecolor=NOISE_COLOR, label=f"noise (n={int(noise.sum())})"))
    for cid in sorted(int(c) for c in np.unique(labels) if c >= 0):
        mask = labels == cid
        dom = dominant_class(mask, truth, category_names)
        ax_clust.scatter(
            xy[mask, 0],
            xy[mask, 1],
            c=cluster_colors_map[cid],
            s=20,
            alpha=0.85,
            linewidths=0.25,
            edgecolors="white",
            zorder=2,
        )
        clust_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=cluster_colors_map[cid],
                markersize=7,
                label=f"C{cid} · {topic_short(dom)} · n={int(mask.sum())}",
            )
        )
    n_clusters = len([c for c in np.unique(labels) if c >= 0])
    ax_clust.set_title(f"Rust EVoC · {n_clusters} clusters · n={n_samples}", fontsize=12)
    ax_clust.set_xlabel("PCA 1")
    ax_clust.set_ylabel("PCA 2")
    ax_clust.set_facecolor("#fafafa")
    ax_clust.legend(
        handles=clust_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=7,
        title="Cluster",
    )

    fig.suptitle(f"{dataset_name} (hashed bag-of-words features)", fontsize=13, y=1.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def render_scatter_figure(
    title: str,
    labels: np.ndarray,
    embedding: np.ndarray,
    truth: np.ndarray,
    category_names: list[str],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = cluster_colors(labels)
    xy = pca2(embedding)

    fig, ax = plt.subplots(figsize=(11, 8), facecolor="white")
    noise = labels < 0
    if noise.any():
        ax.scatter(xy[noise, 0], xy[noise, 1], c=NOISE_COLOR, s=10, alpha=0.35, linewidths=0)

    handles = []
    for cid in sorted(int(c) for c in np.unique(labels) if c >= 0):
        mask = labels == cid
        dom = dominant_class(mask, truth, category_names)
        short = dom.split(".")[-1][:18]
        ax.scatter(xy[mask, 0], xy[mask, 1], c=colors[cid], s=18, alpha=0.8, linewidths=0.2, edgecolors="white")
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=colors[cid],
                markersize=8,
                label=f"C{cid} · {short} · n={int(mask.sum())}",
            )
        )

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_facecolor("#fafafa")
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def render_synthetic_figure(out_path: Path, n: int, seed: int) -> None:
    from sklearn.datasets import make_blobs

    data, _ = make_blobs(n_samples=n, n_features=64, centers=4, cluster_std=1.8, random_state=seed)
    data = data.astype(np.float32)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    data = data / np.maximum(norms, 1e-12)
    data_path = TMP / f"synthetic_{n}_{seed}.npy"
    labels_path = TMP / f"synthetic_{n}_{seed}_labels.npy"
    emb_path = TMP / f"synthetic_{n}_{seed}_emb.npy"
    np.save(data_path, data)
    cargo("--bin", "mnist_labels", "--", str(data_path), str(seed), str(labels_path), str(emb_path))
    labels = np.load(labels_path).astype(np.int64)
    embedding = np.load(emb_path).astype(np.float32)
    render_scatter_figure(
        f"Synthetic blobs · Rust EVoC · n={n}",
        labels,
        embedding,
        np.zeros(n, dtype=np.int64),
        [str(i) for i in range(10)],
        out_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-mnist", type=int, default=2000)
    parser.add_argument("--n-fashion", type=int, default=2000)
    parser.add_argument("--n-news", type=int, default=3000)
    parser.add_argument("--n-bbc", type=int, default=2225)
    parser.add_argument("--n-synthetic", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--only",
        choices=("mnist", "fashion", "news", "bbc", "synthetic", "all"),
        default="all",
    )
    args = parser.parse_args()

    try:
        import matplotlib  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError as e:
        raise SystemExit("pip install matplotlib scikit-learn") from e

    IMAGES.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)

    if args.only in ("mnist", "all"):
        print("=== MNIST ===")
        labels, emb, truth, pixels = run_rust_mnist(args.n_mnist, args.seed)
        render_digit_figure(
            f"MNIST · Rust EVoC · n={args.n_mnist} · seed={args.seed}",
            labels,
            emb,
            truth,
            pixels,
            [str(i) for i in range(10)],
            IMAGES / "mnist.png",
            seed=args.seed,
        )

    if args.only in ("fashion", "all"):
        print("=== Fashion-MNIST ===")
        labels, emb, truth, pixels = run_rust_fashion(args.n_fashion, args.seed)
        render_digit_figure(
            f"Fashion-MNIST · Rust EVoC · n={args.n_fashion}",
            labels,
            emb,
            truth,
            pixels,
            FASHION_CLASSES,
            IMAGES / "fashion_mnist.png",
            seed=args.seed + 1,
        )

    if args.only in ("news", "all"):
        print("=== 20 Newsgroups ===")
        labels, emb, truth = run_rust_news(args.n_news, args.seed)
        render_news_figure(
            labels,
            emb,
            truth,
            NEWS20_CATEGORIES,
            IMAGES / "news20.png",
            n_samples=args.n_news,
        )

    if args.only in ("bbc", "all"):
        print("=== BBC News ===")
        labels, emb, truth = run_rust_bbc(args.n_bbc, args.seed)
        render_news_figure(
            labels,
            emb,
            truth,
            BBC_CATEGORIES,
            IMAGES / "bbc_news.png",
            n_samples=args.n_bbc,
            dataset_name="BBC News",
            truth_title="Ground truth · 5 topics",
        )

    if args.only in ("synthetic", "all"):
        print("=== Synthetic ===")
        render_synthetic_figure(IMAGES / "synthetic.png", args.n_synthetic, args.seed)

    print("Done →", IMAGES)


if __name__ == "__main__":
    main()
