# Examples

Regenerate all README figures (Rust EVoC + matplotlib):

```bash
pip install matplotlib scikit-learn
python3 examples/render_readme_figures.py
```

Figures are saved under [`images/`](images/) and embedded below.

## In-memory vectors (`cluster_in_memory.rs`)

No files — build a `Vec<f32>` or `Array2<f32>` in your app and pass it straight to `Evoc`:

```bash
cargo run --release --example cluster_in_memory
```

![Synthetic blob clustering](images/synthetic.png)

## User embeddings (`user_clustering.rs`)

Load a float32 `.npy` matrix or run a synthetic demo.

```bash
cargo run --release --example user_clustering
cargo run --release --example user_clustering -- path/to/embeddings.npy 42
```

Rows are L2-normalized automatically (cosine kNN). After `fit_predict`, inspect
`clusterer.labels_`, `clusterer.embedding_`, and `clusterer.cluster_layers_`.

## Fashion-MNIST (`fashion_mnist_clustering.rs`)

Same pipeline as MNIST — 28×28 apparel images, 10 classes, Rust download.

```bash
cargo run --release --example fashion_mnist_clustering
cargo run --release --example fashion_mnist_clustering -- 3000 42
cargo run --release --bin fashion_mnist_fetch -- 3000 42 data.npy labels.npy
```

![Fashion-MNIST clustering](images/fashion_mnist.png)

Cache: `EVOC_FASHION_MNIST_DIR` or `~/.cache/evoc/fashion-mnist`.

## BBC News (`bbc_news_clustering.rs`)

Five topics (business, entertainment, politics, sport, tech) — small, readable text clustering demo.

```bash
cargo run --release --example bbc_news_clustering
cargo run --release --example bbc_news_clustering -- 2225 42
```

![BBC News: ground truth (left) vs EVoC clusters (right)](images/bbc_news.png)

Cache: `EVOC_BBC_NEWS_DIR` or `~/.cache/evoc/bbc-news` (~2.5 MB zip, [UCD BBC corpus](http://mlg.ucd.ie/datasets/bbc.html)).

## 20 Newsgroups (`news_clustering.rs`)

Downloads the classic by-date training corpus, strips headers, hashed bag-of-words
features (8192-d), then clusters. Reports **topic purity** vs newsgroup name.

```bash
cargo run --release --example news_clustering
cargo run --release --example news_clustering -- 2000 42
```

![20 Newsgroups: ground-truth topics (left) vs EVoC clusters (right)](images/news20.png)

The left panel shows the known newsgroup labels; the right panel shows unsupervised clusters from Rust EVoC.

Cache: `EVOC_NEWS20_DIR` or `~/.cache/evoc/news20` (~14 MB first download).

## MNIST clustering example

Python/Rust parity comparison script (optional, for aligned Python vs Rust figures):

```bash
.venv-parity/bin/pip install matplotlib scikit-learn
.venv-parity/bin/python3 examples/mnist_clusters.py 3000 42
```

Rust-only clustering figure (same style as other examples):

![MNIST clustering](images/mnist.png)

The `mnist_clusters.py` script writes additional side-by-side figures to `examples/output/`.
Rust uses Python-exported parity intermediates when you need **identical** labels on both backends.
