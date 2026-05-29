//! Cluster BBC News articles in pure Rust (5 topics, download → bag-of-words → EVoC).
//!
//! ```bash
//! cargo run --release --example bbc_news_clustering
//! cargo run --release --example bbc_news_clustering -- 2225 42 [labels.npy] [embedding.npy] [truth.npy]
//! ```

use evoc::bbc_news_data::{self, category_name, DEFAULT_FEATURE_DIM};
use evoc::Evoc;
use ndarray::Array1;
use ndarray_npy::WriteNpyExt;
use std::collections::HashMap;
use std::env;
use std::fs::File;

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let n: usize = env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(bbc_news_data::BBC_TRAIN_LEN);
    let seed: u64 = env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);

    let cache = bbc_news_data::cache_dir();
    eprintln!("BBC News n={n} seed={seed} cache={} ...", cache.display());
    eprintln!("(first run downloads ~2.5 MB zip; hashed BoW dim={DEFAULT_FEATURE_DIM})");
    let (data, topic_labels) =
        bbc_news_data::sample_normalized(n, seed, &cache, DEFAULT_FEATURE_DIM)?;
    eprintln!(
        "loaded {} × {} features (L2-normalized)",
        data.nrows(),
        data.ncols()
    );

    let mut clusterer = Evoc {
        random_state: Some(seed),
        n_neighbors: 30,
        base_min_cluster_size: 5,
        min_samples: 5,
        approx_n_clusters: Some(5),
        n_epochs: 80,
        n_label_prop_iter: 30,
        noise_level: 0.3,
        ..Evoc::default()
    };
    let labels = clusterer.fit_predict(data)?;

    if let Some(path) = env::args().nth(3) {
        Array1::from_iter(labels.iter().copied()).write_npy(File::create(path)?)?;
    }
    if let Some(path) = env::args().nth(4) {
        clusterer.embedding_.write_npy(File::create(path)?)?;
    }
    if let Some(path) = env::args().nth(5) {
        topic_labels.write_npy(File::create(path)?)?;
    }

    print_results(&labels, &topic_labels, &clusterer);
    Ok(())
}

fn print_results(labels: &ndarray::Array1<i64>, truth: &ndarray::Array1<u8>, clusterer: &Evoc) {
    let n_clusters = labels
        .iter()
        .filter(|&&l| l >= 0)
        .collect::<std::collections::HashSet<_>>()
        .len();
    let n_noise = labels.iter().filter(|&&l| l == -1).count();
    let purity = mean_topic_purity(labels, truth);

    eprintln!();
    eprintln!("Results");
    eprintln!("-------");
    eprintln!("articles:            {}", labels.len());
    eprintln!("clusters (non-noise): {n_clusters}");
    eprintln!("noise:               {n_noise}");
    eprintln!("mean topic purity:   {purity:.3}");
    eprintln!("resolution layers:   {}", clusterer.cluster_layers_.len());
    eprintln!("embedding dim:       {}", clusterer.embedding_.ncols());

    eprintln!();
    eprintln!("Clusters (dominant BBC topic):");
    let mut sizes: Vec<(i64, usize)> = labels
        .iter()
        .filter(|&&l| l >= 0)
        .fold(HashMap::new(), |mut m, &l| {
            *m.entry(l).or_insert(0) += 1;
            m
        })
        .into_iter()
        .collect();
    sizes.sort_by(|a, b| b.1.cmp(&a.1));
    for (cid, count) in sizes {
        let indices: Vec<usize> = labels
            .iter()
            .enumerate()
            .filter(|(_, &l)| l == cid)
            .map(|(i, _)| i)
            .collect();
        let dom = dominant_topic(&indices, truth);
        eprintln!("  cluster {cid:3} size={count:4}  dominant: {dom}");
    }
}

fn dominant_topic(indices: &[usize], truth: &ndarray::Array1<u8>) -> String {
    let mut counts: HashMap<u8, usize> = HashMap::new();
    for &i in indices {
        *counts.entry(truth[i]).or_insert(0) += 1;
    }
    let (&label, _) = counts.iter().max_by_key(|(_, c)| *c).unwrap();
    format!("{} ({label})", category_name(label))
}

fn mean_topic_purity(labels: &ndarray::Array1<i64>, truth: &ndarray::Array1<u8>) -> f64 {
    let mut by_cluster: HashMap<i64, Vec<usize>> = HashMap::new();
    for (i, &l) in labels.iter().enumerate() {
        if l >= 0 {
            by_cluster.entry(l).or_default().push(i);
        }
    }
    let mut purities = Vec::new();
    for indices in by_cluster.values() {
        let mut counts: HashMap<u8, usize> = HashMap::new();
        for &i in indices {
            *counts.entry(truth[i]).or_insert(0) += 1;
        }
        let total: usize = counts.values().sum();
        let max = counts.values().copied().max().unwrap_or(0);
        purities.push(max as f64 / total as f64);
    }
    if purities.is_empty() {
        f64::NAN
    } else {
        purities.iter().sum::<f64>() / purities.len() as f64
    }
}
