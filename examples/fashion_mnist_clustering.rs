//! Cluster Fashion-MNIST in pure Rust (download → L2-normalize → EVoC).
//!
//! ```bash
//! cargo run --release --example fashion_mnist_clustering
//! cargo run --release --example fashion_mnist_clustering -- 3000 42 [labels.npy] [embedding.npy]
//! ```

use evoc::fashion_mnist_data::{self, class_name};
use evoc::Evoc;
use ndarray::{Array1, Array2};
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
        .unwrap_or(3000);
    let seed: u64 = env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);

    let cache = fashion_mnist_data::cache_dir();
    eprintln!(
        "Fashion-MNIST n={n} seed={seed} cache={} ...",
        cache.display()
    );
    let (data, class_labels, pixels_flat) = fashion_mnist_data::sample_normalized(n, seed, &cache)?;
    eprintln!(
        "loaded {} × {} pixels (L2-normalized)",
        data.nrows(),
        data.ncols()
    );

    let mut clusterer = Evoc {
        random_state: Some(seed),
        n_neighbors: 15,
        base_min_cluster_size: 15,
        min_samples: 15,
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
        class_labels.write_npy(File::create(path)?)?;
    }
    if let Some(path) = env::args().nth(6) {
        let pixels = Array2::from_shape_vec((n, 784), pixels_flat)?;
        pixels.write_npy(File::create(path)?)?;
    }

    print_results(&labels, &class_labels, &clusterer);
    Ok(())
}

fn print_results(labels: &ndarray::Array1<i64>, truth: &ndarray::Array1<u8>, clusterer: &Evoc) {
    let n_clusters = labels
        .iter()
        .filter(|&&l| l >= 0)
        .collect::<std::collections::HashSet<_>>()
        .len();
    let n_noise = labels.iter().filter(|&&l| l == -1).count();
    let purity = mean_class_purity(labels, truth, |t| class_name(*t));

    eprintln!();
    eprintln!("Results");
    eprintln!("-------");
    eprintln!("points:              {}", labels.len());
    eprintln!("clusters (non-noise): {n_clusters}");
    eprintln!("noise points:        {n_noise}");
    eprintln!("mean class purity:   {purity:.3}");
    eprintln!("resolution layers:   {}", clusterer.cluster_layers_.len());
    eprintln!("embedding dim:       {}", clusterer.embedding_.ncols());

    eprintln!();
    eprintln!("Top clusters by size (dominant Fashion-MNIST class):");
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
    for (cid, count) in sizes.into_iter().take(8) {
        let mask: Vec<usize> = labels
            .iter()
            .enumerate()
            .filter(|(_, &l)| l == cid)
            .map(|(i, _)| i)
            .collect();
        let dom = dominant_label(&mask, truth, |t| class_name(*t));
        eprintln!("  cluster {cid:3} size={count:4}  dominant: {dom}");
    }
}

fn dominant_label(
    indices: &[usize],
    truth: &ndarray::Array1<u8>,
    name: impl Fn(&u8) -> &str,
) -> String {
    let mut counts: HashMap<u8, usize> = HashMap::new();
    for &i in indices {
        *counts.entry(truth[i]).or_insert(0) += 1;
    }
    let (&label, _) = counts.iter().max_by_key(|(_, c)| *c).unwrap();
    format!("{} ({label})", name(&label))
}

fn mean_class_purity(
    labels: &ndarray::Array1<i64>,
    truth: &ndarray::Array1<u8>,
    _name: impl Fn(&u8) -> &str,
) -> f64 {
    let mut purities = Vec::new();
    let mut seen = HashMap::new();
    for (i, &l) in labels.iter().enumerate() {
        if l < 0 {
            continue;
        }
        seen.entry(l).or_insert_with(Vec::new).push(i);
    }
    for indices in seen.values() {
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
