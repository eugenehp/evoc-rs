//! Run EVoC on a float32 `.npy` matrix; write cluster labels and embedding.
//!
//! Usage:
//!   cargo run --release --bin mnist_labels -- data.npy seed labels.npy [embedding.npy] [parity_dir]
//!   cargo run --release --bin mnist_labels -- --mnist [n] [seed] labels.npy [embedding.npy] [truth.npy] [pixels.npy] [parity_dir]
//!   cargo run --release --bin mnist_labels -- --fashion-mnist [n] [seed] labels.npy [embedding.npy] [truth.npy] [pixels.npy]

use evoc::fashion_mnist_data;
use evoc::mnist_data;
use evoc::Evoc;
use ndarray::{Array1, Array2};
use ndarray_npy::{ReadNpyExt, WriteNpyExt};
use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;

struct LoadedData {
    data: Array2<f32>,
    seed: u64,
    labels_path: String,
    embedding_path: Option<String>,
    truth_path: Option<String>,
    pixels_export: Option<(String, Vec<u8>)>,
    parity_dir: Option<PathBuf>,
    class_labels: Option<Array1<u8>>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let loaded = if args.first().is_some_and(|a| a == "--mnist") {
        load_mnist_args(&args)?
    } else if args.first().is_some_and(|a| a == "--fashion-mnist") {
        load_fashion_args(&args)?
    } else {
        load_file_args(&args)?
    };

    let parity_graph_coo = loaded.parity_dir.as_ref().map(|d| d.join("graph_coo.npz"));

    let mut clusterer = Evoc {
        random_state: Some(loaded.seed),
        n_neighbors: 15,
        parity_graph_coo,
        ..Evoc::default()
    };

    let t0 = Instant::now();
    let labels = clusterer.fit_predict(loaded.data)?;
    let dt = t0.elapsed().as_secs_f64();

    let label_arr = Array1::from_iter(labels.iter().copied());
    label_arr.write_npy(File::create(&loaded.labels_path)?)?;

    if let Some(emb_path) = &loaded.embedding_path {
        clusterer.embedding_.write_npy(File::create(emb_path)?)?;
        eprintln!("wrote embedding {emb_path}");
    }
    if let (Some(truth_path), Some(class_labels)) = (&loaded.truth_path, &loaded.class_labels) {
        class_labels.write_npy(File::create(truth_path)?)?;
        eprintln!("wrote truth {truth_path}");
    }
    if let Some((pixels_path, pixels_flat)) = &loaded.pixels_export {
        let pixels = Array2::from_shape_vec((pixels_flat.len() / 784, 784), pixels_flat.clone())?;
        pixels.write_npy(File::create(pixels_path)?)?;
        eprintln!("wrote pixels {pixels_path}");
    }

    let n_clusters = labels
        .iter()
        .copied()
        .filter(|&l| l >= 0)
        .collect::<std::collections::HashSet<_>>()
        .len();
    eprintln!(
        "wrote {} (n={} clusters={} layers={} parity={} seconds={:.3})",
        loaded.labels_path,
        labels.len(),
        n_clusters,
        clusterer.cluster_layers_.len(),
        loaded.parity_dir.is_some(),
        dt
    );

    Ok(())
}

fn load_mnist_args(args: &[String]) -> Result<LoadedData, Box<dyn std::error::Error>> {
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(3000);
    let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(42);
    let labels_path = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "labels.npy".to_string());
    let embedding_path = args.get(4).cloned();
    let truth_path = args.get(5).cloned();
    let pixels_path = args.get(6).cloned();
    let parity_dir = args.get(7).map(PathBuf::from);
    eprintln!(
        "loading MNIST n={n} seed={seed} cache={} ...",
        mnist_data::cache_dir().display()
    );
    let (data, class_labels, pixels_flat) =
        mnist_data::sample_normalized(n, seed, &mnist_data::cache_dir())?;
    Ok(LoadedData {
        data,
        seed,
        labels_path,
        embedding_path,
        truth_path,
        pixels_export: pixels_path.map(|p| (p, pixels_flat)),
        parity_dir,
        class_labels: Some(class_labels),
    })
}

fn load_fashion_args(args: &[String]) -> Result<LoadedData, Box<dyn std::error::Error>> {
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(3000);
    let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(42);
    let labels_path = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "labels.npy".to_string());
    let embedding_path = args.get(4).cloned();
    let truth_path = args.get(5).cloned();
    let pixels_path = args.get(6).cloned();
    eprintln!(
        "loading Fashion-MNIST n={n} seed={seed} cache={} ...",
        fashion_mnist_data::cache_dir().display()
    );
    let (data, class_labels, pixels_flat) =
        fashion_mnist_data::sample_normalized(n, seed, &fashion_mnist_data::cache_dir())?;
    Ok(LoadedData {
        data,
        seed,
        labels_path,
        embedding_path,
        truth_path,
        pixels_export: pixels_path.map(|p| (p, pixels_flat)),
        parity_dir: None,
        class_labels: Some(class_labels),
    })
}

fn load_file_args(args: &[String]) -> Result<LoadedData, Box<dyn std::error::Error>> {
    let data_path = args
        .first()
        .cloned()
        .unwrap_or_else(|| "tests/fixtures/large_2000/data.npy".to_string());
    let seed: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(42);
    let labels_path = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "labels.npy".to_string());
    let embedding_path = args.get(3).cloned();
    let parity_dir = args.get(4).map(PathBuf::from);
    let mut f = File::open(&data_path)?;
    let data: Array2<f32> = Array2::read_npy(&mut f)?;
    Ok(LoadedData {
        data,
        seed,
        labels_path,
        embedding_path,
        truth_path: None,
        pixels_export: None,
        parity_dir,
        class_labels: None,
    })
}
