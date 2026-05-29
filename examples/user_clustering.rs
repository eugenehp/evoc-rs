//! Cluster your own embedding vectors with EVoC.
//!
//! Pass a float32 `.npy` matrix `(n_samples, n_features)` with one vector per row.
//! Rows should be L2-normalized for cosine kNN (this example normalizes for you).
//!
//! ```bash
//! # Synthetic demo (three random clusters on the unit sphere)
//! cargo run --release --example user_clustering
//!
//! # Your data
//! cargo run --release --example user_clustering -- data/embeddings.npy 42
//! ```

use evoc::Evoc;
use ndarray::{Array2, Axis};
use ndarray_npy::ReadNpyExt;
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::path::Path;

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    let seed: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(42);

    let mut data = if let Some(path) = args.first() {
        load_user_matrix(path)?
    } else {
        eprintln!("no input file — using synthetic demo data (3 clusters × 400 points, dim=64)");
        synthetic_clusters(3, 400, 64, seed)
    };

    l2_normalize_rows(&mut data);
    eprintln!(
        "input: {} points × {} dims (rows L2-normalized)",
        data.nrows(),
        data.ncols()
    );

    let mut clusterer = Evoc {
        random_state: Some(seed),
        n_neighbors: 15,
        base_min_cluster_size: 10,
        min_samples: 10,
        ..Evoc::default()
    };

    let labels = clusterer.fit_predict(data)?;
    print_summary(&labels, &clusterer);

    Ok(())
}

fn load_user_matrix(path: &str) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    let path = Path::new(path);
    if !path.is_file() {
        return Err(format!("file not found: {}", path.display()).into());
    }
    let mut file = File::open(path)?;
    let data: Array2<f32> = Array2::read_npy(&mut file)?;
    if data.ncols() == 0 || data.nrows() == 0 {
        return Err("matrix must be non-empty".into());
    }
    if data.ncols() == 1 {
        eprintln!("warning: only one feature column — clustering may be degenerate");
    }
    Ok(data)
}

/// Three well-separated clusters on the unit sphere (demo when no file is given).
fn synthetic_clusters(n_clusters: usize, per_cluster: usize, dim: usize, seed: u64) -> Array2<f32> {
    use evoc::check_random_state;

    let n = n_clusters * per_cluster;
    let mut rng = check_random_state(Some(seed));
    let mut centers = Array2::<f32>::zeros((n_clusters, dim));
    for mut row in centers.rows_mut() {
        for x in row.iter_mut() {
            *x = rng.normal_scaled(1.0) as f32;
        }
        let norm = row.dot(&row).sqrt().max(1e-12);
        row.mapv_inplace(|x| x / norm);
    }

    let mut data = Array2::<f32>::zeros((n, dim));
    for c in 0..n_clusters {
        let center = centers.row(c);
        for i in 0..per_cluster {
            let row = c * per_cluster + i;
            for d in 0..dim {
                let noise = rng.normal_scaled(0.08) as f32;
                data[[row, d]] = center[d] + noise;
            }
        }
    }
    data
}

fn l2_normalize_rows(data: &mut Array2<f32>) {
    for mut row in data.axis_iter_mut(Axis(0)) {
        let norm = row.mapv(|x| x * x).sum().sqrt().max(1e-12);
        row.mapv_inplace(|x| x / norm);
    }
}

fn print_summary(labels: &ndarray::Array1<i64>, clusterer: &Evoc) {
    let n = labels.len();
    let mut counts: HashMap<i64, usize> = HashMap::new();
    for &label in labels.iter() {
        *counts.entry(label).or_default() += 1;
    }

    let n_noise = counts.get(&-1).copied().unwrap_or(0);
    let mut cluster_ids: Vec<i64> = counts.keys().copied().filter(|&c| c >= 0).collect();
    cluster_ids.sort_unstable();

    eprintln!();
    eprintln!("Results");
    eprintln!("-------");
    eprintln!("points:              {n}");
    eprintln!("clusters (non-noise): {}", cluster_ids.len());
    eprintln!("noise points:        {n_noise}");
    eprintln!("resolution layers:   {}", clusterer.cluster_layers_.len());
    eprintln!(
        "embedding shape:     {} × {}",
        clusterer.embedding_.nrows(),
        clusterer.embedding_.ncols()
    );

    eprintln!();
    eprintln!("Cluster sizes (id → count):");
    for cid in cluster_ids {
        eprintln!("  {cid:4} → {}", counts[&cid]);
    }
    if n_noise > 0 {
        eprintln!("  noise → {n_noise}");
    }

    eprintln!();
    eprintln!(
        "First 20 labels: {:?}",
        &labels.slice(ndarray::s![..20.min(n)])
    );
    eprintln!();
    eprintln!("Use clusterer.labels_, clusterer.embedding_, and clusterer.cluster_layers_");
    eprintln!("for downstream work after fit_predict.");
}
