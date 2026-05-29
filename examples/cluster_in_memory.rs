//! Cluster embedding vectors held entirely in memory — no `.npy` files.
//!
//! ```bash
//! cargo run --release --example cluster_in_memory
//! ```

use evoc::Evoc;
use ndarray::{Array2, Axis};

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    // -------------------------------------------------------------------------
    // 1. Your data: however you already have it in your app
    // -------------------------------------------------------------------------

    // Option A — flat buffer from another library / FFI / network payload
    let flat: Vec<f32> = load_embeddings_from_your_source();
    let n_samples = 800;
    let n_features = 32;
    let mut data = Array2::from_shape_vec((n_samples, n_features), flat)?;

    // Option B — build directly (e.g. after a model forward pass in-process)
    // let mut data = Array2::from_shape_fn((n_samples, n_features), |(i, j)| { ... });

    // EVoC expects L2-normalized rows for cosine kNN
    l2_normalize_rows(&mut data);

    // -------------------------------------------------------------------------
    // 2. Cluster
    // -------------------------------------------------------------------------
    let mut clusterer = Evoc {
        random_state: Some(42),
        n_neighbors: 15,
        base_min_cluster_size: 15,
        min_samples: 15,
        ..Evoc::default()
    };

    let labels = clusterer.fit_predict(data)?;

    // -------------------------------------------------------------------------
    // 3. Use results in memory
    // -------------------------------------------------------------------------
    for (i, &label) in labels.iter().enumerate().take(5) {
        let strength = clusterer.membership_strengths_[i];
        eprintln!("point {i}: cluster={label} strength={strength:.3}");
    }

    let n_clusters = labels
        .iter()
        .filter(|&&l| l >= 0)
        .collect::<std::collections::HashSet<_>>()
        .len();
    eprintln!(
        "\n{} points → {n_clusters} clusters, {} noise, embedding dim {}",
        labels.len(),
        labels.iter().filter(|&&l| l == -1).count(),
        clusterer.embedding_.ncols()
    );

    // `labels` is the primary layer; finer/coarser views live here:
    eprintln!("resolution layers: {}", clusterer.cluster_layers_.len());

    Ok(())
}

/// Stand-in for however your app obtains vectors (DB, ONNX, custom model, etc.).
fn load_embeddings_from_your_source() -> Vec<f32> {
    use evoc::check_random_state;

    let n_samples = 800;
    let n_features = 32;
    let n_clusters = 4;
    let per_cluster = n_samples / n_clusters;
    let mut rng = check_random_state(Some(7));

    let mut centers = vec![vec![0.0f32; n_features]; n_clusters];
    for center in &mut centers {
        for x in center.iter_mut() {
            *x = rng.normal_scaled(1.0) as f32;
        }
        let norm: f32 = center.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for x in center.iter_mut() {
            *x /= norm;
        }
    }

    let mut flat = Vec::with_capacity(n_samples * n_features);
    for c in 0..n_clusters {
        for _ in 0..per_cluster {
            for d in 0..n_features {
                flat.push(centers[c][d] + rng.normal_scaled(0.06) as f32);
            }
        }
    }
    flat
}

fn l2_normalize_rows(data: &mut Array2<f32>) {
    for mut row in data.axis_iter_mut(Axis(0)) {
        let norm = row.mapv(|x| x * x).sum().sqrt().max(1e-12);
        row.mapv_inplace(|x| x / norm);
    }
}
