use evoc::{ComputeBackend, Evoc};
use ndarray::Array2;
use ndarray_npy::ReadNpyExt;
use std::fs::File;
use std::time::Instant;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "tests/fixtures/large_2000/data.npy".to_string());
    let seed: u64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);

    let mut f = File::open(&path).expect("open data.npy");
    let data: Array2<f32> = Array2::read_npy(&mut f).expect("read npy");

    let threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "default".to_string());

    let backend = ComputeBackend::resolve(None).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });

    let mut clusterer = Evoc {
        random_state: Some(seed),
        n_neighbors: 15,
        // speed bench: don't force parity fixture graphs
        parity_graph_coo: None,
        compute_backend: Some(backend),
        ..Evoc::default()
    };

    let t0 = Instant::now();
    let labels = clusterer.fit_predict(data).expect("fit_predict");
    let dt = t0.elapsed().as_secs_f64();

    let n_clusters = labels
        .iter()
        .copied()
        .filter(|&l| l >= 0)
        .collect::<std::collections::HashSet<_>>()
        .len();
    println!(
        "rust seconds {:.6} backend {} threads {} clusters {} layers {}",
        dt,
        backend,
        threads,
        n_clusters,
        clusterer.cluster_layers_.len()
    );
}
