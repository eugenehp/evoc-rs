//! Compare `fit_predict` wall time across every compiled compute backend.
//!
//! ```text
//! cargo run --release --bin bench_backends --features "cluster,npy,rlx-all" -- large_2000
//! cargo run --release --bin bench_backends --features "cluster,npy,rlx-cuda" -- --runs 3
//! EVOC_BACKEND=mlx cargo run --release --bin bench_backends --features "cluster,npy,rlx-mlx" --
//! ```

use evoc::{ComputeBackend, Evoc};
use ndarray::Array2;
use ndarray_npy::ReadNpyExt;
use std::env;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Clone, Debug)]
struct Row {
    backend: ComputeBackend,
    runs: Vec<f64>,
}

fn usage() -> ! {
    eprintln!(
        "usage: bench_backends [FIXTURE|PATH] [--runs N] [--warmup N] [--json]\n\
         FIXTURE: small_200 | medium_800 | large_2000 (default: large_2000)\n\
         Set EVOC_BACKEND to benchmark a single backend."
    );
    std::process::exit(2);
}

fn parse_args() -> (String, usize, usize, bool) {
    let mut fixture = "large_2000".to_string();
    let mut runs = 5usize;
    let mut warmup = 1usize;
    let mut json = false;
    let mut it = env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--runs" => {
                runs = it
                    .next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| usage())
            }
            "--warmup" => {
                warmup = it
                    .next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| usage())
            }
            "--json" => json = true,
            "-h" | "--help" => usage(),
            other if other.starts_with('-') => usage(),
            other => fixture = other.to_string(),
        }
    }
    (fixture, runs, warmup, json)
}

fn fixture_path(name_or_path: &str) -> PathBuf {
    let p = Path::new(name_or_path);
    if p.extension().is_some() || p.components().count() > 1 {
        return p.to_path_buf();
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name_or_path)
        .join("data.npy")
}

fn load_data(path: &Path) -> Array2<f32> {
    let mut f = File::open(path).unwrap_or_else(|e| {
        eprintln!("open {}: {e}", path.display());
        std::process::exit(1);
    });
    Array2::read_npy(&mut f).expect("read npy")
}

fn median(mut xs: Vec<f64>) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = xs.len() / 2;
    if xs.len() % 2 == 0 {
        (xs[m - 1] + xs[m]) / 2.0
    } else {
        xs[m]
    }
}

fn bench_backend(
    data: &Array2<f32>,
    backend: ComputeBackend,
    seed: u64,
    warmup: usize,
    runs: usize,
) -> Vec<f64> {
    for _ in 0..warmup {
        let mut clusterer = Evoc {
            random_state: Some(seed),
            n_neighbors: 15,
            parity_graph_coo: None,
            compute_backend: Some(backend),
            strict_precision: false,
            ..Evoc::default()
        };
        let _ = clusterer.fit_predict(data.clone()).expect("fit_predict");
    }

    let mut times = Vec::with_capacity(runs);
    for i in 0..runs {
        let mut clusterer = Evoc {
            random_state: Some(seed.wrapping_add(i as u64)),
            n_neighbors: 15,
            parity_graph_coo: None,
            compute_backend: Some(backend),
            strict_precision: false,
            ..Evoc::default()
        };
        let t0 = Instant::now();
        let _ = clusterer.fit_predict(data.clone()).expect("fit_predict");
        times.push(t0.elapsed().as_secs_f64());
    }
    times
}

fn print_table(rows: &[Row], fixture: &str, n_rows: usize, n_cols: usize, runs: usize) {
    let threads = env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "default".into());
    println!("fixture={fixture} n={n_rows} d={n_cols} runs={runs} threads={threads}");
    println!(
        "{:<8} {:>10} {:>10} {:>10}",
        "backend", "median_s", "min_s", "max_s"
    );
    for row in rows {
        let med = median(row.runs.clone());
        let min = row.runs.iter().copied().fold(f64::INFINITY, f64::min);
        let max = row.runs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        println!(
            "{:<8} {:>10.6} {:>10.6} {:>10.6}",
            row.backend, med, min, max
        );
    }
}

#[cfg(feature = "bench-json")]
fn print_json(
    rows: &[Row],
    fixture: &str,
    path: &Path,
    n_rows: usize,
    n_cols: usize,
    runs: usize,
    warmup: usize,
) {
    let backends: Vec<_> = rows
        .iter()
        .map(|r| {
            serde_json::json!({
                "name": r.backend.to_string(),
                "seconds": r.runs,
                "median_seconds": median(r.runs.clone()),
                "min_seconds": r.runs.iter().copied().fold(f64::INFINITY, f64::min),
                "max_seconds": r.runs.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            })
        })
        .collect();
    let doc = serde_json::json!({
        "fixture": fixture,
        "path": path.display().to_string(),
        "n_samples": n_rows,
        "n_features": n_cols,
        "runs": runs,
        "warmup": warmup,
        "rayon_threads": env::var("RAYON_NUM_THREADS").ok(),
        "backends": backends,
    });
    println!("{}", serde_json::to_string_pretty(&doc).expect("json"));
}

fn main() {
    let (fixture, runs, warmup, json) = parse_args();
    let path = fixture_path(&fixture);
    let data = load_data(&path);
    let (n_rows, n_cols) = data.dim();
    let seed: u64 = 42;

    let backends = ComputeBackend::backends_for_run().unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });

    let mut rows = Vec::new();
    for backend in backends {
        eprintln!("benchmarking {backend} …",);
        let times = bench_backend(&data, backend, seed, warmup, runs);
        rows.push(Row {
            backend,
            runs: times,
        });
    }

    if json {
        #[cfg(feature = "bench-json")]
        print_json(&rows, &fixture, &path, n_rows, n_cols, runs, warmup);
        #[cfg(not(feature = "bench-json"))]
        {
            eprintln!("rebuild with --features bench-json for --json");
            print_table(&rows, &fixture, n_rows, n_cols, runs);
        }
    } else {
        print_table(&rows, &fixture, n_rows, n_cols, runs);
    }
}
