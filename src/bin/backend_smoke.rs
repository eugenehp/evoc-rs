//! Run `fit_predict` with a single compute backend.
//!
//! ```text
//! cargo run --release --bin backend_smoke --features "rlx-cuda" -- cuda small_200
//! EVOC_BACKEND=mlx cargo run --release --bin backend_smoke --features "rlx-mlx" -- small_200
//! ```

use evoc::{ComputeBackend, Evoc};
use ndarray::Array2;
use ndarray_npy::ReadNpyExt;
use std::env;
use std::fs::File;
use std::path::PathBuf;

fn usage() -> ! {
    eprintln!(
        "usage: backend_smoke [BACKEND] [FIXTURE]\n\
         BACKEND: strict|cpu|cuda|mlx|metal|rocm|wgpu|gpu (default: EVOC_BACKEND or strict)\n\
         FIXTURE: small_200|medium_800|large_2000 (default: small_200)"
    );
    std::process::exit(2);
}

fn parse_backend(s: &str) -> Option<ComputeBackend> {
    match s {
        "strict" => Some(ComputeBackend::Strict),
        "cpu" => Some(ComputeBackend::Cpu),
        "cuda" => Some(ComputeBackend::Cuda),
        "mlx" | "metal" => Some(ComputeBackend::Mlx),
        "rocm" => Some(ComputeBackend::Rocm),
        "wgpu" | "gpu" => Some(ComputeBackend::Wgpu),
        _ => None,
    }
}

fn load_fixture(name: &str) -> Array2<f32> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
        .join("data.npy");
    let mut f = File::open(&path).unwrap_or_else(|e| {
        eprintln!("open {path:?}: {e}");
        std::process::exit(1);
    });
    Array2::read_npy(&mut f).expect("read npy")
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let (backend_arg, fixture) = match args.as_slice() {
        [] => (None, "small_200"),
        [f] if parse_backend(f).is_some() => (Some(f.as_str()), "small_200"),
        [f] => (None, f.as_str()),
        [b, f] => (Some(b.as_str()), f.as_str()),
        _ => usage(),
    };

    let backend = if let Some(s) = backend_arg {
        parse_backend(s).unwrap_or_else(|| {
            eprintln!("unknown backend: {s}");
            usage();
        })
    } else {
        ComputeBackend::from_env().unwrap_or(ComputeBackend::Strict)
    };

    let backend = ComputeBackend::resolve(Some(backend)).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });

    let data = load_fixture(fixture);
    let mut clusterer = Evoc {
        random_state: Some(42),
        compute_backend: Some(backend),
        strict_precision: false,
        ..Evoc::default()
    };
    let labels = clusterer.fit_predict(data).expect("fit_predict");
    let n = labels.iter().filter(|&&l| l >= 0).count();
    println!("backend={backend} fixture={fixture} labeled={n}");
}
