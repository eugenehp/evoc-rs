#![cfg(feature = "rlx-mlx")]

use evoc::{ComputeBackend, Evoc};
use ndarray::Array2;
use ndarray_npy::ReadNpyExt;
use std::fs::File;
use std::path::PathBuf;

fn load_fixture(name: &str) -> Array2<f32> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
        .join("data.npy");
    let mut f = File::open(path).unwrap();
    Array2::read_npy(&mut f).unwrap()
}

#[test]
fn mlx_backend_runs() {
    let data = load_fixture("small_200");
    let mut clusterer = Evoc {
        random_state: Some(42),
        compute_backend: Some(ComputeBackend::Mlx),
        strict_precision: false,
        ..Evoc::default()
    };
    let _ = clusterer.fit_predict(data).unwrap();
}
