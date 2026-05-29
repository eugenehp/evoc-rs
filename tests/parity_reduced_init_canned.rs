//! Reduced init with Python-canned graph/data/RNG (isolates recursive init).

use evoc::{label_propagation_init, NumpyRandomState};
use ndarray::Array2;
use ndarray_npy::ReadNpyExt;
use sprs::CsMat;
use std::fs::File;
use std::path::PathBuf;

fn load_csr(path: &PathBuf) -> CsMat<f32> {
    let mut npz = ndarray_npy::NpzReader::new(File::open(path).unwrap()).unwrap();
    let indptr: ndarray::Array1<i64> = npz.by_name("indptr").unwrap();
    let indices: ndarray::Array1<i32> = npz.by_name("indices").unwrap();
    let data: ndarray::Array1<f32> = npz.by_name("data").unwrap();
    let shape: ndarray::Array1<i64> = npz.by_name("shape").unwrap();
    unsafe {
        CsMat::new_unchecked(
            sprs::CompressedStorage::CSR,
            (shape[0] as usize, shape[1] as usize),
            indptr
                .as_slice()
                .unwrap()
                .iter()
                .map(|&p| p as usize)
                .collect(),
            indices
                .as_slice()
                .unwrap()
                .iter()
                .map(|&i| i as usize)
                .collect(),
            data.to_vec(),
        )
    }
}

fn load_rng(dir: &PathBuf, tag: &str) -> NumpyRandomState {
    let inter = dir.join("intermediates");
    let mut kf = File::open(inter.join(format!("rng_{tag}_key.npy"))).unwrap();
    let key: ndarray::Array1<u32> = ndarray::Array1::read_npy(&mut kf).unwrap();
    let mut arr = [0u32; 624];
    arr.copy_from_slice(key.as_slice().unwrap());
    let mut npz =
        ndarray_npy::NpzReader::new(File::open(inter.join(format!("rng_{tag}_meta.npz"))).unwrap())
            .unwrap();
    let pos: ndarray::Array0<i32> = npz.by_name("pos").unwrap();
    let has_gauss: ndarray::Array0<i32> = npz.by_name("has_gauss").unwrap();
    let gauss: ndarray::Array0<f64> = npz.by_name("gauss").unwrap();
    NumpyRandomState::from_numpy_state(
        &arr,
        pos.as_slice().unwrap()[0],
        has_gauss.as_slice().unwrap()[0],
        gauss.as_slice().unwrap()[0],
    )
}

fn max_abs(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn parity_reduced_init_canned_large_2000() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/large_2000");
    let inter = dir.join("intermediates");
    if !inter.join("rng_after_partition_key.npy").exists() {
        eprintln!("Skip: re-run scripts/dump_label_prop_stages.py large_2000");
        return;
    }

    let reduced_graph = load_csr(&inter.join("reduced_graph_csr.npz"));
    let reduced_data: Array2<f32> =
        Array2::read_npy(&mut File::open(inter.join("reduced_data.npy")).unwrap()).unwrap();
    let py_init: Array2<f32> =
        Array2::read_npy(&mut File::open(inter.join("reduced_init.npy")).unwrap()).unwrap();

    let mut rng = load_rng(&dir, "after_partition");
    let approx = (8.0 * (2000f64).sqrt()).clamp(256.0, 16384.0) as usize / 4;
    let n_comp = 4usize;
    let init = label_propagation_init(
        &reduced_graph,
        20,
        50,
        approx,
        n_comp,
        0.5,
        0.1,
        0.5,
        &mut rng,
        Some(&reduced_data),
    );
    let diff = max_abs(&init, &py_init);
    eprintln!("canned reduced_init max diff {diff:.6}");
    assert!(diff <= 1e-4, "canned reduced_init max diff {diff}");
}
