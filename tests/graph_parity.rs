//! Exact fuzzy graph vs Python golden COO.

use evoc::neighbor_graph_matrix;
use evoc::parity::smooth_knn_dist;
use ndarray::Array2;
use ndarray_npy::ReadNpyExt;
use sprs::{CsMat, TriMat};
use std::fs::File;
use std::path::PathBuf;

fn load_py_graph() -> CsMat<f32> {
    let inter =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/medium_800/intermediates");
    let mut npz =
        ndarray_npy::NpzReader::new(File::open(inter.join("graph_coo.npz")).unwrap()).unwrap();
    let rows: ndarray::Array1<i32> = npz.by_name("rows").unwrap();
    let cols: ndarray::Array1<i32> = npz.by_name("cols").unwrap();
    let data: ndarray::Array1<f32> = npz.by_name("data").unwrap();
    let shape: ndarray::Array1<i64> = npz.by_name("shape").unwrap();
    let n = shape[0] as usize;
    let mut tri = TriMat::new((n, n));
    for i in 0..rows.len() {
        tri.add_triplet(rows[i] as usize, cols[i] as usize, data[i]);
    }
    tri.to_csr()
}

#[test]
fn smooth_knn_matches_python() {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/medium_800");
    let inter = base.join("intermediates");
    if !inter.join("sigmas.npy").exists() {
        return;
    }
    let mut f2 = File::open(base.join("nn_dists.npy")).unwrap();
    let nn_dists: Array2<f32> = Array2::read_npy(&mut f2).unwrap();
    let (sigmas, rhos) = smooth_knn_dist(&nn_dists, 15.0);
    let sigma_name = if inter.join("sigmas_py.npy").exists() {
        "sigmas_py.npy"
    } else {
        "sigmas.npy"
    };
    let rho_name = if inter.join("rhos_py.npy").exists() {
        "rhos_py.npy"
    } else {
        "rhos.npy"
    };
    let mut sf = File::open(inter.join(sigma_name)).unwrap();
    let mut rf = File::open(inter.join(rho_name)).unwrap();
    let py_s: ndarray::Array1<f32> = ndarray::Array1::read_npy(&mut sf).unwrap();
    let py_r: ndarray::Array1<f32> = ndarray::Array1::read_npy(&mut rf).unwrap();
    let mut max_s = 0.0f32;
    for i in 0..sigmas.len() {
        max_s = max_s.max((sigmas[i] - py_s[i]).abs());
    }
    let mut max_r = 0.0f32;
    for i in 0..rhos.len() {
        max_r = max_r.max((rhos[i] - py_r[i]).abs());
    }
    eprintln!("sigma max diff {max_s}, rho max diff {max_r}");
    assert!(max_s < 1e-6 && max_r < 1e-6);
}

#[test]
fn graph_matches_python_fixture_knn() {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/medium_800");
    let mut f1 = File::open(base.join("nn_inds.npy")).unwrap();
    let mut f2 = File::open(base.join("nn_dists.npy")).unwrap();
    let nn_inds: Array2<i32> = Array2::read_npy(&mut f1).unwrap();
    let nn_dists: Array2<f32> = Array2::read_npy(&mut f2).unwrap();

    let graph = neighbor_graph_matrix(15.0, &nn_inds, &nn_dists, true);
    let py = load_py_graph();

    let mut max_diff = 0.0f32;
    let mut n_diff = 0usize;
    for (&v, (r, c)) in graph.iter() {
        let pv = py.get(r, c).copied().unwrap_or(0.0);
        let d = (v - pv).abs();
        if d > 1e-6 {
            n_diff += 1;
            if n_diff <= 3 {
                eprintln!("diff at ({r},{c}): rust={v} py={pv}");
            }
        }
        max_diff = max_diff.max(d);
    }
    eprintln!("graph diffs: {n_diff} max={max_diff}");
    assert_eq!(n_diff, 0, "graph must match Python exactly");
}
