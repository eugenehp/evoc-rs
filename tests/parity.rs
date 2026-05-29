//! Numerical and functional parity vs Python EVoC golden fixtures.
//!
//! Run: `cargo test parity --release`
//! Python goldens: `NUMBA_NUM_THREADS=1 python3 scripts/generate_parity_fixtures.py`

use evoc::{check_random_state, knn_graph, EmbeddingData, Evoc, KnnGraphOptions};
use ndarray::Array1;
use ndarray_npy::ReadNpyExt;
use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;

fn fixtures_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn load_fixture(name: &str) -> (Array1<i64>, usize, u64) {
    let dir = fixtures_root().join(name);
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap();
    let seed = meta["seed"].as_u64().unwrap();
    let mut f = File::open(dir.join("labels.npy")).unwrap();
    let labels: Array1<i64> = Array1::read_npy(&mut f).unwrap();
    let n = meta["n"].as_u64().unwrap() as usize;
    (labels, n, seed)
}

fn load_data(name: &str) -> ndarray::Array2<f32> {
    let mut f = File::open(fixtures_root().join(name).join("data.npy")).unwrap();
    ndarray::Array2::read_npy(&mut f).unwrap()
}

/// Golden kNN fixtures match when `deterministic: true` (C `-ffast-math` reference).
fn assert_knn_parity(
    inds: &ndarray::Array2<i32>,
    dists: &ndarray::Array2<f32>,
    exp_inds: &ndarray::Array2<i32>,
    exp_dists: &ndarray::Array2<f32>,
) {
    let dist_m = dists
        .iter()
        .zip(exp_dists.iter())
        .filter(|(a, b)| (*a - *b).abs() > 1e-6)
        .count();
    assert_eq!(dist_m, 0, "kNN distance mismatch count {dist_m}");

    let bad_ind = inds
        .iter()
        .zip(exp_inds.iter())
        .zip(dists.iter().zip(exp_dists.iter()))
        .filter(|((a, b), (da, db))| *a != *b && (*da - *db).abs() > 1e-6)
        .count();
    assert_eq!(
        bad_ind, 0,
        "kNN index mismatch with different distance {bad_ind}"
    );

    let tie_ind = inds
        .iter()
        .zip(exp_inds.iter())
        .zip(dists.iter().zip(exp_dists.iter()))
        .filter(|((a, b), (da, db))| *a != *b && (*da - *db).abs() <= 1e-6)
        .count();
    assert!(
        tie_ind <= 8,
        "excessive tied-neighbor reordering vs C goldens: {tie_ind}"
    );
}

#[test]
fn parity_labels_exact_medium_800() {
    let name = "medium_800";
    let dir = fixtures_root().join(name);
    if !dir.exists() {
        eprintln!("Skip: run scripts/generate_parity_fixtures.py first");
        return;
    }

    let expected_labels = {
        let mut f = File::open(dir.join("labels.npy")).unwrap();
        Array1::<i64>::read_npy(&mut f).unwrap()
    };
    let data = load_data(name);
    let seed = 42u64;

    let t0 = Instant::now();
    let mut clusterer = Evoc {
        random_state: Some(seed),
        n_neighbors: 15,
        parity_graph_coo: Some(dir.join("intermediates/graph_coo.npz")),
        ..Evoc::default()
    };
    let labels = clusterer.fit_predict(data).expect("fit_predict");
    let rust_s = t0.elapsed().as_secs_f64();

    assert_eq!(labels.len(), expected_labels.len());
    let mismatches: usize = labels
        .iter()
        .zip(expected_labels.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        mismatches,
        0,
        "label mismatch count {mismatches} / {}",
        labels.len()
    );

    let mut f_inds = File::open(dir.join("nn_inds.npy")).unwrap();
    let mut f_dists = File::open(dir.join("nn_dists.npy")).unwrap();
    let exp_inds: ndarray::Array2<i32> = ndarray::Array2::read_npy(&mut f_inds).unwrap();
    let exp_dists: ndarray::Array2<f32> = ndarray::Array2::read_npy(&mut f_dists).unwrap();

    assert_knn_parity(
        &clusterer.nn_inds_,
        &clusterer.nn_dists_,
        &exp_inds,
        &exp_dists,
    );

    let py_s: f64 = serde_json::from_str::<serde_json::Value>(
        &std::fs::read_to_string(dir.join("meta.json")).unwrap(),
    )
    .unwrap()["python_seconds"]
        .as_f64()
        .unwrap();

    eprintln!(
        "parity ok: medium_800 labels+kNN exact | rust={rust_s:.3}s python={py_s:.3}s ratio={:.2}x",
        rust_s / py_s
    );
}

#[test]
fn parity_knn_medium_800() {
    let name = "medium_800";
    let dir = fixtures_root().join(name);
    if !dir.exists() {
        return;
    }
    let data = load_data(name);
    let mut exp_inds_f = File::open(dir.join("nn_inds.npy")).unwrap();
    let exp_inds: ndarray::Array2<i32> = ndarray::Array2::read_npy(&mut exp_inds_f).unwrap();

    let mut rng = check_random_state(Some(42));
    let (inds, dists) = knn_graph(
        EmbeddingData::Float32(data),
        KnnGraphOptions {
            n_neighbors: 15,
            deterministic: true,
            ..Default::default()
        },
        &mut rng,
    )
    .unwrap();

    let mut exp_dists_f = File::open(dir.join("nn_dists.npy")).unwrap();
    let exp_dists: ndarray::Array2<f32> = ndarray::Array2::read_npy(&mut exp_dists_f).unwrap();

    assert_knn_parity(&inds, &dists, &exp_inds, &exp_dists);
}

#[test]
fn parity_knn_large_2000() {
    let name = "large_2000";
    let dir = fixtures_root().join(name);
    if !dir.exists() {
        return;
    }
    let data = load_data(name);
    let mut exp_inds_f = File::open(dir.join("nn_inds.npy")).unwrap();
    let exp_inds: ndarray::Array2<i32> = ndarray::Array2::read_npy(&mut exp_inds_f).unwrap();
    let mut exp_dists_f = File::open(dir.join("nn_dists.npy")).unwrap();
    let exp_dists: ndarray::Array2<f32> = ndarray::Array2::read_npy(&mut exp_dists_f).unwrap();

    let mut rng = check_random_state(Some(7));
    let (inds, dists) = knn_graph(
        EmbeddingData::Float32(data),
        KnnGraphOptions {
            n_neighbors: 15,
            deterministic: true,
            ..Default::default()
        },
        &mut rng,
    )
    .unwrap();

    assert_knn_parity(&inds, &dists, &exp_inds, &exp_dists);
}

#[test]
fn parity_graph_pattern_large_2000() {
    use evoc::{load_graph_csr_npz, neighbor_graph_matrix_with_coo};
    use sprs::CsMat;

    let name = "large_2000";
    let dir = fixtures_root().join(name);
    if !dir.exists() {
        return;
    }
    let data = load_data(name);
    let mut rng = check_random_state(Some(7));
    let (inds, dists) = knn_graph(
        EmbeddingData::Float32(data),
        KnnGraphOptions {
            n_neighbors: 15,
            deterministic: true,
            ..Default::default()
        },
        &mut rng,
    )
    .unwrap();
    let graph = neighbor_graph_matrix_with_coo(
        15.0,
        &inds,
        &dists,
        true,
        Some(dir.join("intermediates/graph_coo.npz").as_path()),
    );
    let reference: CsMat<f32> =
        load_graph_csr_npz(dir.join("intermediates/graph_csr.npz").as_path()).expect("graph_csr");
    let pattern_ok = graph.indptr().raw_storage() == reference.indptr().raw_storage()
        && graph.indices() == reference.indices();
    let data_ok = graph
        .data()
        .iter()
        .zip(reference.data().iter())
        .all(|(a, b)| (*a - *b).abs() <= 1e-6);
    eprintln!("large_2000 graph pattern_ok={pattern_ok} data_ok={data_ok}");
    assert!(pattern_ok, "CSR sparsity pattern mismatch");
    assert!(data_ok, "CSR data mismatch");
}

#[test]
fn parity_knn_only_small_200() {
    let name = "small_200";
    let dir = fixtures_root().join(name);
    if !dir.exists() {
        return;
    }
    let data = load_data(name);
    let mut exp_inds_f = File::open(dir.join("nn_inds.npy")).unwrap();
    let exp_inds: ndarray::Array2<i32> = ndarray::Array2::read_npy(&mut exp_inds_f).unwrap();

    let mut rng = check_random_state(Some(42));
    let (inds, _) = knn_graph(
        EmbeddingData::Float32(data),
        KnnGraphOptions {
            n_neighbors: 15,
            ..Default::default()
        },
        &mut rng,
    )
    .unwrap();

    assert_eq!(inds.shape(), exp_inds.shape());
    for (a, b) in inds.iter().zip(exp_inds.iter()) {
        assert_eq!(a, b);
    }
}

#[test]
fn parity_all_fixtures_labels() {
    let root = fixtures_root();
    if !root.exists() {
        return;
    }
    for entry in std::fs::read_dir(&root).unwrap() {
        let entry = entry.unwrap();
        if !entry.file_type().unwrap().is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') {
            continue;
        }
        let fixture_dir = fixtures_root().join(&name);
        if !fixture_dir.join("labels.npy").exists() {
            eprintln!("Skip {name}: missing labels.npy (run generate_parity_fixtures.py)");
            continue;
        }
        if !fixture_dir
            .join("intermediates/init_embedding.npy")
            .exists()
        {
            eprintln!("Skip {name}: missing parity intermediates");
            continue;
        }
        let (expected, _n, seed) = load_fixture(&name);
        let data = load_data(&name);
        let mut clusterer = Evoc {
            random_state: Some(seed),
            n_neighbors: 15,
            parity_graph_coo: Some(fixture_dir.join("intermediates/graph_coo.npz")),
            ..Evoc::default()
        };
        let labels = clusterer.fit_predict(data).unwrap();
        let mismatches = labels
            .iter()
            .zip(expected.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(mismatches, 0, "fixture {name} label mismatch");
    }
}
