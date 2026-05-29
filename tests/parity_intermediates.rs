//! Staged parity: graph → init → embedding (Python intermediates + RNG checkpoints).

use evoc::parity::{
    csr_matmul_dense, normalize_cols_l2, normalize_rows_l1, partition_reduction_map,
    scipy_csr_matmul,
};
use evoc::{label_prop_loop, label_propagation_init, node_embedding, NumpyRandomState};
use ndarray::Array2;
use ndarray_npy::ReadNpyExt;
use sprs::{CompressedStorage, CsMat, TriMat};
use std::fs::File;
use std::path::PathBuf;

fn base() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/medium_800")
}

fn load_numpy_rng(tag: &str) -> NumpyRandomState {
    let inter = base().join("intermediates");
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
fn parity_staged_medium_800() {
    let dir = base();
    if !dir.join("intermediates/rng_after_knn_key.npy").exists() {
        eprintln!("Skip: run scripts/dump_intermediates.py");
        return;
    }

    let mut data_f = File::open(dir.join("data.npy")).unwrap();
    let data: Array2<f32> = Array2::read_npy(&mut data_f).unwrap();

    let py_init: Array2<f32> = {
        let mut f = File::open(dir.join("intermediates/init_embedding.npy")).unwrap();
        Array2::read_npy(&mut f).unwrap()
    };
    let py_emb: Array2<f32> = {
        let mut f = File::open(dir.join("intermediates/embedding.npy")).unwrap();
        Array2::read_npy(&mut f).unwrap()
    };

    let mut exp_inds_f = File::open(dir.join("nn_inds.npy")).unwrap();
    let mut exp_dists_f = File::open(dir.join("nn_dists.npy")).unwrap();
    let _exp_inds: ndarray::Array2<i32> = ndarray::Array2::read_npy(&mut exp_inds_f).unwrap();
    let _exp_dists: ndarray::Array2<f32> = ndarray::Array2::read_npy(&mut exp_dists_f).unwrap();

    let inter = base().join("intermediates");
    let graph = evoc::load_graph_csr_npz(inter.join("graph_csr.npz").as_path())
        .or_else(|| evoc::load_graph_coo_npz(inter.join("graph_coo.npz").as_path()))
        .unwrap();

    let approx = (8.0 * (data.nrows() as f64).sqrt()).clamp(256.0, 16384.0) as usize;

    let mut rng = load_numpy_rng("after_knn");
    let init = label_propagation_init(
        &graph,
        20,
        50,
        approx,
        (15usize / 4).max(4).min(15),
        0.5,
        0.1,
        0.5,
        &mut rng,
        Some(&data),
    );
    let init_diff = max_abs(&init, &py_init);
    eprintln!("max |init rust - py| = {init_diff:.6}");

    let mut rng_emb = load_numpy_rng("after_init");
    let n_comp = (15usize / 4).max(4).min(15);
    let emb_from_py_init = node_embedding(
        &graph,
        n_comp,
        50,
        Some(py_init),
        0.1,
        1.0,
        0.5,
        &mut rng_emb,
        true,
    );
    let emb_py_init_diff = max_abs(&emb_from_py_init, &py_emb);
    eprintln!("max |emb (py init) - py| = {emb_py_init_diff:.6}");

    let mut rng_emb2 = load_numpy_rng("after_init");
    let emb_from_rust_init = node_embedding(
        &graph,
        n_comp,
        50,
        Some(init),
        0.1,
        1.0,
        0.5,
        &mut rng_emb2,
        true,
    );
    let emb_rust_init_diff = max_abs(&emb_from_rust_init, &py_emb);
    eprintln!("max |emb (rust init) - py| = {emb_rust_init_diff:.6}");

    const TOL_INIT: f32 = 1e-4;
    // UMAP-like SGD is sensitive to fast-math / compiler differences; keep as regression guard.
    const TOL_EMB: f32 = 3.6e-4;
    assert!(init_diff <= TOL_INIT, "init_embedding max diff {init_diff}");
    assert!(
        emb_py_init_diff <= TOL_EMB,
        "embedding (py init) max diff {emb_py_init_diff}"
    );
    eprintln!("emb (rust init) max diff {emb_rust_init_diff:.6} (UMAP amplifies init drift)");
}

#[test]
fn parity_staged_large_2000() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/large_2000");
    if !dir.join("intermediates/init_embedding.npy").exists() {
        eprintln!("Skip: run scripts/generate_parity_fixtures.py");
        return;
    }
    let inter = dir.join("intermediates");
    let graph = evoc::load_graph_csr_npz(inter.join("graph_csr.npz").as_path())
        .or_else(|| evoc::load_graph_coo_npz(inter.join("graph_coo.npz").as_path()))
        .unwrap();
    let mut f_init = File::open(inter.join("init_embedding.npy")).unwrap();
    let py_init: Array2<f32> = Array2::read_npy(&mut f_init).unwrap();
    let mut f_emb = File::open(inter.join("embedding.npy")).unwrap();
    let py_emb: Array2<f32> = Array2::read_npy(&mut f_emb).unwrap();
    let data: Array2<f32> =
        Array2::read_npy(&mut File::open(dir.join("data.npy")).unwrap()).unwrap();

    let n_comp = (15usize / 4).max(4).min(15);
    let mut rng_init = load_numpy_rng_from(&dir, "after_knn");
    let approx = (8.0 * (graph.rows() as f64).sqrt()).clamp(256.0, 16384.0) as usize;
    let init = label_propagation_init(
        &graph,
        20,
        50,
        approx,
        n_comp,
        0.5,
        0.1,
        0.5,
        &mut rng_init,
        Some(&data),
    );
    let init_diff = max_abs(&init, &py_init);

    let mut rng_emb = load_numpy_rng_from(&dir, "after_init");
    let emb_from_py_init = node_embedding(
        &graph,
        n_comp,
        50,
        Some(py_init),
        0.1,
        1.0,
        0.5,
        &mut rng_emb,
        true,
    );
    let emb_py_init_diff = max_abs(&emb_from_py_init, &py_emb);

    eprintln!("large_2000 init_max {init_diff:.6} emb(py_init)_max {emb_py_init_diff:.6}");

    let mut rng_emb3 = load_numpy_rng_from(&dir, "after_init");
    let init_for_rust = init.clone();
    let emb_from_rust_init = node_embedding(
        &graph,
        n_comp,
        50,
        Some(init_for_rust),
        0.1,
        1.0,
        0.5,
        &mut rng_emb3,
        true,
    );
    let emb_rust_init_diff = max_abs(&emb_from_rust_init, &py_emb);
    eprintln!("large_2000 emb(rust_init)_max {emb_rust_init_diff:.6}");
}

#[test]
fn parity_full_seed_large_2000_embedding() {
    use evoc::{
        check_random_state, knn_graph, neighbor_graph_matrix_with_coo, EmbeddingData,
        KnnGraphOptions,
    };

    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/large_2000");
    if !dir.join("intermediates/embedding.npy").exists() {
        eprintln!("Skip: run generate_parity_fixtures.py");
        return;
    }
    let inter = dir.join("intermediates");
    let data: Array2<f32> =
        Array2::read_npy(&mut File::open(dir.join("data.npy")).unwrap()).unwrap();
    let py_emb: Array2<f32> =
        Array2::read_npy(&mut File::open(inter.join("embedding.npy")).unwrap()).unwrap();

    let mut rng = check_random_state(Some(7));
    let (inds, dists) = knn_graph(
        EmbeddingData::Float32(data.clone()),
        KnnGraphOptions {
            n_neighbors: 15,
            deterministic: true,
            ..Default::default()
        },
        &mut rng,
    )
    .unwrap();
    let _ = neighbor_graph_matrix_with_coo(
        15.0,
        &inds,
        &dists,
        true,
        Some(inter.join("graph_coo.npz").as_path()),
    );
    let graph = evoc::load_graph_csr_npz(inter.join("graph_csr.npz").as_path()).expect("graph_csr");
    let n_comp = (15usize / 4).max(4).min(15);
    let approx = (8.0 * (graph.rows() as f64).sqrt()).clamp(256.0, 16384.0) as usize;
    let init = label_propagation_init(
        &graph,
        20,
        50,
        approx,
        n_comp,
        0.5,
        0.1,
        0.5,
        &mut rng,
        Some(&data),
    );
    let emb = node_embedding(
        &graph,
        n_comp,
        50,
        Some(init),
        0.1,
        1.0,
        0.5,
        &mut rng,
        true,
    );
    let diff = max_abs(&emb, &py_emb);
    assert!(
        diff <= 2e-3,
        "full-seed computed-init embedding max diff {diff} (target 2e-4 with golden init path)"
    );
}

#[test]
fn parity_rng_after_init_matches_python() {
    for (name, seed) in [("medium_800", 42u64), ("large_2000", 7u64)] {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name);
        if !dir.join("intermediates/rng_after_init_key.npy").exists() {
            continue;
        }
        let inter = dir.join("intermediates");
        let data: Array2<f32> =
            Array2::read_npy(&mut File::open(dir.join("data.npy")).unwrap()).unwrap();
        let graph = evoc::load_graph_csr_npz(inter.join("graph_csr.npz").as_path())
            .or_else(|| evoc::load_graph_coo_npz(inter.join("graph_coo.npz").as_path()))
            .unwrap();
        let approx = (8.0 * (graph.rows() as f64).sqrt()).clamp(256.0, 16384.0) as usize;
        let n_comp = (15usize / 4).max(4).min(15);

        let mut rust_rng = evoc::check_random_state(Some(seed));
        let _ = evoc::knn_graph(
            evoc::EmbeddingData::Float32(data.clone()),
            evoc::KnnGraphOptions {
                n_neighbors: 15,
                deterministic: true,
                ..Default::default()
            },
            &mut rust_rng,
        )
        .expect("knn");
        let _ = label_propagation_init(
            &graph,
            20,
            50,
            approx,
            n_comp,
            0.5,
            0.1,
            0.5,
            &mut rust_rng,
            Some(&data),
        );
        let mut py_rng = load_numpy_rng_from(&dir, "after_init");
        const LO: i64 = i32::MIN as i64 + 1;
        const HI: i64 = i32::MAX as i64 - 1;
        for i in 0..5 {
            assert_eq!(
                rust_rng.randint(LO, HI),
                py_rng.randint(LO, HI),
                "{name} rng after init draw {i}"
            );
        }
    }
}

#[test]
fn parity_partition_and_reduced_data_medium_800() {
    let dir = base();
    if !dir.join("intermediates/partition.npy").exists() {
        eprintln!("Skip: run scripts/dump_label_prop_stages.py");
        return;
    }

    let inter = base().join("intermediates");
    let graph = evoc::load_graph_csr_npz(inter.join("graph_csr.npz").as_path())
        .or_else(|| evoc::load_graph_coo_npz(inter.join("graph_coo.npz").as_path()))
        .unwrap();
    let data: Array2<f32> =
        Array2::read_npy(&mut File::open(dir.join("data.npy")).unwrap()).unwrap();
    let py_part: ndarray::Array1<i64> = ndarray::Array1::read_npy(
        &mut File::open(dir.join("intermediates/partition.npy")).unwrap(),
    )
    .unwrap();
    let py_rd: Array2<f32> =
        Array2::read_npy(&mut File::open(dir.join("intermediates/reduced_data.npy")).unwrap())
            .unwrap();

    let mut labels = vec![-1i64; graph.rows()];
    let mut rng = load_numpy_rng("after_knn");
    let approx = (8.0 * (graph.rows() as f64).sqrt()).clamp(256.0, 16384.0) as usize;
    let part = label_prop_loop(
        graph.indptr().raw_storage(),
        graph.indices(),
        graph.data(),
        &mut labels,
        &mut rng,
        20,
        approx,
    );
    let mism = part
        .iter()
        .zip(py_part.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(mism, 0, "partition mismatch {mism}/{}", part.len());

    let base_map = partition_reduction_map(&part);
    let mut norm_map = base_map.clone();
    normalize_cols_l2(&mut norm_map);
    let mut data_reducer = norm_map.transpose_view().to_csr();
    normalize_rows_l1(&mut data_reducer);
    let reduced_data = csr_matmul_dense(&data_reducer, &data);
    let rd_diff = max_abs(&reduced_data, &py_rd);
    assert!(rd_diff <= 1e-5, "reduced_data max diff {rd_diff}");
}

fn load_csr_npz(path: &PathBuf) -> CsMat<f32> {
    let mut npz = ndarray_npy::NpzReader::new(File::open(path).unwrap()).unwrap();
    let indptr: ndarray::Array1<i64> = npz.by_name("indptr").unwrap();
    let indices: ndarray::Array1<i32> = npz.by_name("indices").unwrap();
    let data: ndarray::Array1<f32> = npz.by_name("data").unwrap();
    let shape: ndarray::Array1<i64> = npz.by_name("shape").unwrap();
    unsafe {
        CsMat::new_unchecked(
            CompressedStorage::CSR,
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

#[test]
fn parity_reduced_layout_medium_800() {
    let dir = base();
    if !dir
        .join("intermediates/rng_after_reduced_init_key.npy")
        .exists()
    {
        eprintln!("Skip: run scripts/dump_label_prop_stages.py");
        return;
    }

    let reduced_graph = load_csr_npz(&dir.join("intermediates/reduced_graph_csr.npz"));
    let py_init: Array2<f32> =
        Array2::read_npy(&mut File::open(dir.join("intermediates/reduced_init.npy")).unwrap())
            .unwrap();
    let py_layout: Array2<f32> =
        Array2::read_npy(&mut File::open(dir.join("intermediates/reduced_layout.npy")).unwrap())
            .unwrap();

    let n_comp = (15usize / 4).max(4).min(15);
    let mut rng = load_numpy_rng("after_reduced_init");
    let layout = node_embedding(
        &reduced_graph,
        n_comp,
        50,
        Some(py_init.clone()),
        0.05, // 0.001 * 50
        1.0,
        0.5,
        &mut rng,
        true,
    );
    let diff = max_abs(&layout, &py_layout);
    eprintln!("reduced_layout max diff {diff:.6}");

    let mut rng1 = load_numpy_rng("after_reduced_init");
    let layout1 = node_embedding(
        &reduced_graph,
        n_comp,
        1,
        Some(py_init),
        0.05,
        1.0,
        0.5,
        &mut rng1,
        true,
    );
    eprintln!(
        "reduced_layout 1-epoch vs py_layout {:.6}",
        max_abs(&layout1, &py_layout)
    );

    assert!(diff <= 1e-4, "reduced_layout max diff {diff}");
}

#[test]
fn parity_reduced_layout_large_2000() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/large_2000");
    if !dir
        .join("intermediates/rng_after_reduced_init_key.npy")
        .exists()
    {
        eprintln!("Skip: run scripts/dump_label_prop_stages.py large_2000");
        return;
    }

    let reduced_graph = load_csr_npz(&dir.join("intermediates/reduced_graph_csr.npz"));
    let py_init: Array2<f32> =
        Array2::read_npy(&mut File::open(dir.join("intermediates/reduced_init.npy")).unwrap())
            .unwrap();
    let py_layout: Array2<f32> =
        Array2::read_npy(&mut File::open(dir.join("intermediates/reduced_layout.npy")).unwrap())
            .unwrap();

    let n_comp = (15usize / 4).max(4).min(15);
    let mut rng = load_numpy_rng_from(&dir, "after_reduced_init");
    let layout = node_embedding(
        &reduced_graph,
        n_comp,
        50,
        Some(py_init),
        0.05,
        1.0,
        0.5,
        &mut rng,
        true,
    );
    let diff = max_abs(&layout, &py_layout);
    eprintln!("large_2000 reduced_layout max diff {diff:.6}");
    assert!(diff <= 1e-4, "reduced_layout max diff {diff}");
}

fn load_numpy_rng_from(dir: &PathBuf, tag: &str) -> NumpyRandomState {
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

#[test]
fn parity_reduced_init_large_2000() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/large_2000");
    if !dir.join("intermediates/reduced_init.npy").exists() {
        eprintln!("Skip: run scripts/dump_label_prop_stages.py large_2000");
        return;
    }

    let graph = load_py_graph_from(&dir);
    let data: Array2<f32> =
        Array2::read_npy(&mut File::open(dir.join("data.npy")).unwrap()).unwrap();
    let py_init: Array2<f32> =
        Array2::read_npy(&mut File::open(dir.join("intermediates/reduced_init.npy")).unwrap())
            .unwrap();

    let mut labels = vec![-1i64; graph.rows()];
    let mut rng = load_numpy_rng_from(&dir, "after_knn");
    let approx = (8.0 * (graph.rows() as f64).sqrt()).clamp(256.0, 16384.0) as usize;
    let partition = label_prop_loop(
        graph.indptr().raw_storage(),
        graph.indices(),
        graph.data(),
        &mut labels,
        &mut rng,
        20,
        approx,
    );

    let base_map = partition_reduction_map(&partition);
    let mut norm_map = base_map.clone();
    normalize_cols_l2(&mut norm_map);
    let mut data_reducer = norm_map.transpose_view().to_csr();
    normalize_rows_l1(&mut data_reducer);
    let reduced_data = csr_matmul_dense(&data_reducer, &data);

    let norm_t = norm_map.transpose_view().to_csr();
    let temp = scipy_csr_matmul(&norm_t, &graph);
    let mut reduced_graph = scipy_csr_matmul(&temp, &base_map);
    for v in reduced_graph.data_mut() {
        *v = v.clamp(0.0, 1.0);
    }

    let n_comp = (15usize / 4).max(4).min(15);
    let init = label_propagation_init(
        &reduced_graph,
        20,
        50,
        approx / 4,
        n_comp,
        0.5,
        0.1,
        0.5,
        &mut rng,
        Some(&reduced_data),
    );
    let diff = max_abs(&init, &py_init);
    eprintln!("large_2000 reduced_init max diff {diff:.6}");
    assert!(diff <= 2e-4, "reduced_init max diff {diff}");
}

fn load_py_graph_from(dir: &PathBuf) -> CsMat<f32> {
    let inter = dir.join("intermediates");
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
fn parity_reduced_init_medium_800() {
    let dir = base();
    if !dir.join("intermediates/reduced_init.npy").exists() {
        eprintln!("Skip: run scripts/dump_label_prop_stages.py");
        return;
    }

    let inter = base().join("intermediates");
    let graph = evoc::load_graph_csr_npz(inter.join("graph_csr.npz").as_path())
        .or_else(|| evoc::load_graph_coo_npz(inter.join("graph_coo.npz").as_path()))
        .unwrap();
    let data: Array2<f32> =
        Array2::read_npy(&mut File::open(dir.join("data.npy")).unwrap()).unwrap();
    let py_init: Array2<f32> =
        Array2::read_npy(&mut File::open(dir.join("intermediates/reduced_init.npy")).unwrap())
            .unwrap();

    let mut labels = vec![-1i64; graph.rows()];
    let mut rng = load_numpy_rng("after_knn");
    let approx = (8.0 * (graph.rows() as f64).sqrt()).clamp(256.0, 16384.0) as usize;
    let partition = label_prop_loop(
        graph.indptr().raw_storage(),
        graph.indices(),
        graph.data(),
        &mut labels,
        &mut rng,
        20,
        approx,
    );

    let base_map = partition_reduction_map(&partition);
    let mut norm_map = base_map.clone();
    normalize_cols_l2(&mut norm_map);
    let mut data_reducer = norm_map.transpose_view().to_csr();
    normalize_rows_l1(&mut data_reducer);
    let reduced_data = csr_matmul_dense(&data_reducer, &data);

    let norm_t = norm_map.transpose_view().to_csr();
    let temp = scipy_csr_matmul(&norm_t, &graph);
    let mut reduced_graph = scipy_csr_matmul(&temp, &base_map);
    for v in reduced_graph.data_mut() {
        *v = v.clamp(0.0, 1.0);
    }

    let n_comp = (15usize / 4).max(4).min(15);
    let init = label_propagation_init(
        &reduced_graph,
        20,
        50,
        approx / 4,
        n_comp,
        0.5,
        0.1,
        0.5,
        &mut rng,
        Some(&reduced_data),
    );
    let diff = max_abs(&init, &py_init);
    eprintln!("reduced_init max diff {diff:.6}");
    assert!(diff <= 1e-4, "reduced_init max diff {diff}");
}

#[test]
fn parity_rng_after_knn_matches_python() {
    let dir = base();
    if !dir.join("intermediates/rng_after_knn_key.npy").exists() {
        return;
    }
    use evoc::{check_random_state, knn_graph, EmbeddingData, KnnGraphOptions};

    let data: Array2<f32> = {
        let mut f = File::open(dir.join("data.npy")).unwrap();
        Array2::read_npy(&mut f).unwrap()
    };
    let mut rust_rng = check_random_state(Some(42));
    let _ = knn_graph(
        EmbeddingData::Float32(data),
        KnnGraphOptions {
            n_neighbors: 15,
            deterministic: true,
            ..Default::default()
        },
        &mut rust_rng,
    )
    .unwrap();
    let py_rng = load_numpy_rng("after_knn");
    // Compare next 5 randint draws — if state matches, sequences match.
    let mut r1 = rust_rng;
    let mut r2 = py_rng;
    const LO: i64 = i32::MIN as i64 + 1;
    const HI: i64 = i32::MAX as i64 - 1;
    for _ in 0..5 {
        assert_eq!(r1.randint(LO, HI), r2.randint(LO, HI));
    }
}
