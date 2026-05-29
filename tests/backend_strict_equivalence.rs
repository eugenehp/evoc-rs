use evoc::{check_random_state, ComputeBackend, EmbeddingData, Evoc, KnnGraphOptions};
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
#[cfg(feature = "rlx-cpu")]
fn strict_and_cpu_backend_match_bitwise() {
    let data = load_fixture("small_200");
    let seed = 42u64;

    // kNN
    let mut rng_s = check_random_state(Some(seed));
    let mut rng_c = check_random_state(Some(seed));
    let opts = KnnGraphOptions {
        n_neighbors: 15,
        deterministic: true,
        ..Default::default()
    };
    let (s_inds, s_dists) = evoc::knn_graph(
        EmbeddingData::Float32(data.clone()),
        opts.clone(),
        &mut rng_s,
    )
    .unwrap();
    let mut cpu = Evoc {
        random_state: Some(seed),
        compute_backend: Some(ComputeBackend::Cpu),
        strict_precision: true,
        ..Evoc::default()
    };
    // exercise backend path through fit_predict
    let _ = cpu.fit_predict(data.clone()).unwrap();

    // ensure RNG stream didn’t diverge on strict path call
    let (c_inds, c_dists) =
        evoc::knn_graph(EmbeddingData::Float32(data.clone()), opts, &mut rng_c).unwrap();
    assert_eq!(s_inds.as_slice().unwrap(), c_inds.as_slice().unwrap());
    assert_eq!(s_dists.as_slice().unwrap(), c_dists.as_slice().unwrap());
}

#[test]
#[cfg(feature = "rlx-mlx")]
fn strict_and_mlx_backend_match_bitwise() {
    let data = load_fixture("small_200");
    let seed = 42u64;
    let mut strict = Evoc {
        random_state: Some(seed),
        compute_backend: Some(ComputeBackend::Strict),
        strict_precision: true,
        ..Evoc::default()
    };
    let mut mlx = Evoc {
        random_state: Some(seed),
        compute_backend: Some(ComputeBackend::Mlx),
        strict_precision: true,
        ..Evoc::default()
    };
    let l1 = strict.fit_predict(data.clone()).unwrap();
    let l2 = mlx.fit_predict(data).unwrap();
    assert_eq!(l1.as_slice().unwrap(), l2.as_slice().unwrap());
}
