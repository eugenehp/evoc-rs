use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
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
    let mut f = File::open(path).expect("open fixture data.npy");
    Array2::read_npy(&mut f).expect("read npy")
}

fn bench_fit_predict(c: &mut Criterion) {
    // Keep these benches stable over time by using the same fixture data.
    // (If fixtures are regenerated, expect a discontinuity in results.)
    let fixtures = [
        ("small_200", 42u64),
        ("medium_800", 42u64),
        ("large_2000", 42u64),
    ];

    let backends: Vec<ComputeBackend> = ComputeBackend::backends_for_run().expect("EVOC_BACKEND");

    for backend in &backends {
        let mut group = c.benchmark_group(format!("fit_predict/{}", backend));
        group.sample_size(20);

        for (name, seed) in fixtures {
            let data = load_fixture(name);
            group.bench_with_input(BenchmarkId::new("evoc", name), &data, |b, data| {
                b.iter(|| {
                    let mut clusterer = Evoc {
                        random_state: Some(*seed),
                        n_neighbors: 15,
                        parity_graph_coo: None,
                        compute_backend: Some(*backend),
                        strict_precision: false,
                        ..Evoc::default()
                    };
                    let _labels = clusterer.fit_predict(data.clone()).expect("fit_predict");
                });
            });
        }
        group.finish();
    }
}

criterion_group!(benches, bench_fit_predict);
criterion_main!(benches);
