use evoc::{
    build_cluster_layers, check_random_state, knn_graph, label_propagation_init,
    neighbor_graph_matrix, node_embedding, EmbeddingData, Evoc, KnnGraphOptions,
};
use ndarray::Array2;
use std::time::Instant;

fn make_data(n: usize, d: usize, centers: usize, seed: u64) -> Array2<f32> {
    // Lightweight synthetic: random centers on unit sphere + gaussian noise, then L2 normalize.
    let mut rng = check_random_state(Some(seed));
    let mut center = Array2::<f32>::zeros((centers, d));
    for c in 0..centers {
        let mut norm = 0.0f32;
        for j in 0..d {
            let v = rng.normal_scaled(1.0) as f32;
            center[[c, j]] = v;
            norm += v * v;
        }
        norm = norm.sqrt().max(1e-12);
        for j in 0..d {
            center[[c, j]] /= norm;
        }
    }

    let mut data = Array2::<f32>::zeros((n, d));
    for i in 0..n {
        let c = (rng.randint(0, centers as i64) as usize).min(centers - 1);
        let mut norm = 0.0f32;
        for j in 0..d {
            let v = center[[c, j]] + (rng.normal_scaled(0.15) as f32);
            data[[i, j]] = v;
            norm += v * v;
        }
        norm = norm.sqrt().max(1e-12);
        for j in 0..d {
            data[[i, j]] /= norm;
        }
    }
    data
}

fn main() {
    // Usage:
    //   cargo run --release --bin bench_huge -- <n> <d> <k> <centers> <seed>
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);
    let d: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let k: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(15);
    let centers: usize = std::env::args()
        .nth(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    let seed: u64 = std::env::args()
        .nth(5)
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);

    let t0 = Instant::now();
    let data = make_data(n, d, centers, seed);
    let t_data = t0.elapsed().as_secs_f64();

    let mut rng = check_random_state(Some(seed));
    let t1 = Instant::now();
    let (nn_inds, nn_dists) = knn_graph(
        EmbeddingData::Float32(data.clone()),
        KnnGraphOptions {
            n_neighbors: k,
            deterministic: false,
            ..Default::default()
        },
        &mut rng,
    )
    .unwrap();
    let t_knn = t1.elapsed().as_secs_f64();

    let t2 = Instant::now();
    let graph = neighbor_graph_matrix(k as f32, &nn_inds, &nn_dists, true);
    let t_graph = t2.elapsed().as_secs_f64();

    let n_comp = (k / 4).max(4).min(15);
    let approx = (8.0 * (n as f64).sqrt()).clamp(256.0, 16384.0) as usize;

    let t3 = Instant::now();
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
    let t_init = t3.elapsed().as_secs_f64();

    let t4 = Instant::now();
    let emb = node_embedding(
        &graph,
        n_comp,
        50,
        Some(init),
        0.1,
        1.0,
        0.5,
        &mut rng,
        false,
    );
    let t_emb = t4.elapsed().as_secs_f64();

    let t5 = Instant::now();
    let (_layers, _strengths, _scores) = build_cluster_layers(&emb, 5, 5, None, false, 0.2, 10);
    let t_cluster = t5.elapsed().as_secs_f64();

    // Also time the full high-level API (for comparison, includes repeated work).
    let t6 = Instant::now();
    let mut clusterer = Evoc {
        random_state: Some(seed),
        n_neighbors: k,
        ..Evoc::default()
    };
    let _labels = clusterer.fit_predict(data).unwrap();
    let t_fit = t6.elapsed().as_secs_f64();

    println!(
        "n {n} d {d} k {k} centers {centers} data_s {t_data:.3} knn_s {t_knn:.3} graph_s {t_graph:.3} init_s {t_init:.3} emb_s {t_emb:.3} cluster_s {t_cluster:.3} fit_predict_s {t_fit:.3}"
    );
}
