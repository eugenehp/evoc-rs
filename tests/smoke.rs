use evoc::{check_random_state, Evoc};
use ndarray::Array2;

fn random_unit_embeddings(n: usize, d: usize, seed: u64) -> Array2<f32> {
    let mut rng = check_random_state(Some(seed));
    let mut data = Array2::from_shape_fn((n, d), |_| rng.normal_scaled(1.0) as f32);
    for mut row in data.rows_mut() {
        let norm = row.dot(&row).sqrt().max(1e-12);
        row /= norm;
    }
    data
}

#[test]
fn evoc_fit_predict_smoke() {
    let data = random_unit_embeddings(2_000, 64, 0);
    let mut clusterer = Evoc {
        random_state: Some(42),
        n_neighbors: 15,
        ..Evoc::default()
    };
    let labels = clusterer
        .fit_predict(data)
        .expect("clustering should succeed");
    assert_eq!(labels.len(), 2_000);
    assert!(!clusterer.cluster_layers_.is_empty());
    let clustered = labels.iter().filter(|&&l| l >= 0).count();
    assert!(clustered > 0);
}

#[test]
fn knn_graph_float() {
    use evoc::{knn_graph, EmbeddingData, KnnGraphOptions};
    let data = random_unit_embeddings(500, 32, 1);
    let mut rng = check_random_state(Some(1));
    let (inds, dists) = knn_graph(
        EmbeddingData::Float32(data),
        KnnGraphOptions {
            n_neighbors: 10,
            ..Default::default()
        },
        &mut rng,
    )
    .unwrap();
    assert_eq!(inds.shape(), [500, 10]);
    assert_eq!(dists.shape(), [500, 10]);
}
