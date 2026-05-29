use super::{ComputeBackend, RlxBackend};
use crate::embed::node_embedding as strict_node_embedding;
use crate::knn::knn_graph as strict_knn_graph;
use crate::knn::KnnError;
use crate::label_prop::label_propagation_init as strict_label_propagation_init;
use crate::numpy_rng::NumpyRandomState;
use crate::{EmbeddingData, KnnGraphOptions};
use ndarray::Array2;
use sprs::CsMat;

pub struct StrictBackend;

impl RlxBackend for StrictBackend {
    fn kind(&self) -> ComputeBackend {
        ComputeBackend::Strict
    }

    fn knn_graph(
        &self,
        data: EmbeddingData,
        opts: KnnGraphOptions,
        rng: &mut NumpyRandomState,
        _strict_precision: bool,
    ) -> Result<(Array2<i32>, Array2<f32>), KnnError> {
        strict_knn_graph(data, opts, rng)
    }

    fn label_propagation_init(
        &self,
        graph: &CsMat<f32>,
        n_label_prop_iter: usize,
        n_embedding_epochs: usize,
        approx_n_parts: usize,
        n_components: usize,
        scaling: f32,
        random_scale: f32,
        noise_level: f32,
        rng: &mut NumpyRandomState,
        data: Option<&Array2<f32>>,
        _strict_precision: bool,
    ) -> Array2<f32> {
        strict_label_propagation_init(
            graph,
            n_label_prop_iter,
            n_embedding_epochs,
            approx_n_parts,
            n_components,
            scaling,
            random_scale,
            noise_level,
            rng,
            data,
        )
    }

    fn node_embedding(
        &self,
        graph: &CsMat<f32>,
        n_components: usize,
        n_epochs: usize,
        initial_embedding: Option<Array2<f32>>,
        initial_alpha: f32,
        negative_sample_rate: f32,
        noise_level: f32,
        rng: &mut NumpyRandomState,
        reproducible_flag: bool,
        _strict_precision: bool,
    ) -> Array2<f32> {
        strict_node_embedding(
            graph,
            n_components,
            n_epochs,
            initial_embedding,
            initial_alpha,
            negative_sample_rate,
            noise_level,
            rng,
            reproducible_flag,
        )
    }
}
