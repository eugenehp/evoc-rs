//! Optional RLX acceleration backends for clustering stages.
//!
//! Each backend is enabled via a Cargo feature and maps to an RLX runtime device
//! (`rlx-cpu`, `rlx/cuda`, `rlx/mlx`, `rlx/rocm`, `rlx/gpu`). Until GPU kernels are wired,
//! all RLX backends run the strict reference implementation (bit-exact when
//! `strict_precision` is true).

mod delegate;
mod strict;

use crate::knn::KnnError;
use crate::numpy_rng::NumpyRandomState;
use crate::{EmbeddingData, KnnGraphOptions};
use delegate::DelegateBackend;
use ndarray::Array2;
use sprs::CsMat;
use thiserror::Error;

/// Compute backend selection for [`crate::Evoc`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ComputeBackend {
    /// Pure Rust reference (default).
    Strict,
    /// RLX CPU (`rlx-cpu` / `rlx-runtime` CPU).
    Cpu,
    /// NVIDIA CUDA (`rlx/cuda`).
    Cuda,
    /// Apple MLX (`rlx/mlx`).
    Mlx,
    /// AMD ROCm (`rlx/rocm`).
    Rocm,
    /// Cross-platform wgpu (`rlx/gpu`).
    Wgpu,
}

impl ComputeBackend {
    /// Parse `EVOC_BACKEND` (case-sensitive).
    ///
    /// | Value | Backend |
    /// |-------|---------|
    /// | `strict` | [`Strict`] |
    /// | `cpu` | [`Cpu`] |
    /// | `cuda` | [`Cuda`] |
    /// | `mlx`, `metal` | [`Mlx`] |
    /// | `rocm` | [`Rocm`] |
    /// | `wgpu`, `gpu` | [`Wgpu`] |
    pub fn from_env() -> Option<Self> {
        let v = std::env::var("EVOC_BACKEND").ok()?;
        match v.as_str() {
            "strict" => Some(Self::Strict),
            "cpu" => Some(Self::Cpu),
            "cuda" => Some(Self::Cuda),
            "mlx" | "metal" => Some(Self::Mlx),
            "rocm" => Some(Self::Rocm),
            "wgpu" | "gpu" => Some(Self::Wgpu),
            _ => None,
        }
    }

    /// Backends compiled into this binary (RLX variants require their feature).
    pub fn available() -> Vec<Self> {
        let mut out = vec![Self::Strict];
        if cfg!(feature = "rlx-cpu") {
            out.push(Self::Cpu);
        }
        if cfg!(feature = "rlx-cuda") {
            out.push(Self::Cuda);
        }
        if cfg!(feature = "rlx-mlx") {
            out.push(Self::Mlx);
        }
        if cfg!(feature = "rlx-rocm") {
            out.push(Self::Rocm);
        }
        if cfg!(feature = "rlx-wgpu") {
            out.push(Self::Wgpu);
        }
        out
    }

    /// Backends to run in benchmarks / smoke tools.
    ///
    /// If `EVOC_BACKEND` is set, returns only that backend (after [`Self::resolve`]).
    /// Otherwise returns [`Self::available`].
    pub fn backends_for_run() -> Result<Vec<Self>, BackendError> {
        if std::env::var("EVOC_BACKEND").is_ok() {
            let explicit = Self::from_env().ok_or(BackendError::UnknownEnv)?;
            return Ok(vec![Self::resolve(Some(explicit))?]);
        }
        Ok(Self::available())
    }

    /// Resolve an explicit backend or `EVOC_BACKEND`, defaulting to [`Strict`].
    ///
    /// Errors if a non-strict backend was requested but its Cargo feature is off.
    pub fn resolve(explicit: Option<Self>) -> Result<Self, BackendError> {
        let kind = explicit.or_else(Self::from_env).unwrap_or(Self::Strict);
        if !kind.is_enabled() {
            return Err(BackendError::NotEnabled {
                backend: kind,
                feature: feature_name(kind),
            });
        }
        Ok(kind)
    }

    /// Whether this backend was enabled at compile time.
    pub fn is_enabled(self) -> bool {
        matches!(self, Self::Strict)
            || (cfg!(feature = "rlx-cpu") && self == Self::Cpu)
            || (cfg!(feature = "rlx-cuda") && self == Self::Cuda)
            || (cfg!(feature = "rlx-mlx") && self == Self::Mlx)
            || (cfg!(feature = "rlx-rocm") && self == Self::Rocm)
            || (cfg!(feature = "rlx-wgpu") && self == Self::Wgpu)
    }
}

impl std::fmt::Display for ComputeBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Strict => write!(f, "strict"),
            Self::Cpu => write!(f, "cpu"),
            Self::Cuda => write!(f, "cuda"),
            Self::Mlx => write!(f, "mlx"),
            Self::Rocm => write!(f, "rocm"),
            Self::Wgpu => write!(f, "wgpu"),
        }
    }
}

/// Acceleration backend interface.
///
/// `strict_precision=true` requires bitwise-identical results to [`Strict`], or the
/// implementation must delegate to the strict reference path.
pub trait RlxBackend {
    fn kind(&self) -> ComputeBackend;

    fn knn_graph(
        &self,
        data: EmbeddingData,
        opts: KnnGraphOptions,
        rng: &mut NumpyRandomState,
        strict_precision: bool,
    ) -> Result<(Array2<i32>, Array2<f32>), KnnError>;

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
        strict_precision: bool,
    ) -> Array2<f32>;

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
        strict_precision: bool,
    ) -> Array2<f32>;
}

fn delegate(kind: ComputeBackend) -> Box<dyn RlxBackend + Send + Sync> {
    Box::new(DelegateBackend { kind })
}

fn feature_name(kind: ComputeBackend) -> &'static str {
    match kind {
        ComputeBackend::Cpu => "rlx-cpu",
        ComputeBackend::Cuda => "rlx-cuda",
        ComputeBackend::Mlx => "rlx-mlx",
        ComputeBackend::Rocm => "rlx-rocm",
        ComputeBackend::Wgpu => "rlx-wgpu",
        ComputeBackend::Strict => "strict",
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum BackendError {
    #[error("unknown EVOC_BACKEND (use strict|cpu|cuda|mlx|metal|rocm|wgpu|gpu)")]
    UnknownEnv,
    #[error(
        "compute backend `{backend}` not enabled at compile time (enable Cargo feature `{feature}`)"
    )]
    NotEnabled {
        backend: ComputeBackend,
        feature: &'static str,
    },
}

pub fn make_backend(
    kind: ComputeBackend,
) -> Result<Box<dyn RlxBackend + Send + Sync>, BackendError> {
    let kind = ComputeBackend::resolve(Some(kind))?;
    Ok(match kind {
        ComputeBackend::Strict => Box::new(strict::StrictBackend),
        ComputeBackend::Cpu
        | ComputeBackend::Cuda
        | ComputeBackend::Mlx
        | ComputeBackend::Rocm
        | ComputeBackend::Wgpu => delegate(kind),
    })
}
