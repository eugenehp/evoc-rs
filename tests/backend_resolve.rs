use evoc::ComputeBackend;

#[test]
#[cfg(all(feature = "rlx-mlx", not(feature = "rlx-cuda")))]
fn mlx_build_rejects_cuda_backend() {
    assert!(!ComputeBackend::Cuda.is_enabled());
    assert!(ComputeBackend::resolve(Some(ComputeBackend::Cuda)).is_err());
}

#[test]
#[cfg(feature = "rlx-cuda")]
fn cuda_backend_enabled() {
    assert!(ComputeBackend::Cuda.is_enabled());
    assert!(ComputeBackend::resolve(Some(ComputeBackend::Cuda)).is_ok());
}
