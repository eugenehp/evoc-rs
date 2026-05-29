//! Cosine distance for L2-normalized f32 vectors (Numba `fast_cosine` semantics).
//!
//! - **Matching / deterministic:** C `-ffast-math` (`native/fast_cosine.c`) — bit-compatible
//!   with Python golden kNN fixtures.
//! - **Default fast path:** [`rlx-cpu`](https://docs.rs/rlx-cpu) NEON dot on aarch64.

use ndarray::ArrayView1;
use std::sync::atomic::{AtomicBool, Ordering};

static STRICT_COSINE: AtomicBool = AtomicBool::new(false);

extern "C" {
    fn fast_cosine_numba(x: *const f32, y: *const f32, dim: i32) -> f32;
}

pub const EXP_NEG_INF: f32 = f32::MIN_POSITIVE;

/// Restore previous strict flag when dropped (safe under parallel `cargo test`).
pub fn strict_cosine_guard(strict: bool) -> StrictCosineGuard {
    StrictCosineGuard(STRICT_COSINE.swap(strict, Ordering::Relaxed))
}

pub struct StrictCosineGuard(bool);

impl Drop for StrictCosineGuard {
    fn drop(&mut self) {
        STRICT_COSINE.store(self.0, Ordering::Relaxed);
    }
}

/// Negative dot product when positive; else `f32::MIN_POSITIVE` (matches EVōC / Numba).
#[inline]
pub fn fast_cosine(x: ArrayView1<f32>, y: ArrayView1<f32>) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    if STRICT_COSINE.load(Ordering::Relaxed) {
        return unsafe { fast_cosine_numba(x.as_ptr(), y.as_ptr(), x.len() as i32) };
    }
    let dot = dot_f32_rlx(x.as_slice().unwrap(), y.as_slice().unwrap());
    if dot > 0.0 {
        -dot
    } else {
        EXP_NEG_INF
    }
}

#[inline]
fn dot_f32_rlx(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe {
            rlx_cpu::intrinsics::neon::strided_dot_f32(x.as_ptr(), 1, y.as_ptr(), 1, x.len())
        };
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut sum = 0.0f32;
        for i in 0..x.len() {
            sum += x[i] * y[i];
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn positive_dot_is_negated() {
        let _g = strict_cosine_guard(true);
        let a = Array1::from_vec(vec![1.0, 0.0]);
        let b = Array1::from_vec(vec![0.5, 0.8660254]);
        let d = fast_cosine(a.view(), b.view());
        assert!((d + 0.5).abs() < 1e-6);
    }

    #[test]
    fn non_positive_dot_is_min_positive() {
        let _g = strict_cosine_guard(true);
        let a = Array1::from_vec(vec![1.0, 0.0]);
        let b = Array1::from_vec(vec![-1.0, 0.0]);
        assert_eq!(fast_cosine(a.view(), b.view()), EXP_NEG_INF);
    }
}
