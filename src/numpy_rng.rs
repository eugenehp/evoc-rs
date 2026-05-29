//! NumPy `numpy.random.RandomState` (MT19937 + legacy distributions) for bit-exact parity.

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;
const RK_STATE_LEN: usize = 624;

/// Legacy NumPy / sklearn `RandomState` (seed with `u32` or `i32`).
#[derive(Clone, Debug)]
pub struct NumpyRandomState {
    key: [u32; RK_STATE_LEN],
    pos: i32,
    gauss: f64,
    has_gauss: bool,
}

impl NumpyRandomState {
    pub fn new(seed: u32) -> Self {
        let mut state = Self {
            key: [0; RK_STATE_LEN],
            pos: RK_STATE_LEN as i32,
            gauss: 0.0,
            has_gauss: false,
        };
        state.mt19937_seed(seed);
        state
    }

    pub fn from_seed(seed: u64) -> Self {
        Self::new(seed as u32)
    }

    fn mt19937_seed(&mut self, seed: u32) {
        let mut seed = seed;
        self.key[0] = seed;
        for i in 1..N {
            seed = 1812433253u32
                .wrapping_mul(seed ^ (seed >> 30))
                .wrapping_add(i as u32);
            self.key[i] = seed;
        }
        self.pos = RK_STATE_LEN as i32;
    }

    fn mt19937_gen(&mut self) {
        for i in 0..N - M {
            let y = (self.key[i] & UPPER_MASK) | (self.key[i + 1] & LOWER_MASK);
            self.key[i] = self.key[i + M] ^ (y >> 1) ^ ((y & 1).wrapping_neg() & MATRIX_A);
        }
        for i in N - M..N - 1 {
            let y = (self.key[i] & UPPER_MASK) | (self.key[i + 1] & LOWER_MASK);
            self.key[i] = self.key[i + M - N] ^ (y >> 1) ^ ((y & 1).wrapping_neg() & MATRIX_A);
        }
        let y = (self.key[N - 1] & UPPER_MASK) | (self.key[0] & LOWER_MASK);
        self.key[N - 1] = self.key[M - 1] ^ (y >> 1) ^ ((y & 1).wrapping_neg() & MATRIX_A);
        self.pos = 0;
    }

    fn next_u32(&mut self) -> u32 {
        if self.pos >= RK_STATE_LEN as i32 {
            self.mt19937_gen();
        }
        let mut y = self.key[self.pos as usize];
        self.pos += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    fn next_double(&mut self) -> f64 {
        let a = (self.next_u32() >> 5) as i32;
        let b = (self.next_u32() >> 6) as i32;
        (a as f64 * 67_108_864.0 + b as f64) / 9_007_199_254_740_992.0
    }

    /// `randint(low, high)` — half-open `[low, high)` when both given.
    pub fn randint(&mut self, low: i64, high: i64) -> i64 {
        let range = (high - low) as u64;
        if range == 0 {
            return low;
        }
        let val = self.random_interval(range - 1);
        low + val as i64
    }

    /// `randint(high)` with `low=0`.
    pub fn randint_high(&mut self, high: i64) -> i64 {
        self.randint(0, high)
    }

    fn random_interval(&mut self, max: u64) -> u64 {
        if max == 0 {
            return 0;
        }
        let mut mask = max;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        mask |= mask >> 32;
        loop {
            let value = (self.next_u32() as u64) & mask;
            if value <= max {
                return value;
            }
        }
    }

    /// `normal(size=n)` element (legacy polar / Box-Muller cache).
    pub fn normal(&mut self) -> f64 {
        if self.has_gauss {
            self.has_gauss = false;
            let temp = self.gauss;
            self.gauss = 0.0;
            return temp;
        }
        loop {
            let x1 = 2.0 * self.next_double() - 1.0;
            let x2 = 2.0 * self.next_double() - 1.0;
            let r2 = x1 * x1 + x2 * x2;
            if r2 < 1.0 && r2 != 0.0 {
                let f = (-2.0 * r2.ln() / r2).sqrt();
                self.gauss = f * x1;
                self.has_gauss = true;
                return f * x2;
            }
        }
    }

    /// Fisher–Yates shuffle (NumPy `RandomState.shuffle`).
    pub fn shuffle<T>(&mut self, arr: &mut [T]) {
        for i in (1..arr.len()).rev() {
            let j = self.randint(0, (i + 1) as i64) as usize;
            arr.swap(i, j);
        }
    }

    /// Uniform floats in [0, 1).
    pub fn random(&mut self) -> f64 {
        self.next_double()
    }

    /// `normal(scale=s)` → `s * N(0,1)` (NumPy `normal(loc=0, scale=s)`).
    pub fn normal_scaled(&mut self, scale: f64) -> f64 {
        scale * self.normal()
    }

    /// Restore from NumPy `RandomState.get_state()` tuple fields.
    pub fn from_numpy_state(
        key: &[u32; RK_STATE_LEN],
        pos: i32,
        has_gauss: i32,
        gauss: f64,
    ) -> Self {
        Self {
            key: *key,
            pos,
            gauss,
            has_gauss: has_gauss != 0,
        }
    }

    /// Load RNG checkpoint written by `scripts/dump_intermediates.py`.
    #[cfg(feature = "npy")]
    pub fn from_intermediates_dir(dir: &std::path::Path, tag: &str) -> Option<Self> {
        use ndarray::Array1;
        use ndarray_npy::{NpzReader, ReadNpyExt};
        use std::fs::File;

        let mut kf = File::open(dir.join(format!("rng_{tag}_key.npy"))).ok()?;
        let key: Array1<u32> = Array1::read_npy(&mut kf).ok()?;
        let mut arr = [0u32; RK_STATE_LEN];
        let key_slice = key.as_slice().unwrap_or(&[]);
        if key_slice.len() != RK_STATE_LEN {
            return None;
        }
        arr.copy_from_slice(key_slice);
        let mut npz =
            NpzReader::new(File::open(dir.join(format!("rng_{tag}_meta.npz"))).ok()?).ok()?;
        let pos: ndarray::Array0<i32> = npz.by_name("pos").ok()?;
        let has_gauss: ndarray::Array0<i32> = npz.by_name("has_gauss").ok()?;
        let gauss: ndarray::Array0<f64> = npz.by_name("gauss").ok()?;
        let pos_v = pos.iter().next().copied().unwrap_or(0);
        let has_gauss_v = has_gauss.iter().next().copied().unwrap_or(0);
        let gauss_v = gauss.iter().next().copied().unwrap_or(0.0);
        Some(Self::from_numpy_state(&arr, pos_v, has_gauss_v, gauss_v))
    }

    #[doc(hidden)]
    pub fn key_state(&self) -> &[u32; RK_STATE_LEN] {
        &self.key
    }

    #[doc(hidden)]
    pub fn position(&self) -> i32 {
        self.pos
    }

    pub fn randint3_for_tau(&mut self) -> [i64; 3] {
        const INT32_MIN: i64 = i32::MIN as i64 + 1;
        const INT32_MAX: i64 = i32::MAX as i64 - 1;
        [
            self.randint(INT32_MIN, INT32_MAX),
            self.randint(INT32_MIN, INT32_MAX),
            self.randint(INT32_MIN, INT32_MAX),
        ]
    }
}

/// Match `sklearn.utils.check_random_state`.
pub fn check_random_state(seed: Option<u64>) -> NumpyRandomState {
    match seed {
        Some(s) => NumpyRandomState::from_seed(s),
        None => NumpyRandomState::new(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u32)
                .unwrap_or(0),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_numpy_seed_42() {
        let mut r = NumpyRandomState::new(42);
        const INT32_MIN: i64 = i32::MIN as i64 + 1;
        const INT32_MAX: i64 = i32::MAX as i64 - 1;
        let a: [i64; 3] = [
            r.randint(INT32_MIN, INT32_MAX),
            r.randint(INT32_MIN, INT32_MAX),
            r.randint(INT32_MIN, INT32_MAX),
        ];
        assert_eq!(a, [-538846105, 1273642420, 1935803229]);
        let b: [i64; 5] = std::array::from_fn(|_| r.randint(INT32_MIN, INT32_MAX));
        assert_eq!(
            b,
            [-1359637233, 996406379, 1201263688, 423734973, 415968277]
        );
        let n = [r.normal(), r.normal(), r.normal()];
        assert!((n[0] - (-0.23415337472333597)).abs() < 1e-12);
        assert!((n[1] - (-0.23413695694918055)).abs() < 1e-12);
        assert!((n[2] - 1.5792128155073915).abs() < 1e-12);
    }
}
