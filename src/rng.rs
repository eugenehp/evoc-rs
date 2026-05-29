//! Tau RNG matching `evoc.common_nndescent`.

/// Fast pseudo-random int32 (Python: `tau_rand_int`).
#[inline]
pub fn tau_rand_int(state: &mut [i64; 3]) -> i32 {
    state[0] = ((((state[0] & 4294967294) << 12) & 0xFFFF_FFFF) as i64)
        ^ (((((state[0] << 13) & 0xFFFF_FFFF) as i64) ^ state[0]) >> 19);
    state[1] = ((((state[1] & 4294967288) << 4) & 0xFFFF_FFFF) as i64)
        ^ (((((state[1] << 2) & 0xFFFF_FFFF) as i64) ^ state[1]) >> 25);
    state[2] = ((((state[2] & 4294967280) << 17) & 0xFFFF_FFFF) as i64)
        ^ (((((state[2] << 3) & 0xFFFF_FFFF) as i64) ^ state[2]) >> 11);
    (state[0] ^ state[1] ^ state[2]) as i32
}

/// Uniform float in [0, 1] (Python: `tau_rand`).
#[inline]
pub fn tau_rand(state: &mut [i64; 3]) -> f32 {
    let integer = tau_rand_int(state);
    (integer.abs() as f32) / 0x7FFF_FFFF as f32
}

/// `tau_rand_int(state) % modulus` with Python/Numba signed-mod semantics (`modulus > 0`).
#[inline]
pub fn tau_rand_mod(state: &mut [i64; 3], modulus: i32) -> i32 {
    tau_rand_int(state).rem_euclid(modulus)
}

/// Offset RNG state per thread/index (Python: `rng_state + n`).
pub fn offset_state(base: &[i64; 3], offset: i64) -> [i64; 3] {
    [
        base[0].wrapping_add(offset),
        base[1].wrapping_add(offset),
        base[2].wrapping_add(offset),
    ]
}
