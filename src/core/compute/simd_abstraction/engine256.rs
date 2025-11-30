//! 256‑bit SIMD engine (AVX2)
//!
//! This module provides the AVX2 implementation of the `SimdEngine` trait on
//! x86_64. It offers 32 lanes for 8‑bit operations and 16 lanes for 16‑bit
//! operations, mapping directly to `_mm256_*` intrinsics where possible.
//!
//! Highlights
//! - Variable byte shifts and `alignr` are implemented via small match‑table
//!   macros to satisfy the x86 immediate‑only requirement while accepting a
//!   runtime `num_bytes` parameter.
//! - `movemask_epi16` is derived from two 128‑bit `movemask_epi8` calls on the
//!   low/high halves and folded into a 16‑bit mask.
//! - All functions are `unsafe` and additionally annotated with
//!   `#[target_feature(enable = "avx2")]`. Callers must ensure AVX2 is
//!   available (the crate’s runtime dispatch does this for you).
//!
//! Performance notes
//! - Compared to the 128‑bit engine, most compute‑bound kernels see ~2×
//!   throughput from the doubled vector width.
//! - The macro‑based variable shifts compile to optimal immediate forms.

#[cfg(target_arch = "x86_64")]
use super::SimdEngine;
#[cfg(target_arch = "x86_64")]
use super::types::simd_arch;

// ===== Internal helper macros (AVX2) =====
// These macros de-duplicate the repeated 0..=15 match tables needed for
// byte-wise variable shifts and alignr operations, which require immediate
// operands at the intrinsic level.
#[cfg(target_arch = "x86_64")]
macro_rules! mm256_match_shift_bytes {
    ($a:expr, $n:expr, $op:ident) => {{
        match $n {
            0 => $a,
            1 => simd_arch::$op($a, 1),
            2 => simd_arch::$op($a, 2),
            3 => simd_arch::$op($a, 3),
            4 => simd_arch::$op($a, 4),
            5 => simd_arch::$op($a, 5),
            6 => simd_arch::$op($a, 6),
            7 => simd_arch::$op($a, 7),
            8 => simd_arch::$op($a, 8),
            9 => simd_arch::$op($a, 9),
            10 => simd_arch::$op($a, 10),
            11 => simd_arch::$op($a, 11),
            12 => simd_arch::$op($a, 12),
            13 => simd_arch::$op($a, 13),
            14 => simd_arch::$op($a, 14),
            15 => simd_arch::$op($a, 15),
            _ => simd_arch::_mm256_setzero_si256(),
        }
    }};
}

#[cfg(target_arch = "x86_64")]
macro_rules! mm256_alignr_bytes_match {
    ($a:expr, $b:expr, $n:expr) => {{
        match $n {
            0 => simd_arch::_mm256_alignr_epi8($a, $b, 0),
            1 => simd_arch::_mm256_alignr_epi8($a, $b, 1),
            2 => simd_arch::_mm256_alignr_epi8($a, $b, 2),
            3 => simd_arch::_mm256_alignr_epi8($a, $b, 3),
            4 => simd_arch::_mm256_alignr_epi8($a, $b, 4),
            5 => simd_arch::_mm256_alignr_epi8($a, $b, 5),
            6 => simd_arch::_mm256_alignr_epi8($a, $b, 6),
            7 => simd_arch::_mm256_alignr_epi8($a, $b, 7),
            8 => simd_arch::_mm256_alignr_epi8($a, $b, 8),
            9 => simd_arch::_mm256_alignr_epi8($a, $b, 9),
            10 => simd_arch::_mm256_alignr_epi8($a, $b, 10),
            11 => simd_arch::_mm256_alignr_epi8($a, $b, 11),
            12 => simd_arch::_mm256_alignr_epi8($a, $b, 12),
            13 => simd_arch::_mm256_alignr_epi8($a, $b, 13),
            14 => simd_arch::_mm256_alignr_epi8($a, $b, 14),
            15 => simd_arch::_mm256_alignr_epi8($a, $b, 15),
            _ => simd_arch::_mm256_alignr_epi8($a, $b, 0),
        }
    }};
}

#[cfg(target_arch = "x86_64")]
/// 256-bit SIMD engine (AVX2 on x86_64)
///
/// Provides 32-way parallelism for 8-bit operations and 16-way for 16-bit operations.
/// Requires AVX2 CPU support (Intel Haswell 2013+ or AMD Excavator 2015+).
///
/// Performance: ~2x throughput improvement over SimdEngine128 for compute-bound workloads.
#[derive(Clone, Copy)]
/// AVX2 SIMD backend implementing `SimdEngine` for 256‑bit vectors.
pub struct SimdEngine256;

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)]
impl SimdEngine for SimdEngine256 {
    const WIDTH_8: usize = 32; // 256 bits ÷ 8 bits = 32 lanes
    const WIDTH_16: usize = 16; // 256 bits ÷ 16 bits = 16 lanes

    type Vec8 = simd_arch::__m256i;
    type Vec16 = simd_arch::__m256i;

    // ===== Creation and Initialization =====

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn setzero_vec8() -> Self::Vec8 {
        simd_arch::_mm256_setzero_si256()
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn setzero_epi8() -> Self::Vec8 {
        simd_arch::_mm256_setzero_si256()
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn set1_epi8(a: i8) -> Self::Vec8 {
        simd_arch::_mm256_set1_epi8(a)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn set1_epi16(a: i16) -> Self::Vec16 {
        simd_arch::_mm256_set1_epi16(a)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn set1_epi32(a: i32) -> Self::Vec8 {
        simd_arch::_mm256_set1_epi32(a)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn setzero_epi16() -> Self::Vec16 {
        simd_arch::_mm256_setzero_si256()
    }

    // ===== Extract and Movemask Operations =====

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn extract_epi8(a: Self::Vec8, imm8: i32) -> i8 {
        // Extract the 128-bit lane, then extract the 8-bit element using store-and-load
        if imm8 < 16 {
            let mut tmp = [0i8; 16];
            let v128 = simd_arch::_mm256_extracti128_si256(a, 0); // Extract lower 128-bit lane
            std::arch::x86_64::_mm_storeu_si128(tmp.as_mut_ptr() as *mut _, v128);
            *tmp.get_unchecked(imm8 as usize)
        } else {
            let mut tmp = [0i8; 16];
            let v128 = simd_arch::_mm256_extracti128_si256(a, 1); // Extract upper 128-bit lane
            std::arch::x86_64::_mm_storeu_si128(tmp.as_mut_ptr() as *mut _, v128);
            *tmp.get_unchecked((imm8 - 16) as usize)
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn extract_epi16(a: Self::Vec16, imm8: i32) -> i16 {
        // Extract the 128-bit lane, then extract the 16-bit element using store-and-load
        if imm8 < 8 {
            let mut tmp = [0i16; 8];
            let v128 = simd_arch::_mm256_extracti128_si256(a, 0); // Extract lower 128-bit lane
            std::arch::x86_64::_mm_storeu_si128(tmp.as_mut_ptr() as *mut _, v128);
            *tmp.get_unchecked(imm8 as usize)
        } else {
            let mut tmp = [0i16; 8];
            let v128 = simd_arch::_mm256_extracti128_si256(a, 1); // Extract upper 128-bit lane
            std::arch::x86_64::_mm_storeu_si128(tmp.as_mut_ptr() as *mut _, v128);
            *tmp.get_unchecked((imm8 - 8) as usize)
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn movemask_epi8(a: Self::Vec8) -> i32 {
        simd_arch::_mm256_movemask_epi8(a)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn movemask_epi16(a: Self::Vec16) -> i32 {
        // AVX2 does not have _mm256_movemask_epi16 or _mm_movemask_epi16.
        // We can use _mm_movemask_epi8 on the two 128-bit halves and then combine the results.
        let lower_half = simd_arch::_mm256_extracti128_si256(a, 0); // Extract lower 128-bit lane
        let upper_half = simd_arch::_mm256_extracti128_si256(a, 1); // Extract upper 128-bit lane

        // Get 16 bits (1 per byte) from each 128-bit lane
        let mask_lower_bytes = simd_arch::_mm_movemask_epi8(lower_half);
        let mask_upper_bytes = simd_arch::_mm_movemask_epi8(upper_half);

        // Extract bits for the sign bit (most significant bit) of each 16-bit value
        // We need to keep only the even positions (0, 2, 4, 6, 8, 10, 12, 14)
        let mask_lower = (mask_lower_bytes & 0x55) | ((mask_lower_bytes >> 1) & 0x55);
        let mask_upper = (mask_upper_bytes & 0x55) | ((mask_upper_bytes >> 1) & 0x55);

        // Combine the two 8-bit masks (from the 128-bit halves) into a single 16-bit mask
        (mask_upper << 8) | mask_lower
    }

    // ===== Variable Shift Operations =====

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn slli_bytes(a: Self::Vec8, num_bytes: i32) -> Self::Vec8 {
        mm256_match_shift_bytes!(a, num_bytes, _mm256_slli_si256)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn slli_bytes_16(a: Self::Vec16, num_bytes: i32) -> Self::Vec16 {
        Self::slli_bytes(a, num_bytes)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn srli_bytes(a: Self::Vec8, num_bytes: i32) -> Self::Vec8 {
        mm256_match_shift_bytes!(a, num_bytes, _mm256_srli_si256)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn srli_bytes_16(a: Self::Vec16, num_bytes: i32) -> Self::Vec16 {
        Self::srli_bytes(a, num_bytes)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn alignr_bytes(a: Self::Vec8, b: Self::Vec8, num_bytes: i32) -> Self::Vec8 {
        mm256_alignr_bytes_match!(a, b, num_bytes)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn alignr_bytes_16(a: Self::Vec16, b: Self::Vec16, num_bytes: i32) -> Self::Vec16 {
        mm256_alignr_bytes_match!(a, b, num_bytes)
    }

    // ===== 8-bit Integer Arithmetic =====

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn add_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm256_add_epi8(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn sub_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm256_sub_epi8(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn subs_epu8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm256_subs_epu8(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn subs_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm256_subs_epi8(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn adds_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm256_adds_epi8(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn adds_epu8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm256_adds_epu8(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn max_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm256_max_epi8(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn max_epu8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm256_max_epu8(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn min_epu8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm256_min_epu8(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn min_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm256_min_epi8(a, b)
    }

    // ===== 16-bit Integer Arithmetic =====

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn add_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        simd_arch::_mm256_add_epi16(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn sub_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        simd_arch::_mm256_sub_epi16(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn adds_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        simd_arch::_mm256_adds_epi16(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn subs_epu16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        simd_arch::_mm256_subs_epu16(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn subs_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        simd_arch::_mm256_subs_epi16(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn max_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        simd_arch::_mm256_max_epi16(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn min_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        simd_arch::_mm256_min_epi16(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn max_epu16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        simd_arch::_mm256_max_epu16(a, b)
    }

    // ===== Comparison Operations =====

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn cmpeq_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm256_cmpeq_epi8(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn cmpgt_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm256_cmpgt_epi8(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn cmpgt_epu8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        // AVX2 has no native unsigned comparison; use XOR trick to flip sign bit
        // a >_u b ⟺ (a XOR 0x80) >_s (b XOR 0x80)
        let sign_bit = simd_arch::_mm256_set1_epi8(0x80u8 as i8);
        let a_signed = simd_arch::_mm256_xor_si256(a, sign_bit);
        let b_signed = simd_arch::_mm256_xor_si256(b, sign_bit);
        simd_arch::_mm256_cmpgt_epi8(a_signed, b_signed)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn cmpge_epu8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        // a >=_u b ⟺ max(a, b) == a
        let max_val = simd_arch::_mm256_max_epu8(a, b);
        simd_arch::_mm256_cmpeq_epi8(max_val, a)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn cmpeq_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        simd_arch::_mm256_cmpeq_epi16(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn cmpgt_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        simd_arch::_mm256_cmpgt_epi16(a, b)
    }

    // ===== Blend/Select Operations =====

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn blendv_epi8(a: Self::Vec8, b: Self::Vec8, mask: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm256_blendv_epi8(a, b, mask)
    }

    // ===== Bitwise Operations =====

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn and_si128(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm256_and_si256(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn or_si128(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm256_or_si256(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn xor_si128(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm256_xor_si256(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn andnot_si128(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm256_andnot_si256(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn shuffle_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm256_shuffle_epi8(a, b)
    }

    // ===== Unpack Operations =====

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn unpacklo_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm256_unpacklo_epi8(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn unpackhi_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm256_unpackhi_epi8(a, b)
    }

    // ===== Shift Operations =====

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn slli_epi16(a: Self::Vec16, imm8: i32) -> Self::Vec16 {
        // For runtime variable shift amounts, we need to use a match or lookup table
        match imm8 {
            0 => a,
            1 => simd_arch::_mm256_slli_epi16(a, 1),
            2 => simd_arch::_mm256_slli_epi16(a, 2),
            3 => simd_arch::_mm256_slli_epi16(a, 3),
            4 => simd_arch::_mm256_slli_epi16(a, 4),
            5 => simd_arch::_mm256_slli_epi16(a, 5),
            6 => simd_arch::_mm256_slli_epi16(a, 6),
            7 => simd_arch::_mm256_slli_epi16(a, 7),
            8 => simd_arch::_mm256_slli_epi16(a, 8),
            _ => simd_arch::_mm256_slli_epi16(a, 0),
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn srli_si128_fixed(a: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm256_srli_si256(a, 2)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn alignr_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm256_alignr_epi8(a, b, 1)
    }

    // ===== Memory Operations =====

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn load_si128(p: *const Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm256_load_si256(p)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn load_si128_16(p: *const Self::Vec16) -> Self::Vec16 {
        simd_arch::_mm256_load_si256(p)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn store_si128(p: *mut Self::Vec8, a: Self::Vec8) {
        simd_arch::_mm256_store_si256(p, a)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn store_si128_16(p: *mut Self::Vec16, a: Self::Vec16) {
        simd_arch::_mm256_store_si256(p, a)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn loadu_si128(p: *const Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm256_loadu_si256(p)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn loadu_si128_16(p: *const Self::Vec16) -> Self::Vec16 {
        simd_arch::_mm256_loadu_si256(p)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn storeu_si128(p: *mut Self::Vec8, a: Self::Vec8) {
        simd_arch::_mm256_storeu_si256(p, a)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn storeu_si128_16(p: *mut Self::Vec16, a: Self::Vec16) {
        simd_arch::_mm256_storeu_si256(p, a)
    }
}
