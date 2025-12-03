//! 512‑bit SIMD engine (AVX‑512BW on x86_64)
//!
//! This module implements the widest `SimdEngine` backend, using AVX‑512BW
//! intrinsics on supported x86_64 CPUs. It provides 64 lanes for 8‑bit and 32
//! lanes for 16‑bit operations and maps directly to `_mm512_*` intrinsics.
//!
//! Notes
//! - Requires the `avx512` Cargo feature and CPU support for AVX‑512BW.
//! - Variable byte shifts use generic match‑tables from portable_intrinsics to
//!   accept a runtime `num_bytes` while satisfying immediate‑operand restrictions.
//! - `movemask_epi8`/`movemask_epi16` use native AVX‑512 intrinsics.
//!
//! Safety
//! - All functions are `unsafe` and annotated with the appropriate
//!   `#[target_feature]`. Callers must ensure the selected engine matches CPU
//!   features (the crate's runtime dispatch enforces this in normal use).
//! - Pointers passed to loads/stores must be valid and, for aligned variants,
//!   appropriately aligned.

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
use super::SimdEngine;
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
use super::types::simd_arch;

// ===== Generic match-immediate macros from portable_intrinsics =====
// We use the generic match_shift_immediate! and match_alignr_immediate_or! macros
// instead of defining engine-specific versions. This eliminates ~75 lines of
// duplication per engine while maintaining zero runtime overhead.
// Note: AVX-512 uses _mm512_bslli_epi128 / _mm512_bsrli_epi128 for byte shifts
// within each 128-bit lane (not across the whole register).
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
use crate::{match_alignr_immediate_or, match_shift_immediate};

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
/// AVX‑512BW SIMD backend implementing `SimdEngine` for 512‑bit vectors.
#[derive(Clone, Copy)]
pub struct SimdEngine512;

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[allow(unsafe_op_in_unsafe_fn)]
impl SimdEngine for SimdEngine512 {
    const WIDTH_8: usize = 64; // 64 lanes for 8-bit operations (4x SSE)
    const WIDTH_16: usize = 32; // 32 lanes for 16-bit operations (4x SSE)

    type Vec8 = simd_arch::__m512i;
    type Vec16 = simd_arch::__m512i;

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn setzero_vec8() -> Self::Vec8 {
        simd_arch::_mm512_setzero_si512()
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn setzero_epi8() -> Self::Vec8 {
        simd_arch::_mm512_setzero_si512()
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn set1_epi8(a: i8) -> Self::Vec8 {
        simd_arch::_mm512_set1_epi8(a)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn set1_epi16(a: i16) -> Self::Vec16 {
        simd_arch::_mm512_set1_epi16(a)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn set1_epi32(a: i32) -> Self::Vec8 {
        simd_arch::_mm512_set1_epi32(a)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn setzero_epi16() -> Self::Vec16 {
        simd_arch::_mm512_setzero_si512()
    }

    // ===== Extract and Movemask Operations (Specific to x86) =====
    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn extract_epi8(a: Self::Vec8, imm8: i32) -> i8 {
        // Extract the 128-bit lane, then extract the 8-bit element using store-and-load
        if imm8 < 16 {
            let mut tmp = [0i8; 16];
            let v128 = simd_arch::_mm512_extracti32x4_epi32(a, 0);
            std::arch::x86_64::_mm_storeu_si128(tmp.as_mut_ptr() as *mut _, v128);
            *tmp.get_unchecked(imm8 as usize)
        } else if imm8 < 32 {
            let mut tmp = [0i8; 16];
            let v128 = simd_arch::_mm512_extracti32x4_epi32(a, 1);
            std::arch::x86_64::_mm_storeu_si128(tmp.as_mut_ptr() as *mut _, v128);
            *tmp.get_unchecked((imm8 - 16) as usize)
        } else if imm8 < 48 {
            let mut tmp = [0i8; 16];
            let v128 = simd_arch::_mm512_extracti32x4_epi32(a, 2);
            std::arch::x86_64::_mm_storeu_si128(tmp.as_mut_ptr() as *mut _, v128);
            *tmp.get_unchecked((imm8 - 32) as usize)
        } else {
            // imm8 < 64
            let mut tmp = [0i8; 16];
            let v128 = simd_arch::_mm512_extracti32x4_epi32(a, 3);
            std::arch::x86_64::_mm_storeu_si128(tmp.as_mut_ptr() as *mut _, v128);
            *tmp.get_unchecked((imm8 - 48) as usize)
        }
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn extract_epi16(a: Self::Vec16, imm8: i32) -> i16 {
        // Extract the 128-bit lane, then extract the 16-bit element using store-and-load
        if imm8 < 8 {
            let mut tmp = [0i16; 8];
            let v128 = simd_arch::_mm512_extracti32x4_epi32(a, 0);
            std::arch::x86_64::_mm_storeu_si128(tmp.as_mut_ptr() as *mut _, v128);
            *tmp.get_unchecked(imm8 as usize)
        } else if imm8 < 16 {
            let mut tmp = [0i16; 8];
            let v128 = simd_arch::_mm512_extracti32x4_epi32(a, 1);
            std::arch::x86_64::_mm_storeu_si128(tmp.as_mut_ptr() as *mut _, v128);
            *tmp.get_unchecked((imm8 - 8) as usize)
        } else if imm8 < 24 {
            let mut tmp = [0i16; 8];
            let v128 = simd_arch::_mm512_extracti32x4_epi32(a, 2);
            std::arch::x86_64::_mm_storeu_si128(tmp.as_mut_ptr() as *mut _, v128);
            *tmp.get_unchecked((imm8 - 16) as usize)
        } else {
            // imm8 < 32
            let mut tmp = [0i16; 8];
            let v128 = simd_arch::_mm512_extracti32x4_epi32(a, 3);
            std::arch::x86_64::_mm_storeu_si128(tmp.as_mut_ptr() as *mut _, v128);
            *tmp.get_unchecked((imm8 - 24) as usize)
        }
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn movemask_epi8(a: Self::Vec8) -> i32 {
        // AVX-512 doesn't have movemask_epi8, use movepi8_mask instead
        // This returns a 64-bit mask, we return the low 32 bits for API compatibility
        simd_arch::_mm512_movepi8_mask(a) as i32
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn movemask_epi16(a: Self::Vec16) -> i32 {
        // AVX-512 doesn't have movemask_epi16, use movepi16_mask instead
        simd_arch::_mm512_movepi16_mask(a) as i32
    }

    // ===== Variable Shift Operations =====

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn slli_bytes(a: Self::Vec8, num_bytes: i32) -> Self::Vec8 {
        // AVX-512 uses _mm512_bslli_epi128 for byte shifts within each 128-bit lane
        match_shift_immediate!(
            a,
            num_bytes,
            simd_arch::_mm512_bslli_epi128,
            simd_arch::_mm512_setzero_si512()
        )
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn slli_bytes_16(a: Self::Vec16, num_bytes: i32) -> Self::Vec16 {
        Self::slli_bytes(a, num_bytes)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn srli_bytes(a: Self::Vec8, num_bytes: i32) -> Self::Vec8 {
        // AVX-512 uses _mm512_bsrli_epi128 for byte shifts within each 128-bit lane
        match_shift_immediate!(
            a,
            num_bytes,
            simd_arch::_mm512_bsrli_epi128,
            simd_arch::_mm512_setzero_si512()
        )
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn srli_bytes_16(a: Self::Vec16, num_bytes: i32) -> Self::Vec16 {
        Self::srli_bytes(a, num_bytes)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn alignr_bytes(a: Self::Vec8, b: Self::Vec8, num_bytes: i32) -> Self::Vec8 {
        // _mm512_alignr_epi8 requires const immediate, use match table
        // AVX-512 returns $a for num_bytes >= 16 (different from AVX2)
        match_alignr_immediate_or!(a, b, num_bytes, simd_arch::_mm512_alignr_epi8, a)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn alignr_bytes_16(a: Self::Vec16, b: Self::Vec16, num_bytes: i32) -> Self::Vec16 {
        // AVX-512 returns $a for num_bytes >= 16 (different from AVX2)
        match_alignr_immediate_or!(a, b, num_bytes, simd_arch::_mm512_alignr_epi8, a)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn add_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm512_add_epi8(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn sub_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm512_sub_epi8(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn subs_epu8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm512_subs_epu8(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn subs_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm512_subs_epi8(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn adds_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm512_adds_epi8(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn adds_epu8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm512_adds_epu8(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn max_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm512_max_epi8(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn max_epu8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm512_max_epu8(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn min_epu8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm512_min_epu8(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn min_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm512_min_epi8(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn add_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        simd_arch::_mm512_add_epi16(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn sub_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        simd_arch::_mm512_sub_epi16(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn adds_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        simd_arch::_mm512_adds_epi16(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn subs_epu16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        simd_arch::_mm512_subs_epu16(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn subs_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        simd_arch::_mm512_subs_epi16(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn max_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        simd_arch::_mm512_max_epi16(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn min_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        simd_arch::_mm512_min_epi16(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn max_epu16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        simd_arch::_mm512_max_epu16(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn cmpeq_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        let mask = simd_arch::_mm512_cmpeq_epi8_mask(a, b);
        simd_arch::_mm512_maskz_set1_epi8(mask, -1i8)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn cmpgt_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        let mask = simd_arch::_mm512_cmpgt_epi8_mask(a, b);
        simd_arch::_mm512_maskz_set1_epi8(mask, -1i8)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn cmpgt_epu8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        let mask = simd_arch::_mm512_cmpgt_epu8_mask(a, b);
        simd_arch::_mm512_maskz_set1_epi8(mask, -1i8)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn cmpge_epu8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        let mask = simd_arch::_mm512_cmpge_epu8_mask(a, b);
        simd_arch::_mm512_maskz_set1_epi8(mask, -1i8)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn cmpeq_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        let mask = simd_arch::_mm512_cmpeq_epi16_mask(a, b);
        simd_arch::_mm512_maskz_set1_epi16(mask, -1i16)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn cmpgt_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        let mask = simd_arch::_mm512_cmpgt_epi16_mask(a, b);
        simd_arch::_mm512_maskz_set1_epi16(mask, -1i16)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn blendv_epi8(a: Self::Vec8, b: Self::Vec8, mask: Self::Vec8) -> Self::Vec8 {
        // Convert mask to kmask (AVX-512 uses mask registers)
        let kmask = simd_arch::_mm512_movepi8_mask(mask);
        simd_arch::_mm512_mask_blend_epi8(kmask, a, b)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn and_si128(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm512_and_si512(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn or_si128(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm512_or_si512(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn xor_si128(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm512_xor_si512(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn andnot_si128(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm512_andnot_si512(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn shuffle_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm512_shuffle_epi8(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn slli_epi16(a: Self::Vec16, imm8: i32) -> Self::Vec16 {
        // For runtime variable shift amounts, we need to use a match or lookup table
        match imm8 {
            0 => a,
            1 => simd_arch::_mm512_slli_epi16(a, 1),
            2 => simd_arch::_mm512_slli_epi16(a, 2),
            3 => simd_arch::_mm512_slli_epi16(a, 3),
            4 => simd_arch::_mm512_slli_epi16(a, 4),
            5 => simd_arch::_mm512_slli_epi16(a, 5),
            6 => simd_arch::_mm512_slli_epi16(a, 6),
            7 => simd_arch::_mm512_slli_epi16(a, 7),
            8 => simd_arch::_mm512_slli_epi16(a, 8),
            _ => simd_arch::_mm512_slli_epi16(a, 0),
        }
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn srli_si128_fixed(a: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm512_bsrli_epi128(a, 2)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn alignr_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm512_alignr_epi8(a, b, 1)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn load_si128(p: *const Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm512_load_si512(p)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn load_si128_16(p: *const Self::Vec16) -> Self::Vec16 {
        simd_arch::_mm512_load_si512(p)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn store_si128(p: *mut Self::Vec8, a: Self::Vec8) {
        simd_arch::_mm512_store_si512(p, a)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn store_si128_16(p: *mut Self::Vec16, a: Self::Vec16) {
        simd_arch::_mm512_store_si512(p, a)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn loadu_si128(p: *const Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm512_loadu_si512(p)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn loadu_si128_16(p: *const Self::Vec16) -> Self::Vec16 {
        simd_arch::_mm512_loadu_si512(p)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn storeu_si128(p: *mut Self::Vec8, a: Self::Vec8) {
        simd_arch::_mm512_storeu_si512(p, a)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn storeu_si128_16(p: *mut Self::Vec16, a: Self::Vec16) {
        simd_arch::_mm512_storeu_si512(p, a)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn unpacklo_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm512_unpacklo_epi8(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn unpackhi_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm512_unpackhi_epi8(a, b)
    }
}
