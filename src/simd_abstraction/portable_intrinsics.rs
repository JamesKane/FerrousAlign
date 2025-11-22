//! Portable helpers that mirror a subset of x86 SSE2 intrinsics
//!
//! This module provides small, well‑defined shims used by the 128‑bit engine to
//! implement operations that require an immediate on some ISAs (e.g. byte‑wise
//! variable shifts and `alignr`) using a runtime `num_bytes` parameter. They are
//! intentionally minimal and exist to keep the higher‑level engines simple and
//! uniform.
//!
//! Design notes
//! - All functions here are `unsafe` for the same reasons as the underlying
//!   intrinsics (pointer validity, CPU features).
//! - The variable‑byte shift/align helpers clamp or define behavior for
//!   out‑of‑range `num_bytes` consistently with x86 semantics (shifts by 16 or
//!   more bytes yield zeros; `alignr` selects a window of 16 bytes, clamping to
//!   valid ranges).
//! - On aarch64 these map to equivalent NEON operations or straightforward
//!   scalar fallbacks where needed.
//!
//! These helpers are used by `engine128` and also serve as reference behavior
//! for wider engines when constructing macro‑based match tables.

use super::types::{__m128i, simd_arch};

//
// Start of the portable intrinsic implementations
//

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn _mm_setzero_si128() -> __m128i {
    #[cfg(target_arch = "x86_64")]
    {
        unsafe { simd_arch::_mm_setzero_si128() }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { __m128i(simd_arch::vdupq_n_u8(0)) }
    }
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn _mm_add_epi16(a: __m128i, b: __m128i) -> __m128i {
    #[cfg(target_arch = "x86_64")]
    {
        unsafe { simd_arch::_mm_add_epi16(a, b) }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { __m128i::from_s16(simd_arch::vaddq_s16(a.as_s16(), b.as_s16())) }
    }
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn _mm_set1_epi8(a: i8) -> __m128i {
    #[cfg(target_arch = "x86_64")]
    {
        unsafe { simd_arch::_mm_set1_epi8(a) }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { __m128i(simd_arch::vdupq_n_u8(a as u8)) }
    }
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn _mm_set1_epi16(a: i16) -> __m128i {
    #[cfg(target_arch = "x86_64")]
    {
        unsafe { simd_arch::_mm_set1_epi16(a) }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { __m128i::from_s16(simd_arch::vdupq_n_s16(a)) }
    }
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn _mm_set1_epi32(a: i32) -> __m128i {
    #[cfg(target_arch = "x86_64")]
    {
        unsafe { simd_arch::_mm_set1_epi32(a) }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { __m128i::from_s32(simd_arch::vdupq_n_s32(a)) }
    }
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn _mm_subs_epu8(a: __m128i, b: __m128i) -> __m128i {
    #[cfg(target_arch = "x86_64")]
    {
        unsafe { simd_arch::_mm_subs_epu8(a, b) }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { __m128i(simd_arch::vqsubq_u8(a.0, b.0)) }
    }
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn _mm_subs_epu16(a: __m128i, b: __m128i) -> __m128i {
    #[cfg(target_arch = "x86_64")]
    {
        unsafe { simd_arch::_mm_subs_epu16(a, b) }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { __m128i::from_u16(simd_arch::vqsubq_u16(a.as_u16(), b.as_u16())) }
    }
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn _mm_max_epu8(a: __m128i, b: __m128i) -> __m128i {
    #[cfg(target_arch = "x86_64")]
    {
        unsafe { simd_arch::_mm_max_epu8(a, b) }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { __m128i(simd_arch::vmaxq_u8(a.0, b.0)) }
    }
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn _mm_cmpeq_epi8(a: __m128i, b: __m128i) -> __m128i {
    #[cfg(target_arch = "x86_64")]
    {
        unsafe { simd_arch::_mm_cmpeq_epi8(a, b) }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { __m128i(simd_arch::vceqq_u8(a.0, b.0)) }
    }
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn _mm_load_si128(p: *const __m128i) -> __m128i {
    #[cfg(target_arch = "x86_64")]
    {
        unsafe { simd_arch::_mm_load_si128(p) }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { *(p as *const __m128i) }
    }
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn _mm_load_si128_16(p: *const __m128i) -> __m128i {
    _mm_load_si128(p)
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn _mm_store_si128(p: *mut __m128i, a: __m128i) {
    #[cfg(target_arch = "x86_64")]
    {
        unsafe { simd_arch::_mm_store_si128(p, a) }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            *(p as *mut __m128i) = a;
        }
    }
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn _mm_store_si128_16(p: *mut __m128i, a: __m128i) {
    _mm_store_si128(p, a)
}

// ===== Additional intrinsics for Smith-Waterman =====

/// Load 128-bit value from unaligned memory for 16-bit vectors
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn _mm_loadu_si128_16(p: *const __m128i) -> __m128i {
    _mm_loadu_si128(p)
}

/// Store 128-bit value to unaligned memory for 16-bit vectors
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn _mm_storeu_si128_16(p: *mut __m128i, a: __m128i) {
    _mm_storeu_si128(p, a)
}

/// Shift 128-bit vector left by `num_bytes` (variable)
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn _mm_slli_si128_var(a: __m128i, num_bytes: i32) -> __m128i {
    #[cfg(target_arch = "x86_64")]
    {
        // _mm_slli_si128 requires imm8 to be a compile-time constant.
        // For a runtime num_bytes, we must use a workaround like storing to memory
        // or a series of conditional branches for common values.
        // Given typical usage, num_bytes is often small (1, 2, 4, 8).
        match num_bytes {
            0 => a,
            1 => simd_arch::_mm_slli_si128(a, 1),
            2 => simd_arch::_mm_slli_si128(a, 2),
            3 => simd_arch::_mm_slli_si128(a, 3),
            4 => simd_arch::_mm_slli_si128(a, 4),
            5 => simd_arch::_mm_slli_si128(a, 5),
            6 => simd_arch::_mm_slli_si128(a, 6),
            7 => simd_arch::_mm_slli_si128(a, 7),
            8 => simd_arch::_mm_slli_si128(a, 8),
            9 => simd_arch::_mm_slli_si128(a, 9),
            10 => simd_arch::_mm_slli_si128(a, 10),
            11 => simd_arch::_mm_slli_si128(a, 11),
            12 => simd_arch::_mm_slli_si128(a, 12),
            13 => simd_arch::_mm_slli_si128(a, 13),
            14 => simd_arch::_mm_slli_si128(a, 14),
            15 => simd_arch::_mm_slli_si128(a, 15),
            _ => _mm_setzero_si128(), // Shift by 16 or more bytes results in zero
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if num_bytes >= 16 {
            _mm_setzero_si128()
        } else {
            // vextq_u8 extracts a sub-vector from two concatenated vectors.
            // For left shift, we concatenate `a` with a zero vector and extract.
            __m128i(simd_arch::vextq_u8(
                simd_arch::vdupq_n_u8(0),
                a.0,
                16 - num_bytes as i32,
            ))
        }
    }
}

/// Shift 128-bit vector right by `num_bytes` (variable)
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn _mm_srli_si128_var(a: __m128i, num_bytes: i32) -> __m128i {
    #[cfg(target_arch = "x86_64")]
    {
        // _mm_srli_si128 requires imm8 to be a compile-time constant.
        // Similar workaround as slli_si128_var
        match num_bytes {
            0 => a,
            1 => simd_arch::_mm_srli_si128(a, 1),
            2 => simd_arch::_mm_srli_si128(a, 2),
            3 => simd_arch::_mm_srli_si128(a, 3),
            4 => simd_arch::_mm_srli_si128(a, 4),
            5 => simd_arch::_mm_srli_si128(a, 5),
            6 => simd_arch::_mm_srli_si128(a, 6),
            7 => simd_arch::_mm_srli_si128(a, 7),
            8 => simd_arch::_mm_srli_si128(a, 8),
            9 => simd_arch::_mm_srli_si128(a, 9),
            10 => simd_arch::_mm_srli_si128(a, 10),
            11 => simd_arch::_mm_srli_si128(a, 11),
            12 => simd_arch::_mm_srli_si128(a, 12),
            13 => simd_arch::_mm_srli_si128(a, 13),
            14 => simd_arch::_mm_srli_si128(a, 14),
            15 => simd_arch::_mm_srli_si128(a, 15),
            _ => _mm_setzero_si128(), // Shift by 16 or more bytes results in zero
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if num_bytes >= 16 {
            _mm_setzero_si128()
        } else {
            // vextq_u8 extracts a sub-vector from two concatenated vectors.
            // For right shift, we concatenate a zero vector with `a` and extract.
            __m128i(simd_arch::vextq_u8(
                a.0,
                simd_arch::vdupq_n_u8(0),
                num_bytes as i32,
            ))
        }
    }
}

/// Concatenate and shift right by `num_bytes` (variable)
/// `b` is the most significant 128-bit part, `a` is the least significant part.
/// Result contains the `num_bytes` right shifted concatenation of `b` and `a`.
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn _mm_alignr_epi8_var(a: __m128i, b: __m128i, num_bytes: i32) -> __m128i {
    #[cfg(target_arch = "x86_64")]
    {
        // _mm_alignr_epi8 requires imm8 to be a compile-time constant.
        // Similar workaround as slli/srli_si128_var
        match num_bytes {
            0 => a, // Effectively just 'a'
            1 => simd_arch::_mm_alignr_epi8(a, b, 1),
            2 => simd_arch::_mm_alignr_epi8(a, b, 2),
            3 => simd_arch::_mm_alignr_epi8(a, b, 3),
            4 => simd_arch::_mm_alignr_epi8(a, b, 4),
            5 => simd_arch::_mm_alignr_epi8(a, b, 5),
            6 => simd_arch::_mm_alignr_epi8(a, b, 6),
            7 => simd_arch::_mm_alignr_epi8(a, b, 7),
            8 => simd_arch::_mm_alignr_epi8(a, b, 8),
            9 => simd_arch::_mm_alignr_epi8(a, b, 9),
            10 => simd_arch::_mm_alignr_epi8(a, b, 10),
            11 => simd_arch::_mm_alignr_epi8(a, b, 11),
            12 => simd_arch::_mm_alignr_epi8(a, b, 12),
            13 => simd_arch::_mm_alignr_epi8(a, b, 13),
            14 => simd_arch::_mm_alignr_epi8(a, b, 14),
            15 => simd_arch::_mm_alignr_epi8(a, b, 15),
            _ => _mm_setzero_si128(), // Shift by 16 or more bytes results in zero
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if num_bytes >= 16 {
            // If shift is 16 or more, it means we're effectively shifting 'b' past its own length
            // or past the combined length of 'b' and 'a'. So, result is zero.
            _mm_setzero_si128()
        } else {
            // NEON's vextq_u8 concatenates the second argument (a) and then the first (b)
            // and extracts `num_bytes` from the right.
            // Example: vextq_u8(b, a, N) extracts bytes [N .. N+15] from [a | b] (where a is LSB, b is MSB)
            __m128i(simd_arch::vextq_u8(a.0, b.0, num_bytes as i32))
        }
    }
}
// ===== Additional intrinsics for Smith-Waterman =====

// 8-bit integer operations
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn _mm_add_epi8(a: __m128i, b: __m128i) -> __m128i {
    #[cfg(target_arch = "x86_64")]
    {
        simd_arch::_mm_add_epi8(a, b)
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { __m128i::from_s8(simd_arch::vaddq_s8(a.as_s8(), b.as_s8())) }
    }
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn _mm_sub_epi8(a: __m128i, b: __m128i) -> __m128i {
    #[cfg(target_arch = "x86_64")]
    {
        simd_arch::_mm_sub_epi8(a, b)
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { __m128i::from_s8(simd_arch::vsubq_s8(a.as_s8(), b.as_s8())) }
    }
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn _mm_max_epi8(a: __m128i, b: __m128i) -> __m128i {
    #[cfg(target_arch = "x86_64")]
    {
        simd_arch::_mm_max_epi8(a, b)
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { __m128i::from_s8(simd_arch::vmaxq_s8(a.as_s8(), b.as_s8())) }
    }
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn _mm_blendv_epi8(a: __m128i, b: __m128i, mask: __m128i) -> __m128i {
    #[cfg(target_arch = "x86_64")]
    {
        simd_arch::_mm_blendv_epi8(a, b, mask)
    }
    #[cfg(target_arch = "aarch64")]
    {
        // NEON blend: if mask MSB is 1, select b, otherwise a
        // vbslq_u8 does: if mask bit is 1, select first arg, else second
        // So we need to pass b first, then a
        unsafe { __m128i(simd_arch::vbslq_u8(mask.0, b.0, a.0)) }
    }
}

// 16-bit integer operations
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn _mm_sub_epi16(a: __m128i, b: __m128i) -> __m128i {
    #[cfg(target_arch = "x86_64")]
    {
        simd_arch::_mm_sub_epi16(a, b)
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { __m128i::from_s16(simd_arch::vsubq_s16(a.as_s16(), b.as_s16())) }
    }
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn _mm_max_epi16(a: __m128i, b: __m128i) -> __m128i {
    #[cfg(target_arch = "x86_64")]
    {
        simd_arch::_mm_max_epi16(a, b)
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { __m128i::from_s16(simd_arch::vmaxq_s16(a.as_s16(), b.as_s16())) }
    }
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn _mm_cmpeq_epi16(a: __m128i, b: __m128i) -> __m128i {
    #[cfg(target_arch = "x86_64")]
    {
        simd_arch::_mm_cmpeq_epi16(a, b)
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { __m128i::from_u16(simd_arch::vceqq_s16(a.as_s16(), b.as_s16())) }
    }
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn _mm_max_epu16(a: __m128i, b: __m128i) -> __m128i {
    #[cfg(target_arch = "x86_64")]
    {
        simd_arch::_mm_max_epu16(a, b)
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { __m128i::from_u16(simd_arch::vmaxq_u16(a.as_u16(), b.as_u16())) }
    }
}



// ===== Additional intrinsics for Smith-Waterman optimization =====

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn _mm_min_epu8(a: __m128i, b: __m128i) -> __m128i {
    #[cfg(target_arch = "x86_64")]
    {
        simd_arch::_mm_min_epu8(a, b)
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { __m128i(simd_arch::vminq_u8(a.0, b.0)) }
    }
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn _mm_cmpgt_epi8(a: __m128i, b: __m128i) -> __m128i {
    #[cfg(target_arch = "x86_64")]
    {
        simd_arch::_mm_cmpgt_epi8(a, b)
    }
    #[cfg(target_arch = "aarch64")]
    {
        // vcgtq_s8 returns uint8x16_t (comparison mask), need to reinterpret as int8x16_t
        unsafe {
            let cmp_result = simd_arch::vcgtq_s8(a.as_s8(), b.as_s8());
            __m128i(cmp_result)
        }
    }
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn _mm_subs_epi8(a: __m128i, b: __m128i) -> __m128i {
    #[cfg(target_arch = "x86_64")]
    {
        simd_arch::_mm_subs_epi8(a, b)
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { __m128i::from_s8(simd_arch::vqsubq_s8(a.as_s8(), b.as_s8())) }
    }
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn _mm_adds_epi8(a: __m128i, b: __m128i) -> __m128i {
    #[cfg(target_arch = "x86_64")]
    {
        simd_arch::_mm_adds_epi8(a, b)
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { __m128i::from_s8(simd_arch::vqaddq_s8(a.as_s8(), b.as_s8())) }
    }
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn _mm_adds_epu8(a: __m128i, b: __m128i) -> __m128i {
    #[cfg(target_arch = "x86_64")]
    {
        simd_arch::_mm_adds_epu8(a, b)
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { __m128i(simd_arch::vqaddq_u8(a.0, b.0)) }
    }
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn _mm_adds_epi16(a: __m128i, b: __m128i) -> __m128i {
    #[cfg(target_arch = "x86_64")]
    {
        simd_arch::_mm_adds_epi16(a, b)
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { __m128i::from_s16(simd_arch::vqaddq_s16(a.as_s16(), b.as_s16())) }
    }
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn _mm_subs_epi16(a: __m128i, b: __m128i) -> __m128i {
    #[cfg(target_arch = "x86_64")]
    {
        simd_arch::_mm_subs_epi16(a, b)
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { __m128i::from_s16(simd_arch::vqsubq_s16(a.as_s16(), b.as_s16())) }
    }
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn _mm_min_epi16(a: __m128i, b: __m128i) -> __m128i {
    #[cfg(target_arch = "x86_64")]
    {
        simd_arch::_mm_min_epi16(a, b)
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { __m128i::from_s16(simd_arch::vminq_s16(a.as_s16(), b.as_s16())) }
    }
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn _mm_cmpgt_epi16(a: __m128i, b: __m128i) -> __m128i {
    #[cfg(target_arch = "x86_64")]
    {
        simd_arch::_mm_cmpgt_epi16(a, b)
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { __m128i::from_u16(simd_arch::vcgtq_s16(a.as_s16(), b.as_s16())) }
    }
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn _mm_or_si128(a: __m128i, b: __m128i) -> __m128i {
    #[cfg(target_arch = "x86_64")]
    {
        simd_arch::_mm_or_si128(a, b)
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { __m128i(simd_arch::vorrq_u8(a.0, b.0)) }
    }
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn _mm_andnot_si128(a: __m128i, b: __m128i) -> __m128i {
    #[cfg(target_arch = "x86_64")]
    {
        simd_arch::_mm_andnot_si128(a, b)
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { __m128i(simd_arch::vbicq_u8(b.0, a.0)) }
    }
}

// Note: For x86 intrinsics, shift amounts must be compile-time constants.
// We provide wrapper macros for variable shift amounts, but the trait methods
// use fixed values that match the C++ bwa-mem2 implementation.







// Note: For x86 intrinsics, shift amounts must be compile-time constants.
// We provide wrapper macros for variable shift amounts, but the trait methods
// use fixed values that match the C++ bwa-mem2 implementation.

#[macro_export]
macro_rules! mm_slli_epi16 {
    ($a:expr, $imm8:expr) => {{
        #[cfg(target_arch = "x86_64")]
        unsafe {
            simd_arch::_mm_slli_epi16($a, $imm8)
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            __m128i::from_s16(simd_arch::vshlq_n_s16($a.as_s16(), $imm8))
        }
    }};
}

#[macro_export]
macro_rules! mm_srli_si128 {
    ($a:expr, $imm8:expr) => {{
        #[cfg(target_arch = "x86_64")]
        unsafe {
            simd_arch::_mm_srli_si128($a, $imm8)
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            if $imm8 >= 16 {
                _mm_setzero_si128()
            } else {
                __m128i(simd_arch::vextq_u8($a.0, simd_arch::vdupq_n_u8(0), $imm8))
            }
        }
    }};
}

#[macro_export]
macro_rules! mm_alignr_epi8 {
    ($a:expr, $b:expr, $imm8:expr) => {{
        #[cfg(target_arch = "x86_64")]
        unsafe {
            simd_arch::_mm_alignr_epi8($a, $b, $imm8)
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            if $imm8 >= 16 {
                mm_srli_si128!($a, $imm8 - 16)
            } else {
                __m128i(simd_arch::vextq_u8($b.0, $a.0, $imm8))
            }
        }
    }};
}

/// Bitwise AND of two 128-bit vectors
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn _mm_and_si128(a: __m128i, b: __m128i) -> __m128i {
    #[cfg(target_arch = "x86_64")]
    {
        simd_arch::_mm_and_si128(a, b)
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { __m128i(simd_arch::vandq_u8(a.0, b.0)) }
    }
}

/// Load 128-bit value from unaligned memory
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn _mm_loadu_si128(p: *const __m128i) -> __m128i {
    #[cfg(target_arch = "x86_64")]
    {
        simd_arch::_mm_loadu_si128(p)
    }
    #[cfg(target_arch = "aarch64")]
    {
        // NEON loads are unaligned by default
        unsafe { __m128i(simd_arch::vld1q_u8(p as *const u8)) }
    }
}

/// Store 128-bit value to unaligned memory
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn _mm_storeu_si128(p: *mut __m128i, a: __m128i) {
    #[cfg(target_arch = "x86_64")]
    {
        simd_arch::_mm_storeu_si128(p, a);
    }
    #[cfg(target_arch = "aarch64")]
    {
        // NEON stores are unaligned by default
        unsafe {
            simd_arch::vst1q_u8(p as *mut u8, a.0);
        }
    }
}