//! An abstraction layer for SIMD intrinsics.
//!
//! This module provides a portable interface to SIMD operations,
//! abstracting over the differences between x86-64 (SSE, AVX, AVX2, AVX-512) and aarch64 (NEON).
//!
//! ## SIMD Architecture
//!
//! The `SimdEngine` trait provides a unified interface for different SIMD widths:
//! - `SimdEngine128`: 128-bit vectors (SSE on x86_64, NEON on aarch64) - 16 lanes (8-bit)
//! - `SimdEngine256`: 256-bit vectors (AVX2 on x86_64) - 32 lanes (8-bit)
//! - `SimdEngine512`: 512-bit vectors (AVX-512 on x86_64) - 64 lanes (8-bit)
//!
//! ## Runtime SIMD Dispatch Pattern
//!
//! The codebase uses a **runtime dispatch** pattern for SIMD optimization:
//!
//! 1. **Detection** (once per run): `detect_optimal_simd_engine()` detects CPU features
//! 2. **Batch Sizing**: `get_simd_batch_sizes(engine)` returns optimal batch sizes
//! 3. **Dispatch**: Functions use `match engine` to select the optimal SIMD kernel
//!
//! **Key Dispatch Locations**:
//! - `banded_swa.rs`: `simd_banded_swa_dispatch()` - Smith-Waterman alignment
//! - `align.rs`: `determine_optimal_batch_size()` - Adaptive batch sizing
//! - `mem.rs`: Initial detection for logging
//!
//! This pattern allows binary portability across different CPU generations while
//! maximizing performance on modern hardware.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64 as simd_arch;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64 as simd_arch;

/// Trait abstracting over different SIMD vector widths for Smith-Waterman alignment.
///
/// This trait provides a unified interface for SIMD operations across different
/// architectures (SSE, AVX2, AVX-512) and platforms (x86_64, aarch64).
///
/// The trait uses associated types for vector types and constants for lane counts,
/// allowing generic code to work across different SIMD widths without runtime overhead.
pub trait SimdEngine: Sized {
    /// Number of 8-bit lanes in a vector (16 for SSE, 32 for AVX2, 64 for AVX-512)
    const WIDTH_8: usize;

    /// Number of 16-bit lanes in a vector (8 for SSE, 16 for AVX2, 32 for AVX-512)
    const WIDTH_16: usize;

    /// Vector type for 8-bit operations (__m128i, __m256i, or __m512i)
    type Vec8: Copy + Clone;

    /// Vector type for 16-bit operations (__m128i, __m256i, or __m512i)
    type Vec16: Copy + Clone;

    // ===== Creation and Initialization =====

    /// Create a vector with all elements set to zero
    unsafe fn setzero_vec8() -> Self::Vec8;

    /// Create a vector with all 8-bit elements set to zero
    unsafe fn setzero_epi8() -> Self::Vec8;

    /// Create a vector with all 8-bit elements set to the same value
    unsafe fn set1_epi8(a: i8) -> Self::Vec8;

    /// Create a vector with all 16-bit elements set to the same value
    unsafe fn set1_epi16(a: i16) -> Self::Vec16;

    /// Create a vector with all 32-bit elements set to the same value
    unsafe fn set1_epi32(a: i32) -> Self::Vec8;

    // ===== 8-bit Integer Arithmetic =====

    /// Add packed 8-bit integers
    unsafe fn add_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8;

    /// Subtract packed 8-bit integers
    unsafe fn sub_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8;

    /// Saturating subtract packed unsigned 8-bit integers
    unsafe fn subs_epu8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8;

    /// Saturating subtract packed signed 8-bit integers
    unsafe fn subs_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8;

    /// Saturating add packed signed 8-bit integers
    unsafe fn adds_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8;

    /// Saturating add packed unsigned 8-bit integers
    unsafe fn adds_epu8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8;

    /// Maximum of packed signed 8-bit integers
    unsafe fn max_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8;

    /// Maximum of packed unsigned 8-bit integers
    unsafe fn max_epu8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8;

    /// Minimum of packed unsigned 8-bit integers
    unsafe fn min_epu8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8;

    // ===== 16-bit Integer Arithmetic =====

    /// Add packed 16-bit integers
    unsafe fn add_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16;

    /// Subtract packed 16-bit integers
    unsafe fn sub_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16;

    /// Saturating add packed signed 16-bit integers
    unsafe fn adds_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16;

    /// Saturating subtract packed unsigned 16-bit integers
    unsafe fn subs_epu16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16;

    /// Saturating subtract packed signed 16-bit integers
    unsafe fn subs_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16;

    /// Maximum of packed signed 16-bit integers
    unsafe fn max_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16;

    /// Minimum of packed signed 16-bit integers
    unsafe fn min_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16;

    /// Maximum of packed unsigned 16-bit integers
    unsafe fn max_epu16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16;

    // ===== Comparison Operations =====

    /// Compare packed 8-bit integers for equality
    unsafe fn cmpeq_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8;

    /// Compare packed 8-bit integers for greater-than
    unsafe fn cmpgt_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8;

    /// Compare packed 16-bit integers for equality
    unsafe fn cmpeq_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16;

    /// Compare packed 16-bit integers for greater-than
    unsafe fn cmpgt_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16;

    // ===== Blend/Select Operations =====

    /// Blend packed 8-bit integers based on mask
    /// For each lane: if mask bit is set, take from b, else take from a
    unsafe fn blendv_epi8(a: Self::Vec8, b: Self::Vec8, mask: Self::Vec8) -> Self::Vec8;

    // ===== Bitwise Operations =====

    /// Bitwise AND
    unsafe fn and_si128(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8;

    /// Bitwise OR
    unsafe fn or_si128(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8;

    /// Bitwise AND NOT
    unsafe fn andnot_si128(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8;

    // ===== Shift Operations =====

    /// Shift 16-bit integers left by immediate value
    unsafe fn slli_epi16(a: Self::Vec16, imm8: i32) -> Self::Vec16;

    /// Shift 128-bit vector right by 2 bytes (fixed)
    unsafe fn srli_si128_fixed(a: Self::Vec8) -> Self::Vec8;

    /// Concatenate and shift right by 1 byte
    unsafe fn alignr_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8;

    // ===== Memory Operations =====

    /// Load 128/256/512-bit value from aligned memory
    unsafe fn load_si128(p: *const Self::Vec8) -> Self::Vec8;

    /// Store 128/256/512-bit value to aligned memory
    unsafe fn store_si128(p: *mut Self::Vec8, a: Self::Vec8);

    /// Load 128/256/512-bit value from unaligned memory
    unsafe fn loadu_si128(p: *const Self::Vec8) -> Self::Vec8;

    /// Store 128/256/512-bit value to unaligned memory
    unsafe fn storeu_si128(p: *mut Self::Vec8, a: Self::Vec8);
}

// Type alias for __m128i
#[allow(non_camel_case_types)]
#[cfg(target_arch = "x86_64")]
pub type __m128i = simd_arch::__m128i;

#[allow(non_camel_case_types)]
#[cfg(target_arch = "aarch64")]
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct __m128i(pub simd_arch::uint8x16_t);

// Helper methods for type casting on aarch64
#[cfg(target_arch = "aarch64")]
impl __m128i {
    #[inline(always)]
    pub fn as_s8(self) -> simd_arch::int8x16_t {
        unsafe { simd_arch::vreinterpretq_s8_u8(self.0) }
    }

    #[inline(always)]
    pub fn from_s8(v: simd_arch::int8x16_t) -> Self {
        Self(unsafe { simd_arch::vreinterpretq_u8_s8(v) })
    }

    #[inline(always)]
    pub fn as_s16(self) -> simd_arch::int16x8_t {
        unsafe { simd_arch::vreinterpretq_s16_u8(self.0) }
    }

    #[inline(always)]
    pub fn from_s16(v: simd_arch::int16x8_t) -> Self {
        Self(unsafe { simd_arch::vreinterpretq_u8_s16(v) })
    }

    #[inline(always)]
    pub fn as_u16(self) -> simd_arch::uint16x8_t {
        unsafe { simd_arch::vreinterpretq_u16_u8(self.0) }
    }

    #[inline(always)]
    pub fn from_u16(v: simd_arch::uint16x8_t) -> Self {
        Self(unsafe { simd_arch::vreinterpretq_u8_u16(v) })
    }

    #[inline(always)]
    pub fn as_s32(self) -> simd_arch::int32x4_t {
        unsafe { simd_arch::vreinterpretq_s32_u8(self.0) }
    }

    #[inline(always)]
    pub fn from_s32(v: simd_arch::int32x4_t) -> Self {
        Self(unsafe { simd_arch::vreinterpretq_u8_s32(v) })
    }
}

//
// Start of the portable intrinsic implementations
//

#[inline(always)]
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

#[inline(always)]
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

#[inline(always)]
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

#[inline(always)]
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

#[inline(always)]
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

#[inline(always)]
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

#[inline(always)]
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

#[inline(always)]
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

#[inline(always)]
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

#[inline(always)]
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

#[inline(always)]
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

// ===== Additional intrinsics for Smith-Waterman =====

// 8-bit integer operations
#[inline(always)]
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

#[inline(always)]
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

#[inline(always)]
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

#[inline(always)]
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
#[inline(always)]
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

#[inline(always)]
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

#[inline(always)]
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

#[inline(always)]
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

#[inline(always)]
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

#[inline(always)]
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

#[inline(always)]
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

#[inline(always)]
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

#[inline(always)]
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

#[inline(always)]
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

#[inline(always)]
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

#[inline(always)]
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

#[inline(always)]
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

#[inline(always)]
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

#[inline(always)]
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
#[inline(always)]
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
#[inline(always)]
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
#[inline(always)]
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

// =============================================================================
// SimdEngine Implementations
// =============================================================================

/// 128-bit SIMD engine (SSE on x86_64, NEON on aarch64)
///
/// Provides 16-way parallelism for 8-bit operations and 8-way for 16-bit operations.
/// This is the baseline SIMD implementation that works on all modern CPUs.
pub struct SimdEngine128;

#[allow(unsafe_op_in_unsafe_fn)]
impl SimdEngine for SimdEngine128 {
    const WIDTH_8: usize = 16; // 128 bits ÷ 8 bits = 16 lanes
    const WIDTH_16: usize = 8; // 128 bits ÷ 16 bits = 8 lanes

    type Vec8 = __m128i;
    type Vec16 = __m128i;

    // ===== Creation and Initialization =====

    #[inline(always)]
    unsafe fn setzero_vec8() -> Self::Vec8 {
        unsafe { _mm_setzero_si128() }
    }

    #[inline(always)]
    unsafe fn setzero_epi8() -> Self::Vec8 {
        unsafe { _mm_setzero_si128() }
    }

    #[inline(always)]
    unsafe fn set1_epi8(a: i8) -> Self::Vec8 {
        unsafe { _mm_set1_epi8(a) }
    }

    #[inline(always)]
    unsafe fn set1_epi16(a: i16) -> Self::Vec16 {
        unsafe { _mm_set1_epi16(a) }
    }

    #[inline(always)]
    unsafe fn set1_epi32(a: i32) -> Self::Vec8 {
        unsafe { _mm_set1_epi32(a) }
    }

    // ===== 8-bit Integer Arithmetic =====

    #[inline(always)]
    unsafe fn add_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        unsafe { _mm_add_epi8(a, b) }
    }

    #[inline(always)]
    unsafe fn sub_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        unsafe { _mm_sub_epi8(a, b) }
    }

    #[inline(always)]
    unsafe fn subs_epu8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        unsafe { _mm_subs_epu8(a, b) }
    }

    #[inline(always)]
    unsafe fn subs_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        unsafe { _mm_subs_epi8(a, b) }
    }

    #[inline(always)]
    unsafe fn adds_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        unsafe { _mm_adds_epi8(a, b) }
    }

    #[inline(always)]
    unsafe fn adds_epu8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        unsafe { _mm_adds_epu8(a, b) }
    }

    #[inline(always)]
    unsafe fn max_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        unsafe { _mm_max_epi8(a, b) }
    }

    #[inline(always)]
    unsafe fn max_epu8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        unsafe { _mm_max_epu8(a, b) }
    }

    #[inline(always)]
    unsafe fn min_epu8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        unsafe { _mm_min_epu8(a, b) }
    }

    // ===== 16-bit Integer Arithmetic =====

    #[inline(always)]
    unsafe fn add_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        unsafe { _mm_add_epi16(a, b) }
    }

    #[inline(always)]
    unsafe fn sub_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        unsafe { _mm_sub_epi16(a, b) }
    }

    #[inline(always)]
    unsafe fn adds_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        unsafe { _mm_adds_epi16(a, b) }
    }

    #[inline(always)]
    unsafe fn subs_epu16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        unsafe { _mm_subs_epu16(a, b) }
    }

    #[inline(always)]
    unsafe fn subs_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        unsafe { _mm_subs_epi16(a, b) }
    }

    #[inline(always)]
    unsafe fn max_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        unsafe { _mm_max_epi16(a, b) }
    }

    #[inline(always)]
    unsafe fn min_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        unsafe { _mm_min_epi16(a, b) }
    }

    #[inline(always)]
    unsafe fn max_epu16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        unsafe { _mm_max_epu16(a, b) }
    }

    // ===== Comparison Operations =====

    #[inline(always)]
    unsafe fn cmpeq_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        unsafe { _mm_cmpeq_epi8(a, b) }
    }

    #[inline(always)]
    unsafe fn cmpgt_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        unsafe { _mm_cmpgt_epi8(a, b) }
    }

    #[inline(always)]
    unsafe fn cmpeq_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        unsafe { _mm_cmpeq_epi16(a, b) }
    }

    #[inline(always)]
    unsafe fn cmpgt_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        unsafe { _mm_cmpgt_epi16(a, b) }
    }

    // ===== Blend/Select Operations =====

    #[inline(always)]
    unsafe fn blendv_epi8(a: Self::Vec8, b: Self::Vec8, mask: Self::Vec8) -> Self::Vec8 {
        unsafe { _mm_blendv_epi8(a, b, mask) }
    }

    // ===== Bitwise Operations =====

    #[inline(always)]
    unsafe fn and_si128(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        unsafe { _mm_and_si128(a, b) }
    }

    #[inline(always)]
    unsafe fn or_si128(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        unsafe { _mm_or_si128(a, b) }
    }

    #[inline(always)]
    unsafe fn andnot_si128(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        unsafe { _mm_andnot_si128(a, b) }
    }

    // ===== Shift Operations =====

    #[inline(always)]
    unsafe fn slli_epi16(a: Self::Vec16, imm8: i32) -> Self::Vec16 {
        // For runtime variable shift amounts, we need to use a match or lookup table
        // This is a limitation of x86 intrinsics requiring compile-time constants
        match imm8 {
            0 => a,
            1 => mm_slli_epi16!(a, 1),
            2 => mm_slli_epi16!(a, 2),
            3 => mm_slli_epi16!(a, 3),
            4 => mm_slli_epi16!(a, 4),
            5 => mm_slli_epi16!(a, 5),
            6 => mm_slli_epi16!(a, 6),
            7 => mm_slli_epi16!(a, 7),
            8 => mm_slli_epi16!(a, 8),
            _ => mm_slli_epi16!(a, 0), // Fallback for out of range
        }
    }

    #[inline(always)]
    unsafe fn srli_si128_fixed(a: Self::Vec8) -> Self::Vec8 {
        mm_srli_si128!(a, 2)
    }

    #[inline(always)]
    unsafe fn alignr_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        mm_alignr_epi8!(a, b, 1)
    }

    // ===== Memory Operations =====

    #[inline(always)]
    unsafe fn load_si128(p: *const Self::Vec8) -> Self::Vec8 {
        unsafe { _mm_load_si128(p) }
    }

    #[inline(always)]
    unsafe fn store_si128(p: *mut Self::Vec8, a: Self::Vec8) {
        unsafe { _mm_store_si128(p, a) }
    }

    #[inline(always)]
    unsafe fn loadu_si128(p: *const Self::Vec8) -> Self::Vec8 {
        unsafe { _mm_loadu_si128(p) }
    }

    #[inline(always)]
    unsafe fn storeu_si128(p: *mut Self::Vec8, a: Self::Vec8) {
        unsafe { _mm_storeu_si128(p, a) }
    }
}

// =============================================================================
// SimdEngine256 - AVX2 Implementation (x86_64 only)
// =============================================================================

#[cfg(target_arch = "x86_64")]
/// 256-bit SIMD engine (AVX2 on x86_64)
///
/// Provides 32-way parallelism for 8-bit operations and 16-way for 16-bit operations.
/// Requires AVX2 CPU support (Intel Haswell 2013+ or AMD Excavator 2015+).
///
/// Performance: ~2x throughput improvement over SimdEngine128 for compute-bound workloads.
pub struct SimdEngine256;

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)]
impl SimdEngine for SimdEngine256 {
    const WIDTH_8: usize = 32; // 256 bits ÷ 8 bits = 32 lanes
    const WIDTH_16: usize = 16; // 256 bits ÷ 16 bits = 16 lanes

    type Vec8 = simd_arch::__m256i;
    type Vec16 = simd_arch::__m256i;

    // ===== Creation and Initialization =====

    #[target_feature(enable = "avx2")]
    unsafe fn setzero_vec8() -> Self::Vec8 {
        simd_arch::_mm256_setzero_si256()
    }

    #[target_feature(enable = "avx2")]
    unsafe fn setzero_epi8() -> Self::Vec8 {
        simd_arch::_mm256_setzero_si256()
    }

    #[target_feature(enable = "avx2")]
    unsafe fn set1_epi8(a: i8) -> Self::Vec8 {
        simd_arch::_mm256_set1_epi8(a)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn set1_epi16(a: i16) -> Self::Vec16 {
        simd_arch::_mm256_set1_epi16(a)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn set1_epi32(a: i32) -> Self::Vec8 {
        simd_arch::_mm256_set1_epi32(a)
    }

    // ===== 8-bit Integer Arithmetic =====

    #[target_feature(enable = "avx2")]
    unsafe fn add_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm256_add_epi8(a, b)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn sub_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm256_sub_epi8(a, b)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn subs_epu8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm256_subs_epu8(a, b)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn subs_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm256_subs_epi8(a, b)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn adds_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm256_adds_epi8(a, b)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn adds_epu8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm256_adds_epu8(a, b)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn max_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm256_max_epi8(a, b)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn max_epu8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm256_max_epu8(a, b)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn min_epu8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm256_min_epu8(a, b)
    }

    // ===== 16-bit Integer Arithmetic =====

    #[target_feature(enable = "avx2")]
    unsafe fn add_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        simd_arch::_mm256_add_epi16(a, b)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn sub_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        simd_arch::_mm256_sub_epi16(a, b)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn adds_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        simd_arch::_mm256_adds_epi16(a, b)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn subs_epu16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        simd_arch::_mm256_subs_epu16(a, b)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn subs_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        simd_arch::_mm256_subs_epi16(a, b)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn max_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        simd_arch::_mm256_max_epi16(a, b)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn min_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        simd_arch::_mm256_min_epi16(a, b)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn max_epu16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        simd_arch::_mm256_max_epu16(a, b)
    }

    // ===== Comparison Operations =====

    #[target_feature(enable = "avx2")]
    unsafe fn cmpeq_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm256_cmpeq_epi8(a, b)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn cmpgt_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm256_cmpgt_epi8(a, b)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn cmpeq_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        simd_arch::_mm256_cmpeq_epi16(a, b)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn cmpgt_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        simd_arch::_mm256_cmpgt_epi16(a, b)
    }

    // ===== Blend/Select Operations =====

    #[target_feature(enable = "avx2")]
    unsafe fn blendv_epi8(a: Self::Vec8, b: Self::Vec8, mask: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm256_blendv_epi8(a, b, mask)
    }

    // ===== Bitwise Operations =====

    #[target_feature(enable = "avx2")]
    unsafe fn and_si128(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm256_and_si256(a, b)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn or_si128(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm256_or_si256(a, b)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn andnot_si128(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm256_andnot_si256(a, b)
    }

    // ===== Shift Operations =====

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

    #[target_feature(enable = "avx2")]
    unsafe fn srli_si128_fixed(a: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm256_srli_si256(a, 2)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn alignr_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm256_alignr_epi8(a, b, 1)
    }

    // ===== Memory Operations =====

    #[target_feature(enable = "avx2")]
    unsafe fn load_si128(p: *const Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm256_load_si256(p)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn store_si128(p: *mut Self::Vec8, a: Self::Vec8) {
        simd_arch::_mm256_store_si256(p, a)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn loadu_si128(p: *const Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm256_loadu_si256(p)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn storeu_si128(p: *mut Self::Vec8, a: Self::Vec8) {
        simd_arch::_mm256_storeu_si256(p, a)
    }
}

// ============================================================================
// SimdEngine512: AVX-512 Implementation (64-way parallelism)
// ============================================================================

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub struct SimdEngine512;

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[allow(unsafe_op_in_unsafe_fn)]
impl SimdEngine for SimdEngine512 {
    const WIDTH_8: usize = 64; // 64 lanes for 8-bit operations (4x SSE)
    const WIDTH_16: usize = 32; // 32 lanes for 16-bit operations (4x SSE)

    type Vec8 = simd_arch::__m512i;
    type Vec16 = simd_arch::__m512i;

    #[target_feature(enable = "avx512bw")]
    unsafe fn setzero_vec8() -> Self::Vec8 {
        simd_arch::_mm512_setzero_si512()
    }

    #[target_feature(enable = "avx512bw")]
    unsafe fn setzero_epi8() -> Self::Vec8 {
        simd_arch::_mm512_setzero_si512()
    }

    #[target_feature(enable = "avx512bw")]
    unsafe fn set1_epi8(a: i8) -> Self::Vec8 {
        simd_arch::_mm512_set1_epi8(a)
    }

    #[target_feature(enable = "avx512bw")]
    unsafe fn set1_epi16(a: i16) -> Self::Vec16 {
        simd_arch::_mm512_set1_epi16(a)
    }

    #[target_feature(enable = "avx512bw")]
    unsafe fn set1_epi32(a: i32) -> Self::Vec8 {
        simd_arch::_mm512_set1_epi32(a)
    }

    #[target_feature(enable = "avx512bw")]
    unsafe fn add_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm512_add_epi8(a, b)
    }

    #[target_feature(enable = "avx512bw")]
    unsafe fn sub_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm512_sub_epi8(a, b)
    }

    #[target_feature(enable = "avx512bw")]
    unsafe fn subs_epu8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm512_subs_epu8(a, b)
    }

    #[target_feature(enable = "avx512bw")]
    unsafe fn subs_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm512_subs_epi8(a, b)
    }

    #[target_feature(enable = "avx512bw")]
    unsafe fn adds_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm512_adds_epi8(a, b)
    }

    #[target_feature(enable = "avx512bw")]
    unsafe fn adds_epu8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm512_adds_epu8(a, b)
    }

    #[target_feature(enable = "avx512bw")]
    unsafe fn max_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm512_max_epi8(a, b)
    }

    #[target_feature(enable = "avx512bw")]
    unsafe fn max_epu8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm512_max_epu8(a, b)
    }

    #[target_feature(enable = "avx512bw")]
    unsafe fn min_epu8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm512_min_epu8(a, b)
    }

    #[target_feature(enable = "avx512bw")]
    unsafe fn add_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        simd_arch::_mm512_add_epi16(a, b)
    }

    #[target_feature(enable = "avx512bw")]
    unsafe fn sub_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        simd_arch::_mm512_sub_epi16(a, b)
    }

    #[target_feature(enable = "avx512bw")]
    unsafe fn adds_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        simd_arch::_mm512_adds_epi16(a, b)
    }

    #[target_feature(enable = "avx512bw")]
    unsafe fn subs_epu16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        simd_arch::_mm512_subs_epu16(a, b)
    }

    #[target_feature(enable = "avx512bw")]
    unsafe fn subs_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        simd_arch::_mm512_subs_epi16(a, b)
    }

    #[target_feature(enable = "avx512bw")]
    unsafe fn max_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        simd_arch::_mm512_max_epi16(a, b)
    }

    #[target_feature(enable = "avx512bw")]
    unsafe fn min_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        simd_arch::_mm512_min_epi16(a, b)
    }

    #[target_feature(enable = "avx512bw")]
    unsafe fn max_epu16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        simd_arch::_mm512_max_epu16(a, b)
    }

    #[target_feature(enable = "avx512bw")]
    unsafe fn cmpeq_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        let mask = simd_arch::_mm512_cmpeq_epi8_mask(a, b);
        simd_arch::_mm512_maskz_set1_epi8(mask, -1i8)
    }

    #[target_feature(enable = "avx512bw")]
    unsafe fn cmpgt_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        let mask = simd_arch::_mm512_cmpgt_epi8_mask(a, b);
        simd_arch::_mm512_maskz_set1_epi8(mask, -1i8)
    }

    #[target_feature(enable = "avx512bw")]
    unsafe fn cmpeq_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        let mask = simd_arch::_mm512_cmpeq_epi16_mask(a, b);
        simd_arch::_mm512_maskz_set1_epi16(mask, -1i16)
    }

    #[target_feature(enable = "avx512bw")]
    unsafe fn cmpgt_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        let mask = simd_arch::_mm512_cmpgt_epi16_mask(a, b);
        simd_arch::_mm512_maskz_set1_epi16(mask, -1i16)
    }

    #[target_feature(enable = "avx512bw")]
    unsafe fn blendv_epi8(a: Self::Vec8, b: Self::Vec8, mask: Self::Vec8) -> Self::Vec8 {
        // Convert mask to kmask (AVX-512 uses mask registers)
        let kmask = simd_arch::_mm512_movepi8_mask(mask);
        simd_arch::_mm512_mask_blend_epi8(kmask, a, b)
    }

    #[target_feature(enable = "avx512bw")]
    unsafe fn and_si128(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm512_and_si512(a, b)
    }

    #[target_feature(enable = "avx512bw")]
    unsafe fn or_si128(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm512_or_si512(a, b)
    }

    #[target_feature(enable = "avx512bw")]
    unsafe fn andnot_si128(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm512_andnot_si512(a, b)
    }

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

    #[target_feature(enable = "avx512bw")]
    unsafe fn srli_si128_fixed(a: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm512_bsrli_epi128(a, 2)
    }

    #[target_feature(enable = "avx512bw")]
    unsafe fn alignr_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm512_alignr_epi8(a, b, 1)
    }

    #[target_feature(enable = "avx512bw")]
    unsafe fn load_si128(p: *const Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm512_load_si512(p)
    }

    #[target_feature(enable = "avx512bw")]
    unsafe fn store_si128(p: *mut Self::Vec8, a: Self::Vec8) {
        simd_arch::_mm512_store_si512(p, a)
    }

    #[target_feature(enable = "avx512bw")]
    unsafe fn loadu_si128(p: *const Self::Vec8) -> Self::Vec8 {
        simd_arch::_mm512_loadu_si512(p)
    }

    #[target_feature(enable = "avx512bw")]
    unsafe fn storeu_si128(p: *mut Self::Vec8, a: Self::Vec8) {
        simd_arch::_mm512_storeu_si512(p, a)
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Test basic arithmetic operations for SimdEngine128
    #[test]
    fn test_simd_engine_128_basic_ops() {
        unsafe {
            // Test set1 and add
            let a = SimdEngine128::set1_epi8(5);
            let b = SimdEngine128::set1_epi8(3);
            let sum = SimdEngine128::add_epi8(a, b);

            // Extract and verify (8 is expected since 5 + 3 = 8)
            let mut result = [0u8; 16];
            SimdEngine128::storeu_si128(result.as_mut_ptr() as *mut _, sum);
            assert_eq!(result[0], 8);
            assert_eq!(result[15], 8);
        }
    }

    /// Test saturating arithmetic for SimdEngine128
    #[test]
    fn test_simd_engine_128_saturating_ops() {
        unsafe {
            // Test saturating add (unsigned)
            let a = SimdEngine128::set1_epi8(250_u8 as i8);
            let b = SimdEngine128::set1_epi8(10_u8 as i8);
            let sum = SimdEngine128::adds_epu8(a, b);

            let mut result = [0u8; 16];
            SimdEngine128::storeu_si128(result.as_mut_ptr() as *mut _, sum);
            // 250 + 10 = 260, but should saturate to 255
            assert_eq!(result[0], 255);

            // Test saturating subtract (unsigned)
            let a = SimdEngine128::set1_epi8(10_u8 as i8);
            let b = SimdEngine128::set1_epi8(20_u8 as i8);
            let diff = SimdEngine128::subs_epu8(a, b);

            SimdEngine128::storeu_si128(result.as_mut_ptr() as *mut _, diff);
            // 10 - 20 would be negative, but should saturate to 0
            assert_eq!(result[0], 0);
        }
    }

    /// Test max/min operations for SimdEngine128
    #[test]
    fn test_simd_engine_128_max_min() {
        unsafe {
            let a = SimdEngine128::set1_epi8(10_u8 as i8);
            let b = SimdEngine128::set1_epi8(20_u8 as i8);

            let max_val = SimdEngine128::max_epu8(a, b);
            let min_val = SimdEngine128::min_epu8(a, b);

            let mut max_result = [0u8; 16];
            let mut min_result = [0u8; 16];
            SimdEngine128::storeu_si128(max_result.as_mut_ptr() as *mut _, max_val);
            SimdEngine128::storeu_si128(min_result.as_mut_ptr() as *mut _, min_val);

            assert_eq!(max_result[0], 20);
            assert_eq!(min_result[0], 10);
        }
    }

    /// Test comparison operations for SimdEngine128
    #[test]
    fn test_simd_engine_128_compare() {
        unsafe {
            let a = SimdEngine128::set1_epi8(42);
            let b = SimdEngine128::set1_epi8(42);
            let c = SimdEngine128::set1_epi8(10);

            // Test equality
            let eq_mask = SimdEngine128::cmpeq_epi8(a, b);
            let mut eq_result = [0u8; 16];
            SimdEngine128::storeu_si128(eq_result.as_mut_ptr() as *mut _, eq_mask);
            // All bits should be set (0xFF) for equal values
            assert_eq!(eq_result[0], 0xFF);

            // Test greater than (signed comparison)
            let gt_mask = SimdEngine128::cmpgt_epi8(a, c);
            let mut gt_result = [0u8; 16];
            SimdEngine128::storeu_si128(gt_result.as_mut_ptr() as *mut _, gt_mask);
            // 42 > 10, so mask should be 0xFF
            assert_eq!(gt_result[0], 0xFF);
        }
    }

    /// Test bitwise operations for SimdEngine128
    #[test]
    fn test_simd_engine_128_bitwise() {
        unsafe {
            let a = SimdEngine128::set1_epi8(0b11110000_u8 as i8);
            let b = SimdEngine128::set1_epi8(0b10101010_u8 as i8);

            // Test AND
            let and_val = SimdEngine128::and_si128(a, b);
            let mut and_result = [0u8; 16];
            SimdEngine128::storeu_si128(and_result.as_mut_ptr() as *mut _, and_val);
            assert_eq!(and_result[0], 0b10100000);

            // Test OR
            let or_val = SimdEngine128::or_si128(a, b);
            let mut or_result = [0u8; 16];
            SimdEngine128::storeu_si128(or_result.as_mut_ptr() as *mut _, or_val);
            assert_eq!(or_result[0], 0b11111010);

            // Test AND-NOT (a & ~b)
            let andnot_val = SimdEngine128::andnot_si128(b, a);
            let mut andnot_result = [0u8; 16];
            SimdEngine128::storeu_si128(andnot_result.as_mut_ptr() as *mut _, andnot_val);
            assert_eq!(andnot_result[0], 0b01010000);
        }
    }

    /// Test memory load/store operations for SimdEngine128
    #[test]
    fn test_simd_engine_128_memory_ops() {
        unsafe {
            let data = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

            // Test unaligned load
            let vec = SimdEngine128::loadu_si128(data.as_ptr() as *const _);

            // Test unaligned store
            let mut result = [0u8; 16];
            SimdEngine128::storeu_si128(result.as_mut_ptr() as *mut _, vec);

            assert_eq!(data, result);
        }
    }

    /// Test zero vector creation
    #[test]
    fn test_simd_engine_128_zero() {
        unsafe {
            let zero_vec = SimdEngine128::setzero_epi8();
            let mut result = [0xFFu8; 16];
            SimdEngine128::storeu_si128(result.as_mut_ptr() as *mut _, zero_vec);

            assert_eq!(result, [0u8; 16]);
        }
    }

    /// Test SimdEngine256 basic operations (x86_64 only)
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_simd_engine_256_basic_ops() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("Skipping AVX2 test - CPU does not support AVX2");
            return;
        }

        unsafe {
            // Test set1 and add
            let a = SimdEngine256::set1_epi8(5);
            let b = SimdEngine256::set1_epi8(3);
            let sum = SimdEngine256::add_epi8(a, b);

            // Extract and verify (32 lanes for AVX2)
            let mut result = [0u8; 32];
            SimdEngine256::storeu_si128(result.as_mut_ptr() as *mut _, sum);
            assert_eq!(result[0], 8);
            assert_eq!(result[15], 8);
            assert_eq!(result[31], 8);
        }
    }

    /// Test SimdEngine256 max/min operations (x86_64 only)
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_simd_engine_256_max_min() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("Skipping AVX2 test - CPU does not support AVX2");
            return;
        }

        unsafe {
            let a = SimdEngine256::set1_epi8(10_u8 as i8);
            let b = SimdEngine256::set1_epi8(20_u8 as i8);

            let max_val = SimdEngine256::max_epu8(a, b);
            let min_val = SimdEngine256::min_epu8(a, b);

            let mut max_result = [0u8; 32];
            let mut min_result = [0u8; 32];
            SimdEngine256::storeu_si128(max_result.as_mut_ptr() as *mut _, max_val);
            SimdEngine256::storeu_si128(min_result.as_mut_ptr() as *mut _, min_val);

            assert_eq!(max_result[0], 20);
            assert_eq!(max_result[31], 20);
            assert_eq!(min_result[0], 10);
            assert_eq!(min_result[31], 10);
        }
    }

    /// Test SimdEngine512 basic operations (x86_64 with avx512 feature only)
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[test]
    fn test_simd_engine_512_basic_ops() {
        if !is_x86_feature_detected!("avx512bw") {
            eprintln!("Skipping AVX-512 test - CPU does not support AVX-512BW");
            return;
        }

        unsafe {
            // Test set1 and add
            let a = SimdEngine512::set1_epi8(5);
            let b = SimdEngine512::set1_epi8(3);
            let sum = SimdEngine512::add_epi8(a, b);

            // Extract and verify (64 lanes for AVX-512)
            let mut result = [0u8; 64];
            SimdEngine512::storeu_si128(result.as_mut_ptr() as *mut _, sum);
            assert_eq!(result[0], 8);
            assert_eq!(result[31], 8);
            assert_eq!(result[63], 8);
        }
    }


    /// Test that all engines can handle the same data pattern correctly
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_cross_engine_consistency() {
        // Test data pattern
        let test_pattern = [5u8; 16];

        // Test with SSE (always available)
        let mut sse_result = [0u8; 16];
        unsafe {
            let vec = SimdEngine128::loadu_si128(test_pattern.as_ptr() as *const _);
            let doubled = SimdEngine128::add_epi8(vec, vec);
            SimdEngine128::storeu_si128(sse_result.as_mut_ptr() as *mut _, doubled);
        }

        // All values should be 10 (5 + 5)
        for i in 0..16 {
            assert_eq!(sse_result[i], 10, "SSE lane {} incorrect", i);
        }

        // Test with AVX2 if available
        if is_x86_feature_detected!("avx2") {
            let mut avx2_test_pattern = [5u8; 32];
            let mut avx2_result = [0u8; 32];

            unsafe {
                let vec = SimdEngine256::loadu_si128(avx2_test_pattern.as_ptr() as *const _);
                let doubled = SimdEngine256::add_epi8(vec, vec);
                SimdEngine256::storeu_si128(avx2_result.as_mut_ptr() as *mut _, doubled);
            }

            // All values should be 10 (5 + 5)
            for i in 0..32 {
                assert_eq!(avx2_result[i], 10, "AVX2 lane {} incorrect", i);
            }
        }

        // Test with AVX-512 if available and feature enabled
        #[cfg(feature = "avx512")]
        {
            if is_x86_feature_detected!("avx512bw") {
                let mut avx512_test_pattern = [5u8; 64];
                let mut avx512_result = [0u8; 64];

                unsafe {
                    let vec = SimdEngine512::loadu_si128(avx512_test_pattern.as_ptr() as *const _);
                    let doubled = SimdEngine512::add_epi8(vec, vec);
                    SimdEngine512::storeu_si128(avx512_result.as_mut_ptr() as *mut _, doubled);
                }

                // All values should be 10 (5 + 5)
                for i in 0..64 {
                    assert_eq!(avx512_result[i], 10, "AVX-512 lane {} incorrect", i);
                }
            }
        }
    }
}
