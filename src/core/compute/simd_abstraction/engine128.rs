//! 128‑bit SIMD engine (SSE2/SSSE3 on x86_64; NEON on aarch64)
//!
//! This module implements the 128‑bit `SimdEngine` backend. On x86_64 it maps to
//! SSE2/SSSE3 intrinsics via `std::arch::x86_64`; on aarch64 it maps to NEON
//! intrinsics via `std::arch::aarch64` with a small wrapper type `__m128i` to
//! ease reinterpretation between element widths.
//!
//! Highlights
//! - Provides the baseline SIMD width used on all supported CPUs.
//! - Variable byte shifts (`slli_bytes`, `srli_bytes`) and `alignr_bytes` are
//!   implemented in portable helpers (`portable_intrinsics.rs`) to accept a
//!   runtime `num_bytes` on ISAs where the underlying instruction requires an
//!   immediate.
//! - `movemask` helpers are present for i8 and i16; the 16‑bit variant falls
//!   back to a straightforward store‑and‑test on SSE2.
//!
//! Safety
//! - All functions are `unsafe` and expect the caller to execute them on a CPU
//!   that supports the underlying ISA. The crate’s runtime dispatch ensures this
//!   in normal use.
//! - Pointer arguments to loads/stores must be valid for the accessed size and
//!   (for the aligned variants) appropriately aligned.

use super::portable_intrinsics::_mm_storeu_si128;
use super::portable_intrinsics::*;
use std::arch::x86_64::_mm_min_epi8;

use super::types::{__m128i, simd_arch};

use super::SimdEngine;

use crate::{mm_alignr_epi8, mm_slli_epi16, mm_srli_si128};

/// 128-bit SIMD engine (SSE on x86_64, NEON on aarch64)
///
/// Provides 16-way parallelism for 8-bit operations and 8-way for 16-bit operations.
/// This is the baseline SIMD implementation that works on all modern CPUs.
#[derive(Clone, Copy)]
pub struct SimdEngine128;

#[allow(unsafe_op_in_unsafe_fn)]
impl SimdEngine for SimdEngine128 {
    const WIDTH_8: usize = 16; // 128 bits ÷ 8 bits = 16 lanes
    const WIDTH_16: usize = 8; // 128 bits ÷ 16 bits = 8 lanes

    type Vec8 = __m128i;
    type Vec16 = __m128i;

    // ===== Creation and Initialization =====

    #[inline]
    unsafe fn setzero_vec8() -> Self::Vec8 {
        unsafe { _mm_setzero_si128() }
    }

    #[inline]
    unsafe fn setzero_epi8() -> Self::Vec8 {
        unsafe { _mm_setzero_si128() }
    }

    #[inline]
    unsafe fn set1_epi8(a: i8) -> Self::Vec8 {
        unsafe { _mm_set1_epi8(a) }
    }

    #[inline]
    unsafe fn set1_epi16(a: i16) -> Self::Vec16 {
        unsafe { _mm_set1_epi16(a) }
    }

    #[inline]
    unsafe fn set1_epi32(a: i32) -> Self::Vec8 {
        unsafe { _mm_set1_epi32(a) }
    }

    #[inline]
    unsafe fn setzero_epi16() -> Self::Vec16 {
        unsafe { _mm_setzero_si128() }
    }

    // ===== 8-bit Integer Arithmetic =====

    #[inline]
    unsafe fn extract_epi8(a: Self::Vec8, imm8: i32) -> i8 {
        #[cfg(target_arch = "x86_64")]
        {
            // Store to a temporary array and extract to avoid imm8 non-constant error

            let mut tmp = [0i8; 16];

            _mm_storeu_si128(tmp.as_mut_ptr() as *mut _, a); // Use portable intrinsic for consistency

            *tmp.get_unchecked(imm8 as usize)
        }

        #[cfg(target_arch = "aarch64")]
        {
            // On NEON, vgetq_lane_* requires a compile-time constant lane index.
            // Use a portable fallback: store to a temporary array and index at runtime.
            let mut tmp = [0i8; 16];
            _mm_storeu_si128(tmp.as_mut_ptr() as *mut _, a);
            *tmp.get_unchecked(imm8 as usize)
        }
    }

    #[inline]
    unsafe fn extract_epi16(a: Self::Vec16, imm8: i32) -> i16 {
        #[cfg(target_arch = "x86_64")]
        {
            // Store to a temporary array and extract to avoid imm8 non-constant error

            let mut tmp = [0i16; 8];

            _mm_storeu_si128(tmp.as_mut_ptr() as *mut _, a); // Use portable intrinsic for consistency

            *tmp.get_unchecked(imm8 as usize)
        }

        #[cfg(target_arch = "aarch64")]
        {
            // NEON lane access requires a constant index. Fallback via temporary array.
            let mut tmp = [0i16; 8];
            _mm_storeu_si128(tmp.as_mut_ptr() as *mut _, a);
            *tmp.get_unchecked(imm8 as usize)
        }
    }

    // ===== 16-bit Integer Arithmetic =====

    #[inline]
    unsafe fn movemask_epi8(a: Self::Vec8) -> i32 {
        #[cfg(target_arch = "x86_64")]
        {
            simd_arch::_mm_movemask_epi8(a)
        }

        #[cfg(target_arch = "aarch64")]
        {
            // NEON equivalent for _mm_movemask_epi8
            // Use portable path: store to array and check sign bits.
            let mut res = 0;
            let mut tmp = [0i8; 16];
            _mm_storeu_si128(tmp.as_mut_ptr() as *mut _, a);
            for i in 0..16 {
                if tmp[i as usize] < 0 {
                    res |= 1 << i;
                }
            }
            res
        }
    }

    #[inline]
    unsafe fn movemask_epi16(a: Self::Vec16) -> i32 {
        #[cfg(target_arch = "x86_64")]
        {
            // SSE2 does not have _mm_movemask_epi16.

            // Manual extraction of sign bits and combining them into an integer.

            let mut mask = 0;

            let mut tmp = [0i16; 8];

            _mm_storeu_si128(tmp.as_mut_ptr() as *mut _, a);

            for i in 0..8 {
                if tmp.get_unchecked(i) < &0 {
                    // Check if sign bit is set

                    mask |= 1 << i;
                }
            }

            mask
        }

        #[cfg(target_arch = "aarch64")]
        {
            // NEON equivalent for _mm_movemask_epi16
            // Use portable path: store to array and check sign bits.
            let mut res = 0;
            let mut tmp = [0i16; 8];
            _mm_storeu_si128(tmp.as_mut_ptr() as *mut _, a);
            for i in 0..8 {
                if tmp[i as usize] < 0 {
                    res |= 1 << i;
                }
            }
            res
        }
    }

    // ===== Shift Operations =====

    #[inline]
    unsafe fn slli_bytes(a: Self::Vec8, num_bytes: i32) -> Self::Vec8 {
        _mm_slli_si128_var(a, num_bytes)
    }

    #[inline]
    unsafe fn slli_bytes_16(a: Self::Vec16, num_bytes: i32) -> Self::Vec16 {
        _mm_slli_si128_var(a, num_bytes)
    }

    #[inline]
    unsafe fn srli_bytes(a: Self::Vec8, num_bytes: i32) -> Self::Vec8 {
        _mm_srli_si128_var(a, num_bytes)
    }

    #[inline]
    unsafe fn srli_bytes_16(a: Self::Vec16, num_bytes: i32) -> Self::Vec16 {
        _mm_srli_si128_var(a, num_bytes)
    }

    #[inline]
    unsafe fn alignr_bytes(a: Self::Vec8, b: Self::Vec8, num_bytes: i32) -> Self::Vec8 {
        _mm_alignr_epi8_var(a, b, num_bytes)
    }

    #[inline]
    unsafe fn alignr_bytes_16(a: Self::Vec16, b: Self::Vec16, num_bytes: i32) -> Self::Vec16 {
        _mm_alignr_epi8_var(a, b, num_bytes)
    }
    // ===== 8-bit Integer Arithmetic =====

    #[inline]
    unsafe fn add_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        unsafe { _mm_add_epi8(a, b) }
    }

    #[inline]
    unsafe fn sub_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        unsafe { _mm_sub_epi8(a, b) }
    }

    #[inline]
    unsafe fn subs_epu8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        unsafe { _mm_subs_epu8(a, b) }
    }

    #[inline]
    unsafe fn subs_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        unsafe { _mm_subs_epi8(a, b) }
    }

    #[inline]
    unsafe fn adds_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        unsafe { _mm_adds_epi8(a, b) }
    }

    #[inline]
    unsafe fn adds_epu8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        unsafe { _mm_adds_epu8(a, b) }
    }

    #[inline]
    unsafe fn max_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        unsafe { _mm_max_epi8(a, b) }
    }

    #[inline]
    unsafe fn max_epu8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        unsafe { _mm_max_epu8(a, b) }
    }

    #[inline]
    unsafe fn min_epu8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        unsafe { _mm_min_epu8(a, b) }
    }

    #[inline]
    unsafe fn min_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        unsafe { _mm_min_epi8(a, b) }
    }

    // ===== 16-bit Integer Arithmetic =====

    #[inline]
    unsafe fn add_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        unsafe { _mm_add_epi16(a, b) }
    }

    #[inline]
    unsafe fn sub_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        unsafe { _mm_sub_epi16(a, b) }
    }

    #[inline]
    unsafe fn adds_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        unsafe { _mm_adds_epi16(a, b) }
    }

    #[inline]
    unsafe fn subs_epu16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        unsafe { _mm_subs_epu16(a, b) }
    }

    #[inline]
    unsafe fn subs_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        unsafe { _mm_subs_epi16(a, b) }
    }

    #[inline]
    unsafe fn max_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        unsafe { _mm_max_epi16(a, b) }
    }

    #[inline]
    unsafe fn min_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        unsafe { _mm_min_epi16(a, b) }
    }

    #[inline]
    unsafe fn max_epu16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        unsafe { _mm_max_epu16(a, b) }
    }

    // ===== Comparison Operations =====

    #[inline]
    unsafe fn cmpeq_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        unsafe { _mm_cmpeq_epi8(a, b) }
    }

    #[inline]
    unsafe fn cmpgt_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        unsafe { _mm_cmpgt_epi8(a, b) }
    }

    #[inline]
    unsafe fn cmpgt_epu8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        // SSE has no native unsigned comparison; use XOR trick to flip sign bit
        // a >_u b ⟺ (a XOR 0x80) >_s (b XOR 0x80)
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::*;
            let sign_bit = _mm_set1_epi8(0x80u8 as i8);
            let a_signed = _mm_xor_si128(a, sign_bit);
            let b_signed = _mm_xor_si128(b, sign_bit);
            _mm_cmpgt_epi8(a_signed, b_signed)
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use std::arch::aarch64::*;
            // NEON: vcgt_u8 - unsigned greater-than
            let result_u8 = vcgtq_u8(a.as_u8(), b.as_u8());
            __m128i::from_u8(result_u8)
        }
    }

    #[inline]
    unsafe fn cmpge_epu8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        // a >=_u b ⟺ max(a, b) == a
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::*;
            let max_val = _mm_max_epu8(a, b);
            _mm_cmpeq_epi8(max_val, a)
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use std::arch::aarch64::*;
            // NEON: vcge_u8 - unsigned greater-or-equal
            let result_u8 = vcgeq_u8(a.as_u8(), b.as_u8());
            __m128i::from_u8(result_u8)
        }
    }

    #[inline]
    unsafe fn cmpeq_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        unsafe { _mm_cmpeq_epi16(a, b) }
    }

    #[inline]
    unsafe fn cmpgt_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        unsafe { _mm_cmpgt_epi16(a, b) }
    }

    // ===== Blend/Select Operations =====

    #[inline]
    unsafe fn blendv_epi8(a: Self::Vec8, b: Self::Vec8, mask: Self::Vec8) -> Self::Vec8 {
        unsafe { _mm_blendv_epi8(a, b, mask) }
    }

    // ===== Bitwise Operations =====

    #[inline]
    unsafe fn and_si128(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        unsafe { _mm_and_si128(a, b) }
    }

    #[inline]
    unsafe fn or_si128(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        unsafe { _mm_or_si128(a, b) }
    }

    #[inline]
    unsafe fn xor_si128(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::*;
            _mm_xor_si128(a, b)
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use std::arch::aarch64::*;
            __m128i::from_u8(veorq_u8(a.as_u8(), b.as_u8()))
        }
    }

    #[inline]
    unsafe fn andnot_si128(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        unsafe { _mm_andnot_si128(a, b) }
    }

    #[inline]
    unsafe fn shuffle_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::*;
            _mm_shuffle_epi8(a, b)
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use std::arch::aarch64::*;
            // Emulate SSSE3 _mm_shuffle_epi8 (pshufb) semantics on NEON.
            // SSSE3 behavior:
            //  - For each lane i, if control byte b[i] has high bit set (b[i] & 0x80), output is 0.
            //  - Else, lower 4 bits select source byte from a (index 0..15).
            let ctrl = b.as_u8();
            let idx = vandq_u8(ctrl, vdupq_n_u8(0x0F));
            let mut out = vqtbl1q_u8(a.as_u8(), idx);
            // Build a per-lane mask of 0xFF where high bit is set in control
            let high = vshrq_n_u8(ctrl, 7); // 0x00 or 0x01
            let lane_mask = vmulq_u8(high, vdupq_n_u8(0xFF)); // 0x00 or 0xFF
            // Clear lanes where mask is 0xFF
            out = vbicq_u8(out, lane_mask);
            __m128i::from_u8(out)
        }
    }

    // ===== Unpack Operations =====

    #[inline]
    unsafe fn unpacklo_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::*;
            _mm_unpacklo_epi8(a, b)
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use std::arch::aarch64::*;
            // NEON: vzip1q_u8 interleaves low halves
            __m128i::from_u8(vzip1q_u8(a.as_u8(), b.as_u8()))
        }
    }

    #[inline]
    unsafe fn unpackhi_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::*;
            _mm_unpackhi_epi8(a, b)
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use std::arch::aarch64::*;
            // NEON: vzip2q_u8 interleaves high halves
            __m128i::from_u8(vzip2q_u8(a.as_u8(), b.as_u8()))
        }
    }

    // ===== Shift Operations =====

    #[inline]
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

    #[inline]
    unsafe fn srli_si128_fixed(a: Self::Vec8) -> Self::Vec8 {
        mm_srli_si128!(a, 2)
    }

    #[inline]
    unsafe fn alignr_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8 {
        mm_alignr_epi8!(a, b, 1)
    }

    // ===== Memory Operations =====

    #[inline]
    unsafe fn load_si128(p: *const Self::Vec8) -> Self::Vec8 {
        unsafe { _mm_load_si128(p) }
    }

    #[inline]
    unsafe fn load_si128_16(p: *const Self::Vec16) -> Self::Vec16 {
        unsafe { _mm_load_si128_16(p) }
    }

    #[inline]
    unsafe fn store_si128(p: *mut Self::Vec8, a: Self::Vec8) {
        unsafe { _mm_store_si128(p, a) }
    }

    #[inline]
    unsafe fn store_si128_16(p: *mut Self::Vec16, a: Self::Vec16) {
        unsafe { _mm_store_si128_16(p, a) }
    }

    #[inline]
    unsafe fn loadu_si128(p: *const Self::Vec8) -> Self::Vec8 {
        unsafe { _mm_loadu_si128(p) }
    }

    #[inline]
    unsafe fn loadu_si128_16(p: *const Self::Vec16) -> Self::Vec16 {
        unsafe { _mm_loadu_si128_16(p) }
    }

    #[inline]
    unsafe fn storeu_si128(p: *mut Self::Vec8, a: Self::Vec8) {
        unsafe { _mm_storeu_si128(p, a) }
    }

    #[inline]
    unsafe fn storeu_si128_16(p: *mut Self::Vec16, a: Self::Vec16) {
        unsafe { _mm_storeu_si128_16(p, a) }
    }
}
