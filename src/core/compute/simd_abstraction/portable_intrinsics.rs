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

/// Portable prefetch for read with T0 locality hint (all cache levels)
///
/// Wraps architecture-specific intrinsics.
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn prefetch_read_t0(addr: *const i8) {
    #[cfg(target_arch = "x86_64")]
    {
        // _MM_HINT_T0: Prefetch data into all levels of the cache hierarchy.
        simd_arch::_mm_prefetch(addr, simd_arch::_MM_HINT_T0);
    }
    #[cfg(target_arch = "aarch64")]
    {
        // ARM prefetch intrinsics are unstable in stable Rust.
        // Use inline assembly for prefetch on aarch64, which is stable.
        // PRFM PLDL1KEEP is equivalent to prefetch for read with high locality.
        core::arch::asm!(
            "prfm pldl1keep, [{addr}]",
            addr = in(reg) addr,
            options(nostack, preserves_flags)
        );
    }
    // Other architectures: No-op
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
        // Implement via portable byte copies to avoid NEON immediates.
        let shift = num_bytes.clamp(0, 16) as usize;
        if shift == 0 {
            return a;
        }
        if shift >= 16 {
            return _mm_setzero_si128();
        }
        let mut bytes = [0u8; 16];
        let mut src = [0u8; 16];
        simd_arch::vst1q_u8(src.as_mut_ptr(), a.0);
        // Left shift by N bytes: move lower (16-N) bytes up, zero-fill bottom N bytes
        for i in 0..(16 - shift) {
            bytes[i + shift] = src[i];
        }
        __m128i(unsafe { simd_arch::vld1q_u8(bytes.as_ptr()) })
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
        // Implement via portable byte copies to avoid NEON immediates.
        let shift = num_bytes.clamp(0, 16) as usize;
        if shift == 0 {
            return a;
        }
        if shift >= 16 {
            return _mm_setzero_si128();
        }
        let mut bytes = [0u8; 16];
        let mut src = [0u8; 16];
        simd_arch::vst1q_u8(src.as_mut_ptr(), a.0);
        // Right shift by N bytes: move upper (16-N) bytes down, zero-fill top N bytes
        for i in 0..(16 - shift) {
            bytes[i] = src[i + shift];
        }
        __m128i(unsafe { simd_arch::vld1q_u8(bytes.as_ptr()) })
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
        // Implement via portable byte copies: result = concat(b|a) >> num_bytes
        let shift = num_bytes.clamp(0, 32) as usize;
        if shift >= 32 {
            return _mm_setzero_si128();
        }
        // Dump both vectors to bytes
        let mut bytes_a = [0u8; 16];
        let mut bytes_b = [0u8; 16];
        simd_arch::vst1q_u8(bytes_a.as_mut_ptr(), a.0);
        simd_arch::vst1q_u8(bytes_b.as_mut_ptr(), b.0);
        // Build concatenation [b | a]
        let mut cat = [0u8; 32];
        cat[..16].copy_from_slice(&bytes_b);
        cat[16..].copy_from_slice(&bytes_a);
        // Extract 16 bytes starting at offset `shift`
        let mut out = [0u8; 16];
        let end = shift + 16;
        if end <= 32 {
            out.copy_from_slice(&cat[shift..end]);
        } else {
            // If shift > 16, some bytes would be beyond; fill remaining with zeros.
            let available = 32usize.saturating_sub(shift);
            out[..available].copy_from_slice(&cat[shift..(shift + available)]);
            // rest remain zero
        }
        __m128i(unsafe { simd_arch::vld1q_u8(out.as_ptr()) })
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
pub unsafe fn _mm_min_epi8(a: __m128i, b: __m128i) -> __m128i {
    #[cfg(target_arch = "x86_64")]
    {
        simd_arch::_mm_min_epi8(a, b)
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { __m128i::from_s8(simd_arch::vminq_s8(a.as_s8(), b.as_s8())) }
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
        // SSE blendv_epi8 semantics: if mask byte's MSB (bit 7) is 1, select b; else select a
        // NEON vbslq_u8 semantics: for each BIT, if mask bit is 1, select first arg; else second
        //
        // To emulate SSE blendv, we need to expand the MSB of each byte to fill all 8 bits.
        // We do this by arithmetic right shift by 7 (replicates sign bit).
        //
        // vshrq_n_s8 does arithmetic shift, so -128 (0x80) >> 7 = -1 (0xFF)
        // and 127 (0x7F) >> 7 = 0 (0x00)
        unsafe {
            let mask_expanded = simd_arch::vshrq_n_s8::<7>(mask.as_s8());
            // vbslq_u8(mask, a, b): where mask bit is 1 → select a; else → select b
            // We want: where MSB was 1 (now 0xFF) → select b; else → select a
            // So we need: vbslq_u8(mask_expanded, b, a)
            let mask_u8 = simd_arch::vreinterpretq_u8_s8(mask_expanded);
            __m128i(simd_arch::vbslq_u8(mask_u8, b.0, a.0))
        }
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
            std::arch::x86_64::_mm_slli_epi16($a, $imm8)
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            $crate::compute::simd_abstraction::types::__m128i::from_s16(
                std::arch::aarch64::vshlq_n_s16($a.as_s16(), $imm8),
            )
        }
    }};
}

#[macro_export]
macro_rules! mm_srli_si128 {
    ($a:expr, $imm8:expr) => {{
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_srli_si128($a, $imm8)
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            if $imm8 >= 16 {
                $crate::compute::simd_abstraction::portable_intrinsics::_mm_setzero_si128()
            } else {
                $crate::compute::simd_abstraction::types::__m128i(std::arch::aarch64::vextq_u8(
                    std::arch::aarch64::vdupq_n_u8(0),
                    $a.0,
                    $imm8,
                ))
            }
        }
    }};
}

#[macro_export]
macro_rules! mm_alignr_epi8 {
    ($a:expr, $b:expr, $imm8:expr) => {{
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_alignr_epi8($a, $b, $imm8)
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            if $imm8 >= 16 {
                $crate::compute::simd_abstraction::portable_intrinsics::_mm_srli_si128_var(
                    $a,
                    $imm8 - 16,
                )
            } else {
                $crate::compute::simd_abstraction::types::__m128i(std::arch::aarch64::vextq_u8(
                    $b.0, $a.0, $imm8,
                ))
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

//
// Generic match table macros for immediate-requiring intrinsics
//
// These macros consolidate the duplicated match statements in engine256 and
// engine512 that handle runtime immediate values for shift and alignr
// operations. The intrinsics themselves require compile-time constant
// immediates, but we need runtime flexibility, so we expand to 16-case match
// statements that the compiler can optimize (constant propagation + dead code
// elimination).
//

/// Generate match table for single-operand shifts with zero fallback
///
/// # Examples
/// ```ignore
/// let result = match_shift_immediate!(
///     vector,
///     shift_count,
///     simd_arch::_mm256_bslli_epi128,
///     simd_arch::_mm256_setzero_si256()
/// );
/// ```
#[macro_export]
macro_rules! match_shift_immediate {
    ($a:expr, $n:expr, $shift_op:path, $zero:expr) => {{
        match $n {
            0 => $a,
            1 => $shift_op($a, 1),
            2 => $shift_op($a, 2),
            3 => $shift_op($a, 3),
            4 => $shift_op($a, 4),
            5 => $shift_op($a, 5),
            6 => $shift_op($a, 6),
            7 => $shift_op($a, 7),
            8 => $shift_op($a, 8),
            9 => $shift_op($a, 9),
            10 => $shift_op($a, 10),
            11 => $shift_op($a, 11),
            12 => $shift_op($a, 12),
            13 => $shift_op($a, 13),
            14 => $shift_op($a, 14),
            15 => $shift_op($a, 15),
            _ => $zero,
        }
    }};
}

/// Generate match table for alignr with two operands
///
/// # Examples
/// ```ignore
/// let result = match_alignr_immediate!(
///     hi,
///     lo,
///     byte_count,
///     simd_arch::_mm256_alignr_epi8
/// );
/// ```
#[macro_export]
macro_rules! match_alignr_immediate {
    ($a:expr, $b:expr, $n:expr, $alignr_op:path) => {{
        match $n {
            0 => $alignr_op($a, $b, 0),
            1 => $alignr_op($a, $b, 1),
            2 => $alignr_op($a, $b, 2),
            3 => $alignr_op($a, $b, 3),
            4 => $alignr_op($a, $b, 4),
            5 => $alignr_op($a, $b, 5),
            6 => $alignr_op($a, $b, 6),
            7 => $alignr_op($a, $b, 7),
            8 => $alignr_op($a, $b, 8),
            9 => $alignr_op($a, $b, 9),
            10 => $alignr_op($a, $b, 10),
            11 => $alignr_op($a, $b, 11),
            12 => $alignr_op($a, $b, 12),
            13 => $alignr_op($a, $b, 13),
            14 => $alignr_op($a, $b, 14),
            15 => $alignr_op($a, $b, 15),
            _ => $alignr_op($a, $b, 0), // Out of range: return default
        }
    }};
}

/// Generate match table for alignr with custom out-of-range behavior
///
/// Used by AVX-512 which returns $a for byte counts >= 16.
///
/// # Examples
/// ```ignore
/// let result = match_alignr_immediate_or!(
///     hi,
///     lo,
///     byte_count,
///     simd_arch::_mm512_alignr_epi8,
///     hi  // Return hi for out-of-range
/// );
/// ```
#[macro_export]
macro_rules! match_alignr_immediate_or {
    ($a:expr, $b:expr, $n:expr, $alignr_op:path, $default:expr) => {{
        match $n {
            0 => $b, // Special case: 0 bytes from $a means all of $b
            1 => $alignr_op($a, $b, 1),
            2 => $alignr_op($a, $b, 2),
            3 => $alignr_op($a, $b, 3),
            4 => $alignr_op($a, $b, 4),
            5 => $alignr_op($a, $b, 5),
            6 => $alignr_op($a, $b, 6),
            7 => $alignr_op($a, $b, 7),
            8 => $alignr_op($a, $b, 8),
            9 => $alignr_op($a, $b, 9),
            10 => $alignr_op($a, $b, 10),
            11 => $alignr_op($a, $b, 11),
            12 => $alignr_op($a, $b, 12),
            13 => $alignr_op($a, $b, 13),
            14 => $alignr_op($a, $b, 14),
            15 => $alignr_op($a, $b, 15),
            _ => $default, // Out of range: custom behavior
        }
    }};
}
