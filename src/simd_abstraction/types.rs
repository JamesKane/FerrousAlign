//! SIMD type aliases and architecture bindings
//!
//! This module provides a tiny portability layer so the rest of the
//! `simd_abstraction` code can refer to `simd_arch` and `__m128i` uniformly on
//! both x86_64 and aarch64.
//!
//! - On x86_64 we re-export `std::arch::x86_64` as `simd_arch` and use the
//!   native `__m128i` type.
//! - On aarch64 we re-export `std::arch::aarch64` as `simd_arch` and define a
//!   transparent wrapper `__m128i` backed by `uint8x16_t`. Helper methods allow
//!   lossless reinterpretation between common element widths used by the
//!   portable intrinsics.
//!
//! Safety: all methods that reinterpret vector types are simple bitcasts (no
//! lane reordering), implemented via NEON `vreinterpret` intrinsics.
 
#[cfg(target_arch = "x86_64")]
pub(crate) use std::arch::x86_64 as simd_arch;

#[cfg(target_arch = "aarch64")]
pub(crate) use std::arch::aarch64 as simd_arch;

/// Type alias for `__m128i` on x86_64.
#[allow(non_camel_case_types)]
#[cfg(target_arch = "x86_64")]
pub type __m128i = simd_arch::__m128i;

/// Transparent `__m128i` wrapper on aarch64 (NEON), backed by `uint8x16_t`.
#[allow(non_camel_case_types)]
#[cfg(target_arch = "aarch64")]
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct __m128i(pub simd_arch::uint8x16_t);

/// Helper methods for common reinterpret casts on aarch64.
#[cfg(target_arch = "aarch64")]
impl __m128i {
    /// View the underlying 128‑bit storage as signed 8‑bit lanes.
    #[inline]
    pub fn as_s8(self) -> simd_arch::int8x16_t {
        unsafe { simd_arch::vreinterpretq_s8_u8(self.0) }
    }

    /// Construct from signed 8‑bit lanes by reinterpretation.
    #[inline]
    pub fn from_s8(v: simd_arch::int8x16_t) -> Self {
        Self(unsafe { simd_arch::vreinterpretq_u8_s8(v) })
    }

    /// View as signed 16‑bit lanes.
    #[inline]
    pub fn as_s16(self) -> simd_arch::int16x8_t {
        unsafe { simd_arch::vreinterpretq_s16_u8(self.0) }
    }

    /// Construct from signed 16‑bit lanes by reinterpretation.
    #[inline]
    pub fn from_s16(v: simd_arch::int16x8_t) -> Self {
        Self(unsafe { simd_arch::vreinterpretq_u8_s16(v) })
    }

    /// View as unsigned 16‑bit lanes.
    #[inline]
    pub fn as_u16(self) -> simd_arch::uint16x8_t {
        unsafe { simd_arch::vreinterpretq_u16_u8(self.0) }
    }

    /// Construct from unsigned 16‑bit lanes by reinterpretation.
    #[inline]
    pub fn from_u16(v: simd_arch::uint16x8_t) -> Self {
        Self(unsafe { simd_arch::vreinterpretq_u8_u16(v) })
    }

    /// View as signed 32‑bit lanes.
    #[inline]
    pub fn as_s32(self) -> simd_arch::int32x4_t {
        unsafe { simd_arch::vreinterpretq_s32_u8(self.0) }
    }

    /// Construct from signed 32‑bit lanes by reinterpretation.
    #[inline]
    pub fn from_s32(v: simd_arch::int32x4_t) -> Self {
        Self(unsafe { simd_arch::vreinterpretq_u8_s32(v) })
    }
}
