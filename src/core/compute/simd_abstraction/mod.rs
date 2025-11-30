//! SIMD abstraction layer
//!
//! This module exposes a single, portable surface area for a handful of hot SIMD
//! operations used by the alignment kernels. It hides ISA differences between
//! x86_64 (SSE2/SSSE3/AVX2/AVX‑512BW) and aarch64 (NEON) behind the `SimdEngine`
//! trait, while preserving zero‑cost calls to architecture intrinsics.
//!
//! The abstraction is intentionally small and opinionated: all functions are
//! unsafe and operate on architecture‑specific vector types, but with a uniform
//! API and identical semantics across widths. This keeps call sites simple and
//! allows us to switch engines at runtime without littering ISA checks
//! throughout the codebase.
//!
//! ## Engines and widths
//!
//! The `SimdEngine` trait provides a unified interface for different SIMD
//! widths and corresponding lane counts:
//! - `SimdEngine128`: 128‑bit vectors — 16 lanes of i8 / 8 lanes of i16
//!   (SSE on x86_64; NEON on aarch64)
//! - `SimdEngine256`: 256‑bit vectors — 32 lanes of i8 / 16 lanes of i16
//!   (AVX2 on x86_64)
//! - `SimdEngine512`: 512‑bit vectors — 64 lanes of i8 / 32 lanes of i16
//!   (AVX‑512BW on x86_64)
//!
//! Each engine implements the same trait and can thus be selected via runtime
//! dispatch.
//!
//! ## Runtime dispatch pattern
//!
//! We use feature detection once at program start and keep the chosen engine in
//! a lightweight enum. Performance‑critical paths switch on that enum to call
//! the best available implementation.
//!
//! 1) Detect features: `detect_optimal_simd_engine()`
//! 2) Size batches: `get_simd_batch_sizes(engine)`
//! 3) Dispatch: `match engine { .. }` selects the implementation
//!
//! Key call sites include:
//! - `alignment/banded_swa.rs`: `simd_banded_swa_dispatch()`
//! - `alignment/align.rs`: `determine_optimal_batch_size()`
//! - `alignment/mem.rs` (logging / diagnostics)
//!
//! This pattern keeps binaries portable while maximizing performance on newer
//! CPUs.
//!
//! ## Safety model
//!
//! All trait functions are `unsafe` because they may:
//! - require specific CPU features (e.g. AVX2, AVX‑512BW),
//! - dereference raw pointers for loads/stores, and
//! - assume certain alignment constraints on pointers.
//!
//! Callers must ensure that the chosen `SimdEngine` implementation matches the
//! CPU’s supported features (handled by runtime detection in this crate), and
//! that any pointer arguments are valid for the required size and alignment.

/// Trait for a generic SIMD engine, providing an interface for common SIMD operations.
///
/// This trait allows different SIMD instruction sets (SSE, AVX2, AVX-512, NEON)
/// to be used interchangeably through a common API.
pub trait SimdEngine: Sized + Copy {
    /// Number of 8‑bit lanes in the engine’s native vector type.
    const WIDTH_8: usize;
    /// Number of 16‑bit lanes in the engine’s native vector type.
    const WIDTH_16: usize;

    /// Architecture‑specific 8‑bit vector type used by this engine.
    type Vec8: Copy + Clone;
    /// Architecture‑specific 16‑bit vector type used by this engine.
    type Vec16: Copy + Clone;

    // ===== Creation and Initialization =====
    /// Set all 8‑bit lanes to zero.
    ///
    /// Safety: requires the engine to be used on a matching CPU feature set.
    unsafe fn setzero_vec8() -> Self::Vec8;
    /// Alias of `setzero_vec8` for 8‑bit vectors.
    unsafe fn setzero_epi8() -> Self::Vec8;
    /// Broadcast a scalar i8 into all lanes.
    unsafe fn set1_epi8(a: i8) -> Self::Vec8;
    /// Broadcast a scalar i16 into all lanes.
    unsafe fn set1_epi16(a: i16) -> Self::Vec16;
    /// Broadcast a scalar i32 into 8‑bit vector lanes as 32‑bit elements
    /// (used for masks/constants).
    unsafe fn set1_epi32(a: i32) -> Self::Vec8;
    /// Set all 16‑bit lanes to zero.
    unsafe fn setzero_epi16() -> Self::Vec16;

    // ===== Extract and Movemask Operations =====
    /// Extract a single i8 lane by index (0‑based).
    ///
    /// Valid indices are `[0, WIDTH_8)`. Out‑of‑range values are UB.
    unsafe fn extract_epi8(a: Self::Vec8, imm8: i32) -> i8;
    /// Extract a single i16 lane by index (0‑based).
    ///
    /// Valid indices are `[0, WIDTH_16)`. Out‑of‑range values are UB.
    unsafe fn extract_epi16(a: Self::Vec16, imm8: i32) -> i16;
    /// Return a 32‑bit bitmask with the sign bit of each i8 lane.
    unsafe fn movemask_epi8(a: Self::Vec8) -> i32;
    /// Return a 16‑bit mask with the sign bit of each i16 lane (packed in a
    /// 32‑bit integer). Lane 0 corresponds to bit 0.
    unsafe fn movemask_epi16(a: Self::Vec16) -> i32;

    // ===== Variable Shift Operations =====
    /// Shift the vector left by `num_bytes` bytes within the 128/256/512‑bit lane.
    /// Bytes shifted out are zeroed. For `num_bytes >= 16`, the result is all zeros.
    ///
    /// Valid range: `0..=15` on x86 variable‑byte helpers; other values are
    /// defined to produce zeros for convenience.
    unsafe fn slli_bytes(a: Self::Vec8, num_bytes: i32) -> Self::Vec8;
    /// Same as `slli_bytes` but typed for 16‑bit lanes.
    unsafe fn slli_bytes_16(a: Self::Vec16, num_bytes: i32) -> Self::Vec16;
    /// Shift the vector right by `num_bytes` bytes.
    unsafe fn srli_bytes(a: Self::Vec8, num_bytes: i32) -> Self::Vec8;
    /// Same as `srli_bytes` but typed for 16‑bit lanes.
    unsafe fn srli_bytes_16(a: Self::Vec16, num_bytes: i32) -> Self::Vec16;
    /// Concatenate `b|a`, then extract a 16‑byte window starting at
    /// `num_bytes`. Semantics match x86 `alignr`.
    unsafe fn alignr_bytes(a: Self::Vec8, b: Self::Vec8, num_bytes: i32) -> Self::Vec8;
    /// 16‑bit typed variant of `alignr_bytes`.
    unsafe fn alignr_bytes_16(a: Self::Vec16, b: Self::Vec16, num_bytes: i32) -> Self::Vec16;

    // ===== 8-bit Integer Arithmetic =====
    /// Per‑lane add of i8 lanes (wrapping semantics as per the underlying ISA intrinsic).
    unsafe fn add_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8;
    /// Per‑lane subtract of i8 lanes (wrapping semantics).
    unsafe fn sub_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8;
    /// Per‑lane saturated subtract of unsigned i8 lanes.
    unsafe fn subs_epu8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8;
    /// Per‑lane saturated subtract of signed i8 lanes.
    unsafe fn subs_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8;
    /// Per‑lane saturated add of signed i8 lanes.
    unsafe fn adds_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8;
    /// Per‑lane saturated add of unsigned i8 lanes.
    unsafe fn adds_epu8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8;
    /// Per‑lane max of signed i8 lanes.
    unsafe fn max_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8;
    /// Per‑lane max of unsigned i8 lanes.
    unsafe fn max_epu8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8;
    /// Per‑lane min of unsigned i8 lanes.
    unsafe fn min_epu8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8;
    /// Per‑lane min of signed i8 lanes.
    unsafe fn min_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8;

    // ===== 16-bit Integer Arithmetic =====
    /// Per‑lane add of i16 lanes (wrapping semantics).
    unsafe fn add_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16;
    /// Per‑lane subtract of i16 lanes (wrapping semantics).
    unsafe fn sub_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16;
    /// Per‑lane saturated add of signed i16 lanes.
    unsafe fn adds_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16;
    /// Per‑lane saturated subtract of unsigned i16 lanes.
    unsafe fn subs_epu16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16;
    /// Per‑lane saturated subtract of signed i16 lanes.
    unsafe fn subs_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16;
    /// Per‑lane max of signed i16 lanes.
    unsafe fn max_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16;
    /// Per‑lane min of signed i16 lanes.
    unsafe fn min_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16;
    /// Per‑lane max of unsigned i16 lanes.
    unsafe fn max_epu16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16;

    // ===== Comparison Operations =====
    /// Compare equal on i8 lanes; result is an all‑ones/-zeros mask vector.
    unsafe fn cmpeq_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8;
    /// Compare greater‑than on signed i8 lanes; result is an all‑ones/-zeros mask vector.
    unsafe fn cmpgt_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8;
    /// Compare greater‑than on unsigned i8 lanes; result is an all‑ones/-zeros mask vector.
    unsafe fn cmpgt_epu8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8;
    /// Compare greater‑or‑equal on unsigned i8 lanes; result is an all‑ones/-zeros mask vector.
    unsafe fn cmpge_epu8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8;
    /// Compare equal on i16 lanes; result is an all‑ones/-zeros mask vector.
    unsafe fn cmpeq_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16;
    /// Compare greater‑than on signed i16 lanes; result is an all‑ones/-zeros mask vector.
    unsafe fn cmpgt_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16;

    // ===== Blend/Select Operations =====
    /// Select per‑byte from `a` or `b` using the high bit of `mask` bytes.
    unsafe fn blendv_epi8(a: Self::Vec8, b: Self::Vec8, mask: Self::Vec8) -> Self::Vec8;

    // ===== Bitwise Operations =====
    /// Per‑byte bitwise AND.
    unsafe fn and_si128(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8;
    /// Per‑byte bitwise OR.
    unsafe fn or_si128(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8;
    /// Per‑byte bitwise XOR.
    unsafe fn xor_si128(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8;
    /// Per‑byte bitwise AND‑NOT: `~a & b` on a per‑byte basis.
    unsafe fn andnot_si128(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8;

    // ===== Shuffle Operations =====
    /// Shuffle bytes in `a` using indices in `b`.
    /// Maps to `_mm_shuffle_epi8` (SSSE3), `_mm256_shuffle_epi8` (AVX2), `_mm512_shuffle_epi8` (AVX-512).
    unsafe fn shuffle_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8;

    // ===== Unpack Operations =====
    /// Interleave low 8 bytes of `a` and `b`.
    /// Result: [a0,b0,a1,b1,a2,b2,a3,b3,a4,b4,a5,b5,a6,b6,a7,b7]
    unsafe fn unpacklo_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8;
    /// Interleave high 8 bytes of `a` and `b`.
    /// Result: [a8,b8,a9,b9,a10,b10,a11,b11,a12,b12,a13,b13,a14,b14,a15,b15]
    unsafe fn unpackhi_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8;

    // ===== Fixed/portable Shift Operations =====
    /// Shift each i16 lane left by a fixed immediate `imm8` (backend may accept
    /// only certain ranges; out‑of‑range immediates are backend‑defined).
    unsafe fn slli_epi16(a: Self::Vec16, imm8: i32) -> Self::Vec16;
    /// Shift the 128‑bit register right by 16 bytes (drop low lane) — used in
    /// some algorithms as a fast lane rotate/advance.
    unsafe fn srli_si128_fixed(a: Self::Vec8) -> Self::Vec8;
    /// Concatenate `b|a` and extract the upper 16 bytes — fixed form used in
    /// a few hot loops where the offset is constant 16.
    unsafe fn alignr_epi8(a: Self::Vec8, b: Self::Vec8) -> Self::Vec8;

    // ===== Memory Operations =====
    /// Load a vector from an aligned pointer.
    /// Caller must ensure the pointer is valid and properly aligned for the
    /// backend vector type.
    unsafe fn load_si128(p: *const Self::Vec8) -> Self::Vec8;
    /// 16‑bit typed variant of `load_si128`.
    unsafe fn load_si128_16(p: *const Self::Vec16) -> Self::Vec16;
    /// Store a vector to an aligned pointer.
    unsafe fn store_si128(p: *mut Self::Vec8, a: Self::Vec8);
    /// 16‑bit typed variant of `store_si128`.
    unsafe fn store_si128_16(p: *mut Self::Vec16, a: Self::Vec16);
    /// Load a vector from an unaligned pointer (safe for any alignment, but may
    /// be slower on some ISAs).
    unsafe fn loadu_si128(p: *const Self::Vec8) -> Self::Vec8;
    /// 16‑bit typed variant of `loadu_si128`.
    unsafe fn loadu_si128_16(p: *const Self::Vec16) -> Self::Vec16;
    /// Store a vector to an unaligned pointer.
    unsafe fn storeu_si128(p: *mut Self::Vec8, a: Self::Vec8);
    /// 16‑bit typed variant of `storeu_si128`.
    unsafe fn storeu_si128_16(p: *mut Self::Vec16, a: Self::Vec16);
}

pub mod engine128;
pub mod engine256;
pub mod engine512;
pub mod portable_intrinsics;
pub mod simd;
pub mod tests;
pub mod types;

// Re-exports for internal module use (may be used by submodules or tests)
#[allow(unused_imports)]
use portable_intrinsics::*;
#[allow(unused_imports)]
use types::__m128i;

// Re-export the SimdEngine implementations
pub use engine128::SimdEngine128;
#[cfg(target_arch = "x86_64")]
pub use engine256::SimdEngine256;
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub use engine512::SimdEngine512;
