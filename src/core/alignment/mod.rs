//! Core alignment kernels - SIMD-accelerated Smith-Waterman implementations.
//!
//! These modules provide the computational heart of the aligner and are
//! agnostic to reference structure. They can be used for aligning reads
//! to paths extracted from graphs or linear references.

pub mod banded_swa;
pub mod banded_swa_sse_neon;
pub mod types; // Shared alignment types (e.g., ExtensionDirection) — not re-exported yet to avoid name clashes // Baseline 128-bit vertical SIMD kernel (SSE/NEON)
// Note: batch_extension moved to pipelines::linear since it depends on linear-specific types
pub mod cigar;
pub mod edit_distance;
pub mod ksw_affine_gap;
pub mod kswv_batch; // Horizontal SIMD batching infrastructure
pub mod kswv_sse_neon; // Baseline 128-bit horizontal SIMD kernel
pub mod utils;
pub mod workspace; // Thread-local buffer pools for allocation reuse

// AVX2-specific SIMD implementations (x86_64 only)
#[cfg(target_arch = "x86_64")]
pub mod banded_swa_avx2;
#[cfg(target_arch = "x86_64")]
pub mod kswv_avx2;

// AVX-512-specific SIMD implementations (x86_64 only, requires avx512 feature flag)
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub mod banded_swa_avx512;
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub mod kswv_avx512;

// Shared helpers for banded SW (SoA transforms, padding, result packing, etc.)
// Introduced as part of duplication reduction; currently internal and optional.
pub mod banded_swa_shared;

// Generic kernel surface (trait + params) for shared DP implementation.
// This is currently compile-only (no callers yet) and will be adopted by
// per-ISA wrappers in subsequent steps.
pub mod banded_swa_kernel;
