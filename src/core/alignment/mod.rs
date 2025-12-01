//! Core alignment kernels - SIMD-accelerated Smith-Waterman implementations.
//!
//! These modules provide the computational heart of the aligner and are
//! agnostic to reference structure. They can be used for aligning reads
//! to paths extracted from graphs or linear references.

pub mod banded_swa;
pub mod cigar;
pub mod edit_distance;
pub mod ksw_affine_gap;
pub mod kswv_batch; // Horizontal SIMD batching infrastructure
pub mod kswv_sse_neon; // Baseline 128-bit horizontal SIMD kernel
pub mod kswv; // kswv shared macros/adapters
pub mod utils;
pub mod workspace; // Thread-local buffer pools for allocation reuse
pub mod shared_types; // Shared SoA carriers, config bundles, and arena traits

// AVX2-specific SIMD implementations (x86_64 only)
#[cfg(target_arch = "x86_64")]
pub mod kswv_avx2;

// AVX-512-specific SIMD implementations (x86_64 only, requires avx512 feature flag)
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub mod kswv_avx512;
