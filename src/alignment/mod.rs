pub mod banded_swa;
pub mod chaining;
pub mod cigar;
pub mod coordinates;
pub mod edit_distance;
pub mod finalization;
pub mod ksw_affine_gap;
pub mod kswv_batch;  // Horizontal SIMD batching infrastructure
pub mod kswv_sse_neon;  // Baseline 128-bit horizontal SIMD kernel
pub mod mem;
pub mod mem_opt;
pub mod paired;
pub mod pipeline;
pub mod region;
pub mod seeding;
pub mod single_end;
pub mod utils;

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
