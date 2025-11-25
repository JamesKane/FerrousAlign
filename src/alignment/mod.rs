pub mod single_end;
pub mod paired;
pub mod banded_swa;
pub mod chaining;
pub mod extension;
pub mod finalization;
pub mod ksw_affine_gap;
pub mod pipeline;
pub mod region;
pub mod seeding;
pub mod utils;
pub mod mem;
pub mod mem_opt;

// AVX2-specific SIMD implementations (x86_64 only)
#[cfg(target_arch = "x86_64")]
pub mod banded_swa_avx2;

// AVX-512-specific SIMD implementations (x86_64 only, requires avx512 feature flag)
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub mod banded_swa_avx512;
