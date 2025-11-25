// Enable unstable features for AVX-512 support (requires nightly Rust)
#![cfg_attr(feature = "avx512", feature(stdarch_x86_avx512))]
#![cfg_attr(feature = "avx512", feature(avx512_target_feature))]

pub mod alignment;
pub mod compute; // Heterogeneous compute abstraction (CPU SIMD/GPU/NPU integration points)
pub mod defaults;
pub mod utils;

// New modules
pub mod io;
pub mod index;

// Test modules
#[cfg(test)]
#[path = "io/fastq_reader_test.rs"]
mod fastq_reader_test;

// Note: SAIS implementation removed - we use the `bio` crate's suffix array construction instead
