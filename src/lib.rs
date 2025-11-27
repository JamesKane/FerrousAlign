// Enable unstable features for AVX-512 support (requires nightly Rust)
#![cfg_attr(feature = "avx512", feature(stdarch_x86_avx512))]
#![cfg_attr(feature = "avx512", feature(avx512_target_feature))]

// Core reusable components (reference-agnostic)
pub mod core;

// Alignment pipelines (reference-specific)
pub mod pipelines;

// Shared defaults
pub mod defaults;

// Re-exports for backwards compatibility
// These allow existing code to use the old paths while we transition
pub use core::alignment;
pub use core::compute;
pub use core::io;
pub use core::utils;
pub use pipelines::linear::index;

// Test modules
#[cfg(test)]
#[path = "core/io/fastq_reader_test.rs"]
mod fastq_reader_test;

// Note: SAIS implementation removed - we use the `bio` crate's suffix array construction instead
