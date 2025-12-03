//! Alignment region module.
//!
//! This module implements the "alignment region" concept from BWA-MEM2, which
//! stores alignment boundaries (rb, re, qb, qe) WITHOUT generating CIGAR.
//!
//! ## BWA-MEM2 Reference
//!
//! This is the Rust equivalent of C++ `mem_alnreg_t` (bwamem.h:137-158):
//! - Extension phase computes boundaries via SIMD batch scoring
//! - CIGAR is generated LATER only for surviving alignments (~10-20%)
//! - This architecture eliminates ~80-90% of CIGAR computation
//!
//! ## Module Organization
//!
//! - `types` - Core data structures (`AlignmentRegion`, `ScoreOnlyExtensionResult`)
//! - `extension` - Chain extension to regions (`extend_chains_to_regions`)
//! - `merge` - Score merging for cross-read batching
//! - `cigar` - CIGAR regeneration from boundaries
//!
//! ## Heterogeneous Compute Integration
//!
//! The `extend_chains_to_regions()` function includes dispatch points for:
//! - CPU SIMD (active): SSE/AVX2/AVX-512/NEON batch scoring
//! - GPU (placeholder): Metal/CUDA/ROCm kernel dispatch
//! - NPU (placeholder): ANE/ONNX accelerated seed filtering

mod cigar;
mod extension;
mod merge;
mod types;

// Re-export types
pub use types::{
    AlignmentRegion, ChainExtensionMapping, ScoreOnlyExtensionResult, SeedExtensionMapping,
};

// Re-export extension functions
pub use extension::extend_chains_to_regions;

// Re-export merge functions
pub use merge::merge_extension_scores_to_regions;

// Re-export CIGAR functions
pub use cigar::generate_cigar_from_region;
