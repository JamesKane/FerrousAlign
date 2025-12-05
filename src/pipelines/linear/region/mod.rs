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
//! - `merge` - Score merging for cross-read batching
//! - `cigar` - CIGAR regeneration from boundaries
//!
//! ## Active Extension Pipeline (v0.8.0+)
//!
//! Extension is now handled by the unified SoA pipeline:
//! - `batch_extension/collect_soa.rs` - Job collection from SoA chains
//! - `batch_extension/dispatch.rs` - SIMD batch scoring dispatch
//! - `batch_extension/finalize_soa.rs` - Alignment record generation
//! - `stages/extension/mod.rs` - Stage wrapper for pipeline integration

mod cigar;
mod merge;
mod types;

// Re-export types (still used by batch_extension)
pub use types::{
    AlignmentRegion, ChainExtensionMapping, ScoreOnlyExtensionResult, SeedExtensionMapping,
};

// Re-export merge functions (still used by batch_extension)
pub use merge::merge_extension_scores_to_regions;

// Re-export CIGAR functions (still used by batch_extension)
pub use cigar::generate_cigar_from_region;
