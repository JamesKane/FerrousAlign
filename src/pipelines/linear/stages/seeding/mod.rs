//! Stage 1: SMEM extraction (Seeding)
//!
//! Generates Super-Maximal Exact Matches (SMEMs) from read sequences using
//! FM-Index backward search. This is the first computational stage after loading.
//!
//! # Algorithm Overview
//!
//! The seeding stage implements BWA-MEM2's multi-pass seeding strategy:
//!
//! 1. **Pass 1 (Initial SMEMs)**: Generate supermaximal exact matches using
//!    bidirectional FM-Index search on the original query sequence.
//!
//! 2. **Pass 2 (Re-seeding)**: For long unique SMEMs (length >= split_len,
//!    interval_size <= split_width), re-seed from the middle position to
//!    detect chimeric reads and split alignments.
//!
//! 3. **Pass 3 (Forward-only)**: When `max_mem_intv > 0`, run a simpler
//!    forward-only seeding strategy to find seeds missed by the supermaximal
//!    algorithm in highly repetitive regions.
//!
//! # Data Flow
//!
//! ```text
//! Input:  SoAReadBatch (sequences, qualities, names)
//! Output: (SoASeedBatch, SoAEncodedQueryBatch, SoAEncodedQueryBatch)
//!         Seeds + encoded queries (forward and reverse complement)
//! ```
//!
//! # Implementation Notes
//!
//! This module wraps the existing `find_seeds_batch()` function from
//! `pipelines::linear::seeding`. The actual implementation is preserved
//! to maintain bit-exact compatibility with existing behavior.
//!
//! Future work may split the implementation into submodules:
//! - `smem.rs` - Core SMEM algorithm
//! - `reseeding.rs` - Chimeric detection and re-seeding
//! - `forward_only.rs` - 3rd round seeding strategy
//! - `sa_lookup.rs` - Suffix array lookup utilities

pub use crate::core::io::soa_readers::SoAReadBatch;
use crate::pipelines::linear::seeding::{SoAEncodedQueryBatch, SoASeedBatch, find_seeds_batch};

use super::{PipelineStage, StageContext, StageError};

// Re-export types for downstream consumers
pub use crate::pipelines::linear::seeding::{
    SoAEncodedQueryBatch as EncodedQueryBatch, SoASeedBatch as SeedBatch,
};

/// Output from the seeding stage.
///
/// Contains seeds and encoded query sequences needed for downstream stages.
#[derive(Debug, Clone, Default)]
pub struct SeedingOutput {
    /// Seeds in SoA format with per-read boundaries
    pub seeds: SoASeedBatch,

    /// Encoded query sequences (forward strand)
    pub encoded_queries: SoAEncodedQueryBatch,

    /// Encoded query sequences (reverse complement)
    pub encoded_queries_rc: SoAEncodedQueryBatch,
}

impl SeedingOutput {
    /// Create a new empty seeding output.
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if the output is empty (no seeds generated).
    pub fn is_empty(&self) -> bool {
        self.seeds.query_pos.is_empty()
    }

    /// Get the number of seeds.
    pub fn num_seeds(&self) -> usize {
        self.seeds.query_pos.len()
    }

    /// Get the number of reads that were processed.
    pub fn num_reads(&self) -> usize {
        self.seeds.read_seed_boundaries.len()
    }
}

/// Seeding stage implementation.
///
/// Wraps `find_seeds_batch()` with the `PipelineStage` interface.
/// This provides a clean abstraction while maintaining compatibility
/// with the existing implementation.
#[derive(Debug, Clone, Copy, Default)]
pub struct SeedingStage;

impl SeedingStage {
    /// Create a new seeding stage.
    pub fn new() -> Self {
        Self
    }
}

impl PipelineStage for SeedingStage {
    type Input = SoAReadBatch;
    type Output = SeedingOutput;

    fn process(&self, input: Self::Input, ctx: &StageContext) -> Result<Self::Output, StageError> {
        // Validate input
        self.validate(&input)?;

        // Handle empty batch
        if input.is_empty() {
            return Ok(SeedingOutput::new());
        }

        // Delegate to existing implementation
        let (seeds, encoded_queries, encoded_queries_rc) =
            find_seeds_batch(ctx.index, &input, ctx.options);

        Ok(SeedingOutput {
            seeds,
            encoded_queries,
            encoded_queries_rc,
        })
    }

    fn name(&self) -> &'static str {
        "Seeding"
    }

    fn validate(&self, input: &Self::Input) -> Result<(), StageError> {
        // Check for reasonable input size
        if input.len() > 10_000_000 {
            return Err(StageError::ValidationFailed(format!(
                "Batch size {} exceeds maximum allowed (10M reads)",
                input.len()
            )));
        }

        Ok(())
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seeding_stage_name() {
        let stage = SeedingStage::new();
        assert_eq!(stage.name(), "Seeding");
    }

    #[test]
    fn test_seeding_output_empty() {
        let output = SeedingOutput::new();
        assert!(output.is_empty());
        assert_eq!(output.num_seeds(), 0);
        assert_eq!(output.num_reads(), 0);
    }

    #[test]
    fn test_seeding_stage_validation_empty_batch() {
        let stage = SeedingStage::new();
        let empty_batch = SoAReadBatch::new();

        // Empty batch should pass validation (will be handled in process)
        assert!(stage.validate(&empty_batch).is_ok());
    }
}
