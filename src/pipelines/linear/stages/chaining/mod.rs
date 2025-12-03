//! Stage 2: Seed chaining
//!
//! Groups compatible seeds into chains using O(n²) dynamic programming.
//! Chains represent candidate alignment regions for extension.
//!
//! # Algorithm Overview
//!
//! The chaining algorithm builds chains from seeds by:
//! 1. Sorting seeds by query position
//! 2. Using a B-tree for efficient chain lookup by reference position
//! 3. Testing seed-chain compatibility (gap penalties, strand matching)
//! 4. Merging compatible seeds into existing chains or creating new ones
//!
//! After chain construction, filtering is applied:
//! - Weight calculation based on seed coverage
//! - Removal of low-quality and redundant chains
//! - Keeping only chains likely to produce good alignments
//!
//! # Data Flow
//!
//! ```text
//! Input:  SeedingOutput (SoASeedBatch + encoded queries)
//! Output: ChainingOutput (SoAChainBatch + context for extension)
//! ```
//!
//! # Implementation Notes
//!
//! This module wraps the existing functions from `pipelines::linear::chaining`:
//! - `chain_seeds_batch()` - B-tree based seed chaining
//! - `filter_chains_batch()` - Chain filtering and weight calculation

use super::{PipelineStage, StageContext, StageError};

// Re-export key types
pub use crate::pipelines::linear::chaining::{Chain, SoAChainBatch};
pub use crate::pipelines::linear::seeding::{SoAEncodedQueryBatch, SoASeedBatch};

// Import functions we'll wrap
use crate::pipelines::linear::chaining::{chain_seeds_batch, filter_chains_batch};

// Import seeding output for input type
use super::seeding::SeedingOutput;

/// Output from the chaining stage.
///
/// Contains chains and all context needed for the extension stage.
#[derive(Debug, Clone, Default)]
pub struct ChainingOutput {
    /// Chains in SoA format with per-read boundaries
    pub chains: SoAChainBatch,

    /// Seeds (passed through from seeding stage)
    pub seeds: SoASeedBatch,

    /// Encoded query sequences (forward strand)
    pub encoded_queries: SoAEncodedQueryBatch,

    /// Encoded query sequences (reverse complement)
    pub encoded_queries_rc: SoAEncodedQueryBatch,

    /// Query lengths per read
    pub query_lengths: Vec<i32>,
}

impl ChainingOutput {
    /// Create a new empty chaining output.
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if the output is empty (no chains generated).
    pub fn is_empty(&self) -> bool {
        self.chains.score.is_empty()
    }

    /// Get the total number of chains.
    pub fn num_chains(&self) -> usize {
        self.chains.score.len()
    }

    /// Get the number of reads.
    pub fn num_reads(&self) -> usize {
        self.chains.read_chain_boundaries.len()
    }

    /// Get the number of kept chains (after filtering).
    pub fn num_kept_chains(&self) -> usize {
        self.chains.kept.iter().filter(|&&k| k > 0).count()
    }

    /// Get the number of seeds.
    pub fn num_seeds(&self) -> usize {
        self.seeds.query_pos.len()
    }
}

/// Input to the chaining stage.
pub type ChainingInput = SeedingOutput;

/// Chaining stage implementation.
///
/// Converts seeds into chains via O(n²) dynamic programming:
/// 1. B-tree based chain construction from sorted seeds
/// 2. Chain weight calculation (seed coverage)
/// 3. Chain filtering (remove low-quality, redundant chains)
///
/// This wraps `chain_seeds_batch()` and `filter_chains_batch()`
/// with the `PipelineStage` interface.
#[derive(Debug, Clone, Copy, Default)]
pub struct ChainingStage;

impl ChainingStage {
    /// Create a new chaining stage.
    pub fn new() -> Self {
        Self
    }
}

impl PipelineStage for ChainingStage {
    type Input = ChainingInput;
    type Output = ChainingOutput;

    fn process(&self, input: Self::Input, ctx: &StageContext) -> Result<Self::Output, StageError> {
        // Handle empty input
        if input.is_empty() {
            return Ok(ChainingOutput::new());
        }

        let l_pac = ctx.index.bns.packed_sequence_length;

        // Step 1: Chain seeds using B-tree algorithm
        let mut chains = chain_seeds_batch(&input.seeds, ctx.options, l_pac);

        log::debug!(
            "ChainingStage: Generated {} chains from {} seeds",
            chains.score.len(),
            input.seeds.query_pos.len()
        );

        // Step 2: Calculate query lengths for filtering
        let query_lengths: Vec<i32> = input
            .seeds
            .read_seed_boundaries
            .iter()
            .enumerate()
            .map(|(read_idx, _)| {
                // Get query length from encoded query boundaries
                if read_idx < input.encoded_queries.query_boundaries.len() {
                    input.encoded_queries.query_boundaries[read_idx].1 as i32
                } else {
                    0
                }
            })
            .collect();

        // Step 3: Filter chains
        filter_chains_batch(&mut chains, &input.seeds, ctx.options, &query_lengths);

        let kept_chains = chains.kept.iter().filter(|&&k| k > 0).count();
        log::debug!("ChainingStage: {} chains kept after filtering", kept_chains);

        Ok(ChainingOutput {
            chains,
            seeds: input.seeds,
            encoded_queries: input.encoded_queries,
            encoded_queries_rc: input.encoded_queries_rc,
            query_lengths,
        })
    }

    fn name(&self) -> &'static str {
        "Chaining"
    }

    fn validate(&self, input: &Self::Input) -> Result<(), StageError> {
        // Check for reasonable input size
        if input.num_seeds() > 100_000_000 {
            return Err(StageError::ValidationFailed(format!(
                "Seed count {} exceeds maximum allowed (100M seeds)",
                input.num_seeds()
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
    fn test_chaining_stage_name() {
        let stage = ChainingStage::new();
        assert_eq!(stage.name(), "Chaining");
    }

    #[test]
    fn test_chaining_output_empty() {
        let output = ChainingOutput::new();
        assert!(output.is_empty());
        assert_eq!(output.num_chains(), 0);
        assert_eq!(output.num_reads(), 0);
        assert_eq!(output.num_kept_chains(), 0);
        assert_eq!(output.num_seeds(), 0);
    }

    #[test]
    fn test_chaining_output_counts() {
        let mut output = ChainingOutput::new();

        // Add some mock chain data
        output.chains.score.push(100);
        output.chains.kept.push(3); // Primary
        output.chains.score.push(50);
        output.chains.kept.push(0); // Discarded
        output.chains.score.push(75);
        output.chains.kept.push(1); // Shadowed (still kept)

        output.chains.read_chain_boundaries.push((0, 3));

        assert_eq!(output.num_chains(), 3);
        assert_eq!(output.num_reads(), 1);
        assert_eq!(output.num_kept_chains(), 2); // Only kept > 0
    }
}
