//! Stage 3: Smith-Waterman extension
//!
//! Extends seed chains into full alignments using banded Smith-Waterman.
//! This is the most compute-intensive stage and benefits heavily from SIMD.
//!
//! # Algorithm Overview
//!
//! The extension stage processes kept chains through:
//! 1. **Job Collection**: Extract left/right extension segments from chains
//! 2. **SIMD Scoring**: Batch banded Smith-Waterman across all extension jobs
//! 3. **Score Integration**: Merge left/right scores back to chains
//! 4. **Alignment Generation**: Convert scored regions to alignment records
//!
//! # Data Flow
//!
//! ```text
//! Input:  ChainingOutput (chains + seeds + encoded queries)
//! Output: SoAAlignmentResult (finalized alignments in SoA format)
//! ```
//!
//! # SIMD Engine Selection
//!
//! The stage uses runtime CPU detection to select the optimal SIMD engine:
//! - **128-bit**: SSE2 (x86_64), NEON (aarch64) - 16 lanes
//! - **256-bit**: AVX2 (x86_64) - 32 lanes
//! - **512-bit**: AVX-512 (feature-gated) - 64 lanes
//!
//! # Implementation Notes
//!
//! This module wraps the existing functions from `pipelines::linear::batch_extension`:
//! - `collect_extension_jobs_batch_soa()` - Job collection from SoA chains
//! - `execute_batch_simd_scoring()` - SIMD Smith-Waterman dispatch
//! - `finalize_alignments_soa()` - Alignment record generation

use std::sync::Arc;

use super::{PipelineStage, StageContext, StageError};

// Re-export key types
pub use crate::pipelines::linear::batch_extension::types::SoAAlignmentResult;

// Import functions we'll wrap
use crate::core::alignment::banded_swa::BandedPairWiseSW;
use crate::core::compute::simd_abstraction::simd::detect_optimal_simd_engine;
use crate::pipelines::linear::batch_extension::types::{
    ExtensionJobBatch, SoAReadExtensionContext,
};
use crate::pipelines::linear::batch_extension::{
    collect_extension_jobs_batch_soa, execute_batch_simd_scoring, finalize_alignments_soa,
};

// Import chaining output for input type
use super::chaining::ChainingOutput;

/// Output from the extension stage.
///
/// Contains SoA alignment results ready for finalization stage.
pub type ExtensionOutput = SoAAlignmentResult;

/// Input to the extension stage.
pub type ExtensionInput = ChainingOutput;

/// Extension stage implementation.
///
/// Converts chains into alignments via batched Smith-Waterman:
/// 1. Build extension context from chaining output
/// 2. Collect left/right extension jobs
/// 3. Execute SIMD scoring
/// 4. Finalize alignment records
///
/// This wraps the SoA-native extension pipeline with the `PipelineStage` interface.
#[derive(Debug, Clone, Copy, Default)]
pub struct ExtensionStage;

impl ExtensionStage {
    /// Create a new extension stage.
    pub fn new() -> Self {
        Self
    }
}

impl PipelineStage for ExtensionStage {
    type Input = ExtensionInput;
    type Output = ExtensionOutput;

    fn process(&self, input: Self::Input, ctx: &StageContext) -> Result<Self::Output, StageError> {
        // Handle empty input
        if input.is_empty() {
            return Ok(SoAAlignmentResult::new());
        }

        let batch_size = input.num_reads();
        let opt = ctx.options;
        let bwa_idx = ctx.index;
        let pac_data: &[u8] = &bwa_idx.bns.pac_data;

        // Build SoAReadExtensionContext from ChainingOutput
        let mut soa_context = SoAReadExtensionContext {
            read_boundaries: input.seeds.read_seed_boundaries.clone(),
            query_names: Vec::new(), // Not available from chaining output, will be set by finalization
            query_lengths: input.query_lengths.clone(),
            encoded_queries: input.encoded_queries.encoded_seqs,
            encoded_query_boundaries: input.encoded_queries.query_boundaries,
            encoded_queries_rc: input.encoded_queries_rc.encoded_seqs,
            encoded_query_rc_boundaries: input.encoded_queries_rc.query_boundaries,
            soa_seed_batch: input.seeds,
            soa_chain_batch: input.chains,
            chain_ref_segments: Vec::new(), // Will be populated during job collection
        };

        // Phase 1: Collect extension jobs
        let mut left_batch = ExtensionJobBatch::with_capacity(batch_size * 10, batch_size * 1024);
        let mut right_batch = ExtensionJobBatch::with_capacity(batch_size * 10, batch_size * 1024);

        let mappings = collect_extension_jobs_batch_soa(
            bwa_idx,
            opt,
            &mut soa_context,
            &mut left_batch,
            &mut right_batch,
        );

        log::debug!(
            "ExtensionStage: Collected {} left jobs, {} right jobs",
            left_batch.len(),
            right_batch.len()
        );

        // Phase 2: Execute SIMD scoring
        let sw_params = BandedPairWiseSW::new(
            opt.o_del,
            opt.e_del,
            opt.o_ins,
            opt.e_ins,
            opt.zdrop,
            5,
            opt.pen_clip5,
            opt.pen_clip3,
            opt.mat,
            opt.a as i8,
            -(opt.b as i8),
        );

        // Get SIMD engine from compute context or detect optimal
        let engine = ctx
            .compute_ctx
            .backend
            .simd_engine()
            .unwrap_or_else(detect_optimal_simd_engine);
        let left_results = execute_batch_simd_scoring(&sw_params, &mut left_batch, engine);
        let right_results = execute_batch_simd_scoring(&sw_params, &mut right_batch, engine);

        log::debug!(
            "ExtensionStage: SIMD scoring complete: {} left, {} right results",
            left_results.len(),
            right_results.len()
        );

        // Phase 3: Distribute results to per-read score vectors
        let per_read_left_scores =
            convert_batch_results_to_outscores(&left_results, &left_batch, batch_size);
        let per_read_right_scores =
            convert_batch_results_to_outscores(&right_results, &right_batch, batch_size);

        // Phase 4: Finalize alignments
        // finalize_alignments_soa expects Arc<&T> references for thread-safety in parallel processing
        let bwa_idx_arc = Arc::new(bwa_idx);
        let pac_data_arc = Arc::new(pac_data);
        let opt_arc = Arc::new(opt);
        let batch_start_id = ctx.batch_id;

        let alignments = finalize_alignments_soa(
            &bwa_idx_arc,
            &pac_data_arc,
            &opt_arc,
            &soa_context,
            mappings,
            per_read_left_scores,
            per_read_right_scores,
            batch_start_id,
        );

        log::debug!(
            "ExtensionStage: Finalization complete, {} reads with {} alignments",
            alignments.num_reads(),
            alignments.len()
        );

        Ok(alignments)
    }

    fn name(&self) -> &'static str {
        "Extension"
    }

    fn validate(&self, input: &Self::Input) -> Result<(), StageError> {
        // Check for reasonable input size
        if input.num_chains() > 50_000_000 {
            return Err(StageError::ValidationFailed(format!(
                "Chain count {} exceeds maximum allowed (50M chains)",
                input.num_chains()
            )));
        }

        Ok(())
    }
}

/// Convert batch extension results to per-read OutScore vectors.
///
/// This helper distributes results back to per-read vectors for finalization.
fn convert_batch_results_to_outscores(
    results: &[crate::pipelines::linear::batch_extension::types::BatchExtensionResult],
    batch: &ExtensionJobBatch,
    num_reads: usize,
) -> Vec<Vec<crate::core::alignment::banded_swa::OutScore>> {
    use crate::core::alignment::banded_swa::OutScore;

    // Pre-allocate per-read result vectors
    let mut per_read_scores: Vec<Vec<OutScore>> = vec![Vec::new(); num_reads];

    // Determine size of each read's score vector from the batch jobs
    let mut read_job_counts: Vec<usize> = vec![0; num_reads];
    for job in &batch.jobs {
        read_job_counts[job.read_idx] += 1;
    }

    for (read_idx, count) in read_job_counts.iter().enumerate() {
        per_read_scores[read_idx].reserve(*count);
    }

    // Distribute results back to per-read vectors
    for result in results {
        per_read_scores[result.read_idx].push(OutScore {
            score: result.score,
            query_end_pos: result.query_end,
            target_end_pos: result.ref_end,
            global_score: result.gscore,
            gtarget_end_pos: result.gref_end,
            max_offset: result.max_off,
        });
    }

    per_read_scores
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extension_stage_name() {
        let stage = ExtensionStage::new();
        assert_eq!(stage.name(), "Extension");
    }

    #[test]
    fn test_extension_output_empty() {
        let output = SoAAlignmentResult::new();
        assert!(output.is_empty());
        assert_eq!(output.len(), 0);
        assert_eq!(output.num_reads(), 0);
    }

    #[test]
    fn test_extension_output_with_capacity() {
        let output = SoAAlignmentResult::with_capacity(100, 10);
        assert!(output.is_empty());
        // Capacity is pre-allocated but length is 0
        assert_eq!(output.len(), 0);
    }
}
