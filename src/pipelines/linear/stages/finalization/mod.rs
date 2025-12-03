//! Stage 4: Alignment finalization
//!
//! Generates final alignment records including CIGAR strings, MD tags,
//! MAPQ scores, and SAM flags. This stage transforms raw alignment regions
//! into SAM-compatible `Alignment` structs.
//!
//! # Architecture: Hybrid AoS/SoA
//!
//! The finalization stage operates at the boundary between SoA (compute-heavy)
//! and AoS (logic-heavy) processing:
//!
//! ```text
//! Single-End:
//!   [SoA] Extension → [SoA] Finalization → [AoS] Secondary marking → [AoS] Output
//!
//! Paired-End:
//!   [SoA] Extension → [SoA] Finalization → [AoS] Pairing → [SoA] Mate Rescue → [AoS] Output
//! ```
//!
//! **Key Discovery (v0.7.0)**: Pure SoA processing for pairing causes 96% duplicate
//! reads due to index boundary corruption. The AoS conversion is mandatory for
//! correctness in paired-end processing.
//!
//! **Unified Approach**: Both SE and PE use the same hybrid path for consistency.
//! The ~2% overhead from AoS conversion is acceptable for code simplicity.
//!
//! # Responsibilities
//!
//! The finalization stage handles:
//! - **CIGAR generation**: Converting DP traceback into CIGAR strings (in extension)
//! - **MD tag computation**: Recording mismatches and deletions for variant calling
//! - **MAPQ calculation**: Estimating mapping quality based on alignment scores
//! - **Flag handling**: Setting SECONDARY, SUPPLEMENTARY, and other SAM flags
//! - **XA/SA tags**: Generating alternative alignment information
//! - **SoA↔AoS conversion**: Transitioning between data layouts
//!
//! # Data Flow
//!
//! ```text
//! Input:  SoAAlignmentResult (from extension stage)
//! Output: Vec<Vec<Alignment>> (AoS format, per-read alignment vectors)
//! ```
//!
//! # Implementation Notes
//!
//! This module wraps:
//! - `SoAAlignmentResult::to_aos()` - SoA to AoS conversion
//! - `mark_secondary_alignments()` - Post-processing (MAPQ, flags, tags)
//! - `remove_redundant_alignments()` - Deduplication
//!
//! The actual CIGAR/MD generation happens in the extension stage via
//! `finalize_alignments_soa()`. This stage focuses on the SoA→AoS transition
//! and subsequent per-read processing.

use super::{PipelineStage, StageContext, StageError};

// Re-export key types
pub use crate::pipelines::linear::batch_extension::types::SoAAlignmentResult;
pub use crate::pipelines::linear::finalization::{Alignment, sam_flags};

// Import functions we'll use
use crate::pipelines::linear::finalization::{
    mark_secondary_alignments, remove_redundant_alignments,
};

/// Output from the finalization stage.
///
/// Contains per-read alignment vectors in AoS format, ready for
/// pairing (PE) or direct output (SE).
#[derive(Debug, Clone, Default)]
pub struct FinalizationOutput {
    /// Per-read alignment vectors (outer vec = reads, inner vec = alignments per read)
    pub alignments_per_read: Vec<Vec<Alignment>>,
}

impl FinalizationOutput {
    /// Create a new empty finalization output.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create finalization output from per-read alignments.
    pub fn from_alignments(alignments_per_read: Vec<Vec<Alignment>>) -> Self {
        Self {
            alignments_per_read,
        }
    }

    /// Check if the output is empty (no reads processed).
    pub fn is_empty(&self) -> bool {
        self.alignments_per_read.is_empty()
    }

    /// Get the number of reads.
    pub fn num_reads(&self) -> usize {
        self.alignments_per_read.len()
    }

    /// Get the total number of alignments across all reads.
    pub fn total_alignments(&self) -> usize {
        self.alignments_per_read.iter().map(|v| v.len()).sum()
    }

    /// Get a flat iterator over all alignments.
    pub fn iter_alignments(&self) -> impl Iterator<Item = &Alignment> {
        self.alignments_per_read.iter().flatten()
    }

    /// Get the number of primary alignments.
    pub fn num_primary(&self) -> usize {
        self.iter_alignments()
            .filter(|a| a.flag & (sam_flags::SECONDARY | sam_flags::SUPPLEMENTARY) == 0)
            .count()
    }

    /// Get the number of secondary alignments.
    pub fn num_secondary(&self) -> usize {
        self.iter_alignments()
            .filter(|a| a.flag & sam_flags::SECONDARY != 0)
            .count()
    }

    /// Get the number of supplementary alignments.
    pub fn num_supplementary(&self) -> usize {
        self.iter_alignments()
            .filter(|a| a.flag & sam_flags::SUPPLEMENTARY != 0)
            .count()
    }

    /// Convert back to SoA format (for mate rescue or other SoA operations).
    pub fn to_soa(&self) -> SoAAlignmentResult {
        SoAAlignmentResult::from_aos(&self.alignments_per_read)
    }
}

/// Input to the finalization stage: SoA alignment results from extension.
pub type FinalizationInput = SoAAlignmentResult;

/// Finalization stage implementation.
///
/// Converts SoA alignment results to AoS format and applies post-processing:
/// 1. SoA → AoS conversion (per-read alignment vectors)
/// 2. Remove redundant alignments (deduplication)
/// 3. Mark secondary/supplementary alignments based on query overlap
/// 4. Calculate MAPQ for all alignments
/// 5. Generate and attach XA/SA tags
///
/// Both single-end and paired-end pipelines use this unified approach.
/// The ~2% overhead from AoS conversion is acceptable for code simplicity
/// and correctness (pure SoA pairing causes index corruption).
#[derive(Debug, Clone, Copy, Default)]
pub struct FinalizationStage {
    /// Whether to apply secondary/supplementary marking.
    /// Set to false if downstream stages (e.g., pairing) will handle this.
    apply_secondary_marking: bool,
}

impl FinalizationStage {
    /// Create a new finalization stage with secondary marking enabled.
    pub fn new() -> Self {
        Self {
            apply_secondary_marking: true,
        }
    }

    /// Create a finalization stage without secondary marking.
    ///
    /// Use this when downstream stages (e.g., paired-end pairing) will
    /// handle secondary/supplementary marking after additional processing.
    pub fn without_secondary_marking() -> Self {
        Self {
            apply_secondary_marking: false,
        }
    }

    /// Process SoA results to AoS with optional post-processing.
    fn process_internal(
        &self,
        soa_input: SoAAlignmentResult,
        opt: &crate::pipelines::linear::mem_opt::MemOpt,
    ) -> Vec<Vec<Alignment>> {
        // Step 1: Convert SoA → AoS (per-read alignment vectors)
        let mut alignments_per_read = soa_input.to_aos();

        if self.apply_secondary_marking {
            // Step 2-5: Apply post-processing to each read's alignments
            for alignments in alignments_per_read.iter_mut() {
                if !alignments.is_empty() {
                    // Remove redundant alignments
                    remove_redundant_alignments(alignments, opt);

                    // Mark secondary/supplementary and calculate MAPQ
                    // This also handles XA/SA tag generation
                    mark_secondary_alignments(alignments, opt);
                }
            }
        }

        alignments_per_read
    }
}

impl PipelineStage for FinalizationStage {
    type Input = FinalizationInput;
    type Output = FinalizationOutput;

    fn process(&self, input: Self::Input, ctx: &StageContext) -> Result<Self::Output, StageError> {
        // Handle empty input
        if input.num_reads() == 0 {
            return Ok(FinalizationOutput::new());
        }

        // Process SoA → AoS with post-processing
        let alignments_per_read = self.process_internal(input, ctx.options);

        Ok(FinalizationOutput::from_alignments(alignments_per_read))
    }

    fn name(&self) -> &'static str {
        "Finalization"
    }

    fn validate(&self, input: &Self::Input) -> Result<(), StageError> {
        // Check for reasonable input size
        let total_alignments = input.len();
        if total_alignments > 10_000_000 {
            return Err(StageError::ValidationFailed(format!(
                "Alignment count {} exceeds maximum allowed (10M alignments)",
                total_alignments
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
    fn test_finalization_stage_name() {
        let stage = FinalizationStage::new();
        assert_eq!(stage.name(), "Finalization");
    }

    #[test]
    fn test_finalization_stage_variants() {
        let with_marking = FinalizationStage::new();
        assert!(with_marking.apply_secondary_marking);

        let without_marking = FinalizationStage::without_secondary_marking();
        assert!(!without_marking.apply_secondary_marking);
    }

    #[test]
    fn test_finalization_output_empty() {
        let output = FinalizationOutput::new();
        assert!(output.is_empty());
        assert_eq!(output.num_reads(), 0);
        assert_eq!(output.total_alignments(), 0);
        assert_eq!(output.num_primary(), 0);
        assert_eq!(output.num_secondary(), 0);
        assert_eq!(output.num_supplementary(), 0);
    }

    #[test]
    fn test_finalization_output_from_alignments() {
        let alignments = vec![
            vec![create_test_alignment("read1", 0)],
            vec![
                create_test_alignment("read2", 0),
                create_test_alignment("read2", sam_flags::SECONDARY),
            ],
        ];

        let output = FinalizationOutput::from_alignments(alignments);
        assert_eq!(output.num_reads(), 2);
        assert_eq!(output.total_alignments(), 3);
        assert_eq!(output.num_primary(), 2);
        assert_eq!(output.num_secondary(), 1);
        assert_eq!(output.num_supplementary(), 0);
    }

    #[test]
    fn test_finalization_output_to_soa_roundtrip() {
        let alignments = vec![vec![create_test_alignment("read1", 0)]];

        let output = FinalizationOutput::from_alignments(alignments.clone());
        let soa = output.to_soa();

        assert_eq!(soa.num_reads(), 1);
        assert_eq!(soa.len(), 1);
    }

    /// Create a minimal test alignment
    fn create_test_alignment(name: &str, flags: u16) -> Alignment {
        Alignment {
            query_name: name.to_string(),
            flag: flags,
            ref_name: "chr1".to_string(),
            ref_id: 0,
            pos: 100,
            mapq: 60,
            score: 100,
            cigar: vec![(b'M', 100)],
            rnext: "*".to_string(),
            pnext: 0,
            tlen: 0,
            seq: String::new(),
            qual: String::new(),
            tags: Vec::new(),
            query_start: 0,
            query_end: 100,
            seed_coverage: 50,
            hash: 0,
            frac_rep: 0.0,
            is_alt: false,
        }
    }
}
