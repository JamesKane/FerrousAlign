//! Paired-end pipeline orchestrator
//!
//! Coordinates the execution of pipeline stages for paired-end reads.
//! Uses a hybrid AoS/SoA architecture as discovered during v0.7.0:
//!
//! # Pipeline Flow
//!
//! ```text
//! [R1/R2 SoAReadBatch] → Seeding → Chaining → Extension → [SoA]
//!                                                          ↓
//!                                                    [AoS] Pairing
//!                                                          ↓
//!                                                    [SoA] Mate Rescue
//!                                                          ↓
//!                                                    [AoS] Output
//! ```
//!
//! # Hybrid Architecture Rationale
//!
//! Pure SoA pairing causes 96% duplicate reads due to index boundary corruption.
//! The mandatory AoS conversion at the pairing stage maintains per-read alignment
//! boundaries correctly.
//!
//! # Usage
//!
//! ```ignore
//! let orchestrator = PairedEndOrchestrator::new(&index, &options, &compute_ctx);
//! let stats = orchestrator.run(&[r1_file, r2_file], &mut output)?;
//! ```

mod helpers;

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use super::{
    OrchestratorError, PipelineMode, PipelineOrchestrator, PipelineStatistics, PipelineTimer,
};
use crate::core::compute::ComputeContext;
use crate::core::compute::simd_abstraction::simd::SimdEngineType;
use crate::core::io::soa_readers::{SoAReadBatch, SoaFastqReader};
use crate::core::utils::cputime;
use crate::pipelines::linear::batch_extension::types::SoAAlignmentResult;
use crate::pipelines::linear::index::index::BwaIndex;
use crate::pipelines::linear::mem_opt::MemOpt;
use crate::pipelines::linear::paired::insert_size::{
    InsertSizeStats, bootstrap_insert_size_stats_soa,
};
use crate::pipelines::linear::finalization::mark_secondary_alignments;
use crate::pipelines::linear::stages::chaining::ChainingStage;
use crate::pipelines::linear::stages::extension::ExtensionStage;
use crate::pipelines::linear::stages::finalization::FinalizationStage;
use crate::pipelines::linear::stages::seeding::SeedingStage;
use crate::pipelines::linear::stages::{PipelineStage, StageContext};

// Batch processing constants
const BOOTSTRAP_BATCH_SIZE: usize = 512;

/// Paired-end pipeline orchestrator.
///
/// Coordinates the paired-end alignment pipeline using hybrid AoS/SoA processing.
/// The pairing stage MUST use AoS to maintain correct per-read alignment boundaries.
pub struct PairedEndOrchestrator<'a> {
    /// Reference genome index
    pub(super) index: &'a BwaIndex,
    /// Alignment options
    pub(super) options: &'a MemOpt,
    /// Compute backend context
    compute_ctx: &'a ComputeContext,
    /// Pipeline stages
    seeder: SeedingStage,
    chainer: ChainingStage,
    extender: ExtensionStage,
    /// Finalization stage (reserved for future use)
    #[allow(dead_code)]
    finalizer: FinalizationStage,
    /// Insert size statistics (bootstrapped from first batch)
    pub(super) insert_stats: [InsertSizeStats; 4],
    /// Accumulated statistics
    stats: PipelineStatistics,
    /// SIMD engine type
    pub(super) simd_engine: SimdEngineType,
}

impl<'a> PairedEndOrchestrator<'a> {
    /// Create a new paired-end orchestrator.
    pub fn new(index: &'a BwaIndex, options: &'a MemOpt, compute_ctx: &'a ComputeContext) -> Self {
        let simd_engine = compute_ctx
            .backend
            .simd_engine()
            .unwrap_or(SimdEngineType::Engine128);

        log::debug!(
            "PairedEndOrchestrator: batch_size={}, bootstrap_size={}, simd_engine={:?}",
            options.batch_size,
            BOOTSTRAP_BATCH_SIZE,
            simd_engine
        );

        Self {
            index,
            options,
            compute_ctx,
            seeder: SeedingStage::new(),
            chainer: ChainingStage::new(),
            extender: ExtensionStage::new(),
            finalizer: FinalizationStage::without_secondary_marking(),
            insert_stats: [InsertSizeStats::default(); 4],
            stats: PipelineStatistics::new(),
            simd_engine,
        }
    }

    /// Process R1 and R2 batches through the SoA pipeline stages in parallel.
    fn process_pair_batches(
        &self,
        batch1: SoAReadBatch,
        batch2: SoAReadBatch,
        batch_start_id: u64,
    ) -> Result<(SoAAlignmentResult, SoAAlignmentResult), OrchestratorError> {
        let ctx1 = StageContext::new(self.index, self.options, self.compute_ctx, batch_start_id);
        let ctx2 = StageContext::new(self.index, self.options, self.compute_ctx, batch_start_id);

        // Process R1 and R2 in parallel using rayon::join
        let (result1, result2) = rayon::join(
            || self.process_single_batch(batch1, &ctx1),
            || self.process_single_batch(batch2, &ctx2),
        );

        Ok((result1?, result2?))
    }

    /// Process a single batch through pipeline stages, returning SoA result.
    fn process_single_batch(
        &self,
        batch: SoAReadBatch,
        ctx: &StageContext,
    ) -> Result<SoAAlignmentResult, OrchestratorError> {
        let seeding_output = self.seeder.process(batch, ctx)?;
        let chaining_output = self.chainer.process(seeding_output, ctx)?;
        let extension_output = self.extender.process(chaining_output, ctx)?;
        Ok(extension_output)
    }

    /// Run the complete pipeline with a boxed writer.
    pub fn run_boxed(
        &mut self,
        input_files: &[PathBuf],
        output: &mut Box<dyn Write + '_>,
    ) -> Result<PipelineStatistics, OrchestratorError> {
        if input_files.len() != 2 {
            return Err(OrchestratorError::InvalidInput(format!(
                "Paired-end requires exactly 2 input files, got {}",
                input_files.len()
            )));
        }

        let timer = PipelineTimer::start();
        let start_cpu = cputime();

        let r1_path = input_files[0].to_string_lossy();
        let r2_path = input_files[1].to_string_lossy();

        log::info!(
            "PairedEndOrchestrator: Processing {} and {} using {:?}",
            r1_path,
            r2_path,
            self.compute_ctx.backend
        );

        let mut reader1 = SoaFastqReader::new(&r1_path).map_err(OrchestratorError::Io)?;
        let mut reader2 = SoaFastqReader::new(&r2_path).map_err(OrchestratorError::Io)?;

        let l_pac = self.index.bns.packed_sequence_length as i64;
        let mut pairs_processed: u64 = 0;
        let mut total_rescued: usize = 0;

        // Check if insert size override is provided (BWA-MEM2 -I flag behavior)
        let use_insert_override = self.options.insert_size_override.is_some();
        if let Some(ref is_override) = self.options.insert_size_override {
            log::info!(
                "Using manual insert size override: mean={:.1}, std={:.1}",
                is_override.mean,
                is_override.stddev
            );
            self.insert_stats = [
                InsertSizeStats::default(),
                InsertSizeStats {
                    avg: is_override.mean,
                    std: is_override.stddev,
                    low: is_override.min,
                    high: is_override.max,
                    failed: false,
                },
                InsertSizeStats::default(),
                InsertSizeStats::default(),
            ];
        }

        // === Uniform batch processing loop (BWA-MEM2 behavior) ===
        log::info!(
            "Processing paired-end reads (batch_size={})",
            self.options.batch_size
        );

        let mut batch_num = 0u64;

        loop {
            let mut batch1 = reader1
                .read_batch(self.options.batch_size)
                .map_err(OrchestratorError::Io)?;
            let mut batch2 = reader2
                .read_batch(self.options.batch_size)
                .map_err(OrchestratorError::Io)?;

            // Add /1 and /2 suffixes to read names for unambiguous tracking through the pipeline
            for name in batch1.names.iter_mut() {
                name.push_str("/1");
            }
            for name in batch2.names.iter_mut() {
                name.push_str("/2");
            }

            if batch1.len() != batch2.len() {
                return Err(OrchestratorError::PairedEndMismatch {
                    r1_count: batch1.len(),
                    r2_count: batch2.len(),
                });
            }

            if batch1.is_empty() {
                break;
            }

            batch_num += 1;
            let batch_size = batch1.len();
            let batch_bp: usize = batch1
                .read_boundaries
                .iter()
                .map(|(_, len)| *len)
                .sum::<usize>()
                + batch2
                    .read_boundaries
                    .iter()
                    .map(|(_, len)| *len)
                    .sum::<usize>();

            log::info!(
                "Batch {}: read_chunk: {}, work_chunk_size: {}, nseq: {}",
                batch_num,
                self.options.batch_size,
                batch_bp,
                batch_size * 2
            );

            let batch_start = Instant::now();

            let batch1_for_output = batch1.clone();
            let batch2_for_output = batch2.clone();

            let (soa_result1, soa_result2) =
                self.process_pair_batches(batch1, batch2, pairs_processed)?;

            // Infer insert size from current batch (BWA-MEM2 behavior: per-batch unless override)
            if !use_insert_override {
                self.insert_stats = bootstrap_insert_size_stats_soa(&soa_result1, &soa_result2, l_pac);
            }

            let mut alignments1 = soa_result1.to_aos();
            let mut alignments2 = soa_result2.to_aos();

            // Mark secondary alignments and generate XA tags (equivalent to BWA-MEM2's mem_mark_primary_se)
            // This must happen BEFORE pairing to correctly identify primary vs secondary alignments
            for alns in alignments1.iter_mut() {
                mark_secondary_alignments(alns, self.options);
            }
            for alns in alignments2.iter_mut() {
                mark_secondary_alignments(alns, self.options);
            }

            self.pair_alignments_aos(&mut alignments1, &mut alignments2, pairs_processed);

            let rescued = self.perform_mate_rescue(
                &mut alignments1,
                &mut alignments2,
                &batch1_for_output,
                &batch2_for_output,
            );
            total_rescued += rescued;

            self.write_paired_output(
                &alignments1,
                &alignments2,
                &batch1_for_output,
                &batch2_for_output,
                output,
            )?;

            self.stats.total_reads += batch_size * 2;
            self.stats.total_bases += batch_bp;
            self.stats.batches_processed += 1;
            pairs_processed += batch_size as u64;

            log::info!(
                "Processed {} reads in {:.3}s",
                batch_size * 2,
                batch_start.elapsed().as_secs_f64()
            );

            if batch_size < self.options.batch_size {
                break;
            }
        }

        self.stats.wall_time_secs = timer.stop();
        self.stats.cpu_time_secs = cputime() - start_cpu;
        self.stats.properly_paired = total_rescued;

        log::info!(
            "PairedEndOrchestrator complete: {} (rescued: {} pairs)",
            self.stats,
            total_rescued
        );

        Ok(self.stats.clone())
    }

    /// Get the pipeline mode.
    pub fn mode(&self) -> PipelineMode {
        PipelineMode::PairedEnd
    }
}

impl PipelineOrchestrator for PairedEndOrchestrator<'_> {
    fn run(
        &mut self,
        input_files: &[PathBuf],
        output: &mut dyn Write,
    ) -> Result<PipelineStatistics, OrchestratorError> {
        let mut boxed: Box<dyn Write + '_> = Box::new(WriteAdapter(output));
        self.run_boxed(input_files, &mut boxed)
    }

    fn mode(&self) -> PipelineMode {
        PipelineMode::PairedEnd
    }
}

/// Adapter to wrap &mut dyn Write as a concrete type for Boxing
struct WriteAdapter<'a>(&'a mut dyn Write);

impl Write for WriteAdapter<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.0.flush()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paired_end_orchestrator_mode() {
        assert_eq!(PipelineMode::PairedEnd, PipelineMode::PairedEnd);
    }
}
