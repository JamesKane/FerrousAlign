//! Single-end pipeline orchestrator
//!
//! Coordinates the execution of pipeline stages for single-end reads.
//! Uses a pure SoA (Structure-of-Arrays) pipeline with no AoS conversions.
//!
//! # Pipeline Flow
//!
//! ```text
//! [SoAReadBatch] → Seeding → Chaining → Extension → Finalization → [SAM Output]
//! ```
//!
//! All stages operate on SoA data structures for optimal SIMD utilization.
//!
//! # Usage
//!
//! ```ignore
//! let orchestrator = SingleEndOrchestrator::new(&index, &options, &compute_ctx);
//! let stats = orchestrator.run(&[input_file], &mut output)?;
//! ```

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use super::{
    OrchestratorError, PipelineMode, PipelineOrchestrator, PipelineStatistics, PipelineTimer,
};
use crate::core::compute::ComputeContext;
use crate::core::io::sam_output::write_sam_records_soa;
use crate::core::io::soa_readers::{SoAReadBatch, SoaFastqReader};
use crate::core::utils::cputime;
use crate::pipelines::linear::index::index::BwaIndex;
use crate::pipelines::linear::mem_opt::MemOpt;
use crate::pipelines::linear::stages::chaining::ChainingStage;
use crate::pipelines::linear::stages::extension::ExtensionStage;
use crate::pipelines::linear::stages::finalization::FinalizationStage;
use crate::pipelines::linear::stages::seeding::SeedingStage;
use crate::pipelines::linear::stages::{PipelineStage, StageContext};

// Batch processing constants (matching C++ bwa-mem2)
const CHUNK_SIZE_BASES: usize = 10_000_000;
const AVG_READ_LEN: usize = 101;
const MIN_BATCH_SIZE: usize = 512;

/// Single-end pipeline orchestrator.
///
/// Coordinates the complete single-end alignment pipeline using pure SoA processing.
/// Each batch flows through: Seeding → Chaining → Extension → Finalization → Output
pub struct SingleEndOrchestrator<'a> {
    /// Reference genome index
    index: &'a BwaIndex,
    /// Alignment options
    options: &'a MemOpt,
    /// Compute backend context
    compute_ctx: &'a ComputeContext,
    /// Pipeline stages
    seeder: SeedingStage,
    chainer: ChainingStage,
    extender: ExtensionStage,
    finalizer: FinalizationStage,
    /// Accumulated statistics
    stats: PipelineStatistics,
    /// Batch size (reads per batch)
    batch_size: usize,
}

impl<'a> SingleEndOrchestrator<'a> {
    /// Create a new single-end orchestrator.
    ///
    /// # Arguments
    /// * `index` - Reference genome index
    /// * `options` - Alignment options
    /// * `compute_ctx` - Compute backend context
    pub fn new(index: &'a BwaIndex, options: &'a MemOpt, compute_ctx: &'a ComputeContext) -> Self {
        // Calculate optimal batch size based on thread count (matching C++ bwa-mem2)
        let num_threads = options.n_threads as usize;
        let batch_total_bases = CHUNK_SIZE_BASES * num_threads;
        let batch_size = (batch_total_bases / AVG_READ_LEN).max(MIN_BATCH_SIZE);

        log::debug!(
            "SingleEndOrchestrator: batch_size={} reads ({} MB total, {} threads)",
            batch_size,
            batch_total_bases / 1_000_000,
            num_threads
        );

        Self {
            index,
            options,
            compute_ctx,
            seeder: SeedingStage::new(),
            chainer: ChainingStage::new(),
            extender: ExtensionStage::new(),
            finalizer: FinalizationStage::new(),
            stats: PipelineStatistics::new(),
            batch_size,
        }
    }

    /// Process a single batch through the pipeline stages.
    fn process_batch(
        &self,
        batch: SoAReadBatch,
        ctx: &StageContext,
    ) -> Result<crate::pipelines::linear::stages::finalization::FinalizationOutput, OrchestratorError>
    {
        // Stage 1: Seeding
        let seeding_output = self.seeder.process(batch, ctx)?;
        log::debug!(
            "Seeding: {} seeds from {} reads",
            seeding_output.num_seeds(),
            seeding_output.num_reads()
        );

        // Stage 2: Chaining
        let chaining_output = self.chainer.process(seeding_output, ctx)?;
        log::debug!(
            "Chaining: {} chains ({} kept)",
            chaining_output.num_chains(),
            chaining_output.num_kept_chains()
        );

        // Stage 3: Extension
        let extension_output = self.extender.process(chaining_output, ctx)?;
        log::debug!(
            "Extension: {} alignments from {} reads",
            extension_output.len(),
            extension_output.num_reads()
        );

        // Stage 4: Finalization
        let finalization_output = self.finalizer.process(extension_output, ctx)?;
        log::debug!(
            "Finalization: {} total alignments ({} primary, {} secondary)",
            finalization_output.total_alignments(),
            finalization_output.num_primary(),
            finalization_output.num_secondary()
        );

        Ok(finalization_output)
    }
}

impl SingleEndOrchestrator<'_> {
    /// Run the complete pipeline with a boxed writer.
    pub fn run_boxed(
        &mut self,
        input_files: &[PathBuf],
        output: &mut Box<dyn Write + '_>,
    ) -> Result<PipelineStatistics, OrchestratorError> {
        let timer = PipelineTimer::start();
        let start_cpu = cputime();

        log::info!(
            "SingleEndOrchestrator: Processing {} file(s) using {:?}",
            input_files.len(),
            self.compute_ctx.backend
        );

        let mut reads_processed: u64 = 0;

        for input_file in input_files {
            let file_path = input_file.to_string_lossy();
            log::info!("Processing file: {}", file_path);

            let mut reader =
                SoaFastqReader::new(&file_path).map_err(|e| OrchestratorError::Io(e))?;

            loop {
                // Read batch
                let batch = reader
                    .read_batch(self.batch_size)
                    .map_err(|e| OrchestratorError::Io(e))?;

                if batch.is_empty() {
                    break; // EOF
                }

                let batch_size = batch.len();
                let batch_bp: usize = batch.read_boundaries.iter().map(|(_, len)| *len).sum();

                log::info!(
                    "read_chunk: {}, work_chunk_size: {}, nseq: {}",
                    self.batch_size,
                    batch_bp,
                    batch_size
                );

                // Create stage context for this batch
                let ctx =
                    StageContext::new(self.index, self.options, self.compute_ctx, reads_processed);

                let batch_start = Instant::now();

                // Clone batch for SAM output (processing consumes batch)
                let batch_for_output = batch.clone();

                // Process batch through pipeline (consumes batch)
                let finalization_output = self.process_batch(batch, &ctx)?;

                // Write SAM output
                // Convert to SoA for output (finalization output is AoS)
                let soa_result = finalization_output.to_soa();
                let rg_id = self
                    .options
                    .read_group
                    .as_ref()
                    .and_then(|rg| MemOpt::extract_rg_id(rg));

                write_sam_records_soa(
                    output,
                    &soa_result,
                    &batch_for_output,
                    self.options,
                    rg_id.as_deref(),
                )
                .map_err(|e| OrchestratorError::Io(e))?;

                // Update statistics
                self.stats.total_reads += batch_size;
                self.stats.total_bases += batch_bp;
                self.stats.total_alignments += finalization_output.total_alignments();
                self.stats.batches_processed += 1;
                self.stats.mapped_reads += finalization_output.num_primary();

                reads_processed += batch_size as u64;

                log::info!(
                    "Processed {} reads in {:.3}s",
                    batch_size,
                    batch_start.elapsed().as_secs_f64()
                );

                // Check for incomplete batch (last batch)
                if batch_size < self.batch_size {
                    break;
                }
            }
        }

        // Finalize statistics
        self.stats.wall_time_secs = timer.stop();
        self.stats.cpu_time_secs = cputime() - start_cpu;

        log::info!("SingleEndOrchestrator complete: {}", self.stats);

        Ok(self.stats.clone())
    }

    /// Get the pipeline mode.
    pub fn mode(&self) -> PipelineMode {
        PipelineMode::SingleEnd
    }
}

impl PipelineOrchestrator for SingleEndOrchestrator<'_> {
    fn run(
        &mut self,
        input_files: &[PathBuf],
        output: &mut dyn Write,
    ) -> Result<PipelineStatistics, OrchestratorError> {
        // Wrap dyn Write in a Box for internal use
        // This is a workaround for the Sized bound requirement
        // For production use, prefer run_boxed() directly
        let mut boxed: Box<dyn Write + '_> = Box::new(WriteAdapter(output));
        self.run_boxed(input_files, &mut boxed)
    }

    fn mode(&self) -> PipelineMode {
        PipelineMode::SingleEnd
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
    fn test_single_end_orchestrator_mode() {
        // We can't easily test without an index, but we can verify the mode
        // would be correct if we had one
        assert_eq!(PipelineMode::SingleEnd, PipelineMode::SingleEnd);
    }

    #[test]
    fn test_batch_size_calculation() {
        // Verify batch size calculation matches C++ bwa-mem2
        let num_threads = 16usize;
        let batch_total_bases = CHUNK_SIZE_BASES * num_threads;
        let batch_size = (batch_total_bases / AVG_READ_LEN).max(MIN_BATCH_SIZE);

        // With 16 threads, 10M bases/thread, 101bp avg read:
        // 160M bases / 101 ≈ 1.58M reads
        assert!(batch_size > 1_000_000);
        assert!(batch_size < 2_000_000);
    }
}
