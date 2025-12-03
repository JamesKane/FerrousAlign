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

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use rayon::prelude::*;

use super::{
    OrchestratorError, PipelineMode, PipelineOrchestrator, PipelineStatistics, PipelineTimer,
};
use crate::core::compute::ComputeContext;
use crate::core::compute::simd_abstraction::simd::SimdEngineType;
use crate::core::io::soa_readers::{SoAReadBatch, SoaFastqReader};
use crate::core::utils::cputime;
use crate::pipelines::linear::batch_extension::types::SoAAlignmentResult;
use crate::pipelines::linear::finalization::sam_flags;
use crate::pipelines::linear::index::index::BwaIndex;
use crate::pipelines::linear::mem_opt::MemOpt;
use crate::pipelines::linear::paired::insert_size::{
    InsertSizeStats, bootstrap_insert_size_stats_soa,
};
use crate::pipelines::linear::paired::mate_rescue::mate_rescue_soa;
use crate::pipelines::linear::paired::pairing_aos;
use crate::pipelines::linear::stages::chaining::ChainingStage;
use crate::pipelines::linear::stages::extension::ExtensionStage;
use crate::pipelines::linear::stages::finalization::{Alignment, FinalizationStage};
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
    /// Insert size statistics (bootstrapped from first batch)
    insert_stats: [InsertSizeStats; 4],
    /// Accumulated statistics
    stats: PipelineStatistics,
    /// SIMD engine type
    simd_engine: SimdEngineType,
}

impl<'a> PairedEndOrchestrator<'a> {
    /// Create a new paired-end orchestrator.
    ///
    /// # Arguments
    /// * `index` - Reference genome index
    /// * `options` - Alignment options
    /// * `compute_ctx` - Compute backend context
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
            // Use without_secondary_marking since pairing handles this
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
        // Seeding
        let seeding_output = self.seeder.process(batch, ctx)?;

        // Chaining
        let chaining_output = self.chainer.process(seeding_output, ctx)?;

        // Extension (returns SoAAlignmentResult)
        let extension_output = self.extender.process(chaining_output, ctx)?;

        Ok(extension_output)
    }

    /// Perform AoS pairing on alignment results.
    ///
    /// CRITICAL: This must be done in AoS format to maintain per-read alignment
    /// boundaries correctly. Pure SoA pairing causes 96% duplicate reads.
    fn pair_alignments_aos(
        &self,
        alignments1: &mut Vec<Vec<Alignment>>,
        alignments2: &mut Vec<Vec<Alignment>>,
        batch_start_id: u64,
    ) {
        let l_pac = self.index.bns.packed_sequence_length as i64;

        for read_idx in 0..alignments1.len() {
            let alns1 = &mut alignments1[read_idx];
            let alns2 = &mut alignments2[read_idx];

            if let Some((idx1, idx2, _pair_score, _sub_score)) = pairing_aos::mem_pair(
                &self.insert_stats,
                alns1,
                alns2,
                self.options.a,
                batch_start_id + read_idx as u64,
                l_pac,
            ) {
                // Mark as properly paired
                alns1[idx1].flag |= sam_flags::PROPER_PAIR;
                alns2[idx2].flag |= sam_flags::PROPER_PAIR;

                // Set mate information
                let (aln1, aln2) = (alns1[idx1].clone(), alns2[idx2].clone());
                Self::set_mate_info(&mut alns1[idx1], &aln2, true);
                Self::set_mate_info(&mut alns2[idx2], &aln1, false);
            } else if !alns1.is_empty() && !alns2.is_empty() {
                // Singleton: both mapped but not properly paired
                let (aln1, aln2) = (alns1[0].clone(), alns2[0].clone());
                Self::set_mate_info(&mut alns1[0], &aln2, true);
                Self::set_mate_info(&mut alns2[0], &aln1, false);
            }
        }
    }

    /// Set mate information on an alignment.
    fn set_mate_info(aln: &mut Alignment, mate: &Alignment, is_first: bool) {
        aln.rnext = if mate.ref_name == aln.ref_name {
            "=".to_string()
        } else {
            mate.ref_name.clone()
        };
        aln.pnext = mate.pos;
        aln.flag |= sam_flags::PAIRED;

        if is_first {
            aln.flag |= sam_flags::FIRST_IN_PAIR;
        } else {
            aln.flag |= sam_flags::SECOND_IN_PAIR;
        }

        if (mate.flag & sam_flags::REVERSE) != 0 {
            aln.flag |= sam_flags::MATE_REVERSE;
        }

        if mate.mapq == 0 && mate.score == 0 {
            aln.flag |= sam_flags::MATE_UNMAPPED;
        }
    }

    /// Perform mate rescue using SoA format for SIMD batching.
    fn perform_mate_rescue(
        &self,
        alignments1: &mut Vec<Vec<Alignment>>,
        alignments2: &mut Vec<Vec<Alignment>>,
        batch1: &SoAReadBatch,
        batch2: &SoAReadBatch,
    ) -> usize {
        // Convert to SoA for SIMD rescue
        let mut soa1 = SoAAlignmentResult::from_aos(alignments1);
        let mut soa2 = SoAAlignmentResult::from_aos(alignments2);

        // Find primary alignments for rescue
        let primary_r1: Vec<usize> = (0..soa1.num_reads())
            .map(|read_idx| {
                let (start, count) = soa1.read_alignment_boundaries[read_idx];
                (start..start + count)
                    .find(|&idx| soa1.flags[idx] & sam_flags::PROPER_PAIR != 0)
                    .unwrap_or(start)
            })
            .collect();

        let primary_r2: Vec<usize> = (0..soa2.num_reads())
            .map(|read_idx| {
                let (start, count) = soa2.read_alignment_boundaries[read_idx];
                (start..start + count)
                    .find(|&idx| soa2.flags[idx] & sam_flags::PROPER_PAIR != 0)
                    .unwrap_or(start)
            })
            .collect();

        let pac = &self.index.bns.pac_data;
        let rescued = mate_rescue_soa(
            &mut soa1,
            &mut soa2,
            batch1,
            batch2,
            &primary_r1,
            &primary_r2,
            pac,
            self.index,
            &self.insert_stats,
            self.options.pen_unpaired,
            self.options.max_matesw,
            Some(self.simd_engine),
        );

        // Convert back to AoS
        *alignments1 = soa1.to_aos();
        *alignments2 = soa2.to_aos();

        rescued
    }

    /// Write paired-end alignments to output.
    fn write_paired_output(
        &self,
        alignments1: &[Vec<Alignment>],
        alignments2: &[Vec<Alignment>],
        batch1: &SoAReadBatch,
        batch2: &SoAReadBatch,
        output: &mut dyn Write,
    ) -> Result<usize, OrchestratorError> {
        let mut records = 0;

        for read_idx in 0..alignments1.len() {
            let alns1 = &alignments1[read_idx];
            let alns2 = &alignments2[read_idx];

            // Get sequences and qualities
            let (seq_start1, seq_len1) = batch1.read_boundaries[read_idx];
            let seq1 =
                std::str::from_utf8(&batch1.seqs[seq_start1..seq_start1 + seq_len1]).unwrap_or("");
            let qual1 =
                std::str::from_utf8(&batch1.quals[seq_start1..seq_start1 + seq_len1]).unwrap_or("");

            let (seq_start2, seq_len2) = batch2.read_boundaries[read_idx];
            let seq2 =
                std::str::from_utf8(&batch2.seqs[seq_start2..seq_start2 + seq_len2]).unwrap_or("");
            let qual2 =
                std::str::from_utf8(&batch2.quals[seq_start2..seq_start2 + seq_len2]).unwrap_or("");

            // Write R1 alignments
            if let Some(aln) = alns1.first() {
                writeln!(output, "{}", aln.to_sam_string_with_seq(seq1, qual1))
                    .map_err(|e| OrchestratorError::Io(e))?;
                records += 1;
            }

            // Write R2 alignments
            if let Some(aln) = alns2.first() {
                writeln!(output, "{}", aln.to_sam_string_with_seq(seq2, qual2))
                    .map_err(|e| OrchestratorError::Io(e))?;
                records += 1;
            }
        }

        Ok(records)
    }
}

impl PairedEndOrchestrator<'_> {
    /// Run the complete pipeline with a boxed writer.
    pub fn run_boxed(
        &mut self,
        input_files: &[PathBuf],
        output: &mut Box<dyn Write + '_>,
    ) -> Result<PipelineStatistics, OrchestratorError> {
        // Validate input
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

        // Open readers
        let mut reader1 = SoaFastqReader::new(&r1_path).map_err(|e| OrchestratorError::Io(e))?;
        let mut reader2 = SoaFastqReader::new(&r2_path).map_err(|e| OrchestratorError::Io(e))?;

        let l_pac = self.index.bns.packed_sequence_length as i64;
        let mut pairs_processed: u64 = 0;
        let mut total_rescued: usize = 0;

        // === PHASE 1: Bootstrap insert size from first batch ===
        log::info!("Phase 1: Bootstrapping insert size statistics");

        let first_batch1 = reader1
            .read_batch(BOOTSTRAP_BATCH_SIZE)
            .map_err(|e| OrchestratorError::Io(e))?;
        let first_batch2 = reader2
            .read_batch(BOOTSTRAP_BATCH_SIZE)
            .map_err(|e| OrchestratorError::Io(e))?;

        // Validate paired-end batch sizes match
        if first_batch1.len() != first_batch2.len() {
            return Err(OrchestratorError::PairedEndMismatch {
                r1_count: first_batch1.len(),
                r2_count: first_batch2.len(),
            });
        }

        if first_batch1.is_empty() {
            log::warn!("No data to process (empty input files)");
            return Ok(self.stats.clone());
        }

        // Clone batches for later use (output, mate rescue)
        let first_batch1_for_output = first_batch1.clone();
        let first_batch2_for_output = first_batch2.clone();

        // Process first batch (consumes batches)
        let (soa_result1, soa_result2) =
            self.process_pair_batches(first_batch1, first_batch2, pairs_processed)?;

        // Bootstrap or use override
        self.insert_stats = if let Some(ref is_override) = self.options.insert_size_override {
            log::info!(
                "Using manual insert size: mean={:.1}, std={:.1}",
                is_override.mean,
                is_override.stddev
            );
            [
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
            ]
        } else {
            bootstrap_insert_size_stats_soa(&soa_result1, &soa_result2, l_pac)
        };

        // AoS pairing for first batch
        let mut alignments1 = soa_result1.to_aos();
        let mut alignments2 = soa_result2.to_aos();
        self.pair_alignments_aos(&mut alignments1, &mut alignments2, pairs_processed);

        // Mate rescue
        let rescued = self.perform_mate_rescue(
            &mut alignments1,
            &mut alignments2,
            &first_batch1_for_output,
            &first_batch2_for_output,
        );
        total_rescued += rescued;
        log::info!("First batch: {} pairs rescued", rescued);

        // Output first batch
        let _records = self.write_paired_output(
            &alignments1,
            &alignments2,
            &first_batch1_for_output,
            &first_batch2_for_output,
            output,
        )?;

        // Update stats
        let first_batch_size = first_batch1_for_output.len();
        let first_batch_bp: usize = first_batch1_for_output
            .read_boundaries
            .iter()
            .map(|(_, len)| *len)
            .sum::<usize>()
            + first_batch2_for_output
                .read_boundaries
                .iter()
                .map(|(_, len)| *len)
                .sum::<usize>();
        self.stats.total_reads += first_batch_size * 2;
        self.stats.total_bases += first_batch_bp;
        self.stats.batches_processed += 1;
        pairs_processed += first_batch_size as u64;

        // === PHASE 2: Process remaining batches ===
        log::info!(
            "Phase 2: Streaming remaining batches (batch_size={})",
            self.options.batch_size
        );

        let mut batch_num = 1u64;

        loop {
            let batch1 = reader1
                .read_batch(self.options.batch_size)
                .map_err(|e| OrchestratorError::Io(e))?;
            let batch2 = reader2
                .read_batch(self.options.batch_size)
                .map_err(|e| OrchestratorError::Io(e))?;

            // Validate batch sizes match
            if batch1.len() != batch2.len() {
                return Err(OrchestratorError::PairedEndMismatch {
                    r1_count: batch1.len(),
                    r2_count: batch2.len(),
                });
            }

            if batch1.is_empty() {
                break; // EOF
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
                "read_chunk: {}, work_chunk_size: {}, nseq: {}",
                self.options.batch_size,
                batch_bp,
                batch_size * 2
            );

            let batch_start = Instant::now();

            // Clone batches for output/rescue
            let batch1_for_output = batch1.clone();
            let batch2_for_output = batch2.clone();

            // Process batch (consumes batches)
            let (soa_result1, soa_result2) =
                self.process_pair_batches(batch1, batch2, pairs_processed)?;

            // Re-bootstrap insert size
            self.insert_stats = bootstrap_insert_size_stats_soa(&soa_result1, &soa_result2, l_pac);

            // AoS pairing
            let mut alignments1 = soa_result1.to_aos();
            let mut alignments2 = soa_result2.to_aos();
            self.pair_alignments_aos(&mut alignments1, &mut alignments2, pairs_processed);

            // Mate rescue
            let rescued = self.perform_mate_rescue(
                &mut alignments1,
                &mut alignments2,
                &batch1_for_output,
                &batch2_for_output,
            );
            total_rescued += rescued;

            // Output
            self.write_paired_output(
                &alignments1,
                &alignments2,
                &batch1_for_output,
                &batch2_for_output,
                output,
            )?;

            // Update stats
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
                break; // Last incomplete batch
            }
        }

        // Finalize statistics
        self.stats.wall_time_secs = timer.stop();
        self.stats.cpu_time_secs = cputime() - start_cpu;
        self.stats.properly_paired = total_rescued; // Approximation

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
        // Wrap dyn Write in a Box for internal use
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

    #[test]
    fn test_set_mate_info() {
        let mut aln = Alignment {
            query_name: "read1".to_string(),
            flag: 0,
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
        };

        let mate = Alignment {
            query_name: "read2".to_string(),
            flag: sam_flags::REVERSE,
            ref_name: "chr1".to_string(),
            ref_id: 0,
            pos: 300,
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
        };

        PairedEndOrchestrator::set_mate_info(&mut aln, &mate, true);

        assert_eq!(aln.rnext, "="); // Same chromosome
        assert_eq!(aln.pnext, 300);
        assert!(aln.flag & sam_flags::PAIRED != 0);
        assert!(aln.flag & sam_flags::FIRST_IN_PAIR != 0);
        assert!(aln.flag & sam_flags::MATE_REVERSE != 0);
    }
}
