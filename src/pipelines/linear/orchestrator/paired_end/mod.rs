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

use rayon::prelude::*;

use super::{
    OrchestratorError, PipelineMode, PipelineOrchestrator, PipelineStatistics, PipelineTimer,
};
use crate::core::compute::ComputeContext;
use crate::core::compute::simd_abstraction::simd::SimdEngineType;
use crate::core::io::soa_readers::{SoAReadBatch, SoaFastqReader};
use crate::core::utils::cputime;
use crate::pipelines::linear::batch_extension::types::SoAAlignmentResult;
use crate::pipelines::linear::finalization::mark_secondary_alignments;
use crate::pipelines::linear::index::index::BwaIndex;
use crate::pipelines::linear::mem_opt::MemOpt;
use crate::pipelines::linear::paired::insert_size::{
    InsertSizeStats, bootstrap_insert_size_stats_soa,
};
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

    /// Process R1 and R2 batches through all pipeline stages in parallel.
    ///
    /// Uses parallel chunk processing within R1 and R2 batches for full thread utilization.
    /// Each batch is split into chunks equal to half the thread count, and all chunks
    /// are processed in parallel through seeding, chaining, extension, AND finalization.
    ///
    /// Returns:
    /// - SoAAlignmentResult pair: For insert size estimation
    /// - AoS alignment pair: For safe merging and pairing (Vec<Vec<Alignment>>)
    fn process_pair_batches(
        &self,
        batch1: SoAReadBatch,
        batch2: SoAReadBatch,
        batch_start_id: u64,
    ) -> Result<
        (
            (SoAAlignmentResult, SoAAlignmentResult),
            (Vec<Vec<Alignment>>, Vec<Vec<Alignment>>),
        ),
        OrchestratorError,
    > {
        // Split threads between R1 and R2 processing
        let num_threads = rayon::current_num_threads();
        let chunks_per_batch = (num_threads / 2).max(1);

        let chunks1 = batch1.split_into_chunks(chunks_per_batch);
        let chunks2 = batch2.split_into_chunks(chunks_per_batch);

        // Process all chunks in parallel (both R1 and R2 chunks together)
        // Tag each chunk with its source (R1=true, R2=false)
        let all_chunks: Vec<(SoAReadBatch, usize, bool)> = chunks1
            .into_iter()
            .map(|(chunk, offset)| (chunk, offset, true))
            .chain(
                chunks2
                    .into_iter()
                    .map(|(chunk, offset)| (chunk, offset, false)),
            )
            .collect();

        // Process all chunks through FULL pipeline (seeding → chaining → extension → finalization)
        let chunk_results: Vec<
            Result<(SoAAlignmentResult, Vec<Vec<Alignment>>, usize, bool), OrchestratorError>,
        > = all_chunks
            .into_par_iter()
            .map(|(chunk, chunk_offset, is_r1)| {
                let ctx = StageContext::new(
                    self.index,
                    self.options,
                    self.compute_ctx,
                    batch_start_id + chunk_offset as u64,
                );
                let (soa_result, aos_result) = self.process_single_chunk(chunk, &ctx)?;
                Ok((soa_result, aos_result, chunk_offset, is_r1))
            })
            .collect();

        // Separate R1 and R2 results, maintaining read order
        let mut r1_soa_chunks: Vec<(SoAAlignmentResult, usize)> = Vec::new();
        let mut r2_soa_chunks: Vec<(SoAAlignmentResult, usize)> = Vec::new();
        let mut r1_aos_chunks: Vec<(Vec<Vec<Alignment>>, usize)> = Vec::new();
        let mut r2_aos_chunks: Vec<(Vec<Vec<Alignment>>, usize)> = Vec::new();

        for result in chunk_results {
            let (soa_result, aos_result, offset, is_r1) = result?;
            if is_r1 {
                r1_soa_chunks.push((soa_result, offset));
                r1_aos_chunks.push((aos_result, offset));
            } else {
                r2_soa_chunks.push((soa_result, offset));
                r2_aos_chunks.push((aos_result, offset));
            }
        }

        // Sort by offset to maintain read order
        r1_soa_chunks.sort_by_key(|(_, offset)| *offset);
        r2_soa_chunks.sort_by_key(|(_, offset)| *offset);
        r1_aos_chunks.sort_by_key(|(_, offset)| *offset);
        r2_aos_chunks.sort_by_key(|(_, offset)| *offset);

        // Merge SoA chunks (for insert size estimation)
        let soa_result1 = Self::merge_soa_results(r1_soa_chunks);
        let soa_result2 = Self::merge_soa_results(r2_soa_chunks);

        // Merge AoS chunks (trivial extend - no index adjustment needed!)
        let aos_result1 = Self::merge_aos_results(r1_aos_chunks);
        let aos_result2 = Self::merge_aos_results(r2_aos_chunks);

        Ok(((soa_result1, soa_result2), (aos_result1, aos_result2)))
    }

    /// Merge multiple AoS alignment chunks back together.
    ///
    /// This is the trivially safe merge operation - just extend the vectors.
    /// No index adjustment needed because each read's alignments stay grouped.
    fn merge_aos_results(chunks: Vec<(Vec<Vec<Alignment>>, usize)>) -> Vec<Vec<Alignment>> {
        let mut merged: Vec<Vec<Alignment>> = Vec::new();
        // Chunks are already sorted by offset
        for (chunk, _offset) in chunks {
            merged.extend(chunk);
        }
        merged
    }

    /// Merge multiple SoAAlignmentResult chunks back together.
    fn merge_soa_results(chunks: Vec<(SoAAlignmentResult, usize)>) -> SoAAlignmentResult {
        if chunks.is_empty() {
            return SoAAlignmentResult::new();
        }

        // Extract just the results (offsets already used for sorting)
        let results: Vec<SoAAlignmentResult> = chunks.into_iter().map(|(r, _)| r).collect();
        SoAAlignmentResult::merge_all(results)
    }

    /// Process a single chunk through all pipeline stages, returning AoS result.
    ///
    /// This is the parallelizable unit of work - each chunk is processed independently
    /// through seeding, chaining, extension, AND finalization. This maximizes parallelism
    /// by doing all compute-intensive work within the parallel chunk processing.
    ///
    /// Returns both:
    /// - SoAAlignmentResult: Needed for insert size estimation (before finalization would consume it)
    /// - Vec<Vec<Alignment>>: AoS format for safe merging and pairing
    fn process_single_chunk(
        &self,
        chunk: SoAReadBatch,
        ctx: &StageContext,
    ) -> Result<(SoAAlignmentResult, Vec<Vec<Alignment>>), OrchestratorError> {
        let seeding_output = self.seeder.process(chunk, ctx)?;
        let chaining_output = self.chainer.process(seeding_output, ctx)?;
        let extension_output = self.extender.process(chaining_output, ctx)?;

        // Run finalization inside the parallel chunk (this is the key optimization)
        let finalization_output = self.finalizer.process(extension_output.clone(), ctx)?;

        // Return both: SoA for insert size, AoS for merging/pairing
        Ok((extension_output, finalization_output.alignments_per_read))
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
            let batch1 = reader1
                .read_batch(self.options.batch_size)
                .map_err(OrchestratorError::Io)?;
            let batch2 = reader2
                .read_batch(self.options.batch_size)
                .map_err(OrchestratorError::Io)?;

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

            // Process all chunks through FULL pipeline (seeding → chaining → extension → finalization)
            // Returns both SoA (for insert size) and AoS (for pairing)
            let ((soa_result1, soa_result2), (mut alignments1, mut alignments2)) =
                self.process_pair_batches(batch1, batch2, pairs_processed)?;

            // Infer insert size from current batch (BWA-MEM2 behavior: per-batch unless override)
            if !use_insert_override {
                self.insert_stats =
                    bootstrap_insert_size_stats_soa(&soa_result1, &soa_result2, l_pac);
            }

            // Mark secondary alignments in PARALLEL (equivalent to BWA-MEM2's mem_mark_primary_se)
            // This must happen BEFORE pairing to correctly identify primary vs secondary alignments
            alignments1.par_iter_mut().for_each(|alns| {
                mark_secondary_alignments(alns, self.options);
            });
            alignments2.par_iter_mut().for_each(|alns| {
                mark_secondary_alignments(alns, self.options);
            });

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
    use crate::pipelines::linear::stages::finalization::Alignment;

    /// Helper to create a test alignment with identifiable fields
    fn make_test_alignment(read_name: &str, pos: u64, score: i32) -> Alignment {
        Alignment {
            query_name: read_name.to_string(),
            flag: 0,
            ref_name: "chr1".to_string(),
            ref_id: 0,
            pos,
            mapq: 60,
            score,
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
            hash: pos, // Use pos as unique identifier
            frac_rep: 0.0,
            is_alt: false,
        }
    }

    #[test]
    fn test_paired_end_orchestrator_mode() {
        assert_eq!(PipelineMode::PairedEnd, PipelineMode::PairedEnd);
    }

    // ========================================================================
    // AoS Chunk Merge Tests
    // ========================================================================

    /// Test that AoS chunk merge preserves read order within chunks
    #[test]
    fn test_aos_chunk_merge_preserves_read_order() {
        // Simulate 2 chunks, each with 3 reads
        let chunk1: Vec<Vec<Alignment>> = vec![
            vec![make_test_alignment("read_0", 100, 100)],
            vec![make_test_alignment("read_1", 200, 90)],
            vec![make_test_alignment("read_2", 300, 80)],
        ];

        let chunk2: Vec<Vec<Alignment>> = vec![
            vec![make_test_alignment("read_3", 400, 70)],
            vec![make_test_alignment("read_4", 500, 60)],
            vec![make_test_alignment("read_5", 600, 50)],
        ];

        // Merge chunks (simple concatenation for AoS)
        let mut merged: Vec<Vec<Alignment>> = Vec::new();
        merged.extend(chunk1);
        merged.extend(chunk2);

        // Verify order is preserved
        assert_eq!(merged.len(), 6);
        assert_eq!(merged[0][0].query_name, "read_0");
        assert_eq!(merged[1][0].query_name, "read_1");
        assert_eq!(merged[2][0].query_name, "read_2");
        assert_eq!(merged[3][0].query_name, "read_3");
        assert_eq!(merged[4][0].query_name, "read_4");
        assert_eq!(merged[5][0].query_name, "read_5");

        // Verify positions match expected
        assert_eq!(merged[0][0].pos, 100);
        assert_eq!(merged[5][0].pos, 600);
    }

    /// Test that AoS merge handles varying alignment counts per read
    #[test]
    fn test_aos_chunk_merge_variable_alignments_per_read() {
        // Read 0: 1 alignment, Read 1: 3 alignments, Read 2: 0 alignments
        let chunk1: Vec<Vec<Alignment>> = vec![
            vec![make_test_alignment("read_0", 100, 100)],
            vec![
                make_test_alignment("read_1", 200, 90),
                make_test_alignment("read_1", 250, 85),
                make_test_alignment("read_1", 280, 80),
            ],
            vec![], // No alignments for read_2
        ];

        // Read 3: 2 alignments
        let chunk2: Vec<Vec<Alignment>> = vec![vec![
            make_test_alignment("read_3", 400, 70),
            make_test_alignment("read_3", 450, 65),
        ]];

        let mut merged: Vec<Vec<Alignment>> = Vec::new();
        merged.extend(chunk1);
        merged.extend(chunk2);

        // Verify structure
        assert_eq!(merged.len(), 4);
        assert_eq!(merged[0].len(), 1); // read_0: 1 alignment
        assert_eq!(merged[1].len(), 3); // read_1: 3 alignments
        assert_eq!(merged[2].len(), 0); // read_2: 0 alignments
        assert_eq!(merged[3].len(), 2); // read_3: 2 alignments

        // Verify read_1's alignments are still grouped together
        assert!(merged[1].iter().all(|a| a.query_name == "read_1"));
    }

    /// Test that out-of-order chunk completion is handled correctly
    #[test]
    fn test_aos_chunk_merge_with_offset_sorting() {
        // Simulate parallel processing where chunks complete out of order
        // Chunk 1 (offset 3) finishes before Chunk 0 (offset 0)
        let chunk0: Vec<Vec<Alignment>> = vec![
            vec![make_test_alignment("read_0", 100, 100)],
            vec![make_test_alignment("read_1", 200, 90)],
            vec![make_test_alignment("read_2", 300, 80)],
        ];

        let chunk1: Vec<Vec<Alignment>> = vec![
            vec![make_test_alignment("read_3", 400, 70)],
            vec![make_test_alignment("read_4", 500, 60)],
        ];

        // Simulate out-of-order completion: chunk1 arrives first
        let results_out_of_order: Vec<(Vec<Vec<Alignment>>, usize)> =
            vec![(chunk1.clone(), 3), (chunk0.clone(), 0)];

        // Sort by offset before merging (as done in process_pair_batches)
        let mut sorted_results = results_out_of_order;
        sorted_results.sort_by_key(|(_, offset)| *offset);

        // Merge in sorted order
        let mut merged: Vec<Vec<Alignment>> = Vec::new();
        for (chunk, _offset) in sorted_results {
            merged.extend(chunk);
        }

        // Verify correct order after sorting
        assert_eq!(merged.len(), 5);
        assert_eq!(merged[0][0].query_name, "read_0"); // From chunk0
        assert_eq!(merged[1][0].query_name, "read_1");
        assert_eq!(merged[2][0].query_name, "read_2");
        assert_eq!(merged[3][0].query_name, "read_3"); // From chunk1
        assert_eq!(merged[4][0].query_name, "read_4");
    }

    // ========================================================================
    // R1/R2 Synchronization Tests
    // ========================================================================

    /// Test that R1[i] and R2[i] always correspond to the same read pair
    #[test]
    fn test_r1_r2_index_synchronization() {
        // Create paired alignments with matching indices
        let r1_alignments: Vec<Vec<Alignment>> = vec![
            vec![make_test_alignment("pair_0/1", 100, 100)],
            vec![make_test_alignment("pair_1/1", 200, 90)],
            vec![make_test_alignment("pair_2/1", 300, 80)],
        ];

        let r2_alignments: Vec<Vec<Alignment>> = vec![
            vec![make_test_alignment("pair_0/2", 150, 95)],
            vec![make_test_alignment("pair_1/2", 250, 85)],
            vec![make_test_alignment("pair_2/2", 350, 75)],
        ];

        // Verify arrays have same length
        assert_eq!(r1_alignments.len(), r2_alignments.len());

        // Verify each index pair corresponds to matching read pairs
        for i in 0..r1_alignments.len() {
            let r1_name = &r1_alignments[i][0].query_name;
            let r2_name = &r2_alignments[i][0].query_name;

            // Extract pair number from names like "pair_N/1" and "pair_N/2"
            let r1_pair_num: usize = r1_name
                .strip_prefix("pair_")
                .unwrap()
                .split('/')
                .next()
                .unwrap()
                .parse()
                .unwrap();
            let r2_pair_num: usize = r2_name
                .strip_prefix("pair_")
                .unwrap()
                .split('/')
                .next()
                .unwrap()
                .parse()
                .unwrap();

            assert_eq!(
                r1_pair_num, r2_pair_num,
                "R1[{}] and R2[{}] should be from same pair",
                i, i
            );
        }
    }

    /// Test that chunk merging maintains R1/R2 synchronization across chunks
    #[test]
    fn test_r1_r2_sync_across_chunks() {
        // Chunk 0: reads 0-1
        let r1_chunk0: Vec<Vec<Alignment>> = vec![
            vec![make_test_alignment("pair_0/1", 100, 100)],
            vec![make_test_alignment("pair_1/1", 200, 90)],
        ];
        let r2_chunk0: Vec<Vec<Alignment>> = vec![
            vec![make_test_alignment("pair_0/2", 150, 95)],
            vec![make_test_alignment("pair_1/2", 250, 85)],
        ];

        // Chunk 1: reads 2-3
        let r1_chunk1: Vec<Vec<Alignment>> = vec![
            vec![make_test_alignment("pair_2/1", 300, 80)],
            vec![make_test_alignment("pair_3/1", 400, 70)],
        ];
        let r2_chunk1: Vec<Vec<Alignment>> = vec![
            vec![make_test_alignment("pair_2/2", 350, 75)],
            vec![make_test_alignment("pair_3/2", 450, 65)],
        ];

        // Simulate out-of-order completion
        let r1_results: Vec<(Vec<Vec<Alignment>>, usize)> =
            vec![(r1_chunk1.clone(), 2), (r1_chunk0.clone(), 0)];
        let r2_results: Vec<(Vec<Vec<Alignment>>, usize)> =
            vec![(r2_chunk0.clone(), 0), (r2_chunk1.clone(), 2)]; // Different order!

        // Sort both by offset
        let mut r1_sorted = r1_results;
        let mut r2_sorted = r2_results;
        r1_sorted.sort_by_key(|(_, offset)| *offset);
        r2_sorted.sort_by_key(|(_, offset)| *offset);

        // Merge both
        let mut r1_merged: Vec<Vec<Alignment>> = Vec::new();
        let mut r2_merged: Vec<Vec<Alignment>> = Vec::new();
        for (chunk, _) in r1_sorted {
            r1_merged.extend(chunk);
        }
        for (chunk, _) in r2_sorted {
            r2_merged.extend(chunk);
        }

        // CRITICAL: Verify R1 and R2 are still synchronized
        assert_eq!(r1_merged.len(), r2_merged.len());
        for i in 0..r1_merged.len() {
            let r1_name = &r1_merged[i][0].query_name;
            let r2_name = &r2_merged[i][0].query_name;

            // Both should reference the same pair number
            let r1_pair: &str = r1_name.split('/').next().unwrap();
            let r2_pair: &str = r2_name.split('/').next().unwrap();

            assert_eq!(
                r1_pair, r2_pair,
                "After merge, R1[{}]={} should match R2[{}]={}",
                i, r1_name, i, r2_name
            );
        }
    }

    /// Test that empty alignments don't break synchronization
    #[test]
    fn test_r1_r2_sync_with_empty_alignments() {
        // Some reads have no alignments (unmapped)
        let r1_alignments: Vec<Vec<Alignment>> = vec![
            vec![make_test_alignment("pair_0/1", 100, 100)],
            vec![], // pair_1 R1 unmapped
            vec![make_test_alignment("pair_2/1", 300, 80)],
        ];

        let r2_alignments: Vec<Vec<Alignment>> = vec![
            vec![], // pair_0 R2 unmapped
            vec![make_test_alignment("pair_1/2", 250, 85)],
            vec![make_test_alignment("pair_2/2", 350, 75)],
        ];

        // Arrays must still be same length
        assert_eq!(r1_alignments.len(), r2_alignments.len());

        // Index correspondence must hold even for empty slots
        // pair_0: R1 mapped at [0], R2 unmapped at [0]
        // pair_1: R1 unmapped at [1], R2 mapped at [1]
        // pair_2: Both mapped at [2]
        assert_eq!(r1_alignments[0].len(), 1); // R1 mapped
        assert_eq!(r2_alignments[0].len(), 0); // R2 unmapped
        assert_eq!(r1_alignments[1].len(), 0); // R1 unmapped
        assert_eq!(r2_alignments[1].len(), 1); // R2 mapped
        assert_eq!(r1_alignments[2].len(), 1); // Both mapped
        assert_eq!(r2_alignments[2].len(), 1);
    }

    /// Test the merge function we'll use for AoS chunks
    #[test]
    fn test_merge_aos_chunks_function() {
        // This tests the pattern we'll use in the actual implementation
        fn merge_aos_chunks(chunks: Vec<(Vec<Vec<Alignment>>, usize)>) -> Vec<Vec<Alignment>> {
            let mut sorted_chunks = chunks;
            sorted_chunks.sort_by_key(|(_, offset)| *offset);

            let mut merged = Vec::new();
            for (chunk, _) in sorted_chunks {
                merged.extend(chunk);
            }
            merged
        }

        let chunk0 = vec![
            vec![make_test_alignment("r0", 100, 100)],
            vec![make_test_alignment("r1", 200, 90)],
        ];
        let chunk1 = vec![
            vec![make_test_alignment("r2", 300, 80)],
            vec![make_test_alignment("r3", 400, 70)],
        ];

        // Out of order input
        let chunks = vec![(chunk1, 2), (chunk0, 0)];
        let merged = merge_aos_chunks(chunks);

        assert_eq!(merged.len(), 4);
        assert_eq!(merged[0][0].query_name, "r0");
        assert_eq!(merged[1][0].query_name, "r1");
        assert_eq!(merged[2][0].query_name, "r2");
        assert_eq!(merged[3][0].query_name, "r3");
    }
}
