// Paired-end read processing module
//
// This module handles paired-end FASTQ file processing, including:
// - Insert size statistics bootstrapping from first batch
// - Mate rescue using Smith-Waterman
// - Paired alignment scoring
// - SAM output with proper pair flags
//
// The pipeline is cleanly separated using sam_output module:
// - Computation: align_read_deferred() returns Vec<Alignment>
// - Pairing: mem_pair() scores paired alignments
// - Output: sam_output functions handle flag setting and formatting

use super::super::batch_extension::process_sub_batch_internal_soa;
use super::super::batch_extension::types::SoAAlignmentResult;
use super::super::index::index::BwaIndex;
use super::super::mem_opt::MemOpt;
use super::insert_size::{InsertSizeStats, bootstrap_insert_size_stats_soa};
use super::mate_rescue::mate_rescue_soa;
use super::pairing_aos; // AoS pairing for hybrid architecture
use crate::compute::ComputeContext;
use crate::compute::simd_abstraction::simd::SimdEngineType;
use crate::io::soa_readers::SoaFastqReader;
use crate::pipelines::linear::finalization::sam_flags;
use crate::utils::cputime;
use rayon::prelude::*;
use std::io::Write;
use std::sync::Arc;

// Batch processing constants (matching C++ bwa-mem2)
#[allow(dead_code)]
const CHUNK_SIZE_BASES: usize = 10_000_000;
#[allow(dead_code)]
const AVG_READ_LEN: usize = 101;

// Insert size bootstrap uses small batch (512 pairs) to avoid stalling
const BOOTSTRAP_BATCH_SIZE: usize = 512;

// Main processing uses large batches for better parallelization
// BWA-MEM2 uses 10M bases * n_threads per batch
// With 16 threads and 150bp reads: 160M bases / 150 / 2 = ~533K pairs
// We use 500K pairs to match BWA-MEM2's scale for maximum parallelism
// const PROCESSING_BATCH_SIZE: usize = 500_000; // Removed, now comes from opt.batch_size

// Paired-end alignment constants
#[allow(dead_code)] // Reserved for future use in alignment scoring
const MIN_RATIO: f64 = 0.8; // Minimum ratio for unique alignment

/// Set mate information for a pair of alignments (AoS version)
///
/// Mirrors the logic in finalize_pairs_soa but operates on individual Alignment structs.
/// Sets RNEXT, PNEXT, and mate-related flags (PAIRED, FIRST/SECOND_IN_PAIR, MATE_REVERSE, MATE_UNMAPPED).
///
/// # Arguments
/// * `aln` - The alignment to update with mate information
/// * `mate` - The mate's alignment
/// * `is_first` - True if `aln` is from read 1 (R1), false if from read 2 (R2)
/// * `l_pac` - Length of packed reference (unused but kept for consistency with SoA version)
fn set_mate_info_aos(
    aln: &mut crate::pipelines::linear::finalization::Alignment,
    mate: &crate::pipelines::linear::finalization::Alignment,
    is_first: bool,
    _l_pac: i64,
) {
    // Set mate reference name (use "=" if same chromosome)
    aln.rnext = if mate.ref_name == aln.ref_name {
        "=".to_string()
    } else {
        mate.ref_name.clone()
    };

    // Set mate position
    aln.pnext = mate.pos;

    // Set paired-end flags
    aln.flag |= sam_flags::PAIRED;
    if is_first {
        aln.flag |= sam_flags::FIRST_IN_PAIR;
    } else {
        aln.flag |= sam_flags::SECOND_IN_PAIR;
    }

    // Set mate reverse flag
    if (mate.flag & sam_flags::REVERSE) != 0 {
        aln.flag |= sam_flags::MATE_REVERSE;
    }

    // Set mate unmapped flag
    if mate.mapq == 0 && mate.score == 0 {
        aln.flag |= sam_flags::MATE_UNMAPPED;
    }
}

/// Process a batch in parallel chunks
///
/// Splits the batch into thread-sized chunks and processes them in parallel,
/// then merges the results. This restores full CPU utilization while maintaining
/// SoA benefits within each chunk.
///
/// # Arguments
/// * `batch` - The SoA read batch to process
/// * `bwa_idx` - Reference index
/// * `pac` - Packed reference sequence
/// * `opt` - Alignment options
/// * `batch_start_id` - Starting read ID for hash tie-breaking
/// * `engine` - SIMD engine type
///
/// # Returns
/// Merged SoAAlignmentResult containing all alignments
fn process_batch_parallel(
    batch: &crate::io::soa_readers::SoAReadBatch,
    bwa_idx: &Arc<&BwaIndex>,
    pac: &Arc<&[u8]>,
    opt: &Arc<&MemOpt>,
    batch_start_id: u64,
    engine: SimdEngineType,
) -> SoAAlignmentResult {
    if batch.is_empty() {
        return SoAAlignmentResult::new();
    }

    // Determine optimal chunk size based on thread count
    let num_threads = rayon::current_num_threads();
    let batch_size = batch.len();

    // If batch is smaller than thread count, process sequentially
    if batch_size <= num_threads {
        return process_sub_batch_internal_soa(bwa_idx, pac, opt, batch, batch_start_id, engine);
    }

    // Calculate chunk size (round up to ensure we cover all reads)
    let chunk_size = (batch_size + num_threads - 1) / num_threads;

    log::debug!(
        "Processing batch of {} reads in {} parallel chunks (chunk_size={})",
        batch_size,
        num_threads,
        chunk_size
    );

    // Create chunk boundaries
    let chunks: Vec<(usize, usize)> = (0..batch_size)
        .step_by(chunk_size)
        .map(|start| {
            let end = (start + chunk_size).min(batch_size);
            (start, end)
        })
        .collect();

    // Process chunks in parallel
    let results: Vec<SoAAlignmentResult> = chunks
        .into_par_iter()
        .map(|(start, end)| {
            // Slice the batch (zero-copy for seq/qual data)
            let chunk = batch.slice(start, end);

            // Process chunk with SoA pipeline
            process_sub_batch_internal_soa(
                bwa_idx,
                pac,
                opt,
                &chunk,
                batch_start_id + start as u64,
                engine,
            )
        })
        .collect();

    // Merge results from all chunks
    SoAAlignmentResult::merge_all(results)
}

// Process paired-end reads with parallel batching
// ============================================================================
// HETEROGENEOUS COMPUTE ENTRY POINT - PAIRED-END PROCESSING
// ============================================================================
//
// This function is the main entry point for paired-end alignment processing.
// The compute_ctx parameter controls which hardware backend is used for
// alignment computations.
//
// Compute flow: process_paired_end() → align_read_deferred() → extension
//
// To add GPU/NPU acceleration:
// 1. Pass compute_ctx through to align_read_deferred()
// 2. In extension, route based on compute_ctx.backend
// 3. Implement backend-specific alignment kernel
//
// ============================================================================
pub fn process_paired_end(
    bwa_idx: &BwaIndex,
    read1_file: &str,
    read2_file: &str,
    writer: &mut Box<dyn Write>,
    opt: &MemOpt,
    compute_ctx: &ComputeContext,
) {
    use std::time::Instant;

    // Log the compute backend being used for this processing run
    log::debug!(
        "Paired-end processing using backend: {:?}",
        compute_ctx.backend
    );

    // Wrap index in Arc for thread-safe sharing
    let bwa_idx = Arc::new(bwa_idx);
    let opt = Arc::new(opt);

    // Track overall statistics
    let start_time = Instant::now();
    let start_cpu = cputime();
    let mut total_reads = 0usize;
    let mut total_bases = 0usize;
    let mut pairs_processed = 0u64; // Global pair counter for deterministic hash tie-breaking

    // Two-phase batch size strategy:
    // - Phase 1 (bootstrap): Small batch for insert size estimation (512 pairs)
    // - Phase 2 (main): Large batch for parallelization (50K pairs)
    log::debug!(
        "Using batch sizes: bootstrap={}, processing={}",
        BOOTSTRAP_BATCH_SIZE,
        opt.batch_size
    );

    // PAC data is already loaded into memory in bwa_idx.bns.pac_data
    let pac = &bwa_idx.bns.pac_data;
    log::debug!("Using in-memory PAC data: {} bytes", pac.len());

    // === PHASE 1: Bootstrap insert size statistics from first batch ===
    // Read first batch INLINE (not in thread) with small batch size
    log::info!("Phase 1: Bootstrapping insert size statistics from first batch");

    let mut reader1 = match SoaFastqReader::new(read1_file) {
        Ok(r) => r,
        Err(e) => {
            log::error!("Error opening read1 file {read1_file}: {e}");
            return;
        }
    };
    let mut reader2 = match SoaFastqReader::new(read2_file) {
        Ok(r) => r,
        Err(e) => {
            log::error!("Error opening read2 file {read2_file}: {e}");
            return;
        }
    };

    // Read first batch with small size for insert size bootstrap
    let first_batch1 = match reader1.read_batch(BOOTSTRAP_BATCH_SIZE) {
        Ok(b) => b,
        Err(e) => {
            log::error!("Error reading first batch from read1: {e}");
            return;
        }
    };
    let first_batch2 = match reader2.read_batch(BOOTSTRAP_BATCH_SIZE) {
        Ok(b) => b,
        Err(e) => {
            log::error!("Error reading first batch from read2: {e}");
            return;
        }
    };

    // CRITICAL VALIDATION: Verify batch sizes match to prevent mis-pairing
    // This catches truncated files, missing reads, and other synchronization issues
    if first_batch1.len() != first_batch2.len() {
        log::error!(
            "Paired-end read count mismatch in bootstrap batch: R1={} reads, R2={} reads",
            first_batch1.len(),
            first_batch2.len()
        );
        log::error!(
            "Paired-end FASTQ files must have exactly the same number of reads in the same order."
        );
        log::error!(
            "Common causes: truncated file, missing reads, corrupted data, or mismatched file pairs."
        );
        log::error!(
            "Please verify file integrity with: wc -l {} {}",
            read1_file,
            read2_file
        );
        return;
    }

    // Check for empty input
    if first_batch1.names.is_empty() {
        log::warn!("No data to process (empty input files)");
        return;
    }

    let first_batch_size = first_batch1.names.len();
    let first_batch_bp: usize = first_batch1
        .read_boundaries
        .iter()
        .map(|(_, len)| *len)
        .sum::<usize>()
        + first_batch2
            .read_boundaries
            .iter()
            .map(|(_, len)| *len)
            .sum::<usize>();
    total_reads += first_batch_size * 2;
    total_bases += first_batch_bp;

    log::debug!("[Main] Batch 0: Received {first_batch_size} pairs from channel");
    log::info!(
        "read_chunk: {}, work_chunk_size: {}, nseq: {}",
        BOOTSTRAP_BATCH_SIZE,
        first_batch_bp,
        first_batch_size * 2
    );

    // Track per-batch timing
    let batch_start_cpu = cputime();
    let batch_start_wall = Instant::now();

    // Process first batch using SoA pipeline
    let num_pairs = first_batch1.names.len();
    let bwa_idx_clone = Arc::clone(&bwa_idx);
    let opt_clone = Arc::clone(&opt);
    let batch_start_id = pairs_processed; // Capture for closure

    let simd_engine = compute_ctx
        .backend
        .simd_engine()
        .unwrap_or(SimdEngineType::Engine128);

    // Stage 1: SoA batch alignment for R1 and R2 (parallel processing)
    let pac_ref = Arc::new(&pac[..]);
    let (soa_result1, soa_result2) = rayon::join(
        || {
            process_batch_parallel(
                &first_batch1,
                &bwa_idx_clone,
                &pac_ref,
                &opt_clone,
                batch_start_id,
                simd_engine,
            )
        },
        || {
            process_batch_parallel(
                &first_batch2,
                &bwa_idx_clone,
                &pac_ref,
                &opt_clone,
                batch_start_id,
                simd_engine,
            )
        },
    );

    // Stage 2: Pure SoA pipeline - Bootstrap insert size stats
    // Update global pair counter
    pairs_processed += num_pairs as u64;

    let l_pac = bwa_idx.bns.packed_sequence_length as i64;

    // Bootstrap insert size stats from first batch
    let mut current_stats = if let Some(ref is_override) = opt.insert_size_override {
        // Use manual insert size specification (-I option)
        log::info!(
            "[Paired-end] Using manual insert size: mean={:.1}, std={:.1}, max={}, min={}",
            is_override.mean,
            is_override.stddev,
            is_override.max,
            is_override.min
        );

        [
            InsertSizeStats {
                avg: 0.0,
                std: 0.0,
                low: 0,
                high: 0,
                failed: true,
            },
            InsertSizeStats {
                avg: is_override.mean,
                std: is_override.stddev,
                low: is_override.min,
                high: is_override.max,
                failed: false,
            },
            InsertSizeStats {
                avg: 0.0,
                std: 0.0,
                low: 0,
                high: 0,
                failed: true,
            },
            InsertSizeStats {
                avg: 0.0,
                std: 0.0,
                low: 0,
                high: 0,
                failed: true,
            },
        ]
    } else {
        bootstrap_insert_size_stats_soa(&soa_result1, &soa_result2, l_pac)
    };

    // Stage 3: HYBRID AoS/SoA pipeline
    // Convert SoA results to AoS for pairing (avoids index collision bug)
    let mut alignments1 = soa_result1.to_aos();
    let mut alignments2 = soa_result2.to_aos();

    // Stage 3.1: AoS pairing (correct per-read indexing)
    for read_idx in 0..alignments1.len() {
        let alns1 = &mut alignments1[read_idx];
        let alns2 = &mut alignments2[read_idx];

        if let Some((idx1, idx2, _pair_score, _sub_score)) = pairing_aos::mem_pair(
            &current_stats,
            alns1,
            alns2,
            opt.a, // Match score
            batch_start_id + read_idx as u64,
            l_pac,
        ) {
            // Mark as properly paired
            alns1[idx1].flag |= sam_flags::PROPER_PAIR;
            alns2[idx2].flag |= sam_flags::PROPER_PAIR;

            // Set mate information for both reads
            let (aln1, aln2) = (&alns1[idx1].clone(), &alns2[idx2].clone());
            set_mate_info_aos(&mut alns1[idx1], &aln2, true, l_pac);
            set_mate_info_aos(&mut alns2[idx2], &aln1, false, l_pac);
        } else if !alns1.is_empty() && !alns2.is_empty() {
            // Singleton: both mapped but not properly paired
            // Still need to set mate info for SAM spec compliance (GATK validation)
            let (aln1, aln2) = (&alns1[0].clone(), &alns2[0].clone());
            set_mate_info_aos(&mut alns1[0], &aln2, true, l_pac);
            set_mate_info_aos(&mut alns2[0], &aln1, false, l_pac);
        }
    }

    // Stage 3.2: Mate rescue (convert back to SoA temporarily for SIMD batching)
    // Round-trip: AoS → SoA (for rescue) → AoS (for output)
    let mut soa_for_rescue1 = SoAAlignmentResult::from_aos(&alignments1);
    let mut soa_for_rescue2 = SoAAlignmentResult::from_aos(&alignments2);

    // Track primary alignments for rescue (find index of properly paired alignment per read)
    let primary_r1: Vec<usize> = (0..soa_for_rescue1.num_reads())
        .map(|read_idx| {
            let (start, count) = soa_for_rescue1.read_alignment_boundaries[read_idx];
            (start..start + count)
                .find(|&idx| soa_for_rescue1.flags[idx] & sam_flags::PROPER_PAIR != 0)
                .unwrap_or(start) // Default to first alignment if none properly paired
        })
        .collect();

    let primary_r2: Vec<usize> = (0..soa_for_rescue2.num_reads())
        .map(|read_idx| {
            let (start, count) = soa_for_rescue2.read_alignment_boundaries[read_idx];
            (start..start + count)
                .find(|&idx| soa_for_rescue2.flags[idx] & sam_flags::PROPER_PAIR != 0)
                .unwrap_or(start)
        })
        .collect();

    let rescued_first = mate_rescue_soa(
        &mut soa_for_rescue1,
        &mut soa_for_rescue2,
        &first_batch1,
        &first_batch2,
        &primary_r1,
        &primary_r2,
        pac,
        &bwa_idx,
        &current_stats,
        opt.pen_unpaired,
        opt.max_matesw,
        Some(simd_engine),
    );
    log::info!("First batch: {rescued_first} pairs rescued");

    // Convert back to AoS after rescue
    alignments1 = soa_for_rescue1.to_aos();
    alignments2 = soa_for_rescue2.to_aos();

    // Stage 4: Output first batch using AoS (avoids SoA index bugs)
    // No need to call finalize_pairs_soa since AoS pairing already set correct flags
    let final_alignments1 = alignments1;
    let final_alignments2 = alignments2;

    // Write alignments using direct SAM output (per-read iteration)
    let mut first_batch_records = 0;
    for read_idx in 0..final_alignments1.len() {
        let alns1 = &final_alignments1[read_idx];
        let alns2 = &final_alignments2[read_idx];

        // Get sequences and qualities from original batch
        let (seq_start1, seq_len1) = first_batch1.read_boundaries[read_idx];
        let seq1 = std::str::from_utf8(&first_batch1.seqs[seq_start1..seq_start1 + seq_len1])
            .unwrap_or("");
        let qual1 = std::str::from_utf8(&first_batch1.quals[seq_start1..seq_start1 + seq_len1])
            .unwrap_or("");

        let (seq_start2, seq_len2) = first_batch2.read_boundaries[read_idx];
        let seq2 = std::str::from_utf8(&first_batch2.seqs[seq_start2..seq_start2 + seq_len2])
            .unwrap_or("");
        let qual2 = std::str::from_utf8(&first_batch2.quals[seq_start2..seq_start2 + seq_len2])
            .unwrap_or("");

        // Write R1 primary alignment (or unmapped if none)
        if let Some(primary_aln) = alns1.first() {
            writeln!(
                writer,
                "{}",
                primary_aln.to_sam_string_with_seq(seq1, qual1)
            )
            .unwrap_or_else(|e| log::error!("Write error: {e}"));
            first_batch_records += 1;
        }

        // Write R2 primary alignment (or unmapped if none)
        if let Some(primary_aln) = alns2.first() {
            writeln!(
                writer,
                "{}",
                primary_aln.to_sam_string_with_seq(seq2, qual2)
            )
            .unwrap_or_else(|e| log::error!("Write error: {e}"));
            first_batch_records += 1;
        }
    }
    // Log per-batch timing (matches BWA-MEM2 format)
    let batch_cpu_elapsed = cputime() - batch_start_cpu;
    let batch_wall_elapsed = batch_start_wall.elapsed();
    log::info!(
        "Processed {} reads in {:.3} CPU sec, {:.3} real sec",
        first_batch_size * 2,
        batch_cpu_elapsed,
        batch_wall_elapsed.as_secs_f64()
    );

    // === PHASE 2: Stream remaining batches ===
    log::info!(
        "Phase 2: Streaming remaining batches (batch_size={})",
        opt.batch_size
    );

    let mut batch_num = 1u64;
    let mut total_rescued = rescued_first;
    let mut total_records = first_batch_records;

    loop {
        // Read batch
        let batch1 = match reader1.read_batch(opt.batch_size) {
            Ok(b) => b,
            Err(e) => {
                log::error!("Error reading batch {batch_num} from read1: {e}");
                break;
            }
        };
        let batch2 = match reader2.read_batch(opt.batch_size) {
            Ok(b) => b,
            Err(e) => {
                log::error!("Error reading batch {batch_num} from read2: {e}");
                break;
            }
        };

        // CRITICAL VALIDATION: Verify batch sizes match to prevent mis-pairing
        if batch1.len() != batch2.len() {
            log::error!(
                "Paired-end read count mismatch in batch {}: R1={} reads, R2={} reads",
                batch_num,
                batch1.len(),
                batch2.len()
            );
            log::error!(
                "Paired-end FASTQ files must have exactly the same number of reads in the same order."
            );
            log::error!(
                "Common causes: truncated file, missing reads, corrupted data, or mismatched file pairs."
            );
            log::error!(
                "Please verify file integrity with: wc -l {} {}",
                read1_file,
                read2_file
            );
            log::error!("Aborting to prevent incorrect alignments.");
            break;
        }

        // Check for EOF synchronization
        // If R1 is empty, R2 must also be empty (and vice versa)
        if batch1.is_empty() && !batch2.is_empty() {
            log::error!(
                "R1 file ended but R2 has {} reads remaining in batch {}. Files are not properly paired.",
                batch2.len(),
                batch_num
            );
            log::error!(
                "Please verify files contain the same number of reads: wc -l {} {}",
                read1_file,
                read2_file
            );
            break;
        }
        if !batch1.is_empty() && batch2.is_empty() {
            log::error!(
                "R2 file ended but R1 has {} reads remaining in batch {}. Files are not properly paired.",
                batch1.len(),
                batch_num
            );
            log::error!(
                "Please verify files contain the same number of reads: wc -l {} {}",
                read1_file,
                read2_file
            );
            break;
        }

        // Check for EOF (both files must end together)
        if batch1.is_empty() {
            log::debug!("[Main] EOF reached after {batch_num} batches");
            break;
        }

        batch_num += 1;

        let batch_size = batch1.names.len();
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
        total_reads += batch_size * 2;
        total_bases += batch_bp;

        log::debug!(
            "[Main] Batch {batch_num}: Received {batch_size} pairs from channel (cumulative: {total_reads} reads)"
        );
        log::info!(
            "read_chunk: {}, work_chunk_size: {}, nseq: {}",
            opt.batch_size,
            batch_bp,
            batch_size * 2
        );

        // Track per-batch timing with phase breakdown
        let batch_start_cpu = cputime();
        let batch_start_wall = Instant::now();

        // PHASE 1: Alignment (SoA pipeline)
        let align_start = Instant::now();
        let align_start_cpu = cputime();
        let num_pairs = batch1.names.len();
        let bwa_idx_clone = Arc::clone(&bwa_idx);
        let opt_clone = Arc::clone(&opt);
        let batch_start_id = pairs_processed; // Capture for closure

        // SoA batch alignment for R1 and R2 (parallel processing)
        let pac_ref = Arc::new(&pac[..]);
        let (soa_result1, soa_result2) = rayon::join(
            || {
                process_batch_parallel(
                    &batch1,
                    &bwa_idx_clone,
                    &pac_ref,
                    &opt_clone,
                    batch_start_id,
                    simd_engine,
                )
            },
            || {
                process_batch_parallel(
                    &batch2,
                    &bwa_idx_clone,
                    &pac_ref,
                    &opt_clone,
                    batch_start_id,
                    simd_engine,
                )
            },
        );

        // Pure SoA pipeline - no AoS conversions
        let align_cpu = cputime() - align_start_cpu;
        let align_wall = align_start.elapsed();

        // Update global pair counter
        pairs_processed += num_pairs as u64;

        // PHASE 1.5: Re-bootstrap insert size statistics (pure SoA)
        // Sample first BOOTSTRAP_BATCH_SIZE pairs for stats update
        // SoA functions handle partial batches internally
        current_stats = bootstrap_insert_size_stats_soa(&soa_result1, &soa_result2, l_pac);

        // PHASE 2: HYBRID AoS/SoA Pairing
        let pairing_start = Instant::now();
        let pairing_start_cpu = cputime();

        // Convert SoA results to AoS for pairing (avoids index collision bug)
        let mut alignments1 = soa_result1.to_aos();
        let mut alignments2 = soa_result2.to_aos();

        // PHASE 2.1: AoS pairing (correct per-read indexing)
        for read_idx in 0..alignments1.len() {
            let alns1 = &mut alignments1[read_idx];
            let alns2 = &mut alignments2[read_idx];

            if let Some((idx1, idx2, _pair_score, _sub_score)) = pairing_aos::mem_pair(
                &current_stats,
                alns1,
                alns2,
                opt.a, // Match score
                batch_start_id + read_idx as u64,
                l_pac,
            ) {
                // Mark as properly paired
                alns1[idx1].flag |= sam_flags::PROPER_PAIR;
                alns2[idx2].flag |= sam_flags::PROPER_PAIR;

                // Set mate information for both reads
                let (aln1, aln2) = (&alns1[idx1].clone(), &alns2[idx2].clone());
                set_mate_info_aos(&mut alns1[idx1], &aln2, true, l_pac);
                set_mate_info_aos(&mut alns2[idx2], &aln1, false, l_pac);
            } else if !alns1.is_empty() && !alns2.is_empty() {
                // Singleton: both mapped but not properly paired
                // Still need to set mate info for SAM spec compliance (GATK validation)
                let (aln1, aln2) = (&alns1[0].clone(), &alns2[0].clone());
                set_mate_info_aos(&mut alns1[0], &aln2, true, l_pac);
                set_mate_info_aos(&mut alns2[0], &aln1, false, l_pac);
            }
        }

        // PHASE 2.5: Mate rescue (convert to SoA temporarily for SIMD batching)
        let mut soa_for_rescue1 = SoAAlignmentResult::from_aos(&alignments1);
        let mut soa_for_rescue2 = SoAAlignmentResult::from_aos(&alignments2);

        // Track primary alignments for rescue
        let primary_r1: Vec<usize> = (0..soa_for_rescue1.num_reads())
            .map(|read_idx| {
                let (start, count) = soa_for_rescue1.read_alignment_boundaries[read_idx];
                (start..start + count)
                    .find(|&idx| soa_for_rescue1.flags[idx] & sam_flags::PROPER_PAIR != 0)
                    .unwrap_or(start)
            })
            .collect();

        let primary_r2: Vec<usize> = (0..soa_for_rescue2.num_reads())
            .map(|read_idx| {
                let (start, count) = soa_for_rescue2.read_alignment_boundaries[read_idx];
                (start..start + count)
                    .find(|&idx| soa_for_rescue2.flags[idx] & sam_flags::PROPER_PAIR != 0)
                    .unwrap_or(start)
            })
            .collect();

        let rescued = mate_rescue_soa(
            &mut soa_for_rescue1,
            &mut soa_for_rescue2,
            &batch1,
            &batch2,
            &primary_r1,
            &primary_r2,
            pac,
            &bwa_idx,
            &current_stats,
            opt.pen_unpaired,
            opt.max_matesw,
            Some(simd_engine),
        );
        total_rescued += rescued;

        // Convert back to AoS after rescue
        alignments1 = soa_for_rescue1.to_aos();
        alignments2 = soa_for_rescue2.to_aos();

        let pairing_cpu = cputime() - pairing_start_cpu;
        let pairing_wall = pairing_start.elapsed();

        // PHASE 3: Output using AoS (avoids SoA index bugs)
        // No need to call finalize_pairs_soa since AoS pairing already set correct flags
        let output_start = Instant::now();
        let output_start_cpu = cputime();

        let final_alignments1 = alignments1;
        let final_alignments2 = alignments2;

        // Write alignments using direct SAM output (per-read iteration)
        let mut records = 0;
        for read_idx in 0..final_alignments1.len() {
            let alns1 = &final_alignments1[read_idx];
            let alns2 = &final_alignments2[read_idx];

            // Get sequences and qualities from original batch
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

            // Write R1 primary alignment (or unmapped if none)
            if let Some(primary_aln) = alns1.first() {
                writeln!(
                    writer,
                    "{}",
                    primary_aln.to_sam_string_with_seq(seq1, qual1)
                )
                .unwrap_or_else(|e| log::error!("Write error: {e}"));
                records += 1;
            }

            // Write R2 primary alignment (or unmapped if none)
            if let Some(primary_aln) = alns2.first() {
                writeln!(
                    writer,
                    "{}",
                    primary_aln.to_sam_string_with_seq(seq2, qual2)
                )
                .unwrap_or_else(|e| log::error!("Write error: {e}"));
                records += 1;
            }
        }
        let output_cpu = cputime() - output_start_cpu;
        let output_wall = output_start.elapsed();
        total_records += records;

        // Log per-batch timing with phase breakdown
        let batch_cpu_elapsed = cputime() - batch_start_cpu;
        let batch_wall_elapsed = batch_start_wall.elapsed();

        // Phase timing breakdown (only at INFO level for visibility)
        log::info!(
            "  Phases: align={:.1}s/{:.1}s ({:.0}%), pairing+rescue={:.1}s/{:.1}s ({:.0}%), output={:.1}s/{:.1}s ({:.0}%)",
            align_cpu,
            align_wall.as_secs_f64(),
            100.0 * align_cpu / batch_cpu_elapsed.max(0.001),
            pairing_cpu,
            pairing_wall.as_secs_f64(),
            100.0 * pairing_cpu / batch_cpu_elapsed.max(0.001),
            output_cpu,
            output_wall.as_secs_f64(),
            100.0 * output_cpu / batch_cpu_elapsed.max(0.001),
        );
        log::info!(
            "Processed {} reads in {:.3} CPU sec, {:.3} real sec",
            batch_size * 2,
            batch_cpu_elapsed,
            batch_wall_elapsed.as_secs_f64()
        );

        // Log progress every 10 batches
        if batch_num % 10 == 0 {
            log::info!(
                "Processed {batch_num} batches, {total_records} records written, {total_rescued} rescued"
            );
        }

        // batch_alignments and batch dropped here, memory freed
    }

    // Print summary statistics with total CPU + wall time
    let total_cpu = cputime() - start_cpu;
    let elapsed = start_time.elapsed();
    log::info!(
        "Complete: {} batches, {} reads ({} bp), {} records, {} pairs rescued in {:.3} CPU sec, {:.3} real sec",
        batch_num,
        total_reads,
        total_bases,
        total_records,
        total_rescued,
        total_cpu,
        elapsed.as_secs_f64()
    );
}
