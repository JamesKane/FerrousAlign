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

use super::insert_size::{InsertSizeStats, bootstrap_insert_size_stats};
use super::mate_rescue::{
    MateRescueJob, execute_mate_rescue_batch_with_engine, prepare_mate_rescue_jobs_for_anchor,
    result_to_alignment,
};
use super::pairing::mem_pair;
use super::super::finalization::Alignment;
use super::super::finalization::mark_secondary_alignments;
use super::super::finalization::sam_flags;
use super::super::mem_opt::MemOpt;
use super::super::pipeline::align_read_deferred;
use crate::alignment::utils::encode_sequence;
use crate::compute::ComputeContext;
use crate::compute::simd_abstraction::simd::SimdEngineType;
use super::super::index::index::BwaIndex;
use crate::io::sam_output::{
    PairedFlagContext, create_unmapped_paired, prepare_paired_alignment_read1,
    prepare_paired_alignment_read2, write_sam_record,
};
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

    let mut reader1 = match crate::io::fastq_reader::FastqReader::new(read1_file) {
        Ok(r) => r,
        Err(e) => {
            log::error!("Error opening read1 file {}: {}", read1_file, e);
            return;
        }
    };
    let mut reader2 = match crate::io::fastq_reader::FastqReader::new(read2_file) {
        Ok(r) => r,
        Err(e) => {
            log::error!("Error opening read2 file {}: {}", read2_file, e);
            return;
        }
    };

    // Read first batch with small size for insert size bootstrap
    let first_batch1 = match reader1.read_batch(BOOTSTRAP_BATCH_SIZE) {
        Ok(b) => b,
        Err(e) => {
            log::error!("Error reading first batch from read1: {}", e);
            return;
        }
    };
    let first_batch2 = match reader2.read_batch(BOOTSTRAP_BATCH_SIZE) {
        Ok(b) => b,
        Err(e) => {
            log::error!("Error reading first batch from read2: {}", e);
            return;
        }
    };

    if first_batch1.names.is_empty() {
        log::warn!("No data to process (empty input files)");
        return;
    }

    let first_batch_size = first_batch1.names.len();
    let first_batch_bp: usize = first_batch1.seqs.iter().map(|s| s.len()).sum::<usize>()
        + first_batch2.seqs.iter().map(|s| s.len()).sum::<usize>();
    total_reads += first_batch_size * 2;
    total_bases += first_batch_bp;

    log::debug!(
        "[Main] Batch 0: Received {} pairs from channel",
        first_batch_size
    );
    log::info!(
        "read_chunk: {}, work_chunk_size: {}, nseq: {}",
        BOOTSTRAP_BATCH_SIZE,
        first_batch_bp,
        first_batch_size * 2
    );

    // Track per-batch timing
    let batch_start_cpu = cputime();
    let batch_start_wall = Instant::now();

    // Process first batch in parallel
    let num_pairs = first_batch1.names.len();
    let bwa_idx_clone = Arc::clone(&bwa_idx);
    let pac_clone = &pac; // Borrow pac for this scope
    let opt_clone = Arc::clone(&opt);
    let batch_start_id = pairs_processed; // Capture for closure

    let compute_backend = compute_ctx.backend.clone();

    let mut first_batch_alignments: Vec<(Vec<Alignment>, Vec<Alignment>)> = (0..num_pairs)
        .into_par_iter()
        .map(|i| {
            // Global read ID for deterministic hash tie-breaking (matches C++ bwamem_pair.cpp:416-417)
            // BWA-MEM2 uses (pair_id << 1) | 0 for read1, (pair_id << 1) | 1 for read2
            let pair_id = batch_start_id + i as u64;

            let a1 = align_read_deferred(
                &bwa_idx_clone,
                pac_clone,
                &first_batch1.names[i],
                &first_batch1.seqs[i],
                &first_batch1.quals[i],
                &opt_clone,
                compute_backend.clone(),
                (pair_id << 1) | 0,
                true, // skip secondary marking - done after pairing
            );
            let a2 = align_read_deferred(
                &bwa_idx_clone,
                pac_clone,
                &first_batch2.names[i],
                &first_batch2.seqs[i],
                &first_batch2.quals[i],
                &opt_clone,
                compute_backend.clone(),
                (pair_id << 1) | 1,
                true, // skip secondary marking - done after pairing
            );
            (a1, a2)
        })
        .collect();

    // Update global pair counter
    pairs_processed += num_pairs as u64;

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
        bootstrap_insert_size_stats(
            &first_batch_alignments,
            bwa_idx.bns.packed_sequence_length as i64,
        )
    };

    // Prepare sequences for mate rescue and output
    let first_batch_seqs1 = first_batch1.as_tuple_refs();
    let first_batch_seqs2 = first_batch2.as_tuple_refs();

    // Mate rescue on first batch (BWA-MEM2: run unconditionally on top alignments)
    // Use horizontal SIMD for batched mate rescue when available
    let simd_engine = compute_ctx.backend.simd_engine();
    let rescued_first = mate_rescue_batch(
        &mut first_batch_alignments,
        &first_batch_seqs1,
        &first_batch_seqs2,
        &pac,
        &current_stats,
        &bwa_idx,
        opt.max_matesw as usize,
        simd_engine,
    );
    log::info!("First batch: {} pairs rescued", rescued_first);

    // Prepare sequences for output (need owned versions)
    let first_batch_seqs1_owned: Vec<_> = first_batch1
        .names
        .into_iter()
        .zip(first_batch1.seqs)
        .zip(first_batch1.quals)
        .map(|((n, s), q)| (n, s, q))
        .collect();
    let first_batch_seqs2_owned: Vec<_> = first_batch2
        .names
        .into_iter()
        .zip(first_batch2.seqs)
        .zip(first_batch2.quals)
        .map(|((n, s), q)| (n, s, q))
        .collect();

    // Output first batch immediately
    let l_pac = bwa_idx.bns.packed_sequence_length as i64;
    let first_batch_records = output_batch_paired(
        first_batch_alignments,
        &first_batch_seqs1_owned,
        &first_batch_seqs2_owned,
        &current_stats,
        writer,
        &opt,
        0,
        l_pac,
    )
    .unwrap_or_else(|e| {
        log::error!("Error writing first batch: {}", e);
        0
    });
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
                log::error!("Error reading batch {} from read1: {}", batch_num, e);
                break;
            }
        };
        let batch2 = match reader2.read_batch(opt.batch_size) {
            Ok(b) => b,
            Err(e) => {
                log::error!("Error reading batch {} from read2: {}", batch_num, e);
                break;
            }
        };

        // Check for EOF
        if batch1.names.is_empty() {
            log::debug!("[Main] EOF reached after {} batches", batch_num);
            break;
        }

        batch_num += 1;

        let batch_size = batch1.names.len();
        let batch_bp: usize = batch1.seqs.iter().map(|s| s.len()).sum::<usize>()
            + batch2.seqs.iter().map(|s| s.len()).sum::<usize>();
        total_reads += batch_size * 2;
        total_bases += batch_bp;

        log::debug!(
            "[Main] Batch {}: Received {} pairs from channel (cumulative: {} reads)",
            batch_num,
            batch_size,
            total_reads
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

        // PHASE 1: Alignment (parallel)
        let align_start = Instant::now();
        let align_start_cpu = cputime();
        let num_pairs = batch1.names.len();
        let bwa_idx_clone = Arc::clone(&bwa_idx);
        let pac_clone = &pac; // Borrow pac for this scope
        let opt_clone = Arc::clone(&opt);
        let batch_start_id = pairs_processed; // Capture for closure

        let mut batch_alignments: Vec<(Vec<Alignment>, Vec<Alignment>)> = (0..num_pairs)
            .into_par_iter()
            .map(|i| {
                // Global read ID for deterministic hash tie-breaking (matches C++ bwamem_pair.cpp:416-417)
                // BWA-MEM2 uses (pair_id << 1) | 0 for read1, (pair_id << 1) | 1 for read2
                let pair_id = batch_start_id + i as u64;

                let a1 = align_read_deferred(
                    &bwa_idx_clone,
                    pac_clone,
                    &batch1.names[i],
                    &batch1.seqs[i],
                    &batch1.quals[i],
                    &opt_clone,
                    compute_backend.clone(),
                    (pair_id << 1) | 0,
                    true, // skip secondary marking - done after pairing
                );
                let a2 = align_read_deferred(
                    &bwa_idx_clone,
                    pac_clone,
                    &batch2.names[i],
                    &batch2.seqs[i],
                    &batch2.quals[i],
                    &opt_clone,
                    compute_backend.clone(),
                    (pair_id << 1) | 1,
                    true, // skip secondary marking - done after pairing
                );
                (a1, a2)
            })
            .collect();
        let align_cpu = cputime() - align_start_cpu;
        let align_wall = align_start.elapsed();

        // Update global pair counter
        pairs_processed += num_pairs as u64;

        // PHASE 1.5: Re-bootstrap insert size statistics from current batch
        // Take a sample from the current batch to update statistics
        // Use an iterator to take the first BOOTSTRAP_BATCH_SIZE elements
        let sample_size = BOOTSTRAP_BATCH_SIZE.min(batch_alignments.len());
        let sampled_alignments = batch_alignments.iter().take(sample_size).cloned().collect::<Vec<_>>();

        if !sampled_alignments.is_empty() {
            current_stats = bootstrap_insert_size_stats(
                &sampled_alignments,
                bwa_idx.bns.packed_sequence_length as i64,
            );
        }

        // Prepare sequences for mate rescue
        let batch_seqs1 = batch1.as_tuple_refs();
        let batch_seqs2 = batch2.as_tuple_refs();

        // PHASE 2: Mate rescue (parallel)
        let rescue_start = Instant::now();
        let rescue_start_cpu = cputime();
        let rescued = mate_rescue_batch(
            &mut batch_alignments,
            &batch_seqs1,
            &batch_seqs2,
            &pac,
            &current_stats,
            &bwa_idx,
            opt.max_matesw as usize,
            simd_engine,
        );
        let rescue_cpu = cputime() - rescue_start_cpu;
        let rescue_wall = rescue_start.elapsed();
        total_rescued += rescued;

        // Prepare sequences for output (need owned versions)
        let batch_seqs1_owned: Vec<_> = batch1
            .names
            .into_iter()
            .zip(batch1.seqs)
            .zip(batch1.quals)
            .map(|((n, s), q)| (n, s, q))
            .collect();
        let batch_seqs2_owned: Vec<_> = batch2
            .names
            .into_iter()
            .zip(batch2.seqs)
            .zip(batch2.quals)
            .map(|((n, s), q)| (n, s, q))
            .collect();

        // PHASE 3: Output (parallel format + sequential write)
        let output_start = Instant::now();
        let output_start_cpu = cputime();
        let records = output_batch_paired(
            batch_alignments,
            &batch_seqs1_owned,
            &batch_seqs2_owned,
            &current_stats,
            writer,
            &opt,
            batch_num * opt.batch_size as u64,
            l_pac,
        )
        .unwrap_or_else(|e| {
            log::error!("Error writing batch: {}", e);
            0
        });
        let output_cpu = cputime() - output_start_cpu;
        let output_wall = output_start.elapsed();
        total_records += records;

        // Log per-batch timing with phase breakdown
        let batch_cpu_elapsed = cputime() - batch_start_cpu;
        let batch_wall_elapsed = batch_start_wall.elapsed();

        // Phase timing breakdown (only at INFO level for visibility)
        log::info!(
            "  Phases: align={:.1}s/{:.1}s ({:.0}%), rescue={:.1}s/{:.1}s ({:.0}%), output={:.1}s/{:.1}s ({:.0}%)",
            align_cpu, align_wall.as_secs_f64(), 100.0 * align_cpu / batch_cpu_elapsed.max(0.001),
            rescue_cpu, rescue_wall.as_secs_f64(), 100.0 * rescue_cpu / batch_cpu_elapsed.max(0.001),
            output_cpu, output_wall.as_secs_f64(), 100.0 * output_cpu / batch_cpu_elapsed.max(0.001),
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
                "Processed {} batches, {} records written, {} rescued",
                batch_num,
                total_records,
                total_rescued
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

/// Perform mate rescue on a single batch.
///
/// BWA-MEM2 behavior: Run mate rescue UNCONDITIONALLY on the top alignments from
/// each read, using them as anchors to find potential mates. This is done even
/// when both reads have alignments, because the best individual alignments may
/// not form a concordant pair - mate rescue can find a concordant position.
///
/// Parameters:
/// - max_matesw: Maximum number of alignments to use as anchors (default: 50)
///
/// Returns number of pairs where at least one mate was rescued.
///
/// OPTIMIZATION (Session 48): Three-phase batched mate rescue
/// Phase 1: Collect all SW jobs across all pairs (parallel collection)
/// Phase 2: Execute all SW in parallel using rayon
/// Phase 3: Distribute results back to pairs
fn mate_rescue_batch(
    batch_pairs: &mut [(Vec<Alignment>, Vec<Alignment>)],
    batch_seqs1: &[(&str, &[u8], &str)], // (name, seq, qual) for read1
    batch_seqs2: &[(&str, &[u8], &str)], // (name, seq, qual) for read2
    pac: &[u8],
    stats: &[InsertSizeStats; 4],
    bwa_idx: &BwaIndex,
    max_matesw: usize,
    simd_engine: Option<SimdEngineType>,
) -> usize {
    use std::time::Instant;

    if pac.is_empty() {
        return 0;
    }

    // ========================================================================
    // PHASE 1: Collect all SW jobs across all pairs
    // ========================================================================
    // We collect jobs in parallel from each pair, then flatten into one big list.
    // Each job stores (pair_index, which_read) so we know where to put results.

    let phase1_start = Instant::now();
    let all_jobs: Vec<MateRescueJob> = batch_pairs
        .par_iter()
        .enumerate()
        .flat_map(|(i, (alns1, alns2))| {
            let (name1, seq1, _qual1) = batch_seqs1[i];
            let (name2, seq2, _qual2) = batch_seqs2[i];

            let mut pair_jobs = Vec::new();

            // Collect jobs for rescuing read2 using read1's anchors
            if !alns1.is_empty() {
                let mate_seq = encode_sequence(seq2);
                let num_anchors = alns1.len().min(max_matesw);

                for j in 0..num_anchors {
                    let jobs = prepare_mate_rescue_jobs_for_anchor(
                        bwa_idx, pac, stats, &alns1[j], &mate_seq, name2, alns2, i,
                        false, // rescuing read2
                    );
                    pair_jobs.extend(jobs);
                }
            }

            // Collect jobs for rescuing read1 using read2's anchors
            if !alns2.is_empty() {
                let mate_seq = encode_sequence(seq1);
                let num_anchors = alns2.len().min(max_matesw);

                for j in 0..num_anchors {
                    let jobs = prepare_mate_rescue_jobs_for_anchor(
                        bwa_idx, pac, stats, &alns2[j], &mate_seq, name1, alns1, i,
                        true, // rescuing read1
                    );
                    pair_jobs.extend(jobs);
                }
            }

            pair_jobs
        })
        .collect();

    let phase1_elapsed = phase1_start.elapsed();

    if all_jobs.is_empty() {
        return 0;
    }

    log::debug!(
        "Mate rescue: collected {} SW jobs across {} pairs",
        all_jobs.len(),
        batch_pairs.len()
    );

    // ========================================================================
    // PHASE 2: Execute all SW in parallel
    // ========================================================================
    // Use horizontal SIMD batching when a SIMD engine is specified
    let phase2_start = Instant::now();
    let mut jobs_for_execution = all_jobs;
    let results = execute_mate_rescue_batch_with_engine(&mut jobs_for_execution, simd_engine);
    let phase2_elapsed = phase2_start.elapsed();

    // ========================================================================
    // PHASE 3: Distribute results back to pairs (PARALLEL)
    // ========================================================================
    // OPTIMIZATION (Session 53): Parallelize result distribution
    // The expensive part is result_to_alignment() which computes CIGAR, MD tags, etc.
    // We use parallel fold to:
    // 1. Process results in parallel (compute alignments)
    // 2. Group into thread-local HashMaps by (pair_idx, is_read1)
    // 3. Merge results at the end
    use std::collections::HashMap;

    let phase3_start = Instant::now();

    // Type: HashMap<(pair_idx, is_read1), Vec<Alignment>>
    type ResultMap = HashMap<(usize, bool), Vec<Alignment>>;

    // Parallel fold: each thread builds its own HashMap
    let merged_results: ResultMap = results
        .par_iter()
        .fold(
            || ResultMap::new(),
            |mut acc, result| {
                let job = &jobs_for_execution[result.job_index];

                // Convert SW result to Alignment (expensive - CIGAR, MD, NM)
                if let Some(alignment) = result_to_alignment(job, &result.aln, bwa_idx, pac) {
                    let key = (job.pair_index, job.rescuing_read1);
                    acc.entry(key).or_default().push(alignment);
                }
                acc
            },
        )
        .reduce(
            || ResultMap::new(),
            |mut a, b| {
                // Merge maps: combine vectors for same keys
                for (key, mut alignments) in b {
                    a.entry(key).or_default().append(&mut alignments);
                }
                a
            },
        );

    // Apply merged results to batch_pairs
    let mut pairs_rescued = vec![false; batch_pairs.len()];
    for ((pair_idx, is_read1), alignments) in merged_results {
        if is_read1 {
            batch_pairs[pair_idx].0.extend(alignments);
        } else {
            batch_pairs[pair_idx].1.extend(alignments);
        }
        pairs_rescued[pair_idx] = true;
    }

    let phase3_elapsed = phase3_start.elapsed();

    // Count how many pairs had at least one rescue
    let rescued_count = pairs_rescued.iter().filter(|&&x| x).count();

    // Log phase timing breakdown
    log::info!(
        "  Rescue phases: collect={:.2}s, SW={:.2}s, distribute={:.2}s ({} jobs)",
        phase1_elapsed.as_secs_f64(),
        phase2_elapsed.as_secs_f64(),
        phase3_elapsed.as_secs_f64(),
        results.len()
    );

    log::debug!(
        "Mate rescue: {} SW jobs produced {} rescued alignments in {} pairs",
        results.len(),
        results
            .iter()
            .filter(
                |r| result_to_alignment(&jobs_for_execution[r.job_index], &r.aln, bwa_idx, pac)
                    .is_some()
            )
            .count(),
        rescued_count
    );

    rescued_count
}

// ============================================================================
// PAIRED-END OUTPUT HELPERS
// ============================================================================

/// Filter alignments by score threshold, creating unmapped if all filtered
fn filter_alignments_by_threshold(
    alignments: &mut Vec<Alignment>,
    name: &str,
    seq: &[u8],
    is_first_in_pair: bool,
    threshold: i32,
) {
    alignments.retain(|a| a.score >= threshold);

    if alignments.is_empty() {
        log::debug!(
            "{}: All alignments filtered by score threshold, creating unmapped",
            name
        );
        alignments.push(create_unmapped_paired(name, seq, is_first_in_pair));
    }
}

/// Select best pair indices and determine if properly paired
///
/// # Arguments
/// * `alignments1` - Alignments for read 1
/// * `alignments2` - Alignments for read 2
/// * `stats` - Insert size statistics for each orientation
/// * `pair_id` - Unique pair identifier for tie-breaking
/// * `l_pac` - Length of packed reference (for bidirectional coordinate conversion)
fn select_best_pair(
    alignments1: &[Alignment],
    alignments2: &[Alignment],
    stats: &[InsertSizeStats; 4],
    pair_id: u64,
    l_pac: i64,
) -> (usize, usize, bool) {
    if alignments1.is_empty() || alignments2.is_empty() {
        return (0, 0, false);
    }

    let pair_result = mem_pair(stats, alignments1, alignments2, 2, pair_id, l_pac);

    if let Some((idx1, idx2, _pair_score, _sub_score)) = pair_result {
        log::trace!("mem_pair selected best_idx1={}, best_idx2={}", idx1, idx2);
        (idx1, idx2, true)
    } else {
        // Fallback: even if mem_pair didn't find a valid pair, check if the top hits
        // from both reads constitute a proper pair based on same reference and insert size
        // (Matches bwa-mem2 logic in bwamem_pair.cpp lines 536-540)
        let is_proper = check_proper_pair_fallback(&alignments1[0], &alignments2[0], stats, l_pac);
        log::trace!(
            "No valid pair from mem_pair, fallback proper_pair={}",
            is_proper
        );
        (0, 0, is_proper)
    }
}

/// Fallback check for proper pairing when mem_pair returns None
/// Matches bwa-mem2 logic (bwamem_pair.cpp:536-540): if top hits are on same reference
/// and within insert size bounds, mark as properly paired.
///
/// Uses the same bidirectional coordinate system as BWA-MEM2's mem_infer_dir():
/// - Forward strand: bidir_pos = leftmost coordinate
/// - Reverse strand: bidir_pos = (2*l_pac - 1) - rightmost coordinate
///
/// # Arguments
/// * `aln1` - Best alignment for read 1
/// * `aln2` - Best alignment for read 2
/// * `stats` - Insert size statistics for each orientation
/// * `l_pac` - Length of packed reference (for bidirectional coordinate conversion)
fn check_proper_pair_fallback(
    aln1: &Alignment,
    aln2: &Alignment,
    stats: &[InsertSizeStats; 4],
    l_pac: i64,
) -> bool {
    // Both must be mapped
    if (aln1.flag & sam_flags::UNMAPPED) != 0 || (aln2.flag & sam_flags::UNMAPPED) != 0 {
        return false;
    }

    // Must be on same reference (BWA-MEM2: h[0].rid == h[1].rid)
    if aln1.ref_id != aln2.ref_id {
        return false;
    }

    // Convert SAM positions to bidirectional coordinates
    // This matches BWA-MEM2's a[0].a[0].rb and a[1].a[0].rb
    let is_rev1 = (aln1.flag & sam_flags::REVERSE) != 0;
    let is_rev2 = (aln2.flag & sam_flags::REVERSE) != 0;
    let ref_len1 = aln1.reference_length() as i64;
    let ref_len2 = aln2.reference_length() as i64;

    // BWA-MEM2 bidirectional coordinate conversion:
    // - Forward strand: rb = leftmost position
    // - Reverse strand: rb = (2*l_pac - 1) - rightmost position
    let bidir_pos1 = if is_rev1 {
        let rightmost = aln1.pos as i64 + ref_len1 - 1;
        (l_pac << 1) - 1 - rightmost
    } else {
        aln1.pos as i64
    };

    let bidir_pos2 = if is_rev2 {
        let rightmost = aln2.pos as i64 + ref_len2 - 1;
        (l_pac << 1) - 1 - rightmost
    } else {
        aln2.pos as i64
    };

    // Use infer_orientation which matches BWA-MEM2's mem_infer_dir()
    use super::insert_size::infer_orientation;
    let (orientation, dist) = infer_orientation(l_pac, bidir_pos1, bidir_pos2);

    // Check if this orientation has valid statistics
    if stats[orientation].failed {
        return false;
    }

    let dist = dist as i64;

    // Check if within bounds
    dist >= stats[orientation].low as i64 && dist <= stats[orientation].high as i64
}

/// Extract mate information for flag context
fn get_mate_info(alignments: &[Alignment], best_idx: usize) -> (String, u64, u16, i32) {
    if let Some(aln) = alignments.get(best_idx) {
        let ref_len = aln.reference_length();
        (aln.ref_name.clone(), aln.pos, aln.flag, ref_len)
    } else {
        ("*".to_string(), 0, 0, 0)
    }
}

/// Select which alignments to output for a paired read
fn select_output_indices(
    alignments: &[Alignment],
    best_idx: usize,
    output_all: bool,
    threshold: i32,
) -> Vec<usize> {
    let mut indices = Vec::new();
    for (idx, alignment) in alignments.iter().enumerate() {
        let is_unmapped = alignment.flag & sam_flags::UNMAPPED != 0;
        let is_primary = idx == best_idx;
        let is_supplementary = alignment.flag & sam_flags::SUPPLEMENTARY != 0;

        // BWA-MEM2 mem_reg2sam line 1538: ALL alignments must have score >= opt->T
        // Apply score threshold to supplementary alignments as well
        let should_output = if output_all {
            is_unmapped || alignment.score >= threshold
        } else {
            (is_primary || is_supplementary) && (is_unmapped || alignment.score >= threshold)
        };

        if should_output {
            indices.push(idx);
        }
    }
    indices
}

/// Reorder alignments so that the best pair alignment is at index 0
/// This ensures the pair-selected alignment becomes primary after mark_secondary_alignments
fn reorder_for_best_pair(alignments: &mut Vec<Alignment>, best_idx: usize) {
    if best_idx > 0 && best_idx < alignments.len() {
        // Swap the best pair alignment to position 0
        alignments.swap(0, best_idx);
        log::debug!(
            "Reordered alignment: moved idx {} to position 0 (score: {})",
            best_idx,
            alignments[0].score
        );
    }
}

// Format a batch of paired-end alignments in PARALLEL
// Returns Vec of formatted SAM record strings
//
// This is the key optimization: all CPU-intensive work (pairing, flag setting,
// secondary marking, SAM string formatting) happens in parallel using rayon.
// The caller just needs to write the pre-formatted strings sequentially.
fn format_batch_paired_parallel(
    batch_pairs: Vec<(Vec<Alignment>, Vec<Alignment>)>,
    batch_seqs1: &[(String, Vec<u8>, String)],
    batch_seqs2: &[(String, Vec<u8>, String)],
    stats: &[InsertSizeStats; 4],
    opt: &MemOpt,
    starting_pair_id: u64,
    l_pac: i64,
) -> Vec<String> {
    let rg_id = opt
        .read_group
        .as_ref()
        .and_then(|rg| super::super::mem_opt::MemOpt::extract_rg_id(rg));

    // Process all pairs in parallel - this is where the CPU work happens
    batch_pairs
        .into_par_iter()
        .enumerate()
        .flat_map(|(pair_idx, (mut alignments1, mut alignments2))| {
            let (name1, seq1, qual1) = &batch_seqs1[pair_idx];
            let (name2, seq2, qual2) = &batch_seqs2[pair_idx];
            let pair_id = starting_pair_id + pair_idx as u64;

            // Stage 1: Filter by score threshold
            let pe_threshold = opt.t;
            filter_alignments_by_threshold(&mut alignments1, name1, seq1, true, pe_threshold);
            filter_alignments_by_threshold(&mut alignments2, name2, seq2, false, pe_threshold);

            // Stage 2: Select best pair
            let (best_idx1, best_idx2, is_properly_paired) =
                select_best_pair(&alignments1, &alignments2, stats, pair_id, l_pac);

            // Stage 2b: Reorder so best pair is at index 0
            reorder_for_best_pair(&mut alignments1, best_idx1);
            reorder_for_best_pair(&mut alignments2, best_idx2);

            // Stage 2c: Mark secondary/supplementary
            mark_secondary_alignments(&mut alignments1, opt);
            mark_secondary_alignments(&mut alignments2, opt);

            let best_idx1 = 0;
            let best_idx2 = 0;

            // Stage 3: Build mate contexts
            let (mate2_ref, mate2_pos, mate2_flag, mate2_ref_len) =
                get_mate_info(&alignments2, best_idx2);
            let (mate1_ref, mate1_pos, mate1_flag, mate1_ref_len) =
                get_mate_info(&alignments1, best_idx1);

            let mate2_cigar = alignments2
                .get(best_idx2)
                .map(|a| a.cigar_string())
                .unwrap_or_else(|| "*".to_string());
            let mate1_cigar = alignments1
                .get(best_idx1)
                .map(|a| a.cigar_string())
                .unwrap_or_else(|| "*".to_string());

            let ctx_for_read1 = PairedFlagContext {
                mate_ref: mate2_ref,
                mate_pos: mate2_pos,
                mate_flag: mate2_flag,
                mate_cigar: mate2_cigar,
                mate_ref_len: mate2_ref_len,
                is_properly_paired,
            };

            let ctx_for_read2 = PairedFlagContext {
                mate_ref: mate1_ref,
                mate_pos: mate1_pos,
                mate_flag: mate1_flag,
                mate_cigar: mate1_cigar,
                mate_ref_len: mate1_ref_len,
                is_properly_paired,
            };

            // Stage 4: Select output indices
            let output_indices1 =
                select_output_indices(&alignments1, best_idx1, opt.output_all_alignments, opt.t);
            let output_indices2 =
                select_output_indices(&alignments2, best_idx2, opt.output_all_alignments, opt.t);

            // Stage 5: Format SAM strings for read1
            let seq1_str = std::str::from_utf8(seq1).unwrap_or("");
            let mut records: Vec<String> = Vec::with_capacity(output_indices1.len() + output_indices2.len());

            for idx in output_indices1 {
                let is_primary = idx == best_idx1;
                prepare_paired_alignment_read1(
                    &mut alignments1[idx],
                    is_primary,
                    &ctx_for_read1,
                    rg_id.as_deref(),
                );
                records.push(alignments1[idx].to_sam_string_with_seq(seq1_str, qual1));
            }

            // Stage 6: Format SAM strings for read2
            let seq2_str = std::str::from_utf8(seq2).unwrap_or("");
            for idx in output_indices2 {
                let is_primary = idx == best_idx2;
                prepare_paired_alignment_read2(
                    &mut alignments2[idx],
                    is_primary,
                    &ctx_for_read2,
                    rg_id.as_deref(),
                );
                records.push(alignments2[idx].to_sam_string_with_seq(seq2_str, qual2));
            }

            records
        })
        .collect()
}

// Output a batch of paired-end alignments with proper flags and flushing
// Returns number of records written
//
// # Arguments
// * `batch_pairs` - Vec of (read1 alignments, read2 alignments) for each pair
// * `batch_seqs1` - Sequences for read 1: (name, seq, qual)
// * `batch_seqs2` - Sequences for read 2: (name, seq, qual)
// * `stats` - Insert size statistics for each orientation
// * `writer` - Output writer for SAM records
// * `opt` - Alignment options
// * `starting_pair_id` - Base pair ID for this batch
// * `l_pac` - Length of packed reference (for bidirectional coordinate conversion)
fn output_batch_paired(
    batch_pairs: Vec<(Vec<Alignment>, Vec<Alignment>)>,
    batch_seqs1: &[(String, Vec<u8>, String)], // (name, seq, qual) for read1
    batch_seqs2: &[(String, Vec<u8>, String)], // (name, seq, qual) for read2
    stats: &[InsertSizeStats; 4],
    writer: &mut Box<dyn Write>,
    opt: &MemOpt,
    starting_pair_id: u64,
    l_pac: i64,
) -> std::io::Result<usize> {
    // PARALLEL FORMATTING: All CPU-intensive work happens here in parallel
    let formatted_records = format_batch_paired_parallel(
        batch_pairs,
        batch_seqs1,
        batch_seqs2,
        stats,
        opt,
        starting_pair_id,
        l_pac,
    );

    let records_written = formatted_records.len();

    // SEQUENTIAL WRITES: Just write pre-formatted strings (fast)
    for record in formatted_records {
        writeln!(writer, "{}", record)?;
    }

    // Flush after each batch for incremental output
    writer.flush()?;

    Ok(records_written)
}
