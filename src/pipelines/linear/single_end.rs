// Single-end read processing module
//
// This module handles single-end FASTQ file processing, including:
// - Batched read loading (Stage 0)
// - Parallel alignment using Rayon (Stage 1)
// - SAM output formatting (Stage 2)
//
// The pipeline is cleanly separated:
// - Computation: align_read_deferred() returns Vec<Alignment>
// - Selection: sam_output::select_single_end_alignments() filters output
// - Output: sam_output::write_sam_record() writes to stream

use super::batch_extension::process_sub_batch_internal_soa;
use super::batch_extension::types::SoAAlignmentResult;
use super::index::index::BwaIndex;
use super::mem_opt::MemOpt;
use crate::compute::ComputeBackend;
use crate::compute::ComputeContext;
use crate::compute::simd_abstraction::simd::SimdEngineType;
use crate::io::sam_output::write_sam_records_soa;
use crate::io::soa_readers::SoaFastqReader;
use crate::utils::cputime;
use rayon::prelude::*;
use std::io::Write;
use std::sync::Arc;

// Batch processing constants (matching C++ bwa-mem2)
// Chunk size in base pairs (from C++ bwamem.cpp mem_opt_init: o->chunk_size = 10000000)
const CHUNK_SIZE_BASES: usize = 10_000_000;
// Assumed average read length for batch size calculation (typical Illumina)
const AVG_READ_LEN: usize = 101;
// Minimum batch size (BATCH_SIZE from C++ macro.h)
const MIN_BATCH_SIZE: usize = 512;

/// Process a batch in parallel chunks (single-end version)
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

// ============================================================================
// HETEROGENEOUS COMPUTE ENTRY POINT - SINGLE-END PROCESSING
// ============================================================================
//
// This function is the main entry point for single-end alignment processing.
// The compute_ctx parameter controls which hardware backend is used for
// alignment computations.
//
// Compute flow: process_single_end() → align_read_deferred() → extension
//
// To add GPU/NPU acceleration:
// 1. Pass compute_ctx through to align_read_deferred()
// 2. In extension, route based on compute_ctx.backend
// 3. Implement backend-specific alignment kernel
//
// ============================================================================
pub fn process_single_end(
    bwa_idx: &BwaIndex,
    query_files: &Vec<String>,
    writer: &mut Box<dyn Write>,
    opt: &MemOpt,
    compute_ctx: &ComputeContext,
) {
    use std::time::Instant;

    // Log the compute backend being used for this processing run
    log::debug!(
        "Single-end processing using backend: {:?}",
        compute_ctx.backend
    );

    // Wrap index in Arc for thread-safe sharing
    let bwa_idx = Arc::new(bwa_idx);
    let opt = Arc::new(opt);

    // PAC data is already loaded into memory in bwa_idx.bns.pac_data
    // Just reference it directly - no file I/O needed!
    log::debug!(
        "Using in-memory PAC data: {} bytes",
        bwa_idx.bns.pac_data.len()
    );
    let pac_data = Arc::new(bwa_idx.bns.pac_data.clone());

    // Track overall statistics
    let start_time = Instant::now();
    let start_cpu = cputime();
    let mut total_reads = 0usize;
    let mut total_bases = 0usize;
    let mut reads_processed = 0u64; // Global read counter for deterministic hash tie-breaking

    // Calculate optimal batch size based on thread count (matching C++ bwa-mem2)
    // Formula: (CHUNK_SIZE_BASES * n_threads) / AVG_READ_LEN
    // C++ fastmap.cpp:947: aux.task_size = opt->chunk_size * opt->n_threads
    // C++ fastmap.cpp:459: nreads = aux->actual_chunk_size / READ_LEN + 10
    let num_threads = opt.n_threads as usize;
    let batch_total_bases = CHUNK_SIZE_BASES * num_threads;
    let reads_per_batch = (batch_total_bases / AVG_READ_LEN).max(MIN_BATCH_SIZE);
    log::debug!(
        "Using batch size: {} reads ({} MB total, {} threads × {} MB/thread)",
        reads_per_batch,
        batch_total_bases / 1_000_000,
        num_threads,
        CHUNK_SIZE_BASES / 1_000_000
    );

    // Pure SoA pipeline - no AoS paths
    log::info!("Using end-to-end SoA pipeline");

    for query_file_name in query_files {
        process_single_end_soa(
            &bwa_idx,
            &pac_data,
            &opt,
            query_file_name,
            writer,
            compute_ctx,
            &mut reads_processed,
            &mut total_reads,
            &mut total_bases,
            reads_per_batch,
        );
    }

    // Print summary statistics with total CPU + wall time
    let total_cpu = cputime() - start_cpu;
    let elapsed = start_time.elapsed();
    log::info!(
        "Processed {} reads ({} bp) in {:.3} CPU sec, {:.3} real sec",
        total_reads,
        total_bases,
        total_cpu,
        elapsed.as_secs_f64()
    );
}

/// Process single-end reads using SoA pipeline (PR3 end-to-end)
fn process_single_end_soa(
    bwa_idx: &Arc<&BwaIndex>,
    pac_data: &Arc<Vec<u8>>,
    opt: &Arc<&MemOpt>,
    query_file_name: &str,
    writer: &mut Box<dyn Write>,
    compute_ctx: &ComputeContext,
    reads_processed: &mut u64,
    total_reads: &mut usize,
    total_bases: &mut usize,
    reads_per_batch: usize,
) {
    let mut reader = match SoaFastqReader::new(query_file_name) {
        Ok(r) => r,
        Err(e) => {
            log::error!("Error opening query file {query_file_name}: {e}");
            return;
        }
    };

    let engine = match &compute_ctx.backend {
        ComputeBackend::CpuSimd(e) => *e,
        _ => SimdEngineType::Engine128,
    };

    loop {
        // Stage 0: Read batch of reads into SoA format
        let soa_read_batch = match reader.read_batch(reads_per_batch) {
            Ok(b) => b,
            Err(e) => {
                log::error!("Error reading batch from {query_file_name}: {e}");
                break;
            }
        };

        if soa_read_batch.is_empty() {
            break; // EOF
        }

        let batch_size = soa_read_batch.len();

        // Calculate batch base pairs
        let batch_bp: usize = soa_read_batch
            .read_boundaries
            .iter()
            .map(|(_, len)| *len)
            .sum();
        *total_reads += batch_size;
        *total_bases += batch_bp;

        log::info!(
            "read_chunk: {reads_per_batch}, work_chunk_size: {batch_bp}, nseq: {batch_size}"
        );

        // Track per-batch timing
        let batch_start_cpu = cputime();
        let batch_start_wall = std::time::Instant::now();

        // Stage 1: Process batch using end-to-end SoA pipeline (parallel processing)
        let bwa_idx_clone = Arc::clone(bwa_idx);
        let pac_data_clone = Arc::clone(pac_data);
        let opt_clone = Arc::clone(opt);
        let batch_start_id = *reads_processed;

        let pac_slice: &[u8] = &pac_data_clone;
        let pac_ref = Arc::new(pac_slice);
        let soa_alignments = process_batch_parallel(
            &soa_read_batch,
            &bwa_idx_clone,
            &pac_ref,
            &opt_clone,
            batch_start_id,
            engine,
        );

        // Update global read counter for next batch
        *reads_processed += batch_size as u64;

        // Stage 2: Write output sequentially using SoA-aware SAM writer (PR4)
        let rg_id = opt
            .read_group
            .as_ref()
            .and_then(|rg| super::mem_opt::MemOpt::extract_rg_id(rg));

        if let Err(e) = write_sam_records_soa(
            writer,
            &soa_alignments,
            &soa_read_batch,
            opt,
            rg_id.as_deref(),
        ) {
            log::error!("Error writing SAM records: {e}");
        }

        // Log per-batch timing
        let batch_cpu_elapsed = cputime() - batch_start_cpu;
        let batch_wall_elapsed = batch_start_wall.elapsed();
        log::info!(
            "Processed {} reads in {:.3} CPU sec, {:.3} real sec",
            batch_size,
            batch_cpu_elapsed,
            batch_wall_elapsed.as_secs_f64()
        );

        if batch_size < reads_per_batch {
            break; // Last incomplete batch
        }
    }
}
