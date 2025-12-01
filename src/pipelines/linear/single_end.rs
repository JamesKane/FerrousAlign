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

use super::batch_extension::{process_batch_cross_read, process_batch_parallel_subbatch, process_sub_batch_internal_soa};
use super::finalization::Alignment;
use super::index::index::BwaIndex;
use super::mem_opt::MemOpt;
use super::pipeline::align_read_deferred;
use crate::compute::ComputeBackend;
use crate::compute::ComputeContext;
use crate::compute::simd_abstraction::simd::SimdEngineType;
use crate::io::fastq_reader::FastqReader;
use crate::io::soa_readers::SoaFastqReader;
use crate::io::sam_output::{
    create_unmapped_single_end, prepare_single_end_alignment, select_single_end_alignments,
    write_sam_record,
};
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

    // Check for SoA pipeline mode (PR3 end-to-end SoA)
    let use_soa_pipeline = std::env::var("FERROUS_SOA_PIPELINE").is_ok_and(|v| v == "1");

    if use_soa_pipeline {
        log::info!("Using end-to-end SoA pipeline (PR3)");
    }

    for query_file_name in query_files {
        // Branch on reader type based on pipeline mode
        if use_soa_pipeline {
            // SoA pipeline: Use SoaFastqReader
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
        } else {
            // AoS pipeline: Use FastqReader (existing code)
            process_single_end_aos(
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

/// Process single-end reads using AoS pipeline (existing implementation)
fn process_single_end_aos(
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
    let mut reader = match FastqReader::new(query_file_name) {
            Ok(r) => r,
            Err(e) => {
                log::error!("Error opening query file {query_file_name}: {e}");
                return;
            }
        };

    loop {
        // Stage 0: Read batch of reads (matching C++ kt_pipeline step 0)
        let batch = match reader.read_batch(reads_per_batch) {
            Ok(b) => b,
            Err(e) => {
                log::error!("Error reading batch from {query_file_name}: {e}");
                break;
            }
        };

        if batch.names.is_empty() {
            break; // EOF
        }

        let batch_size = batch.names.len();

        // Calculate batch base pairs
        let batch_bp: usize = batch.seqs.iter().map(|s| s.len()).sum();
        *total_reads += batch_size;
        *total_bases += batch_bp;

        log::info!(
            "read_chunk: {reads_per_batch}, work_chunk_size: {batch_bp}, nseq: {batch_size}"
        );

        // Track per-batch timing
        let batch_start_cpu = cputime();
        let batch_start_wall = std::time::Instant::now();

        // Stage 1: Process batch (matching C++ kt_pipeline step 1)
        let bwa_idx_clone = Arc::clone(bwa_idx);
        let pac_data_clone = Arc::clone(pac_data);
        let opt_clone = Arc::clone(opt);
        let batch_start_id = *reads_processed; // Capture for closure
        let compute_backend = compute_ctx.backend;

        // Check batching mode via environment variables:
        // - FERROUS_CROSS_READ_BATCH=1: Global cross-read batching (experimental, slow)
        // - FERROUS_PARALLEL_SUBBATCH=1: Parallel sub-batch processing (optimized)
        // - Default: Per-read processing with rayon
        let use_cross_read_batching =
            std::env::var("FERROUS_CROSS_READ_BATCH").is_ok_and(|v| v == "1");
        let use_parallel_subbatch =
            std::env::var("FERROUS_PARALLEL_SUBBATCH").is_ok_and(|v| v == "1");

        let alignments: Vec<Vec<Alignment>> = if use_parallel_subbatch {
            // OPTIMIZED: Parallel sub-batch processing (BWA-MEM2 style)
            // Processes ~512-read sub-batches in parallel, with cross-read batching
            // within each sub-batch for better SIMD utilization
            let engine = match &compute_backend {
                ComputeBackend::CpuSimd(e) => *e,
                _ => SimdEngineType::Engine128,
            };

            process_batch_parallel_subbatch(
                *bwa_idx_clone, // Deref Arc<&BwaIndex> -> &BwaIndex
                &pac_data_clone,
                *opt_clone, // Deref Arc<&MemOpt> -> &MemOpt
                &batch.names,
                &batch.seqs,
                &batch.quals,
                batch_start_id,
                engine,
            )
        } else if use_cross_read_batching {
            // EXPERIMENTAL: Global cross-read batching (slow due to sequential phases)
            let engine = match &compute_backend {
                ComputeBackend::CpuSimd(e) => *e,
                _ => SimdEngineType::Engine128,
            };

            process_batch_cross_read(
                *bwa_idx_clone, // Deref Arc<&BwaIndex> -> &BwaIndex
                &pac_data_clone,
                *opt_clone, // Deref Arc<&MemOpt> -> &MemOpt
                &batch.names,
                &batch.seqs,
                &batch.quals,
                batch_start_id,
                engine,
            )
        } else {
            // ORIGINAL: Per-read processing (reference implementation)
            batch
                .names
                .par_iter()
                .zip(batch.seqs.par_iter())
                .zip(batch.quals.par_iter())
                .enumerate()
                .map(|(i, ((name, seq), qual))| {
                    // Global read ID for deterministic hash tie-breaking (matches C++ bwamem.cpp:1325)
                    let read_id = batch_start_id + i as u64;

                    align_read_deferred(
                        &bwa_idx_clone,
                        &pac_data_clone,
                        name,
                        seq,
                        qual,
                        &opt_clone,
                        compute_backend,
                        read_id,
                        false, // don't skip secondary marking
                    )
                })
                .collect()
        };

        // Update global read counter for next batch
        *reads_processed += batch_size as u64;

        // Stage 2: Write output sequentially (matching C++ kt_pipeline step 2)
        // Uses sam_output module for clean separation of concerns
        let rg_id = opt
            .read_group
            .as_ref()
            .and_then(|rg| super::mem_opt::MemOpt::extract_rg_id(rg));

        for (read_idx, mut alignment_vec) in alignments.into_iter().enumerate() {
            // Get original seq/qual from batch
            let orig_seq = std::str::from_utf8(&batch.seqs[read_idx]).unwrap_or("");
            let orig_qual = &batch.quals[read_idx];

            // Select which alignments to output
            let selection = select_single_end_alignments(&alignment_vec, opt);

            if selection.output_as_unmapped {
                // Output unmapped record
                let query_name = alignment_vec
                    .first()
                    .map(|a| a.query_name.as_str())
                    .unwrap_or("unknown");
                let mut unmapped = create_unmapped_single_end(query_name, orig_seq.len());

                if let Some(ref rg) = rg_id {
                    unmapped.tags.push(("RG".to_string(), format!("Z:{rg}")));
                }

                if let Err(e) = write_sam_record(writer, &unmapped, orig_seq, orig_qual) {
                    log::error!("Error writing SAM record: {e}");
                }
                continue;
            }

            // Output selected alignments
            for idx in selection.output_indices {
                let is_primary = idx == selection.primary_idx;
                prepare_single_end_alignment(
                    &mut alignment_vec[idx],
                    is_primary,
                    rg_id.as_deref(),
                );

                if let Err(e) =
                    write_sam_record(writer, &alignment_vec[idx], orig_seq, orig_qual)
                {
                    log::error!("Error writing SAM record: {e}");
                }
            }
        }

        // Log per-batch timing (matches BWA-MEM2 format)
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

        if soa_read_batch.len() == 0 {
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

        // Stage 1: Process batch using end-to-end SoA pipeline
        let bwa_idx_clone = Arc::clone(bwa_idx);
        let pac_data_clone = Arc::clone(pac_data);
        let opt_clone = Arc::clone(opt);
        let batch_start_id = *reads_processed;

        let pac_slice: &[u8] = &pac_data_clone;
        let pac_ref = Arc::new(pac_slice);
        let alignments = process_sub_batch_internal_soa(
            &bwa_idx_clone,
            &pac_ref,
            &opt_clone,
            &soa_read_batch,
            batch_start_id,
            engine,
        );

        // Update global read counter for next batch
        *reads_processed += batch_size as u64;

        // Stage 2: Write output sequentially
        let rg_id = opt
            .read_group
            .as_ref()
            .and_then(|rg| super::mem_opt::MemOpt::extract_rg_id(rg));

        for (read_idx, mut alignment_vec) in alignments.into_iter().enumerate() {
            // Get original seq/qual from SoA batch
            let (seq_start, seq_len) = soa_read_batch.read_boundaries[read_idx];
            let seq_bytes = &soa_read_batch.seqs[seq_start..seq_start + seq_len];
            let orig_seq = std::str::from_utf8(seq_bytes).unwrap_or("");
            let qual_bytes = &soa_read_batch.quals[seq_start..seq_start + seq_len];
            let orig_qual = std::str::from_utf8(qual_bytes).unwrap_or("");

            // Select which alignments to output
            let selection = select_single_end_alignments(&alignment_vec, opt);

            if selection.output_as_unmapped {
                // Output unmapped record
                let query_name = alignment_vec
                    .first()
                    .map(|a| a.query_name.as_str())
                    .unwrap_or("unknown");
                let mut unmapped = create_unmapped_single_end(query_name, orig_seq.len());

                if let Some(ref rg) = rg_id {
                    unmapped.tags.push(("RG".to_string(), format!("Z:{rg}")));
                }

                if let Err(e) = write_sam_record(writer, &unmapped, orig_seq, orig_qual) {
                    log::error!("Error writing SAM record: {e}");
                }
                continue;
            }

            // Output selected alignments
            for idx in selection.output_indices {
                let is_primary = idx == selection.primary_idx;
                prepare_single_end_alignment(
                    &mut alignment_vec[idx],
                    is_primary,
                    rg_id.as_deref(),
                );

                if let Err(e) =
                    write_sam_record(writer, &alignment_vec[idx], orig_seq, orig_qual)
                {
                    log::error!("Error writing SAM record: {e}");
                }
            }
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
