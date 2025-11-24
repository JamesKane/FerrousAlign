// Single-end read processing module
//
// This module handles single-end FASTQ file processing, including:
// - Batched read loading (Stage 0)
// - Parallel alignment using Rayon (Stage 1)
// - SAM output formatting (Stage 2)
//
// The pipeline is cleanly separated:
// - Computation: generate_seeds() returns Vec<Alignment>
// - Selection: sam_output::select_single_end_alignments() filters output
// - Output: sam_output::write_sam_record() writes to stream

use crate::alignment::finalization::Alignment;
use crate::alignment::pipeline::{generate_seeds, align_read_deferred};
use crate::compute::ComputeContext;
use crate::fastq_reader::FastqReader;
use crate::index::BwaIndex;
use crate::mem_opt::MemOpt;
use crate::sam_output::{
    select_single_end_alignments, prepare_single_end_alignment,
    create_unmapped_single_end, write_sam_record,
};
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
// Compute flow: process_single_end() → generate_seeds() → align_read() → extension
//
// To add GPU/NPU acceleration:
// 1. Pass compute_ctx through to generate_seeds()
// 2. In align_read(), route based on compute_ctx.backend
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

    for query_file_name in query_files {
        let mut reader = match FastqReader::new(query_file_name) {
            Ok(r) => r,
            Err(e) => {
                log::error!("Error opening query file {}: {}", query_file_name, e);
                continue;
            }
        };

        loop {
            // Stage 0: Read batch of reads (matching C++ kt_pipeline step 0)
            let batch = match reader.read_batch(reads_per_batch) {
                Ok(b) => b,
                Err(e) => {
                    log::error!("Error reading batch from {}: {}", query_file_name, e);
                    break;
                }
            };

            if batch.names.is_empty() {
                break; // EOF
            }

            let batch_size = batch.names.len();

            // Calculate batch base pairs
            let batch_bp: usize = batch.seqs.iter().map(|s| s.len()).sum();
            total_reads += batch_size;
            total_bases += batch_bp;

            log::info!("Read {} sequences ({} bp)", batch_size, batch_bp);
            log::debug!("Processing batch of {} reads in parallel", batch_size);

            // Stage 1: Process batch in parallel (matching C++ kt_pipeline step 1)
            let bwa_idx_clone = Arc::clone(&bwa_idx);
            let pac_data_clone = Arc::clone(&pac_data);
            let opt_clone = Arc::clone(&opt);
            let batch_start_id = reads_processed; // Capture for closure
            let use_deferred = opt.deferred_cigar;
            let compute_backend = compute_ctx.backend.clone();

            let alignments: Vec<Vec<Alignment>> = batch
                .names
                .par_iter()
                .zip(batch.seqs.par_iter())
                .zip(batch.quals.par_iter())
                .enumerate()
                .map(|(i, ((name, seq), qual))| {
                    // Global read ID for deterministic hash tie-breaking (matches C++ bwamem.cpp:1325)
                    let read_id = batch_start_id + i as u64;

                    if use_deferred {
                        // Deferred CIGAR pipeline (experimental)
                        // Generates CIGARs only for high-scoring alignments
                        align_read_deferred(
                            &bwa_idx_clone,
                            &pac_data_clone,
                            name,
                            seq,
                            qual,
                            &opt_clone,
                            compute_backend.clone(),
                            read_id,
                            false, // don't skip secondary marking
                        )
                    } else {
                        // Standard pipeline
                        generate_seeds(&bwa_idx_clone, &pac_data_clone, name, seq, qual, &opt_clone, read_id)
                    }
                })
                .collect();

            // Update global read counter for next batch
            reads_processed += batch_size as u64;

            // Stage 2: Write output sequentially (matching C++ kt_pipeline step 2)
            // Uses sam_output module for clean separation of concerns
            let rg_id = opt
                .read_group
                .as_ref()
                .and_then(|rg| crate::mem_opt::MemOpt::extract_rg_id(rg));

            for (read_idx, mut alignment_vec) in alignments.into_iter().enumerate() {
                // Get original seq/qual from batch
                let orig_seq = std::str::from_utf8(&batch.seqs[read_idx]).unwrap_or("");
                let orig_qual = &batch.quals[read_idx];

                // Select which alignments to output
                let selection = select_single_end_alignments(&alignment_vec, &opt);

                if selection.output_as_unmapped {
                    // Output unmapped record
                    let query_name = alignment_vec
                        .first()
                        .map(|a| a.query_name.as_str())
                        .unwrap_or("unknown");
                    let mut unmapped = create_unmapped_single_end(query_name, orig_seq.len());

                    if let Some(ref rg) = rg_id {
                        unmapped.tags.push(("RG".to_string(), format!("Z:{}", rg)));
                    }

                    if let Err(e) = write_sam_record(writer, &unmapped, orig_seq, orig_qual) {
                        log::error!("Error writing SAM record: {}", e);
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

                    if let Err(e) = write_sam_record(writer, &alignment_vec[idx], orig_seq, orig_qual) {
                        log::error!("Error writing SAM record: {}", e);
                    }
                }
            }

            if batch_size < reads_per_batch {
                break; // Last incomplete batch
            }
        }
    }

    // Print summary statistics
    let elapsed = start_time.elapsed();
    log::info!(
        "Processed {} reads ({} bp) in {:.2} sec",
        total_reads,
        total_bases,
        elapsed.as_secs_f64()
    );
}
