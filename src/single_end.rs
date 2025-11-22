// Single-end read processing module
//
// This module handles single-end FASTQ file processing, including:
// - Batched read loading
// - Parallel alignment using Rayon
// - SAM output formatting

use crate::alignment::finalization::Alignment;
use crate::alignment::finalization::sam_flags;
use crate::alignment::pipeline::generate_seeds;
use crate::fastq_reader::FastqReader;
use crate::index::BwaIndex;
use crate::mem_opt::MemOpt;
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

pub fn process_single_end(
    bwa_idx: &BwaIndex,
    query_files: &Vec<String>,
    writer: &mut Box<dyn Write>,
    opt: &MemOpt,
) {
    use std::time::Instant;

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
            let alignments: Vec<Vec<Alignment>> = batch
                .names
                .par_iter()
                .zip(batch.seqs.par_iter())
                .zip(batch.quals.par_iter())
                .map(|((name, seq), qual)| {
                    generate_seeds(&bwa_idx_clone, &pac_data_clone, name, seq, qual, &opt_clone)
                })
                .collect();

            // Stage 2: Write output sequentially (matching C++ kt_pipeline step 2)
            // Filter by minimum score threshold (-T)
            // Extract read group ID if specified
            let rg_id = opt
                .read_group
                .as_ref()
                .and_then(|rg| crate::mem_opt::MemOpt::extract_rg_id(rg));

            for alignment_vec in alignments {
                // Find the best (highest scoring) alignment as primary
                let best_idx = alignment_vec
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, aln)| aln.score)
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);

                // CRITICAL FIX: Check if best alignment score is below threshold (matching bwa-mem2 behavior)
                // If so, output an unmapped record instead of the low-scoring alignment
                let best_alignment = &alignment_vec[best_idx];
                let best_is_unmapped = best_alignment.flag & sam_flags::UNMAPPED != 0;
                let all_below_threshold = !best_is_unmapped && best_alignment.score < opt.t;

                if all_below_threshold {
                    // All alignments are below score threshold - output unmapped record
                    // (matching C++ bwa-mem2: bwamem.cpp:1561-1565)
                    log::debug!(
                        "{}: Best alignment score {} below threshold {}, outputting unmapped record",
                        best_alignment.query_name,
                        best_alignment.score,
                        opt.t
                    );

                    // Create unmapped alignment for single-end read
                    let unmapped = Alignment {
                        query_name: best_alignment.query_name.clone(),
                        flag: sam_flags::UNMAPPED, // sam_flags::UNMAPPED
                        ref_name: "*".to_string(),
                        ref_id: 0,
                        pos: 0,
                        mapq: 0,
                        score: 0,
                        cigar: Vec::new(), // Empty CIGAR = "*" in SAM
                        rnext: "*".to_string(),
                        pnext: 0,
                        tlen: 0,
                        seq: best_alignment.seq.clone(),
                        qual: best_alignment.qual.clone(),
                        tags: vec![
                            ("AS".to_string(), "i:0".to_string()),
                            ("NM".to_string(), "i:0".to_string()),
                        ],
                        query_start: 0,
                        query_end: 0,
                        seed_coverage: 0,
                        hash: 0,
                        frac_rep: 0.0,
                    };

                    // Add RG tag if read group is specified
                    let mut output_alignment = unmapped;
                    if let Some(ref rg) = rg_id {
                        output_alignment
                            .tags
                            .push(("RG".to_string(), format!("Z:{}", rg)));
                    }

                    let sam_record = output_alignment.to_sam_string();
                    if let Err(e) = writeln!(writer, "{}", sam_record) {
                        log::error!("Error writing SAM record: {}", e);
                    }

                    continue; // Skip to next read
                }

                for (idx, mut alignment) in alignment_vec.into_iter().enumerate() {
                    let is_unmapped = alignment.flag & sam_flags::UNMAPPED != 0;
                    let is_secondary = alignment.flag & sam_flags::SECONDARY != 0;
                    let is_supplementary = alignment.flag & sam_flags::SUPPLEMENTARY != 0;
                    let is_best = idx == best_idx;

                    // bwa-mem2 output behavior (mem_reg2sam, bwamem.cpp:1534-1560):
                    // - Primary alignments: always output (flag without 0x100 or 0x800 for first, 0x800 for subsequent)
                    // - Supplementary alignments: always output (flag 0x800)
                    // - Secondary alignments: only output with -a flag (flag 0x100)
                    //
                    // The flags were already set correctly by mark_secondary_alignments() in finalization.rs
                    let should_output = if opt.output_all_alignments {
                        // -a flag: output all alignments meeting score threshold
                        is_unmapped || alignment.score >= opt.t
                    } else {
                        // Default: output primary and supplementary, not secondary
                        // Primary = best alignment without SECONDARY flag
                        // Supplementary = non-overlapping alignments with SUPPLEMENTARY flag
                        // Secondary = overlapping alignments with SECONDARY flag (skip these)
                        is_unmapped || (!is_secondary && (is_best || is_supplementary))
                    };

                    if !should_output {
                        continue; // Skip secondary alignments (unless -a flag set)
                    }

                    // Ensure the best alignment doesn't have SECONDARY or SUPPLEMENTARY flags
                    // (it should be the true primary)
                    if is_best && !is_unmapped {
                        alignment.flag &= !sam_flags::SECONDARY;
                        alignment.flag &= !sam_flags::SUPPLEMENTARY;
                    }

                    // Add RG tag if read group is specified
                    if let Some(ref rg) = rg_id {
                        alignment.tags.push(("RG".to_string(), format!("Z:{}", rg)));
                    }

                    let sam_record = alignment.to_sam_string();
                    if let Err(e) = writeln!(writer, "{}", sam_record) {
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
