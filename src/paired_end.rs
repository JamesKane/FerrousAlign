// Paired-end read processing module
//
// This module handles paired-end FASTQ file processing, including:
// - Reader thread for pipeline parallelism
// - Insert size statistics bootstrapping
// - Mate rescue using Smith-Waterman
// - Paired alignment scoring
// - SAM output with proper pair flags

use crate::align;
use crate::fastq_reader::FastqReader;
use crate::index::BwaIndex;
use crate::insert_size::{InsertSizeStats, bootstrap_insert_size_stats};
use crate::mate_rescue::mem_matesw;
use crate::mem_opt::MemOpt;
use crate::pairing::mem_pair;
use crossbeam_channel::{Receiver, Sender, bounded};
use rayon::prelude::*;
use std::io::Write;
use std::sync::Arc;
use std::thread;

// Batch processing constants (matching C++ bwa-mem2)
const CHUNK_SIZE_BASES: usize = 10_000_000;
const AVG_READ_LEN: usize = 101;
const MIN_BATCH_SIZE: usize = 512;

// Paired-end alignment constants
#[allow(dead_code)] // Reserved for future use in alignment scoring
const MIN_RATIO: f64 = 0.8; // Minimum ratio for unique alignment

// Message type for pipeline communication
type PairedBatchMessage = Option<(
    crate::fastq_reader::ReadBatch,
    crate::fastq_reader::ReadBatch,
)>;

pub(crate) fn reader_thread(
    read1_file: String,
    read2_file: String,
    reads_per_batch: usize,
    sender: Sender<PairedBatchMessage>,
) {
    // Open both FASTQ files
    let mut reader1 = match FastqReader::new(&read1_file) {
        Ok(r) => r,
        Err(e) => {
            log::error!("Error opening read1 file {}: {}", read1_file, e);
            let _ = sender.send(None); // Signal error
            return;
        }
    };
    let mut reader2 = match FastqReader::new(&read2_file) {
        Ok(r) => r,
        Err(e) => {
            log::error!("Error opening read2 file {}: {}", read2_file, e);
            let _ = sender.send(None); // Signal error
            return;
        }
    };

    log::debug!(
        "[Reader thread] Started, reading batches of {} read pairs",
        reads_per_batch
    );

    let mut batch_count = 0usize;
    let mut total_pairs_read = 0usize;

    loop {
        // Read batch from both files
        let batch1 = match reader1.read_batch(reads_per_batch) {
            Ok(b) => b,
            Err(e) => {
                log::error!("Error reading batch from read1 file: {}", e);
                let _ = sender.send(None); // Signal error
                break;
            }
        };
        let batch2 = match reader2.read_batch(reads_per_batch) {
            Ok(b) => b,
            Err(e) => {
                log::error!("Error reading batch from read2 file: {}", e);
                let _ = sender.send(None); // Signal error
                break;
            }
        };

        // Check for EOF
        if batch1.names.is_empty() && batch2.names.is_empty() {
            log::debug!("[Reader thread] EOF reached, shutting down");
            let _ = sender.send(None); // Signal EOF
            break;
        }

        // Check for mismatched batch sizes
        if batch1.names.len() != batch2.names.len() {
            log::warn!("Warning: Paired-end files have different number of reads");
            let _ = sender.send(None); // Signal error
            break;
        }

        let batch_size = batch1.names.len();
        total_pairs_read += batch_size;
        log::debug!(
            "[Reader thread] Batch {}: Read {} read pairs (cumulative: {})",
            batch_count,
            batch_size,
            total_pairs_read
        );

        // Send batch through channel
        log::debug!(
            "[Reader thread] Batch {}: Sending to channel...",
            batch_count
        );
        if sender.send(Some((batch1, batch2))).is_err() {
            log::error!(
                "[Reader thread] Batch {}: Channel closed, shutting down",
                batch_count
            );
            break;
        }
        log::debug!("[Reader thread] Batch {}: Successfully sent", batch_count);
        batch_count += 1;

        // If this was a partial batch, it's the last one
        if batch_size < reads_per_batch {
            log::debug!(
                "[Reader thread] Batch {} was partial ({}), sending EOF signal",
                batch_count - 1,
                batch_size
            );
            let _ = sender.send(None); // Signal EOF
            break;
        }
    }

    log::debug!(
        "[Reader thread] Exiting - sent {} batches, {} total pairs",
        batch_count,
        total_pairs_read
    );
}

// Process paired-end reads with parallel batching
pub fn process_paired_end(
    bwa_idx: &BwaIndex,
    read1_file: &str,
    read2_file: &str,
    writer: &mut Box<dyn Write>,
    opt: &MemOpt,
) {
    use std::time::Instant;

    // Wrap index in Arc for thread-safe sharing
    let bwa_idx = Arc::new(bwa_idx);
    let opt = Arc::new(opt);

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

    // Create channel for pipeline communication
    // Buffer size of 2 allows reader to stay 1 batch ahead
    let (sender, receiver): (Sender<PairedBatchMessage>, Receiver<PairedBatchMessage>) = bounded(2);

    // Spawn reader thread for pipeline parallelism
    let read1_file_owned = read1_file.to_string();
    let read2_file_owned = read2_file.to_string();
    let reader_handle = thread::spawn(move || {
        reader_thread(read1_file_owned, read2_file_owned, reads_per_batch, sender);
    });

    log::debug!("[Main] Pipeline started: reader thread spawned");

    // Load PAC file ONCE for all mate rescue operations
    // This eliminates catastrophic I/O that was occurring in the old design
    let pac = if let Some(pac_path) = &bwa_idx.bns.pac_file_path {
        std::fs::read(pac_path).unwrap_or_else(|e| {
            log::warn!(
                "Could not load PAC file: {}. Mate rescue will be disabled.",
                e
            );
            Vec::new()
        })
    } else {
        log::warn!("PAC file path not set. Mate rescue will be disabled.");
        Vec::new()
    };
    log::debug!("Loaded PAC file: {} bytes", pac.len());

    // === PHASE 1: Bootstrap insert size statistics from first batch ===
    log::info!("Phase 1: Bootstrapping insert size statistics from first batch");

    // Receive first batch
    log::debug!("[Main] Waiting for first batch (batch 0)...");
    let first_batch_msg = match receiver.recv() {
        Ok(msg) => msg,
        Err(_) => {
            log::error!("Failed to receive first batch from reader");
            return;
        }
    };

    let (first_batch1, first_batch2) = match first_batch_msg {
        Some((b1, b2)) => (b1, b2),
        None => {
            log::warn!("No data to process (empty input files)");
            return;
        }
    };

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
        "Read {} sequences ({} bp) [first batch]",
        first_batch_size * 2,
        first_batch_bp
    );

    // Process first batch in parallel
    let num_pairs = first_batch1.names.len();
    let bwa_idx_clone = Arc::clone(&bwa_idx);
    let opt_clone = Arc::clone(&opt);

    let mut first_batch_alignments: Vec<(Vec<align::Alignment>, Vec<align::Alignment>)> = (0
        ..num_pairs)
        .into_par_iter()
        .map(|i| {
            let aln1 = align::generate_seeds(
                &bwa_idx_clone,
                &first_batch1.names[i],
                &first_batch1.seqs[i],
                &first_batch1.quals[i],
                &opt_clone,
            );
            let aln2 = align::generate_seeds(
                &bwa_idx_clone,
                &first_batch2.names[i],
                &first_batch2.seqs[i],
                &first_batch2.quals[i],
                &opt_clone,
            );
            (aln1, aln2)
        })
        .collect();

    // Bootstrap insert size stats from first batch
    let stats = if let Some(ref is_override) = opt.insert_size_override {
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

    // Mate rescue on first batch
    let rescued_first = mate_rescue_batch(
        &mut first_batch_alignments,
        &first_batch_seqs1,
        &first_batch_seqs2,
        &pac,
        &stats,
        &bwa_idx,
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
    let first_batch_records = output_batch_paired(
        first_batch_alignments,
        &first_batch_seqs1_owned,
        &first_batch_seqs2_owned,
        &stats,
        writer,
        &opt,
        bwa_idx.bns.packed_sequence_length as i64,
        0,
    )
    .unwrap_or_else(|e| {
        log::error!("Error writing first batch: {}", e);
        0
    });
    log::info!("First batch: {} records written", first_batch_records);

    // === PHASE 2: Stream remaining batches ===
    log::info!("Phase 2: Streaming remaining batches");

    let mut batch_num = 1u64;
    let mut total_rescued = rescued_first;
    let mut total_records = first_batch_records;

    loop {
        // Receive batch from reader thread (blocks until batch available)
        log::debug!("[Main] Waiting for batch {}...", batch_num);
        let batch_msg = match receiver.recv() {
            Ok(msg) => msg,
            Err(_) => {
                log::debug!("[Main] Reader channel closed (after {} batches)", batch_num);
                break;
            }
        };

        // Check for EOF or error signal
        let (batch1, batch2) = match batch_msg {
            Some((b1, b2)) => (b1, b2),
            None => {
                log::debug!(
                    "[Main] Received EOF/error signal from reader (after {} batches)",
                    batch_num
                );
                break;
            }
        };

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
        log::info!("Read {} sequences ({} bp)", batch_size * 2, batch_bp);

        // Process batch in parallel
        let num_pairs = batch1.names.len();
        let bwa_idx_clone = Arc::clone(&bwa_idx);
        let opt_clone = Arc::clone(&opt);

        let mut batch_alignments: Vec<(Vec<align::Alignment>, Vec<align::Alignment>)> = (0
            ..num_pairs)
            .into_par_iter()
            .map(|i| {
                let aln1 = align::generate_seeds(
                    &bwa_idx_clone,
                    &batch1.names[i],
                    &batch1.seqs[i],
                    &batch1.quals[i],
                    &opt_clone,
                );
                let aln2 = align::generate_seeds(
                    &bwa_idx_clone,
                    &batch2.names[i],
                    &batch2.seqs[i],
                    &batch2.quals[i],
                    &opt_clone,
                );
                (aln1, aln2)
            })
            .collect();

        // Prepare sequences for mate rescue
        let batch_seqs1 = batch1.as_tuple_refs();
        let batch_seqs2 = batch2.as_tuple_refs();

        // Mate rescue on this batch
        let rescued = mate_rescue_batch(
            &mut batch_alignments,
            &batch_seqs1,
            &batch_seqs2,
            &pac,
            &stats,
            &bwa_idx,
        );
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

        // Output batch immediately
        let records = output_batch_paired(
            batch_alignments,
            &batch_seqs1_owned,
            &batch_seqs2_owned,
            &stats,
            writer,
            &opt,
            bwa_idx.bns.packed_sequence_length as i64,
            batch_num * reads_per_batch as u64,
        )
        .unwrap_or_else(|e| {
            log::error!("Error writing batch: {}", e);
            0
        });
        total_records += records;

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

    // Wait for reader thread to finish
    log::debug!("[Main] Waiting for reader thread to finish");
    if let Err(e) = reader_handle.join() {
        log::error!("[Main] Reader thread panicked: {:?}", e);
    }
    log::debug!("[Main] Reader thread finished, pipeline complete");

    // Print summary statistics
    let elapsed = start_time.elapsed();
    log::info!(
        "Complete: {} batches, {} reads ({} bp), {} records, {} pairs rescued in {:.2} sec",
        batch_num,
        total_reads,
        total_bases,
        total_records,
        total_rescued,
        elapsed.as_secs_f64()
    );
}

// Perform mate rescue on a single batch
// Returns number of pairs rescued
fn mate_rescue_batch(
    batch_pairs: &mut [(Vec<align::Alignment>, Vec<align::Alignment>)],
    batch_seqs1: &[(&str, &[u8], &str)], // (name, seq, qual) for read1
    batch_seqs2: &[(&str, &[u8], &str)], // (name, seq, qual) for read2
    pac: &[u8],
    stats: &[InsertSizeStats; 4],
    bwa_idx: &BwaIndex,
) -> usize {
    let mut rescued = 0;

    // Encode sequence helper
    // Use align::encode_sequence instead of lambda
    for (i, (alns1, alns2)) in batch_pairs.iter_mut().enumerate() {
        let (name1, seq1, qual1) = batch_seqs1[i];
        let (name2, seq2, qual2) = batch_seqs2[i];

        // DEBUG: Log alignment counts for diagnostic purposes
        log::debug!(
            "Pair {}: {} has {} alignments, {} has {} alignments",
            i,
            name1,
            alns1.len(),
            name2,
            alns2.len()
        );

        // Try to rescue read2 if read1 has good alignments but read2 doesn't
        if !alns1.is_empty() && alns2.is_empty() && !pac.is_empty() {
            let mate_seq = align::encode_sequence(seq2);
            let n = mem_matesw(
                bwa_idx, pac, stats, &alns1[0], // Use best alignment as anchor
                &mate_seq, qual2, name2, alns2,
            );
            if n > 0 {
                rescued += 1;
            }
        }

        // Try to rescue read1 if read2 has good alignments but read1 doesn't
        if !alns2.is_empty() && alns1.is_empty() && !pac.is_empty() {
            let mate_seq = align::encode_sequence(seq1);
            let n = mem_matesw(
                bwa_idx, pac, stats, &alns2[0], // Use best alignment as anchor
                &mate_seq, qual1, name1, alns1,
            );
            if n > 0 {
                rescued += 1;
            }
        }
    }

    rescued
}

// Output a batch of paired-end alignments with proper flags and flushing
// Returns number of records written
fn output_batch_paired(
    batch_pairs: Vec<(Vec<align::Alignment>, Vec<align::Alignment>)>,
    batch_seqs1: &[(String, Vec<u8>, String)], // (name, seq, qual) for read1
    batch_seqs2: &[(String, Vec<u8>, String)], // (name, seq, qual) for read2
    stats: &[InsertSizeStats; 4],
    writer: &mut Box<dyn Write>,
    opt: &MemOpt,
    l_pac: i64,
    starting_pair_id: u64,
) -> std::io::Result<usize> {
    let mut records_written = 0;

    // Extract RG ID once
    let rg_id = opt
        .read_group
        .as_ref()
        .and_then(|rg| crate::mem_opt::MemOpt::extract_rg_id(rg));

    for (pair_idx, (mut alignments1, mut alignments2)) in batch_pairs.into_iter().enumerate() {
        let (name1, seq1, qual1) = &batch_seqs1[pair_idx];
        let (name2, seq2, qual2) = &batch_seqs2[pair_idx];
        let pair_id = starting_pair_id + pair_idx as u64;

        // DEBUG: Log pair processing for diagnostic purposes
        log::debug!(
            "Processing pair {}: {} ({} alns), {} ({} alns)",
            pair_id,
            name1,
            alignments1.len(),
            name2,
            alignments2.len()
        );

        // ===  SCORE THRESHOLD FILTERING (matching C++ bwa-mem2 behavior) ===
        // Filter alignments by score threshold. If ALL alignments filtered,
        // create one unmapped alignment to ensure read is output.
        // This matches C++ bwa-mem2: if (aa.n == 0) { create unmapped record }
        alignments1.retain(|a| a.score >= opt.t);
        alignments2.retain(|a| a.score >= opt.t);

        // If all alignments filtered, create unmapped alignment
        if alignments1.is_empty() {
            log::debug!(
                "{}: All alignments filtered by score threshold, creating unmapped",
                name1
            );
            alignments1.push(align::Alignment {
                query_name: name1.to_string(),
                flag: 0x4, // Unmapped (paired flags will be set later)
                ref_name: "*".to_string(),
                ref_id: 0,
                pos: 0,
                mapq: 0,
                score: 0,
                cigar: Vec::new(),
                rnext: "*".to_string(),
                pnext: 0,
                tlen: 0,
                seq: String::from_utf8_lossy(seq1).to_string(),
                qual: qual1.to_string(),
                tags: Vec::new(),
                query_start: 0,
                query_end: 0,
                seed_coverage: 0,
                hash: 0,
            });
        }

        if alignments2.is_empty() {
            log::debug!(
                "{}: All alignments filtered by score threshold, creating unmapped",
                name2
            );
            alignments2.push(align::Alignment {
                query_name: name2.to_string(),
                flag: 0x4, // Unmapped (paired flags will be set later)
                ref_name: "*".to_string(),
                ref_id: 0,
                pos: 0,
                mapq: 0,
                score: 0,
                cigar: Vec::new(),
                rnext: "*".to_string(),
                pnext: 0,
                tlen: 0,
                seq: String::from_utf8_lossy(seq2).to_string(),
                qual: qual2.to_string(),
                tags: Vec::new(),
                query_start: 0,
                query_end: 0,
                seed_coverage: 0,
                hash: 0,
            });
        }

        // Use mem_pair to score paired alignments
        let pair_result = if !alignments1.is_empty() && !alignments2.is_empty() {
            let result = mem_pair(stats, &alignments1, &alignments2, l_pac, 2, pair_id);

            // DEBUG: Log mem_pair results
            if pair_id < 10 {
                if result.is_some() {
                    log::debug!("mem_pair SUCCESS for pair {}", pair_id);
                } else {
                    log::debug!(
                        "mem_pair FAIL for pair {}. alns1={}, alns2={}, stats[FR]: low={}, high={}, failed={}",
                        pair_id,
                        alignments1.len(),
                        alignments2.len(),
                        stats[1].low,
                        stats[1].high,
                        stats[1].failed
                    );
                }
            }

            result
        } else {
            None
        };

        // Determine best paired alignment indices and check if properly paired
        let (best_idx1, best_idx2, is_properly_paired) =
            if let Some((idx1, idx2, _pair_score, _sub_score)) = pair_result {
                // If mem_pair returned a valid pair, it's properly paired
                // mem_pair already verified: same chromosome, insert size within bounds, valid orientation
                (idx1, idx2, true)
            } else if !alignments1.is_empty() && !alignments2.is_empty() {
                // mem_pair returned None - no valid pair found
                // Use first alignments but NOT properly paired
                // (insert size out of bounds, or wrong orientation)
                (0, 0, false)
            } else {
                // One or both reads unmapped
                (0, 0, false)
            };

        // Extract mate info from best alignments BEFORE creating dummies
        // (needed to populate dummy alignments with mate's position)
        let (mate2_ref_initial, mate2_pos_initial, mate2_flag_initial) =
            if let Some(aln2) = alignments2.get(best_idx2) {
                (aln2.ref_name.clone(), aln2.pos, aln2.flag)
            } else {
                ("*".to_string(), 0, 0)
            };

        let (mate1_ref_initial, mate1_pos_initial, mate1_flag_initial) =
            if let Some(aln1) = alignments1.get(best_idx1) {
                (aln1.ref_name.clone(), aln1.pos, aln1.flag)
            } else {
                ("*".to_string(), 0, 0)
            };

        // Handle unmapped reads - create dummy alignments
        if alignments1.is_empty() {
            let unmapped = align::Alignment::create_unmapped(
                name1.clone(),
                seq1,
                qual1.clone(),
                true, // is_first_in_pair
                &mate2_ref_initial,
                mate2_pos_initial,
                mate2_flag_initial & 0x10 != 0, // mate_is_reverse
            );
            alignments1.push(unmapped);
        }

        if alignments2.is_empty() {
            let unmapped = align::Alignment::create_unmapped(
                name2.clone(),
                seq2,
                qual2.clone(),
                false, // is_first_in_pair (this is read2)
                &mate1_ref_initial,
                mate1_pos_initial,
                mate1_flag_initial & 0x10 != 0, // mate_is_reverse
            );
            alignments2.push(unmapped);
        }

        // Re-extract mate info AFTER creating dummy alignments
        // This ensures mapped reads get correct mate information even when mate is unmapped
        let (mate2_ref, mate2_pos, mate2_flag) = if let Some(aln2) = alignments2.get(best_idx2) {
            (aln2.ref_name.clone(), aln2.pos, aln2.flag)
        } else {
            ("*".to_string(), 0, 0)
        };

        let (mate1_ref, mate1_pos, mate1_flag) = if let Some(aln1) = alignments1.get(best_idx1) {
            (aln1.ref_name.clone(), aln1.pos, aln1.flag)
        } else {
            ("*".to_string(), 0, 0)
        };

        // Set flags and mate information for read1
        for (idx, alignment) in alignments1.iter_mut().enumerate() {
            let is_unmapped = alignment.flag & 0x4 != 0;

            // ALWAYS set paired flag (0x1) - even for unmapped reads
            alignment.flag |= 0x1;

            // ALWAYS set first in pair flag (0x40) - even for unmapped reads
            alignment.flag |= 0x40;

            // Only set proper pair flag for mapped reads in proper pairs
            if !is_unmapped && is_properly_paired && idx == best_idx1 {
                alignment.flag |= 0x2;
            }

            // Set mate unmapped flag if mate is unmapped
            if mate2_ref == "*" {
                alignment.flag |= 0x8;
            }

            // Only set mate position and TLEN for mapped reads
            if !is_unmapped && mate2_ref != "*" {
                alignment.rnext = if alignment.ref_name == mate2_ref {
                    "=".to_string()
                } else {
                    mate2_ref.clone()
                };
                alignment.pnext = mate2_pos + 1;

                if mate2_flag & 0x10 != 0 {
                    alignment.flag |= 0x20;
                }

                if alignment.ref_name == mate2_ref {
                    let mate2_len = alignments2
                        .get(best_idx2)
                        .map(|a| a.reference_length())
                        .unwrap_or(0);
                    alignment.tlen = alignment.calculate_tlen(mate2_pos, mate2_len);
                }
            }
        }

        // Set flags and mate information for read2
        for (idx, alignment) in alignments2.iter_mut().enumerate() {
            let is_unmapped = alignment.flag & 0x4 != 0;

            // ALWAYS set paired flag (0x1) - even for unmapped reads
            alignment.flag |= 0x1;

            // ALWAYS set second in pair flag (0x80) - even for unmapped reads
            alignment.flag |= 0x80;

            // Only set proper pair flag for mapped reads in proper pairs
            if !is_unmapped && is_properly_paired && idx == best_idx2 {
                alignment.flag |= 0x2;
            }

            // Set mate unmapped flag if mate is unmapped
            if mate1_ref == "*" {
                alignment.flag |= 0x8;
            }

            // Only set mate position and TLEN for mapped reads
            if !is_unmapped && mate1_ref != "*" {
                alignment.rnext = if alignment.ref_name == mate1_ref {
                    "=".to_string()
                } else {
                    mate1_ref.clone()
                };
                alignment.pnext = mate1_pos + 1;

                if mate1_flag & 0x10 != 0 {
                    alignment.flag |= 0x20;
                }

                if alignment.ref_name == mate1_ref {
                    let mate1_len = alignments1
                        .get(best_idx1)
                        .map(|a| a.reference_length())
                        .unwrap_or(0);
                    alignment.tlen = alignment.calculate_tlen(mate1_pos, mate1_len);
                }
            }
        }

        // Get mate CIGARs for MC tag
        let mate1_cigar = if !alignments1.is_empty() && best_idx1 < alignments1.len() {
            alignments1[best_idx1].cigar_string()
        } else {
            "*".to_string()
        };

        let mate2_cigar = if !alignments2.is_empty() && best_idx2 < alignments2.len() {
            alignments2[best_idx2].cigar_string()
        } else {
            "*".to_string()
        };

        // Write alignments for read1
        for (idx, mut alignment) in alignments1.into_iter().enumerate() {
            let is_unmapped = alignment.flag & 0x4 != 0;
            // Output unmapped reads always, or reads meeting score threshold
            let should_output = is_unmapped || alignment.score >= opt.t;

            if !should_output {
                continue; // Skip low-scoring alignments
            }

            // Mark non-best alignments as secondary (0x100)
            if idx != best_idx1 {
                alignment.flag |= 0x100; // Secondary alignment flag
            }

            if let Some(ref rg) = rg_id {
                alignment.tags.push(("RG".to_string(), format!("Z:{}", rg)));
            }

            if !mate2_cigar.is_empty() {
                alignment
                    .tags
                    .push(("MC".to_string(), format!("Z:{}", mate2_cigar)));
            }

            let sam_record = alignment.to_sam_string();
            writeln!(writer, "{}", sam_record)?;
            records_written += 1;
        }

        // Write alignments for read2
        for (idx, mut alignment) in alignments2.into_iter().enumerate() {
            let is_unmapped = alignment.flag & 0x4 != 0;
            // Output unmapped reads always, or reads meeting score threshold
            let should_output = is_unmapped || alignment.score >= opt.t;

            if !should_output {
                continue; // Skip low-scoring alignments
            }

            // Mark non-best alignments as secondary (0x100)
            if idx != best_idx2 {
                alignment.flag |= 0x100; // Secondary alignment flag
            }

            if let Some(ref rg) = rg_id {
                alignment.tags.push(("RG".to_string(), format!("Z:{}", rg)));
            }

            if !mate1_cigar.is_empty() {
                alignment
                    .tags
                    .push(("MC".to_string(), format!("Z:{}", mate1_cigar)));
            }

            let sam_record = alignment.to_sam_string();
            writeln!(writer, "{}", sam_record)?;
            records_written += 1;
        }
    }

    // CRITICAL: Flush after each batch for incremental output
    writer.flush()?;

    Ok(records_written)
}
