//! Cross-read batched extension for SIMD utilization
//!
//! This module implements BWA-MEM2's cross-read batching strategy for Smith-Waterman
//! extension. Instead of processing each read's chains individually, we collect
//! extension jobs from ALL reads in a batch, then process them together with SIMD.
//!
//! This maximizes SIMD lane utilization:
//! - AVX-512: 32 lanes (16-bit) fully utilized
//! - AVX2: 16 lanes (16-bit) fully utilized
//!
//! Reference: BWA-MEM2 `mem_chain2aln_across_reads_V2()` in bwamem.cpp

use crate::alignment::banded_swa::{BandedPairWiseSW, OutScore};
use crate::alignment::chaining::{cal_max_gap, Chain};
use crate::alignment::mem_opt::MemOpt;
use crate::alignment::region::{ChainExtensionMapping, SeedExtensionMapping};
use crate::alignment::seeding::Seed;
use crate::compute::simd_abstraction::simd::SimdEngineType;
use crate::index::index::BwaIndex;

/// Direction of extension from seed
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtensionDirection {
    Left,
    Right,
}

/// A single extension job with metadata for result distribution
#[derive(Debug, Clone)]
pub struct BatchedExtensionJob {
    /// Index of the read in the batch (0..batch_size)
    pub read_idx: usize,
    /// Index of the chain within the read's chains
    pub chain_idx: usize,
    /// Index of the seed within the chain (used for primary seed tracking)
    pub seed_idx: usize,
    /// Direction of extension
    pub direction: ExtensionDirection,
    /// Query sequence length for this extension
    pub query_len: i32,
    /// Reference/target sequence length for this extension
    pub ref_len: i32,
    /// Initial score (h0) - seed_len * match_score for left, 0 for right
    pub h0: i32,
    /// Band width for banded alignment
    pub band_width: i32,
    /// Offset into the query sequence buffer
    pub query_offset: usize,
    /// Offset into the reference sequence buffer
    pub ref_offset: usize,
}

/// Result of a batched extension job
#[derive(Debug, Clone)]
pub struct BatchExtensionResult {
    /// Index of the read in the batch
    pub read_idx: usize,
    /// Index of the chain within the read's chains
    pub chain_idx: usize,
    /// Index of the seed within the chain
    pub seed_idx: usize,
    /// Direction of extension
    pub direction: ExtensionDirection,
    /// Alignment score
    pub score: i32,
    /// Query end position (relative to extension segment)
    pub query_end: i32,
    /// Reference end position (relative to extension segment)
    pub ref_end: i32,
    /// Global score (for clipping detection)
    pub gscore: i32,
    /// Global target end position
    pub gref_end: i32,
    /// Max offset (for band width adjustment)
    pub max_off: i32,
}

/// Batch of extension jobs collected from multiple reads
///
/// Uses Structure-of-Arrays (SoA) layout for SIMD-friendly memory access.
/// Sequences are stored contiguously in separate buffers, with per-job
/// offsets for slicing.
#[derive(Debug)]
pub struct ExtensionJobBatch {
    /// Job metadata for result distribution
    pub jobs: Vec<BatchedExtensionJob>,

    /// Contiguous buffer of all query sequences (2-bit encoded)
    pub query_seqs: Vec<u8>,

    /// Contiguous buffer of all reference sequences (2-bit encoded)
    pub ref_seqs: Vec<u8>,

    /// Running offset for next query sequence
    query_offset: usize,

    /// Running offset for next reference sequence
    ref_offset: usize,
}

impl ExtensionJobBatch {
    /// Create a new empty batch with pre-allocated capacity
    pub fn with_capacity(job_capacity: usize, seq_capacity: usize) -> Self {
        Self {
            jobs: Vec::with_capacity(job_capacity),
            query_seqs: Vec::with_capacity(seq_capacity),
            ref_seqs: Vec::with_capacity(seq_capacity),
            query_offset: 0,
            ref_offset: 0,
        }
    }

    /// Create a new empty batch
    pub fn new() -> Self {
        Self::with_capacity(1024, 65536)
    }

    /// Add an extension job to the batch
    ///
    /// Sequences are copied into contiguous buffers.
    pub fn add_job(
        &mut self,
        read_idx: usize,
        chain_idx: usize,
        seed_idx: usize,
        direction: ExtensionDirection,
        query_seq: &[u8],
        ref_seq: &[u8],
        h0: i32,
        band_width: i32,
    ) {
        let query_offset = self.query_offset;
        let ref_offset = self.ref_offset;

        // Copy sequences to contiguous buffers
        self.query_seqs.extend_from_slice(query_seq);
        self.ref_seqs.extend_from_slice(ref_seq);

        // Update offsets
        self.query_offset += query_seq.len();
        self.ref_offset += ref_seq.len();

        // Add job metadata
        self.jobs.push(BatchedExtensionJob {
            read_idx,
            chain_idx,
            seed_idx,
            direction,
            query_len: query_seq.len() as i32,
            ref_len: ref_seq.len() as i32,
            h0,
            band_width,
            query_offset,
            ref_offset,
        });
    }

    /// Get the query sequence slice for a job
    pub fn get_query_seq(&self, job_idx: usize) -> &[u8] {
        let job = &self.jobs[job_idx];
        &self.query_seqs[job.query_offset..job.query_offset + job.query_len as usize]
    }

    /// Get the reference sequence slice for a job
    pub fn get_ref_seq(&self, job_idx: usize) -> &[u8] {
        let job = &self.jobs[job_idx];
        &self.ref_seqs[job.ref_offset..job.ref_offset + job.ref_len as usize]
    }

    /// Number of jobs in the batch
    pub fn len(&self) -> usize {
        self.jobs.len()
    }

    /// Check if the batch is empty
    pub fn is_empty(&self) -> bool {
        self.jobs.is_empty()
    }

    /// Clear the batch for reuse
    pub fn clear(&mut self) {
        self.jobs.clear();
        self.query_seqs.clear();
        self.ref_seqs.clear();
        self.query_offset = 0;
        self.ref_offset = 0;
    }
}

impl Default for ExtensionJobBatch {
    fn default() -> Self {
        Self::new()
    }
}

/// Scores for a chain's left and right extensions
#[derive(Debug, Clone, Default)]
pub struct ChainExtensionScores {
    /// Left extension score
    pub left_score: Option<i32>,
    /// Left extension query end (relative to extension segment)
    pub left_query_end: Option<i32>,
    /// Left extension reference end (relative to extension segment)
    pub left_ref_end: Option<i32>,
    /// Left extension global score
    pub left_gscore: Option<i32>,
    /// Left extension global reference end
    pub left_gref_end: Option<i32>,

    /// Right extension score
    pub right_score: Option<i32>,
    /// Right extension query end
    pub right_query_end: Option<i32>,
    /// Right extension reference end
    pub right_ref_end: Option<i32>,
    /// Right extension global score
    pub right_gscore: Option<i32>,
    /// Right extension global reference end
    pub right_gref_end: Option<i32>,
}

/// Execute batched SIMD scoring for extension jobs
///
/// Processes jobs in chunks matching the SIMD width for maximum lane utilization.
pub fn execute_batch_simd_scoring(
    sw_params: &BandedPairWiseSW,
    batch: &ExtensionJobBatch,
    engine: SimdEngineType,
) -> Vec<BatchExtensionResult> {
    if batch.is_empty() {
        return Vec::new();
    }

    let simd_width = match engine {
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        SimdEngineType::Engine512 => 32,
        #[cfg(target_arch = "x86_64")]
        SimdEngineType::Engine256 => 16,
        SimdEngineType::Engine128 => 8,
    };

    let mut results = Vec::with_capacity(batch.len());

    // Process in chunks of simd_width
    for chunk_start in (0..batch.len()).step_by(simd_width) {
        let chunk_end = (chunk_start + simd_width).min(batch.len());
        let chunk_jobs = &batch.jobs[chunk_start..chunk_end];

        // Build tuple batch for existing SIMD kernel
        // Format: (query_len, query_slice, ref_len, ref_slice, band_width, h0)
        let simd_batch: Vec<(i32, &[u8], i32, &[u8], i32, i32)> = chunk_jobs
            .iter()
            .enumerate()
            .map(|(i, job)| {
                let job_idx = chunk_start + i;
                (
                    job.query_len,
                    batch.get_query_seq(job_idx),
                    job.ref_len,
                    batch.get_ref_seq(job_idx),
                    job.band_width,
                    job.h0,
                )
            })
            .collect();

        // Call engine-specific SIMD kernel
        let scores = dispatch_simd_scoring(sw_params, &simd_batch, engine);

        // Map scores back to results with job metadata
        for (job, score) in chunk_jobs.iter().zip(scores.iter()) {
            results.push(BatchExtensionResult {
                read_idx: job.read_idx,
                chain_idx: job.chain_idx,
                seed_idx: job.seed_idx,
                direction: job.direction,
                score: score.score,
                query_end: score.query_end_pos,
                ref_end: score.target_end_pos,
                gscore: score.global_score,
                gref_end: score.gtarget_end_pos,
                max_off: score.max_offset,
            });
        }
    }

    results
}

/// Dispatch to the appropriate SIMD kernel based on engine type
fn dispatch_simd_scoring(
    sw_params: &BandedPairWiseSW,
    batch: &[(i32, &[u8], i32, &[u8], i32, i32)],
    engine: SimdEngineType,
) -> Vec<OutScore> {
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    {
        match engine {
            SimdEngineType::Engine512 => unsafe {
                crate::alignment::banded_swa_avx512::simd_banded_swa_batch32_int16(
                    batch,
                    sw_params.o_del(),
                    sw_params.e_del(),
                    sw_params.o_ins(),
                    sw_params.e_ins(),
                    sw_params.zdrop(),
                    sw_params.scoring_matrix(),
                    5,
                )
            },
            SimdEngineType::Engine256 => unsafe {
                crate::alignment::banded_swa_avx2::simd_banded_swa_batch16_int16(
                    batch,
                    sw_params.o_del(),
                    sw_params.e_del(),
                    sw_params.o_ins(),
                    sw_params.e_ins(),
                    sw_params.zdrop(),
                    sw_params.scoring_matrix(),
                    5,
                )
            },
            SimdEngineType::Engine128 => sw_params.simd_banded_swa_batch8_int16(batch),
        }
    }

    #[cfg(all(target_arch = "x86_64", not(feature = "avx512")))]
    {
        match engine {
            SimdEngineType::Engine256 => unsafe {
                crate::alignment::banded_swa_avx2::simd_banded_swa_batch16_int16(
                    batch,
                    sw_params.o_del(),
                    sw_params.e_del(),
                    sw_params.o_ins(),
                    sw_params.e_ins(),
                    sw_params.zdrop(),
                    sw_params.scoring_matrix(),
                    5,
                )
            },
            SimdEngineType::Engine128 => sw_params.simd_banded_swa_batch8_int16(batch),
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = engine;
        sw_params.simd_banded_swa_batch8_int16(batch)
    }
}

/// Distribute extension results back to per-read chain scores
pub fn distribute_extension_results(
    results: &[BatchExtensionResult],
    all_chain_scores: &mut [Vec<ChainExtensionScores>],
) {
    for result in results {
        // Ensure the chain_scores vector is large enough
        let chain_scores = &mut all_chain_scores[result.read_idx];
        if chain_scores.len() <= result.chain_idx {
            chain_scores.resize(result.chain_idx + 1, ChainExtensionScores::default());
        }

        let scores = &mut chain_scores[result.chain_idx];
        match result.direction {
            ExtensionDirection::Left => {
                scores.left_score = Some(result.score);
                scores.left_query_end = Some(result.query_end);
                scores.left_ref_end = Some(result.ref_end);
                scores.left_gscore = Some(result.gscore);
                scores.left_gref_end = Some(result.gref_end);
            }
            ExtensionDirection::Right => {
                scores.right_score = Some(result.score);
                scores.right_query_end = Some(result.query_end);
                scores.right_ref_end = Some(result.ref_end);
                scores.right_gscore = Some(result.gscore);
                scores.right_gref_end = Some(result.gref_end);
            }
        }
    }
}

// ============================================================================
// CROSS-READ BATCHING - COLLECTION INFRASTRUCTURE
// ============================================================================
//
// These structures and functions implement BWA-MEM2's cross-read batching strategy:
// 1. Seeding and chaining happen per-read (parallelized)
// 2. Extension jobs are COLLECTED from all reads into a single batch
// 3. SIMD scoring executes on the FULL batch (maximizing lane utilization)
// 4. Results are distributed back to per-read structures
// 5. Finalization happens per-read (parallelized)
//
// ============================================================================

/// Context for a single read during cross-read batched extension
///
/// Contains all the data needed to collect extension jobs and later
/// merge results back into AlignmentRegions.
#[derive(Debug, Clone)]
pub struct ReadExtensionContext {
    /// Read name (for debugging)
    pub query_name: String,
    /// Encoded query sequence (2-bit)
    pub encoded_query: Vec<u8>,
    /// Reverse complement of encoded query
    pub encoded_query_rc: Vec<u8>,
    /// Chains for this read
    pub chains: Vec<Chain>,
    /// Seeds for this read
    pub seeds: Vec<Seed>,
    /// Query length
    pub query_len: i32,
    /// Reference segments for each chain: (rmax_0, rmax_1, rseq)
    pub chain_ref_segments: Vec<Option<(u64, u64, Vec<u8>)>>,
}

/// Per-read mappings from chains/seeds to job indices
///
/// Uses the same SeedExtensionMapping and ChainExtensionMapping structures
/// from region.rs for compatibility with merge_extension_scores_to_regions().
#[derive(Debug, Clone, Default)]
pub struct ReadExtensionMappings {
    pub chain_mappings: Vec<ChainExtensionMapping>,
}

/// Collect extension jobs from a single read's chains
///
/// This extracts the job collection logic from `extend_chains_to_regions()`,
/// storing jobs in the cross-read batches instead of executing immediately.
///
/// **Important**: The job indices stored in SeedExtensionMapping are LOCAL
/// (per-read) indices, not global batch indices. This allows
/// merge_scores_to_regions() to work correctly with per-read score vectors.
pub fn collect_extension_jobs_for_read(
    bwa_idx: &BwaIndex,
    opt: &MemOpt,
    read_idx: usize,
    ctx: &mut ReadExtensionContext,
    left_batch: &mut ExtensionJobBatch,
    right_batch: &mut ExtensionJobBatch,
) -> ReadExtensionMappings {
    let l_pac = bwa_idx.bns.packed_sequence_length;
    let query_len = ctx.query_len;

    let mut mappings = ReadExtensionMappings {
        chain_mappings: Vec::with_capacity(ctx.chains.len()),
    };

    // Track local (per-read) job indices for use by merge_scores_to_regions()
    // These are indices into the per-read score vectors, NOT global batch indices
    let mut left_local_idx = 0usize;
    let mut right_local_idx = 0usize;

    // Pre-allocate chain_ref_segments
    ctx.chain_ref_segments = Vec::with_capacity(ctx.chains.len());

    for (chain_idx, chain) in ctx.chains.iter().enumerate() {
        if chain.seeds.is_empty() {
            mappings
                .chain_mappings
                .push(ChainExtensionMapping { seed_mappings: Vec::new() });
            ctx.chain_ref_segments.push(None);
            continue;
        }

        // Calculate rmax bounds (same as region.rs)
        let (mut rmax_0, mut rmax_1) = (l_pac << 1, 0u64);

        for &seed_idx in &chain.seeds {
            let seed = &ctx.seeds[seed_idx];
            let left_margin = seed.query_pos + cal_max_gap(opt, seed.query_pos);
            let b = seed.ref_pos.saturating_sub(left_margin as u64);
            let remaining_query = query_len - seed.query_pos - seed.len;
            let right_margin = remaining_query + cal_max_gap(opt, remaining_query);
            let e = seed.ref_pos + seed.len as u64 + right_margin as u64;
            rmax_0 = rmax_0.min(b);
            rmax_1 = rmax_1.max(e);
        }

        rmax_1 = rmax_1.min(l_pac << 1);
        if rmax_0 < l_pac && l_pac < rmax_1 {
            if ctx.seeds[chain.seeds[0]].ref_pos < l_pac {
                rmax_1 = l_pac;
            } else {
                rmax_0 = l_pac;
            }
        }

        // Fetch reference segment
        let rseq = match bwa_idx.bns.get_reference_segment(rmax_0, rmax_1 - rmax_0) {
            Ok(seq) => seq,
            Err(_) => {
                mappings
                    .chain_mappings
                    .push(ChainExtensionMapping { seed_mappings: Vec::new() });
                ctx.chain_ref_segments.push(None);
                continue;
            }
        };

        ctx.chain_ref_segments.push(Some((rmax_0, rmax_1, rseq.clone())));

        // Build seed mappings and extension jobs
        let mut seed_mappings = Vec::new();

        for &seed_chain_idx in chain.seeds.iter().rev() {
            let seed = &ctx.seeds[seed_chain_idx];
            let mut left_job_idx = None;
            let mut right_job_idx = None;

            // Left extension
            if seed.query_pos > 0 {
                let tmp = (seed.ref_pos - rmax_0) as usize;
                if tmp > 0 && tmp <= rseq.len() {
                    // Build reversed sequences for left extension
                    let query_seg: Vec<u8> = ctx.encoded_query[0..seed.query_pos as usize]
                        .iter()
                        .rev()
                        .copied()
                        .collect();
                    let target_seg: Vec<u8> = rseq[0..tmp].iter().rev().copied().collect();

                    // Use local (per-read) index, NOT global batch index
                    left_job_idx = Some(left_local_idx);
                    left_local_idx += 1;
                    left_batch.add_job(
                        read_idx,
                        chain_idx,
                        seed_chain_idx,
                        ExtensionDirection::Left,
                        &query_seg,
                        &target_seg,
                        seed.len * opt.a, // h0 = seed_len * match_score
                        opt.w,
                    );
                }
            }

            // Right extension
            let seed_query_end = seed.query_pos + seed.len;
            if seed_query_end < query_len {
                let re = ((seed.ref_pos + seed.len as u64) - rmax_0) as usize;
                if re < rseq.len() {
                    let query_seg: Vec<u8> = ctx.encoded_query[seed_query_end as usize..].to_vec();
                    let target_seg: Vec<u8> = rseq[re..].to_vec();

                    // Use local (per-read) index, NOT global batch index
                    right_job_idx = Some(right_local_idx);
                    right_local_idx += 1;
                    right_batch.add_job(
                        read_idx,
                        chain_idx,
                        seed_chain_idx,
                        ExtensionDirection::Right,
                        &query_seg,
                        &target_seg,
                        0, // h0 = 0 for right extension
                        opt.w,
                    );
                }
            }

            seed_mappings.push(SeedExtensionMapping {
                seed_idx: seed_chain_idx,
                left_job_idx,
                right_job_idx,
            });
        }

        mappings
            .chain_mappings
            .push(ChainExtensionMapping { seed_mappings });
    }

    mappings
}

/// Collect extension jobs from ALL reads in a batch
///
/// This is the main entry point for cross-read batching.
/// Returns the filled job batches and per-read mappings.
pub fn collect_extension_jobs_batch(
    bwa_idx: &BwaIndex,
    opt: &MemOpt,
    read_contexts: &mut [ReadExtensionContext],
) -> (ExtensionJobBatch, ExtensionJobBatch, Vec<ReadExtensionMappings>) {
    // Estimate capacity based on typical chains per read
    let estimated_jobs = read_contexts.len() * 8; // ~4 chains, 2 directions each
    let estimated_seq_bytes = estimated_jobs * 200; // ~100bp per extension

    let mut left_batch = ExtensionJobBatch::with_capacity(estimated_jobs, estimated_seq_bytes);
    let mut right_batch = ExtensionJobBatch::with_capacity(estimated_jobs, estimated_seq_bytes);
    let mut all_mappings = Vec::with_capacity(read_contexts.len());

    for (read_idx, ctx) in read_contexts.iter_mut().enumerate() {
        let mappings = collect_extension_jobs_for_read(
            bwa_idx,
            opt,
            read_idx,
            ctx,
            &mut left_batch,
            &mut right_batch,
        );
        all_mappings.push(mappings);
    }

    log::debug!(
        "CROSS_READ_BATCH: Collected {} left jobs, {} right jobs from {} reads",
        left_batch.len(),
        right_batch.len(),
        read_contexts.len()
    );

    (left_batch, right_batch, all_mappings)
}

/// Convert batch extension results back to per-read OutScore vectors
///
/// This allows us to use the existing `merge_scores_to_regions()` infrastructure
/// after cross-read SIMD scoring.
pub fn convert_batch_results_to_outscores(
    results: &[BatchExtensionResult],
    batch: &ExtensionJobBatch,
    num_reads: usize,
) -> Vec<Vec<OutScore>> {
    // Pre-allocate per-read result vectors
    let mut per_read_scores: Vec<Vec<OutScore>> = vec![Vec::new(); num_reads];

    // Determine size of each read's score vector from the batch jobs
    let mut read_job_counts: Vec<usize> = vec![0; num_reads];
    for job in &batch.jobs {
        read_job_counts[job.read_idx] += 1;
    }

    for (read_idx, count) in read_job_counts.iter().enumerate() {
        per_read_scores[read_idx].reserve(*count);
    }

    // Distribute results back to per-read vectors
    // Note: results are in the same order as batch.jobs
    for result in results {
        per_read_scores[result.read_idx].push(OutScore {
            score: result.score,
            query_end_pos: result.query_end,
            target_end_pos: result.ref_end,
            global_score: result.gscore,
            gtarget_end_pos: result.gref_end,
            max_offset: result.max_off,
        });
    }

    per_read_scores
}

// ============================================================================
// CROSS-READ BATCH ALIGNMENT - MAIN ORCHESTRATION
// ============================================================================
//
// This is the top-level function that replaces per-read align_read_deferred()
// with cross-read batched processing for better SIMD utilization.
//
// ============================================================================

use crate::alignment::finalization::{mark_secondary_alignments, Alignment};
use crate::alignment::pipeline::{build_and_filter_chains, find_seeds};
use crate::alignment::region::{generate_cigar_from_region, merge_extension_scores_to_regions};
use rayon::prelude::*;

/// Process a batch of reads using cross-read SIMD batching
///
/// This is the high-performance alternative to per-read align_read_deferred().
/// Instead of processing each read's extensions independently, we collect
/// extension jobs from ALL reads and process them together with SIMD.
///
/// # Performance
/// - AVX-512: ~2x better SIMD utilization (32 lanes fully used)
/// - AVX2: ~1.5x better SIMD utilization (16 lanes fully used)
///
/// # Arguments
/// * `bwa_idx` - Reference genome index
/// * `pac_data` - Packed reference sequence for CIGAR generation
/// * `opt` - Alignment options
/// * `names` - Read names
/// * `seqs` - Read sequences
/// * `quals` - Quality strings
/// * `batch_start_id` - Starting read ID for deterministic hash tie-breaking
/// * `engine` - SIMD engine type
///
/// # Returns
/// Vector of alignments for each read
pub fn process_batch_cross_read(
    bwa_idx: &BwaIndex,
    pac_data: &[u8],
    opt: &MemOpt,
    names: &[String],
    seqs: &[Vec<u8>],
    _quals: &[String],
    batch_start_id: u64,
    engine: SimdEngineType,
) -> Vec<Vec<Alignment>> {
    use std::time::Instant;

    let batch_size = names.len();
    if batch_size == 0 {
        return Vec::new();
    }

    // ========================================================================
    // Phase 1: Seeding + Chaining (parallel per-read)
    // ========================================================================
    let phase1_start = Instant::now();

    let contexts: Vec<Option<ReadExtensionContext>> = names
        .par_iter()
        .zip(seqs.par_iter())
        .map(|(name, seq)| {
            // Find seeds
            let (seeds, encoded_query, encoded_query_rc) = find_seeds(bwa_idx, name, seq, opt);

            if seeds.is_empty() {
                return None;
            }

            // Build chains
            let (chains, sorted_seeds) = build_and_filter_chains(seeds, opt, seq.len(), name);

            if chains.is_empty() {
                return None;
            }

            Some(ReadExtensionContext {
                query_name: name.clone(),
                encoded_query,
                encoded_query_rc,
                chains,
                seeds: sorted_seeds,
                query_len: seq.len() as i32,
                chain_ref_segments: Vec::new(), // Filled during job collection
            })
        })
        .collect();

    let phase1_time = phase1_start.elapsed();

    // Count reads with valid contexts
    let valid_count = contexts.iter().filter(|c| c.is_some()).count();
    log::debug!(
        "CROSS_READ_BATCH: Phase 1 (seed+chain) took {:.3}s for {} reads ({} valid)",
        phase1_time.as_secs_f64(),
        batch_size,
        valid_count
    );

    // ========================================================================
    // Phase 2: Collect extension jobs across ALL reads
    // ========================================================================
    let phase2_start = Instant::now();

    // Convert Option<Context> to mutable contexts, tracking indices
    let mut read_contexts: Vec<ReadExtensionContext> = Vec::with_capacity(valid_count);
    let mut context_to_read_idx: Vec<usize> = Vec::with_capacity(valid_count);

    for (read_idx, ctx_opt) in contexts.into_iter().enumerate() {
        if let Some(ctx) = ctx_opt {
            read_contexts.push(ctx);
            context_to_read_idx.push(read_idx);
        }
    }

    // Collect jobs
    let (left_batch, right_batch, mappings) =
        collect_extension_jobs_batch(bwa_idx, opt, &mut read_contexts);

    let phase2_time = phase2_start.elapsed();
    log::debug!(
        "CROSS_READ_BATCH: Phase 2 (collect) took {:.3}s - {} left, {} right jobs",
        phase2_time.as_secs_f64(),
        left_batch.len(),
        right_batch.len()
    );

    // ========================================================================
    // Phase 3: Execute SIMD scoring on FULL batch
    // ========================================================================
    let phase3_start = Instant::now();

    let sw_params = BandedPairWiseSW::new(
        opt.o_del,
        opt.e_del,
        opt.o_ins,
        opt.e_ins,
        opt.zdrop,
        5, // end_bonus
        opt.pen_clip5,
        opt.pen_clip3,
        opt.mat,
        opt.a as i8,
        -(opt.b as i8),
    );

    let left_results = execute_batch_simd_scoring(&sw_params, &left_batch, engine);
    let right_results = execute_batch_simd_scoring(&sw_params, &right_batch, engine);

    let phase3_time = phase3_start.elapsed();
    log::debug!(
        "CROSS_READ_BATCH: Phase 3 (SIMD) took {:.3}s",
        phase3_time.as_secs_f64()
    );

    // ========================================================================
    // Phase 4: Distribute results and merge to regions
    // ========================================================================
    let phase4_start = Instant::now();

    let per_read_left_scores = convert_batch_results_to_outscores(
        &left_results,
        &left_batch,
        read_contexts.len(),
    );
    let per_read_right_scores = convert_batch_results_to_outscores(
        &right_results,
        &right_batch,
        read_contexts.len(),
    );

    let phase4_time = phase4_start.elapsed();
    log::debug!(
        "CROSS_READ_BATCH: Phase 4 (distribute) took {:.3}s",
        phase4_time.as_secs_f64()
    );

    // ========================================================================
    // Phase 5: Finalization (parallel per-read)
    // ========================================================================
    let phase5_start = Instant::now();

    let valid_alignments: Vec<(usize, Vec<Alignment>)> = read_contexts
        .into_par_iter()
        .zip(mappings.par_iter())
        .zip(per_read_left_scores.par_iter())
        .zip(per_read_right_scores.par_iter())
        .enumerate()
        .map(|(ctx_idx, (((ctx, mapping), left_scores), right_scores))| {
            let read_idx = context_to_read_idx[ctx_idx];
            let read_id = batch_start_id + read_idx as u64;

            // Merge scores to regions
            let regions = merge_extension_scores_to_regions(
                bwa_idx,
                opt,
                &ctx.chains,
                &ctx.seeds,
                &mapping.chain_mappings,
                &ctx.chain_ref_segments,
                left_scores,
                right_scores,
                ctx.query_len,
            );

            if regions.is_empty() {
                return (read_idx, vec![create_unmapped_alignment_internal(&ctx.query_name)]);
            }

            // Filter regions by score
            let mut filtered_regions: Vec<_> = regions
                .into_iter()
                .filter(|r| r.score >= opt.t)
                .collect();

            if filtered_regions.is_empty() {
                return (read_idx, vec![create_unmapped_alignment_internal(&ctx.query_name)]);
            }

            // Sort by score descending
            filtered_regions.sort_by(|a, b| b.score.cmp(&a.score));

            // Generate CIGAR for surviving regions
            let mut alignments = Vec::new();
            for (idx, region) in filtered_regions.iter().enumerate() {
                let cigar_result =
                    generate_cigar_from_region(bwa_idx, pac_data, &ctx.encoded_query, region, opt);

                let (cigar, nm, md_tag) = match cigar_result {
                    Some(result) => result,
                    None => continue,
                };

                let flag = if region.is_rev {
                    crate::alignment::finalization::sam_flags::REVERSE
                } else {
                    0
                };

                let hash = crate::utils::hash_64(read_id + idx as u64);

                alignments.push(Alignment {
                    query_name: ctx.query_name.clone(),
                    flag,
                    ref_name: region.ref_name.clone(),
                    ref_id: region.rid as usize,
                    pos: region.chr_pos,
                    mapq: 60,
                    score: region.score,
                    cigar,
                    rnext: "*".to_string(),
                    pnext: 0,
                    tlen: 0,
                    seq: String::new(),
                    qual: String::new(),
                    tags: vec![
                        ("AS".to_string(), format!("i:{}", region.score)),
                        ("NM".to_string(), format!("i:{}", nm)),
                        ("MD".to_string(), format!("Z:{}", md_tag)),
                    ],
                    query_start: region.qb,
                    query_end: region.qe,
                    seed_coverage: region.seedcov,
                    hash,
                    frac_rep: region.frac_rep,
                });
            }

            if alignments.is_empty() {
                return (read_idx, vec![create_unmapped_alignment_internal(&ctx.query_name)]);
            }

            // Mark secondary/supplementary
            mark_secondary_alignments(&mut alignments, opt);

            (read_idx, alignments)
        })
        .collect();

    let phase5_time = phase5_start.elapsed();
    log::debug!(
        "CROSS_READ_BATCH: Phase 5 (finalize) took {:.3}s",
        phase5_time.as_secs_f64()
    );

    // ========================================================================
    // Assemble final results
    // ========================================================================

    // Initialize all reads as unmapped
    let mut all_alignments: Vec<Vec<Alignment>> = names
        .iter()
        .map(|name| vec![create_unmapped_alignment_internal(name)])
        .collect();

    // Fill in valid alignments
    for (read_idx, alignments) in valid_alignments {
        all_alignments[read_idx] = alignments;
    }

    all_alignments
}

/// Create an unmapped alignment record (internal helper)
fn create_unmapped_alignment_internal(query_name: &str) -> Alignment {
    use crate::alignment::finalization::sam_flags;

    Alignment {
        query_name: query_name.to_string(),
        flag: sam_flags::UNMAPPED,
        ref_name: "*".to_string(),
        ref_id: 0,
        pos: 0,
        mapq: 0,
        score: 0,
        cigar: Vec::new(),
        rnext: "*".to_string(),
        pnext: 0,
        tlen: 0,
        seq: String::new(),
        qual: String::new(),
        tags: vec![
            ("AS".to_string(), "i:0".to_string()),
            ("NM".to_string(), "i:0".to_string()),
        ],
        query_start: 0,
        query_end: 0,
        seed_coverage: 0,
        hash: 0,
        frac_rep: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extension_job_batch_add_and_retrieve() {
        let mut batch = ExtensionJobBatch::new();

        // Add a job
        let query = vec![0u8, 1, 2, 3]; // ACGT
        let ref_seq = vec![0u8, 1, 2, 3, 0, 1];
        batch.add_job(0, 0, 0, ExtensionDirection::Left, &query, &ref_seq, 10, 50);

        assert_eq!(batch.len(), 1);
        assert_eq!(batch.get_query_seq(0), &query);
        assert_eq!(batch.get_ref_seq(0), &ref_seq);
        assert_eq!(batch.jobs[0].h0, 10);
    }

    #[test]
    fn test_extension_job_batch_multiple_jobs() {
        let mut batch = ExtensionJobBatch::new();

        // Add multiple jobs
        batch.add_job(0, 0, 0, ExtensionDirection::Left, &[0, 1, 2], &[0, 1], 5, 50);
        batch.add_job(0, 1, 0, ExtensionDirection::Right, &[3, 2, 1], &[3, 2, 1, 0], 0, 50);
        batch.add_job(1, 0, 0, ExtensionDirection::Left, &[0, 0, 1, 1], &[2, 2], 8, 50);

        assert_eq!(batch.len(), 3);

        // Verify each job's sequences
        assert_eq!(batch.get_query_seq(0), &[0, 1, 2]);
        assert_eq!(batch.get_ref_seq(0), &[0, 1]);

        assert_eq!(batch.get_query_seq(1), &[3, 2, 1]);
        assert_eq!(batch.get_ref_seq(1), &[3, 2, 1, 0]);

        assert_eq!(batch.get_query_seq(2), &[0, 0, 1, 1]);
        assert_eq!(batch.get_ref_seq(2), &[2, 2]);
    }

    #[test]
    fn test_distribute_extension_results() {
        let results = vec![
            BatchExtensionResult {
                read_idx: 0,
                chain_idx: 0,
                seed_idx: 0,
                direction: ExtensionDirection::Left,
                score: 100,
                query_end: 10,
                ref_end: 15,
                gscore: 95,
                gref_end: 14,
                max_off: 2,
            },
            BatchExtensionResult {
                read_idx: 0,
                chain_idx: 0,
                seed_idx: 0,
                direction: ExtensionDirection::Right,
                score: 80,
                query_end: 20,
                ref_end: 25,
                gscore: 75,
                gref_end: 24,
                max_off: 1,
            },
            BatchExtensionResult {
                read_idx: 1,
                chain_idx: 0,
                seed_idx: 0,
                direction: ExtensionDirection::Left,
                score: 50,
                query_end: 5,
                ref_end: 8,
                gscore: 48,
                gref_end: 7,
                max_off: 0,
            },
        ];

        let mut all_chain_scores: Vec<Vec<ChainExtensionScores>> = vec![vec![], vec![]];

        distribute_extension_results(&results, &mut all_chain_scores);

        // Check read 0, chain 0
        assert_eq!(all_chain_scores[0][0].left_score, Some(100));
        assert_eq!(all_chain_scores[0][0].right_score, Some(80));

        // Check read 1, chain 0
        assert_eq!(all_chain_scores[1][0].left_score, Some(50));
        assert_eq!(all_chain_scores[1][0].right_score, None);
    }
}
