// Mate rescue module
//
// This module handles mate rescue using Smith-Waterman alignment:
// - Region calculation based on insert size distribution (uses full [low, high] range)
// - Full Smith-Waterman alignment (ksw_align2, NOT banded)
// - Rescued alignment creation
//
// CRITICAL: This implements BWA-MEM2's mem_matesw() algorithm exactly.
// The key differences from the old implementation:
// 1. Search region uses full insert size range [low, high], NOT a window around mean
// 2. Uses full Smith-Waterman (ksw_align2), NOT banded SW
// 3. Correct coordinate conversion for reverse strand alignments
//
// OPTIMIZATION (Session 48): Batched mate rescue
// The three-phase approach matching BWA-MEM2's mem_matesw_batch_pre/getScores/mem_matesw_batch_post:
// 1. Collection phase: Gather all SW jobs across all pairs (prepare_mate_rescue_jobs)
// 2. Execution phase: Run all SW alignments in parallel (execute_mate_rescue_batch)
// 3. Distribution phase: Create alignments and add to pairs (distribute_rescue_results)

use super::super::finalization::Alignment;
use super::super::finalization::sam_flags;
use super::super::index::index::BwaIndex;
use super::insert_size::InsertSizeStats;
use crate::core::alignment::edit_distance;
use crate::core::alignment::ksw_affine_gap::{KSW_XBYTE, KSW_XSTART, KSW_XSUBO, Kswr, ksw_align2};

use crate::compute::simd_abstraction::SimdEngine128;
use crate::compute::simd_abstraction::simd::SimdEngineType;
use crate::core::alignment::shared_types::AlignJob;
use crate::core::alignment::workspace::with_workspace;
use crate::pipelines::linear::batch_extension::dispatch::dispatch_kswv_soa;
use rayon::prelude::*;

/// A mate rescue SW job prepared for batch execution
/// Contains all data needed to execute one ksw_align2 call
#[derive(Clone)]
pub struct MateRescueJob {
    /// Index of the pair in the batch
    pub pair_index: usize,
    /// Which read is being rescued (true = read1, false = read2)
    pub rescuing_read1: bool,
    /// Query sequence (mate to be rescued), already rev-comp if needed
    pub query_seq: Vec<u8>,
    /// Reference sequence for alignment
    pub ref_seq: Vec<u8>,
    /// Adjusted reference begin position (after bns_fetch_seq)
    pub adj_rb: i64,
    /// Whether the rescued alignment is on reverse strand
    pub is_rev: bool,
    /// Orientation code (0-3)
    pub orientation: usize,
    /// Anchor reference ID
    pub anchor_ref_id: usize,
    /// Anchor reference name
    pub anchor_ref_name: String,
    /// Mate name (for the rescued read)
    pub mate_name: String,
    /// Mate sequence length
    pub mate_len: i32,
    /// Minimum seed length for score threshold
    pub min_seed_len: i32,
    /// Match score for xtra calculation
    pub match_score: i32,
}

/// Compact mate rescue job that stores indices instead of copying sequences.
/// Saves ~4GB memory for 2.3M jobs (from ~1.9KB/job to ~64 bytes/job).
///
/// Memory layout comparison:
/// - MateRescueJob: ~1900 bytes (query_seq: 150, ref_seq: 1500, strings: 50, overhead: 200)
/// - MateRescueJobCompact: ~64 bytes (all fixed-size fields)
#[derive(Clone, Copy, Debug)]
pub struct MateRescueJobCompact {
    /// Index of the pair in the batch
    pub pair_index: u32,
    /// Which read is being rescued (true = read1, false = read2)
    pub rescuing_read1: bool,
    /// Whether the rescued alignment is on reverse strand
    pub is_rev: bool,
    /// Orientation code (0-3)
    pub orientation: u8,
    /// Anchor reference ID
    pub anchor_ref_id: u16,
    /// Mate sequence length (query length)
    pub mate_len: u16,
    /// Minimum seed length for score threshold
    pub min_seed_len: i16,
    /// PAC coordinates for reference sequence fetch
    pub ref_rb: i64,
    pub ref_re: i64,
    /// Adjusted reference begin position (after bns_fetch_seq)
    pub adj_rb: i64,
    /// Match score for xtra calculation
    pub match_score: i8,
}

/// Shared context for batch mate rescue execution.
/// Contains all data needed to materialize sequences from compact jobs.
pub struct MateRescueBatchContext<'a> {
    /// PAC data for fetching reference sequences
    pub pac: &'a [u8],
    /// BWA index for coordinate conversion
    pub bwa_idx: &'a BwaIndex,
    /// Encoded query sequences for read1 (one per pair in batch)
    pub query_seqs_r1: &'a [Vec<u8>],
    /// Encoded query sequences for read2 (one per pair in batch)
    pub query_seqs_r2: &'a [Vec<u8>],
    /// Read names for read1
    pub names_r1: &'a [&'a str],
    /// Read names for read2
    pub names_r2: &'a [&'a str],
}

/// Scoring matrix for mate rescue (5x5 for DNA: A=0, C=1, G=2, T=3, N=4)
/// Match = 1, Mismatch = -4 (standard BWA-MEM2 defaults)
const MATE_RESCUE_SCORING_MATRIX: [i8; 25] = [
    1, -4, -4, -4, 0, // A
    -4, 1, -4, -4, 0, // C
    -4, -4, 1, -4, 0, // G
    -4, -4, -4, 1, 0, // T
    0, 0, 0, 0, 0, // N
];

/// Mate rescue using FULL Smith-Waterman alignment
/// Equivalent to C++ mem_matesw (bwamem_pair.cpp lines 150-283)
///
/// CRITICAL: This function now matches BWA-MEM2's exact algorithm:
/// 1. Uses FULL insert size range [low, high] for search region (NOT a window around mean)
/// 2. Uses FULL Smith-Waterman (ksw_align2) (NOT banded SW)
/// 3. Correct coordinate conversion for reverse strand alignments
///
/// Returns number of rescued alignments added
pub fn mem_matesw(
    bwa_idx: &BwaIndex,
    pac: &[u8], // Pre-loaded PAC data (passed once, not loaded per call)
    stats: &[InsertSizeStats; 4],
    anchor: &Alignment,
    mate_seq: &[u8],
    _mate_qual: &str,
    mate_name: &str,
    rescued_alignments: &mut Vec<Alignment>,
) -> usize {
    let l_pac = bwa_idx.bns.packed_sequence_length as i64;
    let l_ms = mate_seq.len() as i32;
    let min_seed_len = bwa_idx.min_seed_len;

    // Scoring parameters (matching BWA-MEM2 defaults)
    let match_score = 1i32;
    let o_del = 6;
    let e_del = 1;
    let o_ins = 6;
    let e_ins = 1;

    // Step 1: Initialize skip array from failed orientations (C++ lines 160-162)
    let mut skip = [false; 4];
    for r in 0..4 {
        skip[r] = stats[r].failed;
    }

    // Step 2: Check which orientations already have consistent pairs (C++ lines 164-170)
    // Convert anchor position to bidirectional coordinates first
    let anchor_is_rev = (anchor.flag & sam_flags::REVERSE) != 0;
    let anchor_ref_len = anchor.reference_length() as i64;
    let chr_offset = bwa_idx.bns.annotations[anchor.ref_id].offset as i64;
    let genome_pos = chr_offset + anchor.pos as i64;

    let anchor_rb = if anchor_is_rev {
        let rightmost = genome_pos + anchor_ref_len - 1;
        (l_pac << 1) - 1 - rightmost
    } else {
        genome_pos
    };

    for aln in rescued_alignments.iter() {
        if aln.ref_name == anchor.ref_name {
            // Convert mate alignment to bidirectional coordinates
            let mate_is_rev = (aln.flag & sam_flags::REVERSE) != 0;
            let mate_ref_len = aln.reference_length() as i64;
            let mate_genome_pos = chr_offset + aln.pos as i64;

            let mate_rb = if mate_is_rev {
                let rightmost = mate_genome_pos + mate_ref_len - 1;
                (l_pac << 1) - 1 - rightmost
            } else {
                mate_genome_pos
            };

            let (dir, dist) = mem_infer_dir(l_pac, anchor_rb, mate_rb);
            if dist >= stats[dir].low as i64 && dist <= stats[dir].high as i64 {
                skip[dir] = true;
            }
        }
    }

    // Step 3: Early exit if all orientations already have pairs (C++ line 172)
    if skip.iter().all(|&x| x) {
        return 0;
    }

    let is_debug_read = mate_name.contains("10009:11965");
    if is_debug_read {
        log::debug!(
            "MATE_RESCUE_SW {}: anchor {}:{} is_rev={}, ref_len={}, rb={}, stats[1]=[{},{}]",
            mate_name,
            anchor.ref_name,
            anchor.pos,
            anchor_is_rev,
            anchor_ref_len,
            anchor_rb,
            stats[1].low,
            stats[1].high
        );
    }

    let mut n_rescued = 0;

    // Step 4: Try each non-skipped orientation (C++ lines 175-269)
    for r in 0..4 {
        if skip[r] {
            continue;
        }

        // Decode orientation bits (C++ lines 181-182)
        // r>>1 = anchor strand (0=forward, 1=reverse in bidirectional coords)
        // r&1 = mate strand (0=forward, 1=reverse in bidirectional coords)
        let is_rev = (r >> 1) != (r & 1); // Whether to reverse complement the mate
        let is_larger = (r >> 1) == 0; // Whether mate has larger coordinate

        // Step 5: Prepare mate sequence (reverse complement if needed) (C++ lines 184-193)
        let mut seq: Vec<u8> = if is_rev {
            mate_seq
                .iter()
                .rev()
                .map(|&b| if b < 4 { 3 - b } else { 4 })
                .collect()
        } else {
            mate_seq.to_vec()
        };

        // Step 6: Calculate search region using FULL insert size range [low, high]
        // THIS IS CRITICAL - matches BWA-MEM2 exactly (C++ lines 195-204)
        let (rb, re) = if !is_rev {
            // Same strand as anchor
            let rb = if is_larger {
                anchor_rb + stats[r].low as i64
            } else {
                anchor_rb - stats[r].high as i64
            };
            let re = if is_larger {
                anchor_rb + stats[r].high as i64
            } else {
                anchor_rb - stats[r].low as i64
            } + l_ms as i64;
            (rb, re)
        } else {
            // Opposite strand from anchor
            let rb = if is_larger {
                anchor_rb + stats[r].low as i64
            } else {
                anchor_rb - stats[r].high as i64
            } - l_ms as i64;
            let re = if is_larger {
                anchor_rb + stats[r].high as i64
            } else {
                anchor_rb - stats[r].low as i64
            };
            (rb, re)
        };

        // Clamp to valid range (C++ lines 206-207)
        let rb = rb.max(0);
        let re = re.min(l_pac << 1);

        if is_debug_read {
            log::debug!(
                "MATE_RESCUE_SW {}: orientation {} is_rev={}, is_larger={}, search region [{}, {}), size={}",
                mate_name,
                r,
                is_rev,
                is_larger,
                rb,
                re,
                re - rb
            );
        }

        if rb >= re {
            continue;
        }

        // Step 7: Fetch reference sequence (C++ lines 210-211)
        let (mut ref_seq, adj_rb, adj_re, rid) =
            bwa_idx.bns.bns_fetch_seq(pac, rb, (rb + re) >> 1, re);

        // Step 8: Check if on same reference and region is large enough (C++ lines 214-215)
        if rid as usize != anchor.ref_id || (adj_re - adj_rb) < min_seed_len as i64 {
            if is_debug_read {
                log::debug!(
                    "MATE_RESCUE_SW {}: skipping orientation {} - rid={} (need {}), len={}",
                    mate_name,
                    r,
                    rid,
                    anchor.ref_id,
                    adj_re - adj_rb
                );
            }
            continue;
        }

        let ref_len = ref_seq.len() as i32;

        // Step 9: Perform FULL Smith-Waterman (NOT banded!) using ksw_align2 (C++ lines 217-223)
        // Build xtra flags matching BWA-MEM2
        let xtra = KSW_XSUBO
            | KSW_XSTART
            | if (l_ms * match_score) < 250 {
                KSW_XBYTE
            } else {
                0
            }
            | (min_seed_len * match_score) as u32;

        let aln: Kswr = unsafe {
            ksw_align2::<SimdEngine128>(
                l_ms,
                &mut seq,
                ref_len,
                &mut ref_seq,
                5, // m (alphabet size)
                &MATE_RESCUE_SCORING_MATRIX,
                o_del,
                e_del,
                o_ins,
                e_ins,
                xtra,
            )
        };

        if is_debug_read {
            log::debug!(
                "MATE_RESCUE_SW {}: orientation {} SW result score={}, qb={}, qe={}, tb={}, te={}",
                mate_name,
                r,
                aln.score,
                aln.qb,
                aln.qe,
                aln.tb,
                aln.te
            );
        }

        // Step 10: Check if alignment is good enough (C++ lines 226-227)
        if aln.score < min_seed_len || aln.qb < 0 {
            continue;
        }

        // Step 11: Convert coordinates (C++ lines 229-238)
        // This matches BWA-MEM2's exact coordinate transformation
        let (rescued_rb, _rescued_re, query_start, query_end) = if is_rev {
            // Reverse strand coordinate conversion
            let rb_result = (l_pac << 1) - (adj_rb + aln.te as i64 + 1);
            let re_result = (l_pac << 1) - (adj_rb + aln.tb as i64);
            let qb_result = l_ms - (aln.qe + 1);
            let qe_result = l_ms - aln.qb;
            (rb_result, re_result, qb_result, qe_result)
        } else {
            // Forward strand - straightforward
            let rb_result = adj_rb + aln.tb as i64;
            let re_result = adj_rb + aln.te as i64 + 1;
            (rb_result, re_result, aln.qb, aln.qe + 1)
        };

        // Convert bidirectional position to chromosome-relative
        let (pos_f, _is_rev_depos) = bwa_idx.bns.bns_depos(rescued_rb);
        let rescued_rid = bwa_idx.bns.bns_pos2rid(pos_f);

        if rescued_rid < 0 || rescued_rid as usize != anchor.ref_id {
            if is_debug_read {
                log::debug!(
                    "MATE_RESCUE_SW {}: skipping - rid mismatch: rescued_rid={}, anchor.ref_id={}",
                    mate_name,
                    rescued_rid,
                    anchor.ref_id
                );
            }
            continue;
        }

        let chr_pos = (pos_f - bwa_idx.bns.annotations[rescued_rid as usize].offset as i64) as u64;

        // Build CIGAR from alignment endpoints
        // For mate rescue, we generate a simple cigar based on the alignment extent
        // Use the ksw_align2 result positions (te - tb + 1) for reference aligned length
        let ref_aligned_len = (aln.te - aln.tb + 1).max(0);
        let query_aligned = (query_end - query_start).max(0);

        // Generate CIGAR: soft-clip leading, match region, soft-clip trailing
        // IMPORTANT: CIGAR operations are stored as ASCII characters (b'M', b'S', etc.)
        let mut cigar: Vec<(u8, i32)> = Vec::new();
        if query_start > 0 {
            cigar.push((b'S', query_start)); // Soft clip at start
        }
        if ref_aligned_len > 0 && query_aligned > 0 {
            // Use the smaller of ref and query lengths as match, add indel for difference
            if ref_aligned_len == query_aligned {
                cigar.push((b'M', ref_aligned_len)); // Match
            } else if ref_aligned_len > query_aligned {
                cigar.push((b'M', query_aligned)); // Match
                cigar.push((b'D', ref_aligned_len - query_aligned)); // Deletion
            } else {
                cigar.push((b'M', ref_aligned_len)); // Match
                cigar.push((b'I', query_aligned - ref_aligned_len)); // Insertion
            }
        }
        if query_end < l_ms {
            cigar.push((b'S', l_ms - query_end)); // Soft clip at end
        }

        // Create alignment structure
        let mut flag = 0u16;
        if is_rev {
            flag |= sam_flags::REVERSE;
        }

        // Compute NM and MD tags for the rescued alignment
        let ref_len: i32 = cigar
            .iter()
            .filter_map(|&(op, len)| {
                if matches!(op, b'M' | b'D' | b'=' | b'X') {
                    Some(len)
                } else {
                    None
                }
            })
            .sum();

        // Get aligned query length
        let aligned_query_len: i32 = cigar
            .iter()
            .filter_map(|&(op, len)| {
                if matches!(op, b'M' | b'I' | b'=' | b'X') {
                    Some(len)
                } else {
                    None
                }
            })
            .sum();

        // Calculate leading soft clips
        let mut left_clip = 0i32;
        for &(op, len) in cigar.iter() {
            if op == b'S' || op == b'H' {
                left_clip += len;
            } else {
                break;
            }
        }

        // Get reference sequence at chromosome position
        let forward_ref =
            bwa_idx
                .bns
                .get_forward_ref(pac, anchor.ref_id, chr_pos, ref_len.max(0) as usize);

        // Get aligned query portion
        // For reverse strand, need to handle coordinate transformation
        let (nm, md_tag) = if is_rev {
            // For reverse strand: SEQ = revcomp(original)
            // CIGAR describes SEQ, so aligned portion is SEQ[left_clip..left_clip+aligned_query_len]
            // This maps to original[len-left_clip-aligned_query_len..len-left_clip]
            let orig_start = (l_ms - left_clip - aligned_query_len).max(0) as usize;
            let orig_end = (l_ms - left_clip).max(0) as usize;
            let aligned_query = &mate_seq[orig_start..orig_end.min(mate_seq.len())];
            let query_for_md: Vec<u8> = aligned_query.iter().rev().map(|&b| 3 - b).collect();
            edit_distance::compute_nm_and_md(&forward_ref, &query_for_md, &cigar)
        } else {
            // Forward strand: use query directly
            let start = query_start.max(0) as usize;
            let end = query_end.max(0) as usize;
            let aligned_query = &mate_seq[start..end.min(mate_seq.len())];
            edit_distance::compute_nm_and_md(&forward_ref, aligned_query, &cigar)
        };

        let rescued_aln = Alignment {
            query_name: mate_name.to_string(),
            flag,
            ref_name: anchor.ref_name.clone(),
            ref_id: anchor.ref_id,
            pos: chr_pos,
            mapq: 0, // Will be calculated later
            score: aln.score,
            cigar,
            rnext: String::from("*"),
            pnext: 0,
            tlen: 0,
            seq: String::new(),
            qual: String::new(),
            tags: vec![
                ("AS".to_string(), format!("i:{}", aln.score)),
                ("NM".to_string(), format!("i:{nm}")),
                ("MD".to_string(), format!("Z:{md_tag}")),
            ],
            query_start,
            query_end,
            seed_coverage: (ref_aligned_len.min(query_aligned) >> 1) as i32,
            hash: 0,
            frac_rep: 0.0,
        };

        if is_debug_read {
            log::debug!(
                "MATE_RESCUE_SW {}: RESCUED! pos={}, score={}, cigar={:?}",
                mate_name,
                chr_pos,
                aln.score,
                rescued_aln.cigar
            );
        }

        rescued_alignments.push(rescued_aln);
        n_rescued += 1;
    }

    n_rescued
}

/// Determine orientation and distance between two alignments
/// Equivalent to C++ mem_infer_dir (bwamem_pair.cpp lines 58-65)
///
/// Returns (orientation_code, distance)
/// Orientation codes:
///   0 = FF (forward-forward)
///   1 = FR (forward-reverse) - typical paired-end
///   2 = RF (reverse-forward)
///   3 = RR (reverse-reverse)
#[inline]
fn mem_infer_dir(l_pac: i64, b1: i64, b2: i64) -> (usize, i64) {
    let r1 = if b1 >= l_pac { 1 } else { 0 };
    let r2 = if b2 >= l_pac { 1 } else { 0 };

    // Project b2 onto b1's strand
    let p2 = if r1 == r2 { b2 } else { (l_pac << 1) - 1 - b2 };

    // Calculate absolute distance
    let dist = if p2 > b1 { p2 - b1 } else { b1 - p2 };

    // Calculate orientation code
    let dir = if r1 == r2 { 0 } else { 1 } ^ if p2 > b1 { 0 } else { 3 };

    (dir, dist)
}

// ============================================================================
// BATCHED MATE RESCUE (Session 48 optimization)
// ============================================================================
// Three-phase approach matching BWA-MEM2's batch strategy:
// 1. prepare_mate_rescue_jobs_for_pair - collects SW jobs for one pair
// 2. execute_mate_rescue_batch - runs all SW in parallel
// 3. result_to_alignment - converts SW result to Alignment

/// Result from a single mate rescue SW execution
pub struct MateRescueResult {
    /// Index of the job this result corresponds to
    pub job_index: usize,
    /// The ksw_align2 result
    pub aln: Kswr,
}

/// Execute all mate rescue SW jobs in parallel using rayon
/// This is Phase 2 of the batch strategy
pub fn execute_mate_rescue_batch(jobs: &mut [MateRescueJob]) -> Vec<MateRescueResult> {
    execute_mate_rescue_batch_with_engine(jobs, None)
}

/// Execute all mate rescue SW jobs with optional SIMD engine specification
///
/// When `engine` is Some, uses horizontal SIMD batching for better throughput.
/// When `engine` is None, uses the scalar/rayon fallback.
///
/// # Arguments
/// * `jobs` - Mate rescue jobs to execute
/// * `engine` - Optional SIMD engine type for horizontal batching
///
/// # Horizontal SIMD Batching (engine = Some)
/// - Groups jobs into batches matching SIMD width (16/32/64 depending on engine)
/// - Uses Structure-of-Arrays layout for cache-efficient SIMD processing
/// - Significantly faster for large batches (16+ jobs)
///
/// # Scalar/Rayon Fallback (engine = None)
/// - Uses per-alignment ksw_align2 with rayon parallel iteration
/// - Good for small batches or when SIMD overhead isn't worthwhile
///
/// # Environment Variables
/// - `FERROUS_ALIGN_FORCE_SCALAR=1`: Force scalar path for debugging/validation
pub fn execute_mate_rescue_batch_with_engine(
    jobs: &mut [MateRescueJob],
    engine: Option<SimdEngineType>,
) -> Vec<MateRescueResult> {
    // Check for scalar fallback override via environment variable
    // This is useful for validating SIMD results against scalar baseline
    let force_scalar = std::env::var("FERROUS_ALIGN_FORCE_SCALAR")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false);

    if force_scalar {
        log::info!(
            "FERROUS_ALIGN_FORCE_SCALAR=1: Using scalar path for mate rescue ({} jobs)",
            jobs.len()
        );
        return execute_mate_rescue_batch_scalar(jobs);
    }

    // Use horizontal SIMD batching if engine is specified and we have enough jobs
    // SIMD kernels use two registers for te tracking (te_lo/te_hi) to handle
    // 16-bit te values for all sequences (16 for SSE/NEON, 32 for AVX2).
    if let Some(simd_engine) = engine {
        let min_batch_for_simd = match simd_engine {
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            SimdEngineType::Engine512 => 32, // 50% of 64-way
            #[cfg(target_arch = "x86_64")]
            SimdEngineType::Engine256 => 16, // 50% of 32-way
            SimdEngineType::Engine128 => 8, // 50% of 16-way
        };

        if jobs.len() >= min_batch_for_simd {
            log::debug!(
                "Using horizontal SIMD for mate rescue: {} jobs, engine={:?}",
                jobs.len(),
                simd_engine
            );
            return execute_mate_rescue_batch_simd(jobs, simd_engine);
        }
    }

    // Fallback to scalar/rayon implementation
    execute_mate_rescue_batch_scalar(jobs)
}

/// Scalar implementation using ksw_align2 with rayon parallelism
fn execute_mate_rescue_batch_scalar(jobs: &mut [MateRescueJob]) -> Vec<MateRescueResult> {
    // Scoring parameters (matching BWA-MEM2 defaults)
    let o_del = 6;
    let e_del = 1;
    let o_ins = 6;
    let e_ins = 1;

    jobs.par_iter_mut()
        .enumerate()
        .map(|(idx, job)| {
            let l_ms = job.mate_len;
            let ref_len = job.ref_seq.len() as i32;

            // Build xtra flags matching BWA-MEM2
            let xtra = KSW_XSUBO
                | KSW_XSTART
                | if (l_ms * job.match_score) < 250 {
                    KSW_XBYTE
                } else {
                    0
                }
                | (job.min_seed_len * job.match_score) as u32;

            let aln: Kswr = unsafe {
                ksw_align2::<SimdEngine128>(
                    l_ms,
                    &mut job.query_seq,
                    ref_len,
                    &mut job.ref_seq,
                    5, // m (alphabet size)
                    &MATE_RESCUE_SCORING_MATRIX,
                    o_del,
                    e_del,
                    o_ins,
                    e_ins,
                    xtra,
                )
            };

            MateRescueResult {
                job_index: idx,
                aln,
            }
        })
        .collect()
}

/// Horizontal SIMD batch execution for mate rescue
///
/// This function processes mate rescue jobs using horizontal SIMD, where each SIMD lane
/// handles a different alignment simultaneously. This is significantly faster than
/// the scalar approach for large batches.
///
/// # Algorithm
/// 1. Find max query and reference lengths across all jobs
/// 2. Create SoA (Structure of Arrays) buffer for the batch
/// 3. Transpose sequences into SIMD-friendly layout
/// 4. Execute batch_ksw_align with the specified SIMD engine
/// 5. Convert results back to MateRescueResult format
///
/// # SIMD Width
/// - Engine128 (SSE/NEON): 16 alignments per SIMD operation
/// - Engine256 (AVX2): 32 alignments per SIMD operation
/// - Engine512 (AVX-512): 64 alignments per SIMD operation
fn execute_mate_rescue_batch_simd(
    jobs: &mut [MateRescueJob],
    engine: SimdEngineType,
) -> Vec<MateRescueResult> {
    if jobs.is_empty() {
        return Vec::new();
    }

    // Determine batch size from engine
    let (batch_size, _) = crate::compute::simd_abstraction::simd::get_simd_batch_sizes(engine);

    // Scoring parameters (matching BWA-MEM2 defaults for mate rescue)
    let match_score: i8 = 1;
    let mismatch_penalty: i8 = -4;
    let o_del: i32 = 6;
    let e_del: i32 = 1;
    let o_ins: i32 = 6;
    let e_ins: i32 = 1;

    // Process batches in parallel using rayon
    let results: Vec<MateRescueResult> = jobs
        .par_chunks(batch_size)
        .enumerate()
        .flat_map(|(chunk_idx, chunk)| {
            let chunk_start = chunk_idx * batch_size;
            let chunk_len = chunk.len();

            let align_jobs: Vec<AlignJob> = chunk
                .iter()
                .map(|job| AlignJob {
                    query: &job.query_seq,
                    target: &job.ref_seq,
                    qlen: job.query_seq.len(),
                    tlen: job.ref_seq.len(),
                    band: -1,
                    h0: 0,
                })
                .collect();

            let batch_results = with_workspace(|ws| {
                dispatch_kswv_soa(
                    &align_jobs,
                    ws,
                    batch_size,
                    chunk_len,
                    match_score,
                    mismatch_penalty,
                    o_del,
                    e_del,
                    o_ins,
                    e_ins,
                    0,     // ambig_penalty
                    false, // debug
                    engine,
                )
            });

            // Convert results back to MateRescueResult format
            batch_results
                .iter()
                .take(chunk_len)
                .enumerate()
                .map(|(i, ksw_result)| MateRescueResult {
                    job_index: chunk_start + i,
                    aln: Kswr {
                        score: ksw_result.score,
                        te: ksw_result.te,
                        qe: ksw_result.qe,
                        score2: ksw_result.score2,
                        te2: ksw_result.te2,
                        tb: ksw_result.tb,
                        qb: ksw_result.qb,
                    },
                })
                .collect::<Vec<_>>()
        })
        .collect();

    results
}

/// Convert a successful SW result to an Alignment
/// This is part of Phase 3 of the batch strategy
pub fn result_to_alignment(
    job: &MateRescueJob,
    aln: &Kswr,
    bwa_idx: &BwaIndex,
    pac: &[u8],
) -> Option<Alignment> {
    let l_pac = bwa_idx.bns.packed_sequence_length as i64;
    let l_ms = job.mate_len;

    let is_debug_read = job.mate_name.contains("10009:11965");
    if is_debug_read {
        log::debug!(
            "MATE_RESCUE_RESULT {}: orientation={} adj_rb={} score={} tb={} te={} qb={} qe={} min_seed_len={}",
            job.mate_name,
            job.orientation,
            job.adj_rb,
            aln.score,
            aln.tb,
            aln.te,
            aln.qb,
            aln.qe,
            job.min_seed_len
        );
    }

    // Check if alignment is good enough (C++ lines 226-227)
    if aln.score < job.min_seed_len || aln.qb < 0 {
        if is_debug_read {
            log::debug!(
                "MATE_RESCUE_RESULT {}: REJECTED - score {} < min_seed_len {} or qb {} < 0",
                job.mate_name,
                aln.score,
                job.min_seed_len,
                aln.qb
            );
        }
        return None;
    }

    // Convert coordinates (C++ lines 229-238)
    let (rescued_rb, _rescued_re, query_start, query_end) = if job.is_rev {
        // Reverse strand coordinate conversion
        let rb_result = (l_pac << 1) - (job.adj_rb + aln.te as i64 + 1);
        let _re_result = (l_pac << 1) - (job.adj_rb + aln.tb as i64);
        let qb_result = l_ms - (aln.qe + 1);
        let qe_result = l_ms - aln.qb;
        (rb_result, _re_result, qb_result, qe_result)
    } else {
        // Forward strand - straightforward
        let rb_result = job.adj_rb + aln.tb as i64;
        let re_result = job.adj_rb + aln.te as i64 + 1;
        (rb_result, re_result, aln.qb, aln.qe + 1)
    };

    // Convert bidirectional position to chromosome-relative
    let (pos_f, _is_rev_depos) = bwa_idx.bns.bns_depos(rescued_rb);
    let rescued_rid = bwa_idx.bns.bns_pos2rid(pos_f);

    if is_debug_read {
        log::debug!(
            "MATE_RESCUE_RESULT {}: rescued_rb={} pos_f={} rescued_rid={} anchor_ref_id={}",
            job.mate_name,
            rescued_rb,
            pos_f,
            rescued_rid,
            job.anchor_ref_id
        );
    }

    if rescued_rid < 0 || rescued_rid as usize != job.anchor_ref_id {
        if is_debug_read {
            log::debug!(
                "MATE_RESCUE_RESULT {}: REJECTED - rid mismatch ({} vs {})",
                job.mate_name,
                rescued_rid,
                job.anchor_ref_id
            );
        }
        return None;
    }

    let chr_pos = (pos_f - bwa_idx.bns.annotations[rescued_rid as usize].offset as i64) as u64;

    if is_debug_read {
        log::debug!(
            "MATE_RESCUE_RESULT {}: SUCCESS - chr_pos={} (target ~50141532)",
            job.mate_name,
            chr_pos
        );
    }

    // Build CIGAR from alignment endpoints
    let ref_aligned_len = (aln.te - aln.tb + 1).max(0);
    let query_aligned = (query_end - query_start).max(0);

    let mut cigar: Vec<(u8, i32)> = Vec::new();
    if query_start > 0 {
        cigar.push((b'S', query_start)); // Soft clip at start
    }
    if ref_aligned_len > 0 && query_aligned > 0 {
        if ref_aligned_len == query_aligned {
            cigar.push((b'M', ref_aligned_len));
        } else if ref_aligned_len > query_aligned {
            cigar.push((b'M', query_aligned));
            cigar.push((b'D', ref_aligned_len - query_aligned));
        } else {
            cigar.push((b'M', ref_aligned_len));
            cigar.push((b'I', query_aligned - ref_aligned_len));
        }
    }
    if query_end < l_ms {
        cigar.push((b'S', l_ms - query_end)); // Soft clip at end
    }

    // Create alignment structure
    let mut flag = 0u16;
    if job.is_rev {
        flag |= sam_flags::REVERSE;
    }

    // Compute NM and MD tags for the rescued alignment
    // Get reference length from CIGAR (M/D/=/X operations)
    let ref_len_for_md: i32 = cigar
        .iter()
        .filter_map(|&(op, len)| {
            if matches!(op, b'M' | b'D' | b'=' | b'X') {
                Some(len)
            } else {
                None
            }
        })
        .sum();

    // Validate alignment doesn't extend beyond reference bounds
    // This prevents CIGAR_MAPS_OFF_REFERENCE errors from GATK ValidateSamFile
    let ref_length = bwa_idx.bns.annotations[rescued_rid as usize].sequence_length as u64;
    if chr_pos + ref_len_for_md as u64 > ref_length {
        if is_debug_read {
            log::debug!(
                "MATE_RESCUE_RESULT {}: REJECTED - alignment extends beyond reference (pos {} + ref_len {} > ref_length {})",
                job.mate_name,
                chr_pos,
                ref_len_for_md,
                ref_length
            );
        }
        return None;
    }

    // Get reference sequence at chromosome position
    let forward_ref = bwa_idx.bns.get_forward_ref(
        pac,
        job.anchor_ref_id,
        chr_pos,
        ref_len_for_md.max(0) as usize,
    );

    // Get aligned query portion from job.query_seq
    // IMPORTANT: Use SAM coordinates (query_start/query_end), NOT SW coordinates (aln.qb/qe)
    // For reverse strand, these are different - the CIGAR uses SAM coordinates
    // The aligned portion in SAM SEQ is [query_start..query_end]
    let qb_sam = query_start.max(0) as usize;
    let qe_sam = query_end.max(0) as usize;
    let aligned_query = if qe_sam <= job.query_seq.len() {
        &job.query_seq[qb_sam..qe_sam]
    } else {
        &job.query_seq[qb_sam..]
    };

    let (nm, md_tag) = edit_distance::compute_nm_and_md(&forward_ref, aligned_query, &cigar);

    Some(Alignment {
        query_name: job.mate_name.clone(),
        flag,
        ref_name: job.anchor_ref_name.clone(),
        ref_id: job.anchor_ref_id,
        pos: chr_pos,
        mapq: 0, // Will be calculated later
        score: aln.score,
        cigar,
        rnext: String::from("*"),
        pnext: 0,
        tlen: 0,
        seq: String::new(),
        qual: String::new(),
        tags: vec![
            ("AS".to_string(), format!("i:{}", aln.score)),
            ("NM".to_string(), format!("i:{nm}")),
            ("MD".to_string(), format!("Z:{md_tag}")),
        ],
        query_start,
        query_end,
        seed_coverage: (ref_aligned_len.min(query_aligned) >> 1),
        hash: 0,
        frac_rep: 0.0,
    })
}

/// Prepare mate rescue jobs for a single anchor alignment
/// This is Phase 1 of the batch strategy - collects all valid SW jobs
pub fn prepare_mate_rescue_jobs_for_anchor(
    bwa_idx: &BwaIndex,
    pac: &[u8],
    stats: &[InsertSizeStats; 4],
    anchor: &Alignment,
    mate_seq: &[u8],
    mate_name: &str,
    existing_alignments: &[Alignment],
    pair_index: usize,
    rescuing_read1: bool,
) -> Vec<MateRescueJob> {
    let l_pac = bwa_idx.bns.packed_sequence_length as i64;
    let l_ms = mate_seq.len() as i32;
    let min_seed_len = bwa_idx.min_seed_len;
    let match_score = 1i32;

    let mut jobs = Vec::new();

    // Initialize skip array from failed orientations
    let mut skip = [false; 4];
    for r in 0..4 {
        skip[r] = stats[r].failed;
    }

    // Convert anchor position to bidirectional coordinates
    let anchor_is_rev = (anchor.flag & sam_flags::REVERSE) != 0;
    let anchor_ref_len = anchor.reference_length() as i64;
    let chr_offset = bwa_idx.bns.annotations[anchor.ref_id].offset as i64;
    let genome_pos = chr_offset + anchor.pos as i64;

    let anchor_rb = if anchor_is_rev {
        let rightmost = genome_pos + anchor_ref_len - 1;
        (l_pac << 1) - 1 - rightmost
    } else {
        genome_pos
    };

    let is_debug_read = mate_name.contains("10009:11965");
    if is_debug_read {
        log::debug!(
            "MATE_RESCUE_PREP {}: anchor={}:{} is_rev={} anchor_rb={} rescuing_read1={} existing_alns={}",
            mate_name,
            anchor.ref_name,
            anchor.pos,
            anchor_is_rev,
            anchor_rb,
            rescuing_read1,
            existing_alignments.len()
        );
    }

    // Check which orientations already have consistent pairs
    for aln in existing_alignments.iter() {
        if aln.ref_name == anchor.ref_name {
            let mate_is_rev = (aln.flag & sam_flags::REVERSE) != 0;
            let mate_ref_len = aln.reference_length() as i64;
            let mate_genome_pos = chr_offset + aln.pos as i64;

            let mate_rb = if mate_is_rev {
                let rightmost = mate_genome_pos + mate_ref_len - 1;
                (l_pac << 1) - 1 - rightmost
            } else {
                mate_genome_pos
            };

            let (dir, dist) = mem_infer_dir(l_pac, anchor_rb, mate_rb);
            if is_debug_read {
                log::debug!(
                    "MATE_RESCUE_PREP {}: existing aln {}:{} mate_rb={} dir={} dist={} range=[{},{}] in_range={}",
                    mate_name,
                    aln.ref_name,
                    aln.pos,
                    mate_rb,
                    dir,
                    dist,
                    stats[dir].low,
                    stats[dir].high,
                    dist >= stats[dir].low as i64 && dist <= stats[dir].high as i64
                );
            }
            if dist >= stats[dir].low as i64 && dist <= stats[dir].high as i64 {
                skip[dir] = true;
            }
        }
    }

    // Early exit if all orientations already have pairs
    if skip.iter().all(|&x| x) {
        if is_debug_read {
            log::debug!("MATE_RESCUE_PREP {mate_name}: EARLY EXIT - all orientations have pairs");
        }
        return jobs;
    }

    if is_debug_read {
        log::debug!("MATE_RESCUE_PREP {mate_name}: skip array = {skip:?}");
    }

    // Try each non-skipped orientation
    for r in 0..4 {
        if skip[r] {
            continue;
        }

        let is_rev = (r >> 1) != (r & 1);
        let is_larger = (r >> 1) == 0;

        // Prepare mate sequence
        let seq: Vec<u8> = if is_rev {
            mate_seq
                .iter()
                .rev()
                .map(|&b| if b < 4 { 3 - b } else { 4 })
                .collect()
        } else {
            mate_seq.to_vec()
        };

        // Calculate search region
        let (rb, re) = if !is_rev {
            let rb = if is_larger {
                anchor_rb + stats[r].low as i64
            } else {
                anchor_rb - stats[r].high as i64
            };
            let re = if is_larger {
                anchor_rb + stats[r].high as i64
            } else {
                anchor_rb - stats[r].low as i64
            } + l_ms as i64;
            (rb, re)
        } else {
            let rb = if is_larger {
                anchor_rb + stats[r].low as i64
            } else {
                anchor_rb - stats[r].high as i64
            } - l_ms as i64;
            let re = if is_larger {
                anchor_rb + stats[r].high as i64
            } else {
                anchor_rb - stats[r].low as i64
            };
            (rb, re)
        };

        // Clamp to valid range
        let rb = rb.max(0);
        let re = re.min(l_pac << 1);

        if is_debug_read {
            log::debug!(
                "MATE_RESCUE_PREP {}: orientation r={} is_rev={} is_larger={} rb={} re={} range=[{},{}]",
                mate_name,
                r,
                is_rev,
                is_larger,
                rb,
                re,
                stats[r].low,
                stats[r].high
            );
        }

        if rb >= re {
            if is_debug_read {
                log::debug!("MATE_RESCUE_PREP {mate_name}: SKIP r={r} - rb >= re");
            }
            continue;
        }

        // Fetch reference sequence
        let (ref_seq, adj_rb, adj_re, rid) = bwa_idx.bns.bns_fetch_seq(pac, rb, (rb + re) >> 1, re);

        if is_debug_read {
            log::debug!(
                "MATE_RESCUE_PREP {}: r={} fetched ref: adj_rb={} adj_re={} rid={} anchor.ref_id={} len={}",
                mate_name,
                r,
                adj_rb,
                adj_re,
                rid,
                anchor.ref_id,
                adj_re - adj_rb
            );
        }

        // Check if on same reference and region is large enough
        if rid as usize != anchor.ref_id || (adj_re - adj_rb) < min_seed_len as i64 {
            if is_debug_read {
                log::debug!(
                    "MATE_RESCUE_PREP {}: SKIP r={} - rid mismatch ({} vs {}) or too small ({})",
                    mate_name,
                    r,
                    rid,
                    anchor.ref_id,
                    adj_re - adj_rb
                );
            }
            continue;
        }

        if is_debug_read {
            log::debug!("MATE_RESCUE_PREP {mate_name}: CREATING JOB r={r}");
        }

        jobs.push(MateRescueJob {
            pair_index,
            rescuing_read1,
            query_seq: seq,
            ref_seq,
            adj_rb,
            is_rev,
            orientation: r,
            anchor_ref_id: anchor.ref_id,
            anchor_ref_name: anchor.ref_name.clone(),
            mate_name: mate_name.to_string(),
            mate_len: l_ms,
            min_seed_len,
            match_score,
        });
    }

    jobs
}

// ============================================================================
// COMPACT JOB API (Memory-optimized)
// ============================================================================

impl MateRescueJobCompact {
    /// Materialize query sequence from context
    /// Returns the query with rev-comp applied if needed
    #[inline]
    pub fn get_query_seq(&self, ctx: &MateRescueBatchContext) -> Vec<u8> {
        let pair_idx = self.pair_index as usize;
        let base_seq = if self.rescuing_read1 {
            &ctx.query_seqs_r1[pair_idx]
        } else {
            &ctx.query_seqs_r2[pair_idx]
        };

        if self.is_rev {
            // Reverse complement
            base_seq
                .iter()
                .rev()
                .map(|&b| if b < 4 { 3 - b } else { 4 })
                .collect()
        } else {
            base_seq.clone()
        }
    }

    /// Fetch reference sequence from PAC
    /// Returns (ref_seq, adj_rb, adj_re, rid)
    #[inline]
    pub fn fetch_ref_seq(&self, ctx: &MateRescueBatchContext) -> (Vec<u8>, i64, i64, i32) {
        ctx.bwa_idx.bns.bns_fetch_seq(
            ctx.pac,
            self.ref_rb,
            (self.ref_rb + self.ref_re) >> 1,
            self.ref_re,
        )
    }

    /// Get the mate name from context
    #[inline]
    pub fn get_mate_name<'a>(&self, ctx: &'a MateRescueBatchContext) -> &'a str {
        let pair_idx = self.pair_index as usize;
        if self.rescuing_read1 {
            ctx.names_r1[pair_idx]
        } else {
            ctx.names_r2[pair_idx]
        }
    }

    /// Get the anchor reference name from context
    #[inline]
    pub fn get_anchor_ref_name<'a>(&self, ctx: &MateRescueBatchContext<'a>) -> &'a str {
        &ctx.bwa_idx.bns.annotations[self.anchor_ref_id as usize].name
    }
}

/// Prepare compact mate rescue jobs for a single anchor alignment.
/// Like prepare_mate_rescue_jobs_for_anchor but returns memory-efficient compact jobs.
pub fn prepare_compact_jobs_for_anchor(
    bwa_idx: &BwaIndex,
    pac: &[u8],
    stats: &[InsertSizeStats; 4],
    anchor: &Alignment,
    mate_len: usize,
    existing_alignments: &[Alignment],
    pair_index: usize,
    rescuing_read1: bool,
) -> Vec<MateRescueJobCompact> {
    let l_pac = bwa_idx.bns.packed_sequence_length as i64;
    let l_ms = mate_len as i32;
    let min_seed_len = bwa_idx.min_seed_len;
    let match_score = 1i8;

    let mut jobs = Vec::new();

    // Initialize skip array from failed orientations
    let mut skip = [false; 4];
    for r in 0..4 {
        skip[r] = stats[r].failed;
    }

    // Convert anchor position to bidirectional coordinates
    let anchor_is_rev = (anchor.flag & sam_flags::REVERSE) != 0;
    let anchor_ref_len = anchor.reference_length() as i64;
    let chr_offset = bwa_idx.bns.annotations[anchor.ref_id].offset as i64;
    let genome_pos = chr_offset + anchor.pos as i64;

    let anchor_rb = if anchor_is_rev {
        let rightmost = genome_pos + anchor_ref_len - 1;
        (l_pac << 1) - 1 - rightmost
    } else {
        genome_pos
    };

    // Check which orientations already have consistent pairs
    for aln in existing_alignments.iter() {
        if aln.ref_name == anchor.ref_name {
            let mate_is_rev = (aln.flag & sam_flags::REVERSE) != 0;
            let mate_ref_len = aln.reference_length() as i64;
            let mate_genome_pos = chr_offset + aln.pos as i64;

            let mate_rb = if mate_is_rev {
                let rightmost = mate_genome_pos + mate_ref_len - 1;
                (l_pac << 1) - 1 - rightmost
            } else {
                mate_genome_pos
            };

            let (dir, dist) = mem_infer_dir(l_pac, anchor_rb, mate_rb);
            if dist >= stats[dir].low as i64 && dist <= stats[dir].high as i64 {
                skip[dir] = true;
            }
        }
    }

    // Early exit if all orientations already have pairs
    if skip.iter().all(|&x| x) {
        return jobs;
    }

    // Try each non-skipped orientation
    for r in 0..4 {
        if skip[r] {
            continue;
        }

        let is_rev = (r >> 1) != (r & 1);
        let is_larger = (r >> 1) == 0;

        // Calculate search region (don't fetch ref yet - defer to execution)
        let (rb, re) = if !is_rev {
            let rb = if is_larger {
                anchor_rb + stats[r].low as i64
            } else {
                anchor_rb - stats[r].high as i64
            };
            let re = if is_larger {
                anchor_rb + stats[r].high as i64
            } else {
                anchor_rb - stats[r].low as i64
            } + l_ms as i64;
            (rb, re)
        } else {
            let rb = if is_larger {
                anchor_rb + stats[r].low as i64
            } else {
                anchor_rb - stats[r].high as i64
            } - l_ms as i64;
            let re = if is_larger {
                anchor_rb + stats[r].high as i64
            } else {
                anchor_rb - stats[r].low as i64
            };
            (rb, re)
        };

        // Clamp to valid range
        let rb = rb.max(0);
        let re = re.min(l_pac << 1);

        if rb >= re {
            continue;
        }

        // Pre-check: fetch to validate rid and region size
        // This is needed to avoid creating invalid jobs
        let (_ref_seq, adj_rb, adj_re, rid) =
            bwa_idx.bns.bns_fetch_seq(pac, rb, (rb + re) >> 1, re);

        if rid as usize != anchor.ref_id || (adj_re - adj_rb) < min_seed_len as i64 {
            continue;
        }

        jobs.push(MateRescueJobCompact {
            pair_index: pair_index as u32,
            rescuing_read1,
            is_rev,
            orientation: r as u8,
            anchor_ref_id: anchor.ref_id as u16,
            mate_len: l_ms as u16,
            min_seed_len: min_seed_len as i16,
            ref_rb: rb,
            ref_re: re,
            adj_rb,
            match_score,
        });
    }

    jobs
}

/// Result from a compact mate rescue execution
pub struct MateRescueResultCompact {
    /// Index of the job this result corresponds to
    pub job_index: usize,
    /// The ksw_align2 result
    pub aln: Kswr,
}

/// Execute compact mate rescue jobs using batched SIMD Smith-Waterman
///
/// This implementation uses horizontal SIMD batching (16-way for SSE/NEON)
/// to process multiple alignments in parallel, providing ~8x speedup over
/// the scalar ksw_align2 path.
///
/// # Performance
/// - Processes 16 alignments per SIMD operation (SSE/NEON)
/// - Uses SoA (Structure of Arrays) layout for SIMD efficiency
/// - Parallelizes across batches using Rayon
pub fn execute_compact_batch(
    jobs: &[MateRescueJobCompact],
    ctx: &MateRescueBatchContext,
    engine: Option<SimdEngineType>,
) -> Vec<MateRescueResultCompact> {
    if jobs.is_empty() {
        return Vec::new();
    }

    // Determine SIMD engine and batch size
    let engine = engine.unwrap_or(SimdEngineType::Engine128);
    let (simd_batch_size, _) = crate::compute::simd_abstraction::simd::get_simd_batch_sizes(engine);

    // Scoring parameters (matching BWA-MEM2 defaults)
    let _match_score: i8 = 1;
    let _mismatch_penalty: i8 = -4;
    let o_del: i8 = 6;
    let e_del: i8 = 1;
    let o_ins: i8 = 6;
    let e_ins: i8 = 1;

    // Process in chunks matching SIMD batch size
    // Use Rayon to parallelize across chunks
    let num_chunks = jobs.len().div_ceil(simd_batch_size);

    (0..num_chunks)
        .into_par_iter()
        .flat_map(|chunk_idx| {
            let chunk_start = chunk_idx * simd_batch_size;
            let chunk_end = (chunk_start + simd_batch_size).min(jobs.len());
            let chunk_jobs = &jobs[chunk_start..chunk_end];
            let chunk_size = chunk_jobs.len();

            // Materialize sequences for this chunk
            let mut ref_seqs: Vec<Vec<u8>> = Vec::with_capacity(chunk_size);
            let mut query_seqs: Vec<Vec<u8>> = Vec::with_capacity(chunk_size);

            for job in chunk_jobs {
                let query_seq = job.get_query_seq(ctx);
                let (ref_seq, _, _, _) = job.fetch_ref_seq(ctx);
                query_seqs.push(query_seq);
                ref_seqs.push(ref_seq);
            }

            let align_jobs: Vec<AlignJob> = chunk_jobs
                .iter()
                .zip(query_seqs.iter())
                .zip(ref_seqs.iter())
                .map(|((_job, query), target)| AlignJob {
                    query,
                    target,
                    qlen: query.len(),
                    tlen: target.len(),
                    band: -1,
                    h0: 0,
                })
                .collect();

            let results = with_workspace(|ws| {
                dispatch_kswv_soa(
                    &align_jobs,
                    ws,
                    simd_batch_size,
                    chunk_size,
                    5,
                    o_del,
                    e_del as i32,
                    o_ins as i32,
                    e_ins as i32,
                    0,
                    -1,
                    false,
                    engine,
                )
            });

            // Convert results to MateRescueResultCompact
            let chunk_results: Vec<MateRescueResultCompact> = (0..chunk_size)
                .map(|i| {
                    let job_index = chunk_start + i;
                    let r = &results[i];

                    MateRescueResultCompact {
                        job_index,
                        aln: Kswr {
                            score: r.score,
                            te: r.te,
                            qe: r.qe,
                            score2: r.score2,
                            te2: r.te2,
                            tb: r.tb,
                            qb: r.qb,
                        },
                    }
                })
                .collect();

            chunk_results
        })
        .collect()
}

/// Convert a compact job + SW result to an Alignment
pub fn compact_result_to_alignment(
    job: &MateRescueJobCompact,
    aln: &Kswr,
    ctx: &MateRescueBatchContext,
) -> Option<Alignment> {
    let l_pac = ctx.bwa_idx.bns.packed_sequence_length as i64;
    let l_ms = job.mate_len as i32;
    let mate_name = job.get_mate_name(ctx);

    // Check if alignment is good enough
    if aln.score < job.min_seed_len as i32 || aln.qb < 0 {
        return None;
    }

    // Convert coordinates
    let (rescued_rb, _rescued_re, query_start, query_end) = if job.is_rev {
        let rb_result = (l_pac << 1) - (job.adj_rb + aln.te as i64 + 1);
        let _re_result = (l_pac << 1) - (job.adj_rb + aln.tb as i64);
        let qb_result = l_ms - (aln.qe + 1);
        let qe_result = l_ms - aln.qb;
        (rb_result, _re_result, qb_result, qe_result)
    } else {
        let rb_result = job.adj_rb + aln.tb as i64;
        let re_result = job.adj_rb + aln.te as i64 + 1;
        (rb_result, re_result, aln.qb, aln.qe + 1)
    };

    // Convert bidirectional position to chromosome-relative
    let (pos_f, _is_rev_depos) = ctx.bwa_idx.bns.bns_depos(rescued_rb);
    let rescued_rid = ctx.bwa_idx.bns.bns_pos2rid(pos_f);

    if rescued_rid < 0 || rescued_rid as usize != job.anchor_ref_id as usize {
        return None;
    }

    let chr_pos = (pos_f - ctx.bwa_idx.bns.annotations[rescued_rid as usize].offset as i64) as u64;

    // Build CIGAR from alignment endpoints
    let ref_aligned_len = (aln.te - aln.tb + 1).max(0);
    let query_aligned = (query_end - query_start).max(0);

    let mut cigar: Vec<(u8, i32)> = Vec::new();
    if query_start > 0 {
        cigar.push((b'S', query_start));
    }
    if ref_aligned_len > 0 && query_aligned > 0 {
        if ref_aligned_len == query_aligned {
            cigar.push((b'M', ref_aligned_len));
        } else if ref_aligned_len > query_aligned {
            cigar.push((b'M', query_aligned));
            cigar.push((b'D', ref_aligned_len - query_aligned));
        } else {
            cigar.push((b'M', ref_aligned_len));
            cigar.push((b'I', query_aligned - ref_aligned_len));
        }
    }
    if query_end < l_ms {
        cigar.push((b'S', l_ms - query_end));
    }

    // Create alignment structure
    let mut flag = 0u16;
    if job.is_rev {
        flag |= sam_flags::REVERSE;
    }

    // Compute NM and MD tags
    let ref_len_for_md: i32 = cigar
        .iter()
        .filter_map(|&(op, len)| {
            if matches!(op, b'M' | b'D' | b'=' | b'X') {
                Some(len)
            } else {
                None
            }
        })
        .sum();

    // Validate alignment doesn't extend beyond reference bounds
    let ref_length = ctx.bwa_idx.bns.annotations[rescued_rid as usize].sequence_length as u64;
    if chr_pos + ref_len_for_md as u64 > ref_length {
        return None;
    }

    // Get reference sequence at chromosome position
    let forward_ref = ctx.bwa_idx.bns.get_forward_ref(
        ctx.pac,
        job.anchor_ref_id as usize,
        chr_pos,
        ref_len_for_md.max(0) as usize,
    );

    // Get aligned query portion
    let query_seq = job.get_query_seq(ctx);
    let qb_sam = query_start.max(0) as usize;
    let qe_sam = query_end.max(0) as usize;
    let aligned_query = if qe_sam <= query_seq.len() {
        &query_seq[qb_sam..qe_sam]
    } else {
        &query_seq[qb_sam..]
    };

    let (nm, md_tag) = edit_distance::compute_nm_and_md(&forward_ref, aligned_query, &cigar);

    Some(Alignment {
        query_name: mate_name.to_string(),
        flag,
        ref_name: job.get_anchor_ref_name(ctx).to_string(),
        ref_id: job.anchor_ref_id as usize,
        pos: chr_pos,
        mapq: 0,
        score: aln.score,
        cigar,
        rnext: String::from("*"),
        pnext: 0,
        tlen: 0,
        seq: String::new(),
        qual: String::new(),
        tags: vec![
            ("AS".to_string(), format!("i:{}", aln.score)),
            ("NM".to_string(), format!("i:{nm}")),
            ("MD".to_string(), format!("Z:{md_tag}")),
        ],
        query_start,
        query_end,
        seed_coverage: (ref_aligned_len.min(query_aligned) >> 1),
        hash: 0,
        frac_rep: 0.0,
    })
}

// ============================================================================
// SoA-NATIVE MATE RESCUE (Pure SoA Pipeline - Phase 3)
// ============================================================================

/// SoA-native mate rescue for paired-end reads
///
/// This function performs mate rescue entirely in SoA format, without converting
/// to Alignment structs. It:
/// 1. Collects rescue jobs from SoA alignment results
/// 2. Executes them in batches using horizontal SIMD
/// 3. Appends rescued alignments back to SoA buffers
///
/// # Arguments
/// * `soa_r1` - Mutable SoA alignment results for read 1
/// * `soa_r2` - Mutable SoA alignment results for read 2
/// * `reads_r1` - SoA read batch for read 1 (for sequences)
/// * `reads_r2` - SoA read batch for read 2 (for sequences)
/// * `primary_r1` - Indices of primary alignments for read 1
/// * `primary_r2` - Indices of primary alignments for read 2
/// * `pac` - PAC data for reference sequence fetching
/// * `bwa_idx` - BWA index for coordinate conversion
/// * `stats` - Insert size statistics for each orientation [FF, FR, RF, RR]
/// * `simd_engine` - SIMD engine type for batch execution
///
/// # Returns
/// Number of alignments rescued (total across both reads)
pub fn mate_rescue_soa(
    soa_r1: &mut crate::pipelines::linear::batch_extension::types::SoAAlignmentResult,
    soa_r2: &mut crate::pipelines::linear::batch_extension::types::SoAAlignmentResult,
    reads_r1: &crate::core::io::soa_readers::SoAReadBatch,
    reads_r2: &crate::core::io::soa_readers::SoAReadBatch,
    primary_r1: &[usize],
    primary_r2: &[usize],
    pac: &[u8],
    bwa_idx: &BwaIndex,
    stats: &[InsertSizeStats; 4],
    simd_engine: Option<SimdEngineType>,
) -> usize {
    let num_reads = soa_r1.num_reads();

    // Phase 1: Collect rescue jobs from all read pairs
    let mut jobs: Vec<MateRescueJob> = Vec::new();

    for read_idx in 0..num_reads {
        // Get primary alignments for this pair
        let r1_primary_idx = primary_r1[read_idx];
        let r2_primary_idx = primary_r2[read_idx];

        // Try to rescue R1 using R2 as anchor
        collect_rescue_jobs_for_read_soa(
            soa_r1,
            soa_r2,
            reads_r1,
            read_idx,
            r1_primary_idx,
            r2_primary_idx,
            true, // rescuing R1
            pac,
            bwa_idx,
            stats,
            &mut jobs,
        );

        // Try to rescue R2 using R1 as anchor
        collect_rescue_jobs_for_read_soa(
            soa_r2,
            soa_r1,
            reads_r2,
            read_idx,
            r2_primary_idx,
            r1_primary_idx,
            false, // rescuing R2
            pac,
            bwa_idx,
            stats,
            &mut jobs,
        );
    }

    if jobs.is_empty() {
        return 0;
    }

    log::debug!("[PE] Executing {} mate rescue alignments", jobs.len());

    // Phase 2: Execute all rescue jobs in batch
    let results = execute_mate_rescue_batch_with_engine(&mut jobs, simd_engine);

    // Phase 3: Append rescued alignments to SoA buffers
    let mut rescued_count = 0;
    for result in results {
        let job = &jobs[result.job_index];

        // Check if alignment is good enough
        if result.aln.score < job.min_seed_len || result.aln.qb < 0 {
            continue;
        }

        // Convert SW result to SoA format and append
        if append_rescue_result_to_soa(
            job,
            &result.aln,
            if job.rescuing_read1 { soa_r1 } else { soa_r2 },
            job.pair_index,
            bwa_idx,
        ) {
            rescued_count += 1;
        }
    }

    rescued_count
}

/// Collect rescue jobs for a single read using its mate as anchor
#[allow(clippy::too_many_arguments)]
fn collect_rescue_jobs_for_read_soa(
    soa_rescued: &crate::pipelines::linear::batch_extension::types::SoAAlignmentResult,
    soa_anchor: &crate::pipelines::linear::batch_extension::types::SoAAlignmentResult,
    reads_rescued: &crate::core::io::soa_readers::SoAReadBatch,
    read_idx: usize,
    rescued_primary_idx: usize,
    anchor_primary_idx: usize,
    rescuing_read1: bool,
    pac: &[u8],
    bwa_idx: &BwaIndex,
    stats: &[InsertSizeStats; 4],
    jobs: &mut Vec<MateRescueJob>,
) {
    // Check if this read needs rescue (unmapped or low MAPQ)
    let rescued_mapq = soa_rescued.mapqs[rescued_primary_idx];
    if rescued_mapq >= 20 {
        return; // Good enough alignment, no rescue needed
    }

    // Get anchor alignment info
    let anchor_flag = soa_anchor.flags[anchor_primary_idx];
    let anchor_pos = soa_anchor.positions[anchor_primary_idx];
    let anchor_ref_id = soa_anchor.ref_ids[anchor_primary_idx];
    let anchor_ref_len =
        reference_length_from_cigar_soa_rescue(soa_anchor, anchor_primary_idx) as i64;

    // Get mate sequence
    let (seq_start, seq_len) = reads_rescued.read_boundaries[read_idx];
    let mate_seq = &reads_rescued.seqs[seq_start..seq_start + seq_len];
    let mate_name = &reads_rescued.names[read_idx];

    let l_pac = bwa_idx.bns.packed_sequence_length as i64;
    let min_seed_len = bwa_idx.min_seed_len;
    let match_score = 1i32;

    // Convert anchor position to bidirectional coordinates
    let anchor_is_rev = (anchor_flag & sam_flags::REVERSE) != 0;
    let chr_offset = bwa_idx.bns.annotations[anchor_ref_id].offset as i64;
    let genome_pos = chr_offset + anchor_pos as i64;

    let anchor_rb = if anchor_is_rev {
        let rightmost = genome_pos + anchor_ref_len - 1;
        (l_pac << 1) - 1 - rightmost
    } else {
        genome_pos
    };

    // Initialize skip array from failed orientations
    let mut skip = [false; 4];
    for r in 0..4 {
        skip[r] = stats[r].failed;
    }

    // Check which orientations already have consistent pairs
    let (rescued_start, rescued_count) = soa_rescued.read_alignment_boundaries[read_idx];
    for offset in 0..rescued_count {
        let idx = rescued_start + offset;
        if soa_rescued.ref_ids[idx] == anchor_ref_id {
            let mate_is_rev = (soa_rescued.flags[idx] & sam_flags::REVERSE) != 0;
            let mate_ref_len = reference_length_from_cigar_soa_rescue(soa_rescued, idx) as i64;
            let mate_genome_pos = chr_offset + soa_rescued.positions[idx] as i64;

            let mate_rb = if mate_is_rev {
                let rightmost = mate_genome_pos + mate_ref_len - 1;
                (l_pac << 1) - 1 - rightmost
            } else {
                mate_genome_pos
            };

            let (dir, dist) = mem_infer_dir(l_pac, anchor_rb, mate_rb);
            if dist >= stats[dir].low as i64 && dist <= stats[dir].high as i64 {
                skip[dir] = true;
            }
        }
    }

    // Early exit if all orientations already have pairs
    if skip.iter().all(|&x| x) {
        return;
    }

    let l_ms = mate_seq.len() as i32;

    // Try each non-skipped orientation
    for r in 0..4 {
        if skip[r] {
            continue;
        }

        // Decode orientation bits
        let is_rev = (r >> 1) != (r & 1);
        let is_larger = (r >> 1) == 0;

        // Prepare mate sequence (reverse complement if needed)
        let query_seq: Vec<u8> = if is_rev {
            mate_seq
                .iter()
                .rev()
                .map(|&b| if b < 4 { 3 - b } else { 4 })
                .collect()
        } else {
            mate_seq.to_vec()
        };

        // Calculate search region using FULL insert size range [low, high]
        let (rb, re) = if !is_rev {
            let rb = if is_larger {
                anchor_rb + stats[r].low as i64
            } else {
                anchor_rb - stats[r].high as i64
            };
            let re = if is_larger {
                anchor_rb + stats[r].high as i64
            } else {
                anchor_rb - stats[r].low as i64
            } + l_ms as i64;
            (rb, re)
        } else {
            let rb = if is_larger {
                anchor_rb + stats[r].low as i64
            } else {
                anchor_rb - stats[r].high as i64
            } - l_ms as i64;
            let re = if is_larger {
                anchor_rb + stats[r].high as i64
            } else {
                anchor_rb - stats[r].low as i64
            };
            (rb, re)
        };

        // Clamp to valid range
        let rb = rb.max(0);
        let re = re.min(l_pac << 1);

        if rb >= re {
            continue;
        }

        // Fetch reference sequence
        let (ref_seq, adj_rb, adj_re, rid) = bwa_idx.bns.bns_fetch_seq(pac, rb, (rb + re) >> 1, re);

        // Check if on same reference and region is large enough
        if rid as usize != anchor_ref_id || (adj_re - adj_rb) < min_seed_len as i64 {
            continue;
        }

        // Create rescue job
        jobs.push(MateRescueJob {
            pair_index: read_idx,
            rescuing_read1,
            query_seq,
            ref_seq,
            adj_rb,
            is_rev,
            orientation: r,
            anchor_ref_id,
            anchor_ref_name: soa_anchor.ref_names[anchor_primary_idx].clone(),
            mate_name: mate_name.clone(),
            mate_len: l_ms,
            min_seed_len,
            match_score,
        });
    }
}

/// Append a rescued alignment result to SoA buffers
fn append_rescue_result_to_soa(
    job: &MateRescueJob,
    aln: &Kswr,
    soa: &mut crate::pipelines::linear::batch_extension::types::SoAAlignmentResult,
    read_idx: usize,
    bwa_idx: &BwaIndex,
) -> bool {
    let l_pac = bwa_idx.bns.packed_sequence_length as i64;
    let l_ms = job.mate_len;

    // Convert coordinates
    let (rescued_rb, _rescued_re, query_start, query_end) = if job.is_rev {
        let rb_result = (l_pac << 1) - (job.adj_rb + aln.te as i64 + 1);
        let re_result = (l_pac << 1) - (job.adj_rb + aln.tb as i64);
        let qb_result = l_ms - (aln.qe + 1);
        let qe_result = l_ms - aln.qb;
        (rb_result, re_result, qb_result, qe_result)
    } else {
        let rb_result = job.adj_rb + aln.tb as i64;
        let re_result = job.adj_rb + aln.te as i64 + 1;
        (rb_result, re_result, aln.qb, aln.qe + 1)
    };

    // Convert bidirectional position to chromosome-relative
    let (pos_f, _is_rev_depos) = bwa_idx.bns.bns_depos(rescued_rb);
    let rescued_rid = bwa_idx.bns.bns_pos2rid(pos_f);

    if rescued_rid < 0 || rescued_rid as usize != job.anchor_ref_id {
        return false;
    }

    let chr_pos = (pos_f - bwa_idx.bns.annotations[rescued_rid as usize].offset as i64) as u64;

    // Build CIGAR
    let ref_aligned_len = (aln.te - aln.tb + 1).max(0);
    let query_aligned = (query_end - query_start).max(0);

    let mut cigar_ops = Vec::new();
    let mut cigar_lens = Vec::new();

    if query_start > 0 {
        cigar_ops.push(b'S');
        cigar_lens.push(query_start);
    }
    if ref_aligned_len > 0 && query_aligned > 0 {
        cigar_ops.push(b'M');
        cigar_lens.push(ref_aligned_len.min(query_aligned));
    }
    if query_end < l_ms {
        cigar_ops.push(b'S');
        cigar_lens.push(l_ms - query_end);
    }

    // Set flags
    let mut flag: u16 = 0;
    if job.is_rev {
        flag |= sam_flags::REVERSE;
    }

    // Append to SoA buffers
    let _aln_idx = soa.positions.len();

    soa.query_names.push(job.mate_name.clone());
    soa.flags.push(flag);
    soa.ref_names.push(job.anchor_ref_name.clone());
    soa.ref_ids.push(job.anchor_ref_id);
    soa.positions.push(chr_pos);
    soa.mapqs.push(0); // Rescued alignments get MAPQ 0 initially
    soa.scores.push(aln.score);

    // Append CIGAR
    let cigar_start = soa.cigar_ops.len();
    soa.cigar_ops.extend_from_slice(&cigar_ops);
    soa.cigar_lens.extend_from_slice(&cigar_lens);
    soa.cigar_boundaries.push((cigar_start, cigar_ops.len()));

    // Mate info (will be filled in finalization phase)
    soa.rnexts.push("*".to_string());
    soa.pnexts.push(0);
    soa.tlens.push(0);

    // Sequence and quality (get from original read)
    // Note: For rescued alignments, we don't have the original sequence easily accessible
    // We'll use empty for now and fill in during finalization if needed
    let seq_start = soa.seqs.len();
    soa.seq_boundaries.push((seq_start, 0));

    // Tags (minimal set for rescued alignments)
    let tag_start = soa.tag_names.len();
    soa.tag_names.push("AS".to_string());
    soa.tag_values.push(format!("i:{}", aln.score));
    soa.tag_boundaries.push((tag_start, 1));

    // Internal fields
    soa.query_starts.push(query_start);
    soa.query_ends.push(query_end);
    soa.seed_coverages
        .push(ref_aligned_len.min(query_aligned) >> 1);
    soa.hashes.push(0);
    soa.frac_reps.push(0.0);

    // Update read_alignment_boundaries
    let (old_start, old_count) = soa.read_alignment_boundaries[read_idx];
    soa.read_alignment_boundaries[read_idx] = (old_start, old_count + 1);

    true
}

/// Calculate reference length from CIGAR string in SoA format (for mate rescue module)
///
/// Duplicate to avoid circular dependencies between modules.
fn reference_length_from_cigar_soa_rescue(
    soa: &crate::pipelines::linear::batch_extension::types::SoAAlignmentResult,
    aln_idx: usize,
) -> i32 {
    let (cigar_start, cigar_count) = soa.cigar_boundaries[aln_idx];
    let mut ref_len = 0i32;

    for offset in 0..cigar_count {
        let idx = cigar_start + offset;
        let op = soa.cigar_ops[idx];
        let len = soa.cigar_lens[idx];

        // M, D, N, =, X consume reference
        if op == b'M' || op == b'D' || op == b'N' || op == b'=' || op == b'X' {
            ref_len += len;
        }
    }

    ref_len
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Test CIGAR reference length calculation
    /// This is critical for bounds checking
    #[test]
    fn test_cigar_ref_length_calculation() {
        // Simple match
        let cigar = [(b'M', 100)];
        let ref_len: i32 = cigar
            .iter()
            .filter_map(|&(op, len)| {
                if matches!(op, b'M' | b'D' | b'=' | b'X') {
                    Some(len)
                } else {
                    None
                }
            })
            .sum();
        assert_eq!(ref_len, 100);

        // Match with soft clips (S doesn't consume reference)
        let cigar = [(b'S', 10), (b'M', 80), (b'S', 10)];
        let ref_len: i32 = cigar
            .iter()
            .filter_map(|&(op, len)| {
                if matches!(op, b'M' | b'D' | b'=' | b'X') {
                    Some(len)
                } else {
                    None
                }
            })
            .sum();
        assert_eq!(ref_len, 80);

        // Match with deletion (D consumes reference)
        let cigar = [(b'M', 50), (b'D', 5), (b'M', 45)];
        let ref_len: i32 = cigar
            .iter()
            .filter_map(|&(op, len)| {
                if matches!(op, b'M' | b'D' | b'=' | b'X') {
                    Some(len)
                } else {
                    None
                }
            })
            .sum();
        assert_eq!(ref_len, 100); // 50 + 5 + 45

        // Match with insertion (I doesn't consume reference)
        let cigar = [(b'M', 50), (b'I', 5), (b'M', 50)];
        let ref_len: i32 = cigar
            .iter()
            .filter_map(|&(op, len)| {
                if matches!(op, b'M' | b'D' | b'=' | b'X') {
                    Some(len)
                } else {
                    None
                }
            })
            .sum();
        assert_eq!(ref_len, 100); // 50 + 50, I doesn't count

        // Complex CIGAR with all operations
        let cigar = [
            (b'S', 10),
            (b'M', 30),
            (b'I', 2),
            (b'M', 20),
            (b'D', 3),
            (b'M', 35),
            (b'S', 5),
        ];
        let ref_len: i32 = cigar
            .iter()
            .filter_map(|&(op, len)| {
                if matches!(op, b'M' | b'D' | b'=' | b'X') {
                    Some(len)
                } else {
                    None
                }
            })
            .sum();
        assert_eq!(ref_len, 88); // 30 + 20 + 3 + 35
    }

    /// Test bounds checking logic
    /// Alignment extending past reference end should be rejected
    #[test]
    fn test_bounds_check_within_bounds() {
        let chr_pos: u64 = 1000;
        let ref_len: i32 = 100;
        let ref_length: u64 = 2000;

        // Within bounds: 1000 + 100 = 1100 <= 2000
        assert!(chr_pos + ref_len as u64 <= ref_length);
    }

    #[test]
    fn test_bounds_check_exactly_at_end() {
        let chr_pos: u64 = 1900;
        let ref_len: i32 = 100;
        let ref_length: u64 = 2000;

        // Exactly at end: 1900 + 100 = 2000 <= 2000
        assert!(chr_pos + ref_len as u64 <= ref_length);
    }

    #[test]
    fn test_bounds_check_past_end() {
        let chr_pos: u64 = 1901;
        let ref_len: i32 = 100;
        let ref_length: u64 = 2000;

        // Past end: 1901 + 100 = 2001 > 2000
        assert!(chr_pos + ref_len as u64 > ref_length);
    }

    #[test]
    fn test_bounds_check_chrY_end_case() {
        // This tests the exact case that was causing CIGAR_MAPS_OFF_REFERENCE
        // chrY length: 57,227,415
        // Problematic alignments were at position 57,227,414 with ~100bp CIGAR
        let chr_pos: u64 = 57227414;
        let ref_len: i32 = 99;
        let ref_length: u64 = 57227415;

        // 57227414 + 99 = 57227513 > 57227415 → should be rejected
        assert!(chr_pos + ref_len as u64 > ref_length);
    }

    #[test]
    fn test_bounds_check_valid_chrY_end() {
        // Valid alignment at chrY end
        let chr_pos: u64 = 57227413;
        let ref_len: i32 = 2;
        let ref_length: u64 = 57227415;

        // 57227413 + 2 = 57227415 <= 57227415 → should be accepted
        assert!(chr_pos + ref_len as u64 <= ref_length);
    }

    /// Test mem_infer_dir direction inference
    #[test]
    fn test_mem_infer_dir_ff() {
        let l_pac = 1000i64;
        // Both forward, anchor at 100, mate at 200
        let anchor_rb = 100i64;
        let mate_rb = 200i64;

        let (dir, dist) = mem_infer_dir(l_pac, anchor_rb, mate_rb);
        assert_eq!(dir, 0); // FF orientation
        assert_eq!(dist, 100); // 200 - 100
    }

    #[test]
    fn test_mem_infer_dir_fr() {
        let l_pac = 1000i64;
        // Anchor forward (100), mate reverse (l_pac + 800 = 1800)
        let anchor_rb = 100i64;
        let mate_rb = 1800i64; // In reverse region

        let (dir, dist) = mem_infer_dir(l_pac, anchor_rb, mate_rb);
        assert_eq!(dir, 1); // FR orientation
    }

    #[test]
    fn test_mem_infer_dir_rf() {
        let l_pac = 1000i64;
        // Anchor reverse, mate forward with projected mate BEHIND anchor
        // For RF (code 2): different strands AND p2 <= b1
        // p2 = (2000 - 1 - mate_rb), for p2 <= anchor_rb (1800), need mate_rb >= 199
        let anchor_rb = 1800i64; // In reverse region
        let mate_rb = 200i64; // Forward strand, p2 = 1799 < 1800

        let (dir, dist) = mem_infer_dir(l_pac, anchor_rb, mate_rb);
        assert_eq!(dir, 2); // RF orientation (different strands, mate projected behind anchor)
    }

    #[test]
    fn test_mem_infer_dir_rr() {
        let l_pac = 1000i64;
        // Both reverse
        let anchor_rb = 1800i64;
        let mate_rb = 1700i64;

        let (dir, dist) = mem_infer_dir(l_pac, anchor_rb, mate_rb);
        assert_eq!(dir, 3); // RR orientation
    }

    /// Test scoring matrix
    #[test]
    fn test_scoring_matrix() {
        // Match scores
        assert_eq!(MATE_RESCUE_SCORING_MATRIX[0], 1); // A-A
        assert_eq!(MATE_RESCUE_SCORING_MATRIX[6], 1); // C-C
        assert_eq!(MATE_RESCUE_SCORING_MATRIX[12], 1); // G-G
        assert_eq!(MATE_RESCUE_SCORING_MATRIX[18], 1); // T-T

        // Mismatch penalties
        assert_eq!(MATE_RESCUE_SCORING_MATRIX[1], -4); // A-C
        assert_eq!(MATE_RESCUE_SCORING_MATRIX[5], -4); // C-A
        assert_eq!(MATE_RESCUE_SCORING_MATRIX[7], -4); // C-G

        // N handling (neutral score)
        assert_eq!(MATE_RESCUE_SCORING_MATRIX[4], 0); // A-N
        assert_eq!(MATE_RESCUE_SCORING_MATRIX[24], 0); // N-N
    }
}
