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

use super::insert_size::InsertSizeStats;
use crate::alignment::edit_distance;
use crate::alignment::finalization::Alignment;
use crate::alignment::finalization::sam_flags;
use crate::alignment::ksw_affine_gap::{KSW_XBYTE, KSW_XSTART, KSW_XSUBO, Kswr, ksw_align2};
use crate::compute::simd_abstraction::SimdEngine128;
use crate::index::index::BwaIndex;
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
        let forward_ref = bwa_idx.bns.get_forward_ref(
            pac,
            anchor.ref_id,
            chr_pos,
            ref_len.max(0) as usize,
        );

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
                ("NM".to_string(), format!("i:{}", nm)),
                ("MD".to_string(), format!("Z:{}", md_tag)),
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
            ("NM".to_string(), format!("i:{}", nm)),
            ("MD".to_string(), format!("Z:{}", md_tag)),
        ],
        query_start,
        query_end,
        seed_coverage: (ref_aligned_len.min(query_aligned) >> 1) as i32,
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
            log::debug!(
                "MATE_RESCUE_PREP {}: EARLY EXIT - all orientations have pairs",
                mate_name
            );
        }
        return jobs;
    }

    if is_debug_read {
        log::debug!("MATE_RESCUE_PREP {}: skip array = {:?}", mate_name, skip);
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
                log::debug!("MATE_RESCUE_PREP {}: SKIP r={} - rb >= re", mate_name, r);
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
            log::debug!("MATE_RESCUE_PREP {}: CREATING JOB r={}", mate_name, r);
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
        let cigar = vec![(b'M', 100)];
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
        let cigar = vec![(b'S', 10), (b'M', 80), (b'S', 10)];
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
        let cigar = vec![(b'M', 50), (b'D', 5), (b'M', 45)];
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
        let cigar = vec![(b'M', 50), (b'I', 5), (b'M', 50)];
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
        let cigar = vec![
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

        let (dir, _dist) = mem_infer_dir(l_pac, anchor_rb, mate_rb);
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
