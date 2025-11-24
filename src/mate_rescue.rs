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

use crate::alignment::finalization::Alignment;
use crate::alignment::finalization::sam_flags;
use crate::alignment::ksw_affine_gap::{ksw_align2, Kswr, KSW_XBYTE, KSW_XSTART, KSW_XSUBO};
use crate::index::BwaIndex;
use crate::insert_size::InsertSizeStats;
use crate::simd_abstraction::SimdEngine128;

/// Scoring matrix for mate rescue (5x5 for DNA: A=0, C=1, G=2, T=3, N=4)
/// Match = 1, Mismatch = -4 (standard BWA-MEM2 defaults)
const MATE_RESCUE_SCORING_MATRIX: [i8; 25] = [
    1, -4, -4, -4, 0,   // A
    -4, 1, -4, -4, 0,   // C
    -4, -4, 1, -4, 0,   // G
    -4, -4, -4, 1, 0,   // T
    0, 0, 0, 0, 0,      // N
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
            mate_name, anchor.ref_name, anchor.pos, anchor_is_rev, anchor_ref_len, anchor_rb,
            stats[1].low, stats[1].high
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
        let is_rev = (r >> 1) != (r & 1);  // Whether to reverse complement the mate
        let is_larger = (r >> 1) == 0;      // Whether mate has larger coordinate

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
                mate_name, r, is_rev, is_larger, rb, re, re - rb
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
                    mate_name, r, rid, anchor.ref_id, adj_re - adj_rb
                );
            }
            continue;
        }

        let ref_len = ref_seq.len() as i32;

        // Step 9: Perform FULL Smith-Waterman (NOT banded!) using ksw_align2 (C++ lines 217-223)
        // Build xtra flags matching BWA-MEM2
        let xtra = KSW_XSUBO | KSW_XSTART |
            if (l_ms * match_score) < 250 { KSW_XBYTE } else { 0 } |
            (min_seed_len * match_score) as u32;

        let aln: Kswr = unsafe {
            ksw_align2::<SimdEngine128>(
                l_ms,
                &mut seq,
                ref_len,
                &mut ref_seq,
                5,  // m (alphabet size)
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
                mate_name, r, aln.score, aln.qb, aln.qe, aln.tb, aln.te
            );
        }

        // Step 10: Check if alignment is good enough (C++ lines 226-227)
        if aln.score < min_seed_len || aln.qb < 0 {
            continue;
        }

        // Step 11: Convert coordinates (C++ lines 229-238)
        // This matches BWA-MEM2's exact coordinate transformation
        let (rescued_rb, rescued_re, query_start, query_end) = if is_rev {
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
                    mate_name, rescued_rid, anchor.ref_id
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
            tags: Vec::new(),
            query_start,
            query_end,
            seed_coverage: (ref_aligned_len.min(query_aligned) >> 1) as i32,
            hash: 0,
            frac_rep: 0.0,
        };

        if is_debug_read {
            log::debug!(
                "MATE_RESCUE_SW {}: RESCUED! pos={}, score={}, cigar={:?}",
                mate_name, chr_pos, aln.score, rescued_aln.cigar
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
    let p2 = if r1 == r2 {
        b2
    } else {
        (l_pac << 1) - 1 - b2
    };

    // Calculate absolute distance
    let dist = if p2 > b1 { p2 - b1 } else { b1 - p2 };

    // Calculate orientation code
    let dir = if r1 == r2 { 0 } else { 1 } ^ if p2 > b1 { 0 } else { 3 };

    (dir, dist)
}
