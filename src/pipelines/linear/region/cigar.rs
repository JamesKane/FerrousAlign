//! CIGAR regeneration from alignment boundaries.
//!
//! Implements CIGAR regeneration matching BWA-MEM2's `mem_reg2aln`
//! and `bwa_gen_cigar2` functions.

use super::super::index::index::BwaIndex;
use super::super::mem_opt::MemOpt;
use super::types::AlignmentRegion;
use crate::alignment::banded_swa::scalar::implementation::scalar_banded_swa;
use crate::alignment::edit_distance;
use crate::core::alignment::banded_swa::BandedPairWiseSW;

/// Generate CIGAR string from alignment region boundaries
///
/// This is the Rust equivalent of BWA-MEM2's `mem_reg2aln` (bwamem.cpp:1732-1805)
/// combined with `bwa_gen_cigar2` (bwa.cpp:260-347).
///
/// ## Algorithm
///
/// 1. Extract query segment: `query[qb..qe]`
/// 2. Fetch reference segment from PAC: `bns.get_reference_segment(rb, re-rb)`
/// 3. Run global Smith-Waterman to generate CIGAR
/// 4. Add soft clips for unaligned query ends
/// 5. Compute NM (edit distance) and MD tag
///
/// ## Returns
///
/// Returns `Some((cigar, nm, md_tag, sw_score))` on success, where:
/// - `cigar`: CIGAR operations as (op, length) pairs
/// - `nm`: Edit distance (NM tag value)
/// - `md_tag`: MD tag string
/// - `sw_score`: Smith-Waterman alignment score (for AS tag)
pub fn generate_cigar_from_region(
    bwa_idx: &BwaIndex,
    _pac_data: &[u8],
    query: &[u8],
    region: &AlignmentRegion,
    opt: &MemOpt,
) -> Option<(Vec<(u8, i32)>, i32, String, i32)> {
    // Validate region boundaries
    if region.rb >= region.re || region.qb >= region.qe {
        return None;
    }

    let l_pac = bwa_idx.bns.packed_sequence_length;

    // Check for boundary crossing
    if region.rb < l_pac && region.re > l_pac {
        return None;
    }

    // Extract query segment
    let is_reverse_strand = region.rb >= l_pac;
    let qb = region.qb.max(0) as usize;
    let qe = (region.qe as usize).min(query.len());
    if qb >= qe || qe > query.len() {
        return None;
    }
    let query_segment: Vec<u8> = query[qb..qe].to_vec();

    // Fetch reference segment
    let ref_len = region.re - region.rb;
    let rseq = match bwa_idx.bns.get_reference_segment(region.rb, ref_len) {
        Ok(seq) => seq,
        Err(_) => return None,
    };

    // Prepare sequences for SW (same for both strands)
    let (query_for_sw, rseq_for_sw) = (query_segment.clone(), rseq.clone());

    // Compute band width
    let w = infer_band_width(
        qe as i32 - qb as i32,
        ref_len as i32,
        region.truesc,
        opt.a,
        opt.o_del,
        opt.e_del,
        opt.o_ins,
        opt.e_ins,
        opt.w,
        region.w,
    );

    let sw_params = BandedPairWiseSW::new(
        opt.o_del,
        opt.e_del,
        opt.o_ins,
        opt.e_ins,
        opt.zdrop,
        0,
        opt.pen_clip5,
        opt.pen_clip3,
        opt.mat,
        opt.a as i8,
        -(opt.b as i8),
    );

    // Run global alignment
    let result = scalar_banded_swa(
        &sw_params,
        query_for_sw.len() as i32,
        &query_for_sw,
        rseq_for_sw.len() as i32,
        &rseq_for_sw,
        w,
        0,
    );

    // Extract score from SW result for AS tag
    let sw_score = result.0.score;
    let mut cigar = result.1;

    // If reverse strand, reverse the CIGAR
    if region.rb >= l_pac {
        cigar.reverse();
    }

    // Add soft clips
    let query_len = query.len() as i32;
    let clip5 = if region.is_rev {
        query_len - region.qe
    } else {
        region.qb
    };
    let clip3 = if region.is_rev {
        region.qb
    } else {
        query_len - region.qe
    };

    let mut final_cigar = Vec::new();

    if clip5 > 0 {
        final_cigar.push((b'S', clip5));
    }

    for (op, len) in cigar {
        if let Some((last_op, last_len)) = final_cigar.last_mut() {
            if *last_op == op {
                *last_len += len;
                continue;
            }
        }
        final_cigar.push((op, len));
    }

    if clip3 > 0 {
        if let Some((last_op, last_len)) = final_cigar.last_mut() {
            if *last_op == b'S' {
                *last_len += clip3;
            } else {
                final_cigar.push((b'S', clip3));
            }
        } else {
            final_cigar.push((b'S', clip3));
        }
    }

    // Validate CIGAR has aligned bases
    let has_aligned_bases = final_cigar
        .iter()
        .any(|&(op, _)| matches!(op, b'M' | b'=' | b'X' | b'I' | b'D'));
    if !has_aligned_bases {
        return None;
    }

    // Calculate reference length from CIGAR
    let cigar_ref_len: i32 = final_cigar
        .iter()
        .filter_map(|&(op, len)| {
            if matches!(op as char, 'M' | 'D' | '=' | 'X') {
                Some(len)
            } else {
                None
            }
        })
        .sum();

    // Bounds check - reject alignments extending past reference end
    if region.rid >= 0 && (region.rid as usize) < bwa_idx.bns.annotations.len() {
        let ref_length = bwa_idx.bns.annotations[region.rid as usize].sequence_length as u64;
        if region.chr_pos + cigar_ref_len as u64 > ref_length {
            return None;
        }
    }

    // Get aligned query length
    let aligned_query_len: i32 = final_cigar
        .iter()
        .filter_map(|&(op, len)| {
            if matches!(op, b'M' | b'I' | b'=' | b'X') {
                Some(len)
            } else {
                None
            }
        })
        .sum();

    // Sum leading clips
    let left_clip: i32 = final_cigar
        .iter()
        .take_while(|&&(op, _)| op == b'S' || op == b'H')
        .map(|&(_, len)| len)
        .sum();

    let (nm, md) = if is_reverse_strand {
        let forward_ref = bwa_idx.bns.get_forward_ref(
            &bwa_idx.bns.pac_data,
            region.rid as usize,
            region.chr_pos,
            cigar_ref_len as usize,
        );
        let query_len = query.len() as i32;
        let orig_start = (query_len - left_clip - aligned_query_len).max(0) as usize;
        let orig_end = (query_len - left_clip).max(0) as usize;
        let aligned_original = &query[orig_start..orig_end.min(query.len())];
        let query_for_md: Vec<u8> = aligned_original.iter().rev().map(|&b| 3 - b).collect();
        compute_nm_and_md_local(&final_cigar, &forward_ref, &query_for_md)
    } else {
        let forward_ref = bwa_idx.bns.get_forward_ref(
            &bwa_idx.bns.pac_data,
            region.rid as usize,
            region.chr_pos,
            cigar_ref_len as usize,
        );
        let query_start = left_clip.max(0) as usize;
        let query_end = (left_clip + aligned_query_len).max(0) as usize;
        let aligned_query = &query[query_start..query_end.min(query.len())];
        compute_nm_and_md_local(&final_cigar, &forward_ref, aligned_query)
    };

    Some((final_cigar, nm, md, sw_score))
}

/// Infer band width for global alignment
fn infer_band_width(
    l_query: i32,
    l_ref: i32,
    score: i32,
    match_score: i32,
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    cmd_w: i32,
    region_w: i32,
) -> i32 {
    let tmp_del =
        if l_query == l_ref && l_query * match_score - score < (o_del + e_del - match_score) * 2 {
            0
        } else {
            let min_len = l_query.min(l_ref);
            let w = ((min_len * match_score - score - o_del) as f64 / e_del as f64 + 2.0) as i32;
            w.max((l_query - l_ref).abs())
        };

    let tmp_ins =
        if l_query == l_ref && l_query * match_score - score < (o_ins + e_ins - match_score) * 2 {
            0
        } else {
            let min_len = l_query.min(l_ref);
            let w = ((min_len * match_score - score - o_ins) as f64 / e_ins as f64 + 2.0) as i32;
            w.max((l_query - l_ref).abs())
        };

    let mut w2 = tmp_del.max(tmp_ins);
    if w2 > cmd_w {
        w2 = w2.min(region_w);
    }

    w2.min(cmd_w * 4)
}

/// Compute NM and MD using edit_distance module
#[inline]
fn compute_nm_and_md_local(
    cigar: &[(u8, i32)],
    ref_aligned: &[u8],
    query_aligned: &[u8],
) -> (i32, String) {
    edit_distance::compute_nm_and_md(ref_aligned, query_aligned, cigar)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Calculate reference length from CIGAR
    fn calculate_cigar_ref_len(cigar: &[(u8, i32)]) -> i32 {
        cigar
            .iter()
            .filter_map(|&(op, len)| {
                if matches!(op as char, 'M' | 'D' | '=' | 'X') {
                    Some(len)
                } else {
                    None
                }
            })
            .sum()
    }

    #[test]
    fn test_cigar_ref_len_matches_only() {
        let cigar = vec![(b'M', 100)];
        assert_eq!(calculate_cigar_ref_len(&cigar), 100);
    }

    #[test]
    fn test_cigar_ref_len_with_insertions() {
        let cigar = vec![(b'M', 50), (b'I', 2), (b'M', 48)];
        assert_eq!(calculate_cigar_ref_len(&cigar), 98);
    }

    #[test]
    fn test_cigar_ref_len_with_deletions() {
        let cigar = vec![(b'M', 50), (b'D', 2), (b'M', 48)];
        assert_eq!(calculate_cigar_ref_len(&cigar), 100);
    }

    #[test]
    fn test_cigar_ref_len_with_soft_clips() {
        let cigar = vec![(b'S', 10), (b'M', 80), (b'S', 10)];
        assert_eq!(calculate_cigar_ref_len(&cigar), 80);
    }

    #[test]
    fn test_cigar_ref_len_complex() {
        let cigar = vec![
            (b'S', 5),
            (b'M', 45),
            (b'I', 2),
            (b'D', 3),
            (b'M', 47),
            (b'S', 5),
        ];
        assert_eq!(calculate_cigar_ref_len(&cigar), 95);
    }

    #[test]
    fn test_cigar_ref_len_with_eq_x_ops() {
        let cigar = vec![(b'=', 40), (b'X', 10), (b'=', 50)];
        assert_eq!(calculate_cigar_ref_len(&cigar), 100);
    }

    /// Test bounds check logic
    fn bounds_check_passes(chr_pos: u64, cigar_ref_len: i32, ref_length: u64) -> bool {
        chr_pos + cigar_ref_len as u64 <= ref_length
    }

    #[test]
    fn test_bounds_check_well_within_bounds() {
        assert!(bounds_check_passes(1000, 100, 10000));
    }

    #[test]
    fn test_bounds_check_exactly_at_end() {
        assert!(bounds_check_passes(9900, 100, 10000));
    }

    #[test]
    fn test_bounds_check_one_bp_past_end() {
        assert!(!bounds_check_passes(9901, 100, 10000));
    }

    #[test]
    fn test_bounds_check_far_past_end() {
        assert!(!bounds_check_passes(9950, 100, 10000));
    }

    #[test]
    fn test_bounds_check_chry_end_case() {
        let chry_length: u64 = 57_227_415;
        let chr_pos: u64 = 57_227_414;
        let cigar_ref_len: i32 = 100;
        assert!(!bounds_check_passes(chr_pos, cigar_ref_len, chry_length));
    }

    #[test]
    fn test_bounds_check_chry_valid_alignment() {
        let chry_length: u64 = 57_227_415;
        let chr_pos: u64 = 57_227_314;
        let cigar_ref_len: i32 = 100;
        assert!(bounds_check_passes(chr_pos, cigar_ref_len, chry_length));
    }

    #[test]
    fn test_bounds_check_at_position_zero() {
        assert!(bounds_check_passes(0, 100, 10000));
    }

    #[test]
    fn test_bounds_check_single_base_chromosome() {
        assert!(bounds_check_passes(0, 1, 1));
        assert!(!bounds_check_passes(0, 2, 1));
        assert!(!bounds_check_passes(1, 1, 1));
    }

    #[test]
    fn test_infer_band_width_identical_lengths() {
        let w = infer_band_width(100, 100, 100, 1, 6, 1, 6, 1, 100, 50);
        assert_eq!(w, 0);
    }

    #[test]
    fn test_infer_band_width_different_lengths() {
        let w = infer_band_width(100, 90, 80, 1, 6, 1, 6, 1, 100, 50);
        assert!(w >= 10);
    }

    #[test]
    fn test_infer_band_width_capped_at_4x_cmd() {
        let w = infer_band_width(1000, 500, 100, 1, 6, 1, 6, 1, 25, 200);
        assert!(w <= 100);
    }
}
