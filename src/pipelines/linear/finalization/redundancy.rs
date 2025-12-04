//! Redundant alignment removal.
//!
//! Implements C++ mem_sort_dedup_patch (bwamem.cpp:292-351).

use super::alignment::Alignment;
use super::sam_flags;
use crate::pipelines::linear::mem_opt::MemOpt;

/// Remove redundant alignments that overlap significantly on both reference and query
pub fn remove_redundant_alignments(alignments: &mut Vec<Alignment>, opt: &MemOpt) {
    if alignments.len() <= 1 {
        return;
    }

    let mask_level_redun = opt.mask_level_redun;
    let max_chain_gap = opt.max_chain_gap as u64;

    // Sort by reference end position
    alignments.sort_by(|a, b| {
        let a_ref_end = a.pos + alignment_ref_length(a);
        let b_ref_end = b.pos + alignment_ref_length(b);
        a.ref_id.cmp(&b.ref_id).then_with(|| a_ref_end.cmp(&b_ref_end))
    });

    let mut keep = vec![true; alignments.len()];

    for i in 1..alignments.len() {
        if !keep[i] { continue; }

        let p = &alignments[i];
        let p_ref_start = p.pos;
        let p_ref_end = p.pos + alignment_ref_length(p);

        for j in (0..i).rev() {
            if !keep[j] { continue; }

            let q = &alignments[j];
            if q.ref_id != p.ref_id { break; }

            let q_ref_end = q.pos + alignment_ref_length(q);
            if p_ref_start >= q_ref_end + max_chain_gap { break; }

            let q_ref_start = q.pos;

            let ref_overlap = if p_ref_start < q_ref_end && q_ref_start < p_ref_end {
                (p_ref_end.min(q_ref_end) - p_ref_start.max(q_ref_start)) as i64
            } else { 0 };

            let (p_qb, p_qe) = (p.query_start, p.query_end);
            let (q_qb, q_qe) = (q.query_start, q.query_end);

            let query_overlap = if p_qb < q_qe && q_qb < p_qe {
                p_qe.min(q_qe) - p_qb.max(q_qb)
            } else { 0 };

            let min_ref_len = ((p_ref_end - p_ref_start) as i64).min((q_ref_end - q_ref_start) as i64);
            let min_query_len = (p_qe - p_qb).min(q_qe - q_qb);

            let ref_redundant = ref_overlap as f32 > mask_level_redun * min_ref_len as f32;
            let query_redundant = query_overlap as f32 > mask_level_redun * min_query_len as f32;

            let same_position_duplicate = check_same_position_duplicate(
                p, q, p_ref_start, p_ref_end, q_ref_start, q_ref_end, query_overlap, min_query_len
            );

            let same_region_opposite_strand = check_same_region_opposite_strand(
                p, q, p_ref_start, p_ref_end, q_ref_start, q_ref_end,
                p_qb, p_qe, q_qb, q_qe
            );

            if (ref_redundant && query_redundant) || same_region_opposite_strand || same_position_duplicate {
                if p.score < q.score {
                    keep[i] = false;
                    break;
                } else {
                    keep[j] = false;
                }
            }
        }
    }

    // Remove marked alignments
    let mut write_idx = 0;
    for read_idx in 0..alignments.len() {
        if keep[read_idx] {
            if write_idx != read_idx {
                alignments.swap(write_idx, read_idx);
            }
            write_idx += 1;
        }
    }
    alignments.truncate(write_idx);

    // Sort by score for subsequent processing
    // Match BWA-MEM2's alnreg_slt comparator (bwamem.cpp:290):
    // #define alnreg_slt(a, b) ((a).score > (b).score ||
    //     ((a).score == (b).score && ((a).rb < (b).rb || ((a).rb == (b).rb && (a).qb < (b).qb))))
    // 1. Primary: Score (descending - higher is better)
    // 2. Tie-break 1: Reference position (ascending - lower position wins)
    // 3. Tie-break 2: Query start position (ascending - lower position wins)
    alignments.sort_by(|a, b| {
        b.score.cmp(&a.score)
            .then_with(|| a.pos.cmp(&b.pos))          // rb - reference start position
            .then_with(|| a.query_start.cmp(&b.query_start))  // qb - query start position
    });

    // Remove exact duplicates
    alignments.dedup_by(|a, b| {
        a.score == b.score && a.pos == b.pos && a.ref_id == b.ref_id
            && a.query_start == b.query_start && a.query_end == b.query_end
    });
}

fn check_same_position_duplicate(
    p: &Alignment, q: &Alignment,
    p_ref_start: u64, p_ref_end: u64,
    q_ref_start: u64, q_ref_end: u64,
    query_overlap: i32, min_query_len: i32,
) -> bool {
    let p_is_reverse = (p.flag & sam_flags::REVERSE) != 0;
    let q_is_reverse = (q.flag & sam_flags::REVERSE) != 0;
    let same_strand = p_is_reverse == q_is_reverse;
    let same_pos = p.pos == q.pos;
    let p_ref_len = (p_ref_end - p_ref_start) as i32;
    let q_ref_len = (q_ref_end - q_ref_start) as i32;
    let same_ref_len = (p_ref_len - q_ref_len).abs() <= 2;
    let partial_query_overlap = query_overlap < (min_query_len / 2);
    same_strand && same_pos && same_ref_len && partial_query_overlap
}

fn check_same_region_opposite_strand(
    p: &Alignment, q: &Alignment,
    p_ref_start: u64, p_ref_end: u64,
    q_ref_start: u64, q_ref_end: u64,
    p_qb: i32, p_qe: i32, q_qb: i32, q_qe: i32,
) -> bool {
    let p_is_reverse = (p.flag & sam_flags::REVERSE) != 0;
    let q_is_reverse = (q.flag & sam_flags::REVERSE) != 0;
    if p_is_reverse == q_is_reverse { return false; }

    let max_ref_len = (p_ref_end - p_ref_start).max(q_ref_end - q_ref_start);
    let ref_distance = if p_ref_start > q_ref_end {
        p_ref_start - q_ref_end
    } else {
        q_ref_start.saturating_sub(p_ref_end)
    };
    let regions_close = ref_distance <= max_ref_len;

    let score_ratio = (p.score.min(q.score) as f32) / (p.score.max(q.score).max(1) as f32);
    let scores_similar = score_ratio >= 0.8;

    let query_len = (p_qe.max(q_qe)).max(148);
    let (p_fwd_start, p_fwd_end) = if p_is_reverse {
        (query_len - p_qe, query_len - p_qb)
    } else { (p_qb, p_qe) };
    let (q_fwd_start, q_fwd_end) = if q_is_reverse {
        (query_len - q_qe, query_len - q_qb)
    } else { (q_qb, q_qe) };

    let fwd_b_max = p_fwd_start.max(q_fwd_start);
    let fwd_e_min = p_fwd_end.min(q_fwd_end);
    let fwd_overlap = if fwd_e_min > fwd_b_max { fwd_e_min - fwd_b_max } else { 0 };
    let fwd_min_len = (p_fwd_end - p_fwd_start).min(q_fwd_end - q_fwd_start);
    let query_regions_same = fwd_overlap > 0 && fwd_overlap >= (fwd_min_len as f32 * 0.8) as i32;

    regions_close && scores_similar && query_regions_same
}

/// Calculate approximate reference length from CIGAR
pub fn alignment_ref_length(alignment: &Alignment) -> u64 {
    let mut ref_len = 0u64;
    for &(op, len) in &alignment.cigar {
        match op {
            b'M' | b'D' | b'N' | b'=' | b'X' => ref_len += len as u64,
            _ => {}
        }
    }
    ref_len
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_alignment(
        ref_id: usize, pos: u64, score: i32, cigar: Vec<(u8, i32)>,
        query_start: i32, query_end: i32,
    ) -> Alignment {
        Alignment {
            query_name: "read1".to_string(),
            flag: 0,
            ref_name: format!("chr{}", ref_id),
            ref_id,
            pos,
            mapq: 60,
            score,
            cigar,
            rnext: "*".to_string(),
            pnext: 0,
            tlen: 0,
            seq: "A".repeat((query_end - query_start) as usize),
            qual: "I".repeat((query_end - query_start) as usize),
            tags: vec![],
            query_start,
            query_end,
            seed_coverage: score,
            hash: (pos * 1000 + score as u64),
            frac_rep: 0.0,
            is_alt: false,
        }
    }

    fn default_test_opt() -> MemOpt {
        let mut opt = MemOpt::default();
        opt.mask_level_redun = 0.95;
        opt.max_chain_gap = 10000;
        opt
    }

    #[test]
    fn test_remove_redundant_empty() {
        let opt = default_test_opt();
        let mut alignments: Vec<Alignment> = vec![];
        remove_redundant_alignments(&mut alignments, &opt);
        assert!(alignments.is_empty());
    }

    #[test]
    fn test_remove_redundant_single() {
        let opt = default_test_opt();
        let mut alignments = vec![make_test_alignment(0, 1000, 100, vec![(b'M', 100)], 0, 100)];
        remove_redundant_alignments(&mut alignments, &opt);
        assert_eq!(alignments.len(), 1);
    }

    #[test]
    fn test_remove_redundant_no_overlap() {
        let opt = default_test_opt();
        let mut alignments = vec![
            make_test_alignment(0, 1000, 100, vec![(b'M', 50)], 0, 50),
            make_test_alignment(0, 5000, 80, vec![(b'M', 50)], 50, 100),
        ];
        remove_redundant_alignments(&mut alignments, &opt);
        assert_eq!(alignments.len(), 2);
    }

    #[test]
    fn test_alignment_ref_length_simple() {
        let alignment = make_test_alignment(0, 1000, 100, vec![(b'M', 100)], 0, 100);
        assert_eq!(alignment_ref_length(&alignment), 100);
    }

    #[test]
    fn test_alignment_ref_length_with_insertions() {
        let alignment = make_test_alignment(0, 1000, 100, vec![(b'M', 50), (b'I', 5), (b'M', 45)], 0, 100);
        assert_eq!(alignment_ref_length(&alignment), 95);
    }

    #[test]
    fn test_alignment_ref_length_with_deletions() {
        let alignment = make_test_alignment(0, 1000, 100, vec![(b'M', 50), (b'D', 10), (b'M', 40)], 0, 90);
        assert_eq!(alignment_ref_length(&alignment), 100);
    }
}
