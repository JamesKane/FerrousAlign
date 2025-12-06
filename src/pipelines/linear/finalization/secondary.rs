//! Secondary/supplementary alignment marking and MAPQ calculation.
//!
//! Implements C++ mem_mark_primary_se (bwamem.cpp:1420-1464).

use super::alignment::Alignment;
use super::sam_flags;
use super::tags::attach_xa_sa_tags;
use crate::pipelines::linear::mem_opt::MemOpt;

/// Mark secondary alignments, calculate MAPQ values, and attach XA/SA tags
///
/// BWA-MEM2 behavior (bwamem.cpp:1474-1515):
/// 1. Computes hash for each alignment: hash_64(id + i)
/// 2. Sorts by: score (descending) → is_alt (non-ALT first) → hash (ascending)
/// 3. After sorting, index 0 is the best primary alignment
/// 4. Marks overlapping alignments as secondary
pub fn mark_secondary_alignments(alignments: &mut Vec<Alignment>, opt: &MemOpt) {
    if alignments.is_empty() {
        return;
    }

    // Clear any pre-existing SECONDARY/SUPPLEMENTARY flags
    for aln in alignments.iter_mut() {
        aln.flag &= !(sam_flags::SECONDARY | sam_flags::SUPPLEMENTARY);
    }

    // BWA-MEM2: Sort by score (descending) → is_alt (non-ALT first) → hash (ascending)
    // This ensures the highest-scoring non-ALT alignment is at index 0
    // See bwamem.cpp:155: alnreg_hlt(a, b) comparator
    alignments.sort_by(|a, b| {
        // First: score descending (higher scores first)
        match b.score.cmp(&a.score) {
            std::cmp::Ordering::Equal => {
                // Second: is_alt ascending (non-ALT before ALT)
                match a.is_alt.cmp(&b.is_alt) {
                    std::cmp::Ordering::Equal => {
                        // Third: hash ascending (for deterministic tie-breaking)
                        a.hash.cmp(&b.hash)
                    }
                    other => other,
                }
            }
            other => other,
        }
    });

    let score_gap_threshold = (opt.a + opt.b)
        .max(opt.o_del + opt.e_del)
        .max(opt.o_ins + opt.e_ins);

    let (primary_indices, sub_scores, sub_counts) =
        find_primary_alignments(alignments, opt.mask_level, score_gap_threshold);

    apply_supplementary_flags(alignments, &primary_indices);
    calculate_all_mapq(alignments, &sub_scores, &sub_counts, opt);

    if !primary_indices.is_empty() {
        let primary_score = alignments[primary_indices[0]].score;
        filter_supplementary_by_score(alignments, primary_score, opt.xa_drop_ratio);
    }

    attach_xa_sa_tags(alignments, opt);
}

fn find_primary_alignments(
    alignments: &mut [Alignment],
    mask_level: f32,
    score_gap_threshold: i32,
) -> (Vec<usize>, Vec<i32>, Vec<i32>) {
    let mut primary_indices: Vec<usize> = Vec::with_capacity(alignments.len().min(4));
    let mut sub_scores: Vec<i32> = vec![0; alignments.len()];
    let mut sub_counts: Vec<i32> = vec![0; alignments.len()];

    if alignments.is_empty() {
        return (primary_indices, sub_scores, sub_counts);
    }

    primary_indices.push(0);

    for i in 1..alignments.len() {
        let mut is_secondary = false;

        for &j in &primary_indices {
            if alignments_overlap(&alignments[i], &alignments[j], mask_level) {
                if sub_scores[j] == 0 {
                    sub_scores[j] = alignments[i].score;
                }
                if alignments[j].score - alignments[i].score <= score_gap_threshold {
                    sub_counts[j] += 1;
                }
                alignments[i].flag |= sam_flags::SECONDARY;
                is_secondary = true;
                break;
            }
        }

        if !is_secondary {
            primary_indices.push(i);
        }
    }

    (primary_indices, sub_scores, sub_counts)
}

fn apply_supplementary_flags(alignments: &mut [Alignment], primary_indices: &[usize]) {
    for (idx, &i) in primary_indices.iter().enumerate() {
        if idx > 0 {
            alignments[i].flag &= !sam_flags::SECONDARY;
            alignments[i].flag |= sam_flags::SUPPLEMENTARY;
        }
    }

    if !primary_indices.is_empty() {
        let primary_idx = primary_indices[0];
        alignments[primary_idx].flag &= !sam_flags::SUPPLEMENTARY;
    }
}

fn calculate_all_mapq(
    alignments: &mut [Alignment],
    sub_scores: &[i32],
    sub_counts: &[i32],
    opt: &MemOpt,
) {
    for i in 0..alignments.len() {
        if alignments[i].flag & sam_flags::SECONDARY == 0 {
            alignments[i].mapq = calculate_mapq(
                alignments[i].score,
                sub_scores[i],
                alignments[i].seed_coverage,
                sub_counts[i],
                opt.a,
                opt.b,
                alignments[i].frac_rep,
                opt,
            );
            alignments[i]
                .tags
                .push(("XS".to_string(), format!("i:{}", sub_scores[i])));
        } else {
            alignments[i].mapq = 0;
        }
    }
}

fn filter_supplementary_by_score(
    alignments: &mut Vec<Alignment>,
    primary_score: i32,
    xa_drop_ratio: f32,
) {
    let supp_threshold = (primary_score as f32 * xa_drop_ratio) as i32;
    let mut to_remove: Vec<usize> = Vec::with_capacity(alignments.len().min(4));

    for i in 0..alignments.len() {
        if (alignments[i].flag & sam_flags::SUPPLEMENTARY) != 0
            && alignments[i].score < supp_threshold
        {
            to_remove.push(i);
        }
    }

    for &i in to_remove.iter().rev() {
        alignments.remove(i);
    }
}

/// Check if two alignments overlap significantly on the query sequence
fn alignments_overlap(a1: &Alignment, a2: &Alignment, mask_level: f32) -> bool {
    let (a1_qb, a1_qe) = (a1.query_start, a1.query_end);
    let (a2_qb, a2_qe) = (a2.query_start, a2.query_end);

    let b_max = a1_qb.max(a2_qb);
    let e_min = a1_qe.min(a2_qe);

    if e_min <= b_max {
        return false;
    }

    let overlap = e_min - b_max;
    let min_len = (a1_qe - a1_qb).min(a2_qe - a2_qb);
    let threshold = (min_len as f32 * mask_level) as i32;
    overlap >= threshold
}

/// Calculate MAPQ (mapping quality)
fn calculate_mapq(
    score: i32,
    sub_score: i32,
    seed_coverage: i32,
    sub_count: i32,
    match_score: i32,
    mismatch_penalty: i32,
    frac_rep: f32,
    opt: &MemOpt,
) -> u8 {
    const MEM_MAPQ_COEF: f64 = 30.0;
    const MEM_MAPQ_MAX: u8 = 60;

    let sub = if sub_score > 0 {
        sub_score
    } else {
        opt.min_seed_len * match_score
    };

    if sub >= score {
        return 0;
    }

    let l = seed_coverage;
    let identity = 1.0
        - ((l * match_score - score) as f64)
            / ((match_score + mismatch_penalty) as f64)
            / (l as f64);

    if score == 0 {
        return 0;
    }

    let mut mapq: i32;

    if opt.mapq_coef_len > 0.0 {
        let mut tmp_val: f64;
        if (l as f64) < opt.mapq_coef_len as f64 {
            tmp_val = 1.0;
        } else {
            tmp_val = opt.mapq_coef_fac as f64 / (l as f64).ln();
        }
        tmp_val *= identity * identity;
        mapq =
            (6.02 * (score - sub) as f64 / match_score as f64 * tmp_val * tmp_val + 0.499) as i32;
    } else {
        mapq = (MEM_MAPQ_COEF * (1.0 - (sub as f64) / (score as f64)) * (seed_coverage as f64).ln()
            + 0.499) as i32;
        if identity < 0.95 {
            mapq = (mapq as f64 * identity * identity + 0.499) as i32;
        }
    }

    if sub_count > 0 {
        mapq -= (((sub_count + 1) as f64).ln() * 4.343) as i32;
    }

    mapq = (mapq as f64 * (1.0 - frac_rep as f64) + 0.499) as i32;

    if mapq > MEM_MAPQ_MAX as i32 {
        mapq = MEM_MAPQ_MAX as i32;
    }
    if mapq < 0 {
        mapq = 0;
    }

    mapq as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_alignment(
        ref_id: usize,
        pos: u64,
        score: i32,
        cigar: Vec<(u8, i32)>,
        query_start: i32,
        query_end: i32,
        flag: u16,
    ) -> Alignment {
        Alignment {
            query_name: "read1".to_string(),
            flag,
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
        opt.mask_level = 0.5;
        opt.xa_drop_ratio = 0.8;
        opt
    }

    #[test]
    fn test_mark_secondary_empty() {
        let opt = default_test_opt();
        let mut alignments: Vec<Alignment> = vec![];
        mark_secondary_alignments(&mut alignments, &opt);
        assert!(alignments.is_empty());
    }

    #[test]
    fn test_mark_secondary_single() {
        let opt = default_test_opt();
        let mut alignments = vec![make_test_alignment(
            0,
            1000,
            100,
            vec![(b'M', 100)],
            0,
            100,
            0,
        )];
        mark_secondary_alignments(&mut alignments, &opt);
        assert_eq!(alignments.len(), 1);
        assert_eq!(alignments[0].flag & sam_flags::SECONDARY, 0);
        assert_eq!(alignments[0].flag & sam_flags::SUPPLEMENTARY, 0);
    }

    #[test]
    fn test_mark_secondary_overlapping() {
        let mut opt = default_test_opt();
        opt.mask_level = 0.5;

        let mut alignments = vec![
            make_test_alignment(0, 1000, 100, vec![(b'M', 80)], 0, 80, 0),
            make_test_alignment(0, 2000, 70, vec![(b'M', 80)], 10, 90, 0),
        ];
        mark_secondary_alignments(&mut alignments, &opt);

        assert_eq!(alignments.len(), 2);
        assert_eq!(alignments[0].flag & sam_flags::SECONDARY, 0);
        assert_ne!(alignments[1].flag & sam_flags::SECONDARY, 0);
    }

    #[test]
    fn test_mark_secondary_non_overlapping_supplementary() {
        let mut opt = default_test_opt();
        opt.mask_level = 0.5;

        let mut alignments = vec![
            make_test_alignment(0, 1000, 100, vec![(b'M', 50)], 0, 50, 0),
            make_test_alignment(1, 5000, 80, vec![(b'M', 40)], 60, 100, 0),
        ];
        mark_secondary_alignments(&mut alignments, &opt);

        assert_eq!(alignments.len(), 2);
        assert_eq!(alignments[0].flag & sam_flags::SECONDARY, 0);
        assert_eq!(alignments[1].flag & sam_flags::SECONDARY, 0);
        assert_ne!(alignments[1].flag & sam_flags::SUPPLEMENTARY, 0);
    }
}
