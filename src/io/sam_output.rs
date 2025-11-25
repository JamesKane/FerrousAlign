// SAM output module
//
// Centralizes SAM formatting, flag management, and output logic.
// This module provides a clean separation between alignment computation
// and I/O concerns.

use crate::alignment::finalization::{Alignment, sam_flags};
use crate::alignment::mem_opt::MemOpt;
use std::io::Write;

// ============================================================================
// OUTPUT SELECTION: Which alignments should be written?
// ============================================================================

/// Result of selecting alignments for output from a single read
pub struct SelectedAlignments {
    /// Indices of alignments to output
    pub output_indices: Vec<usize>,
    /// Index of the primary (best) alignment
    pub primary_idx: usize,
    /// Whether the read should be output as unmapped
    pub output_as_unmapped: bool,
}

/// Select which alignments to output for a single-end read
///
/// Implements bwa-mem2 output logic:
/// - If best alignment score < threshold: output unmapped record
/// - Otherwise: output primary, supplementary, and optionally secondary alignments
pub fn select_single_end_alignments(alignments: &[Alignment], opt: &MemOpt) -> SelectedAlignments {
    if alignments.is_empty() {
        return SelectedAlignments {
            output_indices: vec![],
            primary_idx: 0,
            output_as_unmapped: true,
        };
    }

    // Find the primary alignment
    // Priority: The alignment NOT marked as SECONDARY (already determined by mark_secondary_alignments)
    // If all are SECONDARY (shouldn't happen), fall back to highest score
    let primary_idx = alignments
        .iter()
        .enumerate()
        .find(|(_, aln)| aln.flag & sam_flags::SECONDARY == 0)
        .map(|(idx, _)| idx)
        .unwrap_or_else(|| {
            // Fallback: highest scoring alignment
            alignments
                .iter()
                .enumerate()
                .max_by_key(|(_, aln)| aln.score)
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        });

    let best_alignment = &alignments[primary_idx];
    let best_is_unmapped = best_alignment.flag & sam_flags::UNMAPPED != 0;

    // Check if best alignment score is below threshold
    if !best_is_unmapped && best_alignment.score < opt.t {
        return SelectedAlignments {
            output_indices: vec![],
            primary_idx,
            output_as_unmapped: true,
        };
    }

    // Select alignments for output
    let mut output_indices = Vec::new();
    log::debug!(
        "select_single_end_alignments: {} alignments, primary_idx={}, output_all={}",
        alignments.len(),
        primary_idx,
        opt.output_all_alignments
    );
    for (idx, alignment) in alignments.iter().enumerate() {
        let is_unmapped = alignment.flag & sam_flags::UNMAPPED != 0;
        let is_secondary = alignment.flag & sam_flags::SECONDARY != 0;
        let is_supplementary = alignment.flag & sam_flags::SUPPLEMENTARY != 0;
        let is_best = idx == primary_idx;
        log::debug!(
            "  align[{}]: flag={}, score={}, is_secondary={}, is_supp={}, is_best={}",
            idx,
            alignment.flag,
            alignment.score,
            is_secondary,
            is_supplementary,
            is_best
        );

        let should_output = if opt.output_all_alignments {
            // -a flag: output all alignments meeting score threshold
            is_unmapped || alignment.score >= opt.t
        } else {
            // Default: output primary and supplementary, not secondary
            is_unmapped || (!is_secondary && (is_best || is_supplementary))
        };

        if should_output {
            output_indices.push(idx);
        }
    }

    SelectedAlignments {
        output_indices,
        primary_idx,
        output_as_unmapped: false,
    }
}

/// Select which alignments to output for paired-end reads
///
/// Returns (selected1, selected2, is_properly_paired)
pub fn select_paired_end_alignments(
    alignments1: &[Alignment],
    alignments2: &[Alignment],
    best_idx1: usize,
    best_idx2: usize,
    _is_properly_paired: bool,
    opt: &MemOpt,
) -> (Vec<usize>, Vec<usize>) {
    let select_for_read = |alignments: &[Alignment], best_idx: usize| -> Vec<usize> {
        let mut output_indices = Vec::new();
        for (idx, alignment) in alignments.iter().enumerate() {
            let is_unmapped = alignment.flag & sam_flags::UNMAPPED != 0;
            let is_primary = idx == best_idx;
            let is_supplementary = alignment.flag & sam_flags::SUPPLEMENTARY != 0;

            let should_output = if opt.output_all_alignments {
                is_unmapped || alignment.score >= opt.t
            } else {
                is_primary || is_supplementary
            };

            if should_output {
                output_indices.push(idx);
            }
        }
        output_indices
    };

    (
        select_for_read(alignments1, best_idx1),
        select_for_read(alignments2, best_idx2),
    )
}

// ============================================================================
// FLAG MANAGEMENT: Set SAM flags correctly
// ============================================================================

/// Prepare a single-end alignment for output
///
/// Sets correct flags and clears inappropriate ones for the alignment type.
pub fn prepare_single_end_alignment(
    alignment: &mut Alignment,
    is_primary: bool,
    rg_id: Option<&str>,
) {
    let is_unmapped = alignment.flag & sam_flags::UNMAPPED != 0;

    // Ensure the primary alignment doesn't have SECONDARY or SUPPLEMENTARY flags
    if is_primary && !is_unmapped {
        alignment.flag &= !sam_flags::SECONDARY;
        alignment.flag &= !sam_flags::SUPPLEMENTARY;
    }

    // Add RG tag if read group is specified
    if let Some(rg) = rg_id {
        alignment.tags.push(("RG".to_string(), format!("Z:{}", rg)));
    }
}

/// Paired-end flag context for setting flags correctly
pub struct PairedFlagContext {
    pub mate_ref: String,
    pub mate_pos: u64,
    pub mate_flag: u16,
    pub mate_cigar: String,
    pub mate_ref_len: i32,
    pub is_properly_paired: bool,
}

/// Prepare a paired-end alignment for output (read1)
///
/// Sets all paired-end flags, mate information, and TLEN.
pub fn prepare_paired_alignment_read1(
    alignment: &mut Alignment,
    is_primary: bool,
    ctx: &PairedFlagContext,
    rg_id: Option<&str>,
) {
    let is_unmapped = alignment.flag & sam_flags::UNMAPPED != 0;
    let is_supplementary = alignment.flag & sam_flags::SUPPLEMENTARY != 0;

    // ALWAYS set paired flag and first-in-pair flag
    alignment.flag |= sam_flags::PAIRED;
    alignment.flag |= sam_flags::FIRST_IN_PAIR;

    // Set proper pair flag only for mapped reads in proper pairs
    if !is_unmapped && ctx.is_properly_paired && is_primary {
        alignment.flag |= sam_flags::PROPER_PAIR;
    }

    // Determine if mate is unmapped (check both ref_name and flag)
    let mate_is_unmapped =
        ctx.mate_ref == "*" || (ctx.mate_flag & sam_flags::UNMAPPED) != 0;

    // Set mate unmapped flag if mate is unmapped
    if mate_is_unmapped {
        alignment.flag |= sam_flags::MATE_UNMAPPED;
    }

    // Set mate reverse flag if mate is reverse-complemented (and mapped)
    if !mate_is_unmapped && (ctx.mate_flag & sam_flags::REVERSE) != 0 {
        alignment.flag |= sam_flags::MATE_REVERSE;
    }

    // Set mate position info - needed for BOTH mapped and unmapped reads
    // SAM spec: unmapped reads with mapped mates should have RNAME/POS/RNEXT/PNEXT set
    if !mate_is_unmapped {
        // Mate is mapped - set position info
        if is_unmapped {
            // This read is unmapped but mate is mapped:
            // SAM spec says RNAME/POS should be same as mate's position
            alignment.ref_name = ctx.mate_ref.clone();
            alignment.pos = ctx.mate_pos;
            alignment.rnext = "=".to_string();
            alignment.pnext = ctx.mate_pos + 1;
            // Unmapped reads don't have TLEN
        } else {
            // Both reads are mapped
            alignment.rnext = if alignment.ref_name == ctx.mate_ref {
                "=".to_string()
            } else {
                ctx.mate_ref.clone()
            };
            alignment.pnext = ctx.mate_pos + 1;

            // TLEN only for reads on same reference
            if alignment.ref_name == ctx.mate_ref {
                alignment.tlen = alignment.calculate_tlen(ctx.mate_pos, ctx.mate_ref_len);
            }
        }
    } else if is_unmapped {
        // Both reads are unmapped
        alignment.ref_name = "*".to_string();
        alignment.pos = 0;
        alignment.rnext = "*".to_string();
        alignment.pnext = 0;
    } else {
        // This read is mapped but mate is unmapped
        // Unmapped mate will be placed at our position, so point RNEXT/PNEXT to ourselves
        alignment.rnext = "=".to_string();
        alignment.pnext = alignment.pos + 1; // Our own 1-based position
    }

    // Clear or set secondary/supplementary flags based on primary status
    if is_primary {
        alignment.flag &= !sam_flags::SECONDARY;
        alignment.flag &= !sam_flags::SUPPLEMENTARY;
    } else if !is_unmapped && !is_supplementary {
        alignment.flag |= sam_flags::SECONDARY;
    }

    // Add tags
    if let Some(rg) = rg_id {
        alignment.tags.push(("RG".to_string(), format!("Z:{}", rg)));
    }

    // MC tag only for mapped mates
    if !mate_is_unmapped && !ctx.mate_cigar.is_empty() && ctx.mate_cigar != "*" {
        alignment
            .tags
            .push(("MC".to_string(), format!("Z:{}", ctx.mate_cigar)));
    }
}

/// Prepare a paired-end alignment for output (read2)
///
/// Same as read1 but sets SECOND_IN_PAIR instead of FIRST_IN_PAIR.
pub fn prepare_paired_alignment_read2(
    alignment: &mut Alignment,
    is_primary: bool,
    ctx: &PairedFlagContext,
    rg_id: Option<&str>,
) {
    let is_unmapped = alignment.flag & sam_flags::UNMAPPED != 0;
    let is_supplementary = alignment.flag & sam_flags::SUPPLEMENTARY != 0;

    // ALWAYS set paired flag and second-in-pair flag
    alignment.flag |= sam_flags::PAIRED;
    alignment.flag |= sam_flags::SECOND_IN_PAIR;

    // Set proper pair flag only for mapped reads in proper pairs
    if !is_unmapped && ctx.is_properly_paired && is_primary {
        alignment.flag |= sam_flags::PROPER_PAIR;
    }

    // Determine if mate is unmapped (check both ref_name and flag)
    let mate_is_unmapped =
        ctx.mate_ref == "*" || (ctx.mate_flag & sam_flags::UNMAPPED) != 0;

    // Set mate unmapped flag if mate is unmapped
    if mate_is_unmapped {
        alignment.flag |= sam_flags::MATE_UNMAPPED;
    }

    // Set mate reverse flag if mate is reverse-complemented (and mapped)
    if !mate_is_unmapped && (ctx.mate_flag & sam_flags::REVERSE) != 0 {
        alignment.flag |= sam_flags::MATE_REVERSE;
    }

    // Set mate position info - needed for BOTH mapped and unmapped reads
    // SAM spec: unmapped reads with mapped mates should have RNAME/POS/RNEXT/PNEXT set
    if !mate_is_unmapped {
        // Mate is mapped - set position info
        if is_unmapped {
            // This read is unmapped but mate is mapped:
            // SAM spec says RNAME/POS should be same as mate's position
            alignment.ref_name = ctx.mate_ref.clone();
            alignment.pos = ctx.mate_pos;
            alignment.rnext = "=".to_string();
            alignment.pnext = ctx.mate_pos + 1;
            // Unmapped reads don't have TLEN
        } else {
            // Both reads are mapped
            alignment.rnext = if alignment.ref_name == ctx.mate_ref {
                "=".to_string()
            } else {
                ctx.mate_ref.clone()
            };
            alignment.pnext = ctx.mate_pos + 1;

            // TLEN only for reads on same reference
            if alignment.ref_name == ctx.mate_ref {
                alignment.tlen = alignment.calculate_tlen(ctx.mate_pos, ctx.mate_ref_len);
            }
        }
    } else if is_unmapped {
        // Both reads are unmapped
        alignment.ref_name = "*".to_string();
        alignment.pos = 0;
        alignment.rnext = "*".to_string();
        alignment.pnext = 0;
    } else {
        // This read is mapped but mate is unmapped
        // Unmapped mate will be placed at our position, so point RNEXT/PNEXT to ourselves
        alignment.rnext = "=".to_string();
        alignment.pnext = alignment.pos + 1; // Our own 1-based position
    }

    // Clear or set secondary/supplementary flags based on primary status
    if is_primary {
        alignment.flag &= !sam_flags::SECONDARY;
        alignment.flag &= !sam_flags::SUPPLEMENTARY;
    } else if !is_unmapped && !is_supplementary {
        alignment.flag |= sam_flags::SECONDARY;
    }

    // Add tags
    if let Some(rg) = rg_id {
        alignment.tags.push(("RG".to_string(), format!("Z:{}", rg)));
    }

    // MC tag only for mapped mates
    if !mate_is_unmapped && !ctx.mate_cigar.is_empty() && ctx.mate_cigar != "*" {
        alignment
            .tags
            .push(("MC".to_string(), format!("Z:{}", ctx.mate_cigar)));
    }
}

// ============================================================================
// UNMAPPED RECORD CREATION
// ============================================================================

/// Create an unmapped alignment record for single-end reads
pub fn create_unmapped_single_end(query_name: &str, seq_len: usize) -> Alignment {
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
        query_end: seq_len as i32,
        seed_coverage: 0,
        hash: 0,
        frac_rep: 0.0,
    }
}

/// Create an unmapped alignment record for paired-end reads
pub fn create_unmapped_paired(query_name: &str, seq: &[u8], is_first_in_pair: bool) -> Alignment {
    Alignment {
        query_name: query_name.to_string(),
        flag: sam_flags::UNMAPPED
            | sam_flags::PAIRED
            | if is_first_in_pair {
                sam_flags::FIRST_IN_PAIR
            } else {
                sam_flags::SECOND_IN_PAIR
            },
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
        tags: Vec::new(),
        query_start: 0,
        query_end: seq.len() as i32,
        seed_coverage: 0,
        hash: 0,
        frac_rep: 0.0,
    }
}

// ============================================================================
// SAM RECORD WRITING
// ============================================================================

/// Write a single SAM record to the output
#[inline]
pub fn write_sam_record<W: Write>(
    writer: &mut W,
    alignment: &Alignment,
    seq: &str,
    qual: &str,
) -> std::io::Result<()> {
    writeln!(writer, "{}", alignment.to_sam_string_with_seq(seq, qual))
}

/// Write multiple SAM records for single-end alignments
///
/// This is a convenience function that combines selection, preparation, and writing.
pub fn write_single_end_alignments<W: Write>(
    writer: &mut W,
    alignments: Vec<Alignment>,
    seq: &str,
    qual: &str,
    opt: &MemOpt,
    rg_id: Option<&str>,
) -> std::io::Result<usize> {
    let selection = select_single_end_alignments(&alignments, opt);

    // Handle unmapped output case
    if selection.output_as_unmapped {
        let query_name = alignments
            .first()
            .map(|a| a.query_name.as_str())
            .unwrap_or("unknown");
        let mut unmapped = create_unmapped_single_end(query_name, seq.len());

        if let Some(rg) = rg_id {
            unmapped.tags.push(("RG".to_string(), format!("Z:{}", rg)));
        }

        write_sam_record(writer, &unmapped, seq, qual)?;
        return Ok(1);
    }

    let mut records_written = 0;
    let mut alignments = alignments;

    for idx in selection.output_indices {
        let is_primary = idx == selection.primary_idx;
        prepare_single_end_alignment(&mut alignments[idx], is_primary, rg_id);
        write_sam_record(writer, &alignments[idx], seq, qual)?;
        records_written += 1;
    }

    Ok(records_written)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_alignment(name: &str, score: i32, flag: u16) -> Alignment {
        Alignment {
            query_name: name.to_string(),
            flag,
            ref_name: "chr1".to_string(),
            ref_id: 0,
            pos: 1000,
            mapq: 60,
            score,
            cigar: vec![(b'M', 100)],
            rnext: "*".to_string(),
            pnext: 0,
            tlen: 0,
            seq: String::new(),
            qual: String::new(),
            tags: vec![],
            query_start: 0,
            query_end: 100,
            seed_coverage: score,
            hash: 0,
            frac_rep: 0.0,
        }
    }

    #[test]
    fn test_select_single_end_empty() {
        let mut opt = MemOpt::default();
        opt.t = 30;
        let alignments: Vec<Alignment> = vec![];
        let result = select_single_end_alignments(&alignments, &opt);

        assert!(result.output_as_unmapped);
        assert!(result.output_indices.is_empty());
    }

    #[test]
    fn test_select_single_end_below_threshold() {
        let mut opt = MemOpt::default();
        opt.t = 50;
        let alignments = vec![
            make_test_alignment("read1", 30, 0), // Below threshold
        ];
        let result = select_single_end_alignments(&alignments, &opt);

        assert!(result.output_as_unmapped);
    }

    #[test]
    fn test_select_single_end_normal() {
        let mut opt = MemOpt::default();
        opt.t = 30;
        opt.output_all_alignments = false;

        let alignments = vec![
            make_test_alignment("read1", 100, 0), // Primary
            make_test_alignment("read1", 80, sam_flags::SUPPLEMENTARY), // Supplementary
            make_test_alignment("read1", 70, sam_flags::SECONDARY), // Secondary
        ];

        let result = select_single_end_alignments(&alignments, &opt);

        assert!(!result.output_as_unmapped);
        assert_eq!(result.primary_idx, 0);
        // Should include primary (0) and supplementary (1), but not secondary (2)
        assert!(result.output_indices.contains(&0));
        assert!(result.output_indices.contains(&1));
        assert!(!result.output_indices.contains(&2));
    }

    #[test]
    fn test_select_single_end_output_all() {
        let mut opt = MemOpt::default();
        opt.t = 30;
        opt.output_all_alignments = true;

        let alignments = vec![
            make_test_alignment("read1", 100, 0),
            make_test_alignment("read1", 80, sam_flags::SUPPLEMENTARY),
            make_test_alignment("read1", 70, sam_flags::SECONDARY),
        ];

        let result = select_single_end_alignments(&alignments, &opt);

        // With -a flag, all alignments above threshold should be output
        assert_eq!(result.output_indices.len(), 3);
    }

    #[test]
    fn test_select_single_end_secondary_highest_score() {
        // Regression test: When the highest scoring alignment is marked SECONDARY,
        // we should still select the non-SECONDARY alignment as primary
        let mut opt = MemOpt::default();
        opt.t = 30;
        opt.output_all_alignments = false;

        // Alignment 0: lower score, not secondary (should be primary)
        // Alignment 1: higher score, but marked SECONDARY
        let alignments = vec![
            make_test_alignment("read1", 100, 0), // Not secondary
            make_test_alignment("read1", 150, sam_flags::SECONDARY), // Higher score but secondary
        ];

        let result = select_single_end_alignments(&alignments, &opt);

        // Primary should be alignment 0 (the non-SECONDARY one)
        assert_eq!(result.primary_idx, 0);
        assert!(!result.output_as_unmapped);
        // Should output the primary alignment
        assert!(result.output_indices.contains(&0));
        // Should NOT output the secondary (unless -a flag)
        assert!(!result.output_indices.contains(&1));
    }

    #[test]
    fn test_prepare_single_end_primary() {
        let mut alignment = make_test_alignment("read1", 100, sam_flags::SECONDARY);
        prepare_single_end_alignment(&mut alignment, true, None);

        // Primary should have SECONDARY cleared
        assert_eq!(alignment.flag & sam_flags::SECONDARY, 0);
    }

    #[test]
    fn test_prepare_single_end_with_rg() {
        let mut alignment = make_test_alignment("read1", 100, 0);
        prepare_single_end_alignment(&mut alignment, true, Some("sample1"));

        // Should have RG tag
        assert!(
            alignment
                .tags
                .iter()
                .any(|(tag, val)| tag == "RG" && val == "Z:sample1")
        );
    }

    #[test]
    fn test_create_unmapped_single_end() {
        let unmapped = create_unmapped_single_end("read1", 100);

        assert_eq!(unmapped.flag & sam_flags::UNMAPPED, sam_flags::UNMAPPED);
        assert_eq!(unmapped.ref_name, "*");
        assert_eq!(unmapped.pos, 0);
        assert_eq!(unmapped.mapq, 0);
    }

    #[test]
    fn test_create_unmapped_paired() {
        let unmapped_r1 = create_unmapped_paired("read1", b"ACGT", true);
        let unmapped_r2 = create_unmapped_paired("read1", b"ACGT", false);

        // Both should be paired and unmapped
        assert_eq!(unmapped_r1.flag & sam_flags::PAIRED, sam_flags::PAIRED);
        assert_eq!(unmapped_r1.flag & sam_flags::UNMAPPED, sam_flags::UNMAPPED);

        // R1 should be FIRST_IN_PAIR, R2 should be SECOND_IN_PAIR
        assert_eq!(
            unmapped_r1.flag & sam_flags::FIRST_IN_PAIR,
            sam_flags::FIRST_IN_PAIR
        );
        assert_eq!(
            unmapped_r2.flag & sam_flags::SECOND_IN_PAIR,
            sam_flags::SECOND_IN_PAIR
        );
    }
}
