// SAM output module
//
// Centralizes SAM formatting, flag management, and output logic.
// This module provides a clean separation between alignment computation
// and I/O concerns.

use crate::core::io::soa_readers::SoAReadBatch;
use crate::pipelines::linear::batch_extension::SoAAlignmentResult;
use crate::pipelines::linear::finalization::{Alignment, sam_flags};
use crate::pipelines::linear::mem_opt::MemOpt;
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
        alignment.tags.push(("RG".to_string(), format!("Z:{rg}")));
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
    let mate_is_unmapped = ctx.mate_ref == "*" || (ctx.mate_flag & sam_flags::UNMAPPED) != 0;

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
        alignment.tags.push(("RG".to_string(), format!("Z:{rg}")));
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
    let mate_is_unmapped = ctx.mate_ref == "*" || (ctx.mate_flag & sam_flags::UNMAPPED) != 0;

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
        alignment.tags.push(("RG".to_string(), format!("Z:{rg}")));
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
        is_alt: false,  // Unmapped reads don't map to alternate contigs
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
        is_alt: false,  // Unmapped reads don't map to alternate contigs
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
            unmapped.tags.push(("RG".to_string(), format!("Z:{rg}")));
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
// SOA-AWARE SAM WRITING (PR4 Phase 3)
// ============================================================================

/// Write SAM records directly from SoAAlignmentResult (PR4)
///
/// This function writes SAM records batch-wise without converting to individual
/// Alignment structs, providing better memory efficiency for large batches.
///
/// # Arguments
/// * `writer` - Output writer
/// * `soa_result` - SoA alignment results
/// * `soa_read_batch` - Original read batch (for seq/qual data)
/// * `opt` - Alignment options
/// * `rg_id` - Optional read group ID
///
/// # Returns
/// Number of SAM records written
pub fn write_sam_records_soa<W: Write>(
    writer: &mut W,
    soa_result: &SoAAlignmentResult,
    soa_read_batch: &SoAReadBatch,
    opt: &MemOpt,
    rg_id: Option<&str>,
) -> std::io::Result<usize> {
    let mut records_written = 0;

    // Iterate through each read
    for read_idx in 0..soa_result.num_reads() {
        let (alignment_start_idx, num_alignments) = soa_result.read_alignment_boundaries[read_idx];

        // Get original seq/qual from SoA read batch
        let (seq_start, seq_len) = soa_read_batch.read_boundaries[read_idx];
        let seq_bytes = &soa_read_batch.seqs[seq_start..seq_start + seq_len];
        let seq = std::str::from_utf8(seq_bytes).unwrap_or("");
        let qual_bytes = &soa_read_batch.quals[seq_start..seq_start + seq_len];
        let qual = std::str::from_utf8(qual_bytes).unwrap_or("");

        if num_alignments == 0 {
            // No alignments - write unmapped record
            let query_name = &soa_read_batch.names[read_idx];
            let mut unmapped = create_unmapped_single_end(query_name, seq_len);

            if let Some(rg) = rg_id {
                unmapped.tags.push(("RG".to_string(), format!("Z:{rg}")));
            }

            write_sam_record(writer, &unmapped, seq, qual)?;
            records_written += 1;
            continue;
        }

        // Select which alignments to output
        // We need to check scores and flags from SoA data
        let mut best_score = i32::MIN;
        let mut primary_idx_within_read = 0;

        for i in 0..num_alignments {
            let aln_idx = alignment_start_idx + i;
            let is_secondary = soa_result.flags[aln_idx] & sam_flags::SECONDARY != 0;

            // Find primary (not marked as SECONDARY) with highest score
            if !is_secondary && soa_result.scores[aln_idx] > best_score {
                best_score = soa_result.scores[aln_idx];
                primary_idx_within_read = i;
            }
        }

        let primary_aln_idx = alignment_start_idx + primary_idx_within_read;
        let is_unmapped = soa_result.flags[primary_aln_idx] & sam_flags::UNMAPPED != 0;

        // Check if best alignment is below threshold
        if !is_unmapped && best_score < opt.t {
            // Output unmapped record
            let query_name = &soa_read_batch.names[read_idx];
            let mut unmapped = create_unmapped_single_end(query_name, seq_len);

            if let Some(rg) = rg_id {
                unmapped.tags.push(("RG".to_string(), format!("Z:{rg}")));
            }

            write_sam_record(writer, &unmapped, seq, qual)?;
            records_written += 1;
            continue;
        }

        // Write selected alignments
        for i in 0..num_alignments {
            let aln_idx = alignment_start_idx + i;
            let is_primary = i == primary_idx_within_read;
            let is_unmapped = soa_result.flags[aln_idx] & sam_flags::UNMAPPED != 0;
            let is_secondary = soa_result.flags[aln_idx] & sam_flags::SECONDARY != 0;
            let is_supplementary = soa_result.flags[aln_idx] & sam_flags::SUPPLEMENTARY != 0;

            // Determine if should output this alignment
            let should_output = if opt.output_all_alignments {
                is_unmapped || soa_result.scores[aln_idx] >= opt.t
            } else {
                is_unmapped || (!is_secondary && (is_primary || is_supplementary))
            };

            log::debug!(
                "[SAM_OUTPUT] Read {} aln {}/{}: flag={}, score={}, is_primary={}, is_secondary={}, is_supp={}, should_output={}",
                soa_read_batch.names[read_idx],
                i,
                num_alignments,
                soa_result.flags[aln_idx],
                soa_result.scores[aln_idx],
                is_primary,
                is_secondary,
                is_supplementary,
                should_output
            );

            if !should_output {
                continue;
            }

            // Write SAM record directly from SoA data
            write_sam_record_from_soa(writer, soa_result, aln_idx, seq, qual, is_primary, rg_id)?;
            records_written += 1;
        }
    }

    Ok(records_written)
}

/// Helper: Write a single SAM record from SoA data (PR4)
///
/// Formats a SAM record directly from SoAAlignmentResult arrays without
/// constructing an Alignment struct. Replicates the logic from
/// Alignment::to_sam_string_with_seq().
#[inline]
fn write_sam_record_from_soa<W: Write>(
    writer: &mut W,
    soa_result: &SoAAlignmentResult,
    aln_idx: usize,
    seq: &str,
    qual: &str,
    is_primary: bool,
    rg_id: Option<&str>,
) -> std::io::Result<()> {
    let mut flag = soa_result.flags[aln_idx];

    // Adjust flags for primary alignments
    if is_primary {
        flag &= !sam_flags::SECONDARY;
        flag &= !sam_flags::SUPPLEMENTARY;
    }

    // Format CIGAR string
    let (cigar_start, cigar_len) = soa_result.cigar_boundaries[aln_idx];
    let cigar_string = format_cigar_soa(
        &soa_result.cigar_ops[cigar_start..cigar_start + cigar_len],
        &soa_result.cigar_lens[cigar_start..cigar_start + cigar_len],
        flag & sam_flags::SUPPLEMENTARY != 0,
    );

    // Handle reverse complement for SEQ and QUAL
    let (output_seq, output_qual) = if flag & sam_flags::REVERSE != 0 {
        let rev_comp_seq: String = seq
            .chars()
            .rev()
            .map(|c| match c {
                'A' => 'T',
                'T' => 'A',
                'C' => 'G',
                'G' => 'C',
                'N' => 'N',
                _ => c,
            })
            .collect();
        let rev_qual: String = qual.chars().rev().collect();
        (rev_comp_seq, rev_qual)
    } else {
        (seq.to_string(), qual.to_string())
    };

    // Handle hard/soft clips for trimming
    let (trimmed_seq, trimmed_qual) = trim_seq_qual_for_clips(
        &output_seq,
        &output_qual,
        &soa_result.cigar_ops[cigar_start..cigar_start + cigar_len],
        &soa_result.cigar_lens[cigar_start..cigar_start + cigar_len],
        flag & sam_flags::SUPPLEMENTARY != 0,
    );

    // Format SAM position
    let sam_pos = if &soa_result.ref_names[aln_idx] == "*" {
        0
    } else {
        soa_result.positions[aln_idx] + 1
    };

    // Write mandatory fields
    write!(
        writer,
        "{}	{}	{}	{}	{}	{}	{}	{}	{}	{}	{}",
        soa_result.query_names[aln_idx],
        flag,
        soa_result.ref_names[aln_idx],
        sam_pos,
        soa_result.mapqs[aln_idx],
        cigar_string,
        soa_result.rnexts[aln_idx],
        soa_result.pnexts[aln_idx],
        soa_result.tlens[aln_idx],
        trimmed_seq,
        trimmed_qual
    )?;

    // Write tags
    let (tag_start, tag_len) = soa_result.tag_boundaries[aln_idx];
    for i in 0..tag_len {
        write!(
            writer,
            "	{}:{}",
            soa_result.tag_names[tag_start + i],
            soa_result.tag_values[tag_start + i]
        )?;
    }

    // Add RG tag if specified
    if let Some(rg) = rg_id {
        write!(writer, "	RG:Z:{rg}")?;
    }

    writeln!(writer)?;

    Ok(())
}

/// Format CIGAR string from SoA data (PR4)
///
/// Converts soft clips (S) to hard clips (H) for supplementary alignments.
fn format_cigar_soa(ops: &[u8], lens: &[i32], is_supplementary: bool) -> String {
    if ops.is_empty() {
        return "*".to_string();
    }

    ops.iter()
        .zip(lens.iter())
        .map(|(&op, &len)| {
            let op_char = if is_supplementary && op == b'S' {
                'H'
            } else {
                op as char
            };
            format!("{len}{op_char}")
        })
        .collect()
}

/// Trim SEQ and QUAL for hard/soft clips (PR4)
///
/// For supplementary alignments, soft clips (S) are treated as clips requiring trimming.
fn trim_seq_qual_for_clips(
    seq: &str,
    qual: &str,
    ops: &[u8],
    lens: &[i32],
    is_supplementary: bool,
) -> (String, String) {
    let mut leading_clip = 0usize;
    let mut trailing_clip = 0usize;

    // Sum leading clips
    for (&op, &len) in ops.iter().zip(lens.iter()) {
        if op == b'H' || (is_supplementary && op == b'S') {
            leading_clip += len as usize;
        } else {
            break;
        }
    }

    // Sum trailing clips
    for (&op, &len) in ops.iter().zip(lens.iter()).rev() {
        if op == b'H' || (is_supplementary && op == b'S') {
            trailing_clip += len as usize;
        } else {
            break;
        }
    }

    // Trim if needed
    if leading_clip > 0 || trailing_clip > 0 {
        let seq_len = seq.len();
        let start = leading_clip.min(seq_len);
        let end = seq_len.saturating_sub(trailing_clip);

        if start < end {
            (
                seq[start..end].to_string(),
                qual[start..end.min(qual.len())].to_string(),
            )
        } else {
            (String::new(), String::new())
        }
    } else {
        (seq.to_string(), qual.to_string())
    }
}

// ============================================================================
// PAIRED-END SOA SAM WRITING (Critical Fix)
// ============================================================================

/// Write SAM records for paired-end reads directly from SoA data (PR4)
///
/// This function handles mate coordinate assignment, mate flags, and proper
/// paired-end SAM formatting. It processes R1 and R2 simultaneously to ensure
/// correct mate information is set for each alignment.
///
/// # Critical Requirements
/// - `soa_result1` and `soa_result2` must have the same number of reads
/// - Reads at index i in both results must be mates (same read name)
/// - Primary alignments are used for mate coordinate assignment
///
/// # Arguments
/// * `writer` - Output writer
/// * `soa_result1` - SoA alignment results for read1 (R1)
/// * `soa_result2` - SoA alignment results for read2 (R2)
/// * `soa_batch1` - Original read batch for R1 (for seq/qual data)
/// * `soa_batch2` - Original read batch for R2 (for seq/qual data)
/// * `is_properly_paired` - Per-pair proper pair flags (from pairing stage)
/// * `opt` - Alignment options
/// * `rg_id` - Optional read group ID
///
/// # Returns
/// Number of SAM records written
pub fn write_sam_records_paired_soa<W: Write>(
    writer: &mut W,
    soa_result1: &SoAAlignmentResult,
    soa_result2: &SoAAlignmentResult,
    soa_batch1: &SoAReadBatch,
    soa_batch2: &SoAReadBatch,
    is_properly_paired: &[bool], // Per-read proper pair flags
    opt: &MemOpt,
    rg_id: Option<&str>,
) -> std::io::Result<usize> {
    let num_reads = soa_result1.num_reads();

    // Validate input
    if soa_result2.num_reads() != num_reads {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "R1 and R2 must have same number of reads: {} vs {}",
                num_reads,
                soa_result2.num_reads()
            ),
        ));
    }

    if soa_batch1.len() != num_reads || soa_batch2.len() != num_reads {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "Batch sizes must match result sizes: batch1={}, batch2={}, results={}",
                soa_batch1.len(),
                soa_batch2.len(),
                num_reads
            ),
        ));
    }

    if is_properly_paired.len() != num_reads {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "is_properly_paired length must match num_reads: {} vs {}",
                is_properly_paired.len(),
                num_reads
            ),
        ));
    }

    let mut records_written = 0;

    eprintln!(
        "[SAM_OUTPUT] write_sam_records_paired_soa: {} reads, batch1_len={}, batch2_len={}, result1_reads={}, result2_reads={}",
        num_reads,
        soa_batch1.len(),
        soa_batch2.len(),
        soa_result1.num_reads(),
        soa_result2.num_reads()
    );

    // DEBUG: Check for duplicate alignment boundaries
    let mut aln_idx_seen = std::collections::HashSet::new();
    for read_idx in 0..num_reads {
        let (aln_start1, num_alns1) = soa_result1.read_alignment_boundaries[read_idx];
        let (aln_start2, num_alns2) = soa_result2.read_alignment_boundaries[read_idx];

        for i in 0..num_alns1 {
            let aln_idx = aln_start1 + i;
            if !aln_idx_seen.insert(aln_idx) {
                eprintln!(
                    "[SAM_OUTPUT] DUPLICATE alignment index {} for read_idx={} R1",
                    aln_idx, read_idx
                );
            }
        }
        for i in 0..num_alns2 {
            let aln_idx = aln_start2 + i;
            if !aln_idx_seen.insert(aln_idx + 100000) {
                // Offset to separate R1/R2
                eprintln!(
                    "[SAM_OUTPUT] DUPLICATE alignment index {} for read_idx={} R2",
                    aln_idx, read_idx
                );
            }
        }
    }

    // Process each read pair
    for read_idx in 0..num_reads {
        let (aln_start1, num_alns1) = soa_result1.read_alignment_boundaries[read_idx];
        let (aln_start2, num_alns2) = soa_result2.read_alignment_boundaries[read_idx];

        // Get sequences for this pair
        let (seq_start1, seq_len1) = soa_batch1.read_boundaries[read_idx];
        let seq1_bytes = &soa_batch1.seqs[seq_start1..seq_start1 + seq_len1];
        let seq1 = std::str::from_utf8(seq1_bytes).unwrap_or("");
        let qual1_bytes = &soa_batch1.quals[seq_start1..seq_start1 + seq_len1];
        let qual1 = std::str::from_utf8(qual1_bytes).unwrap_or("");

        let (seq_start2, seq_len2) = soa_batch2.read_boundaries[read_idx];
        let seq2_bytes = &soa_batch2.seqs[seq_start2..seq_start2 + seq_len2];
        let seq2 = std::str::from_utf8(seq2_bytes).unwrap_or("");
        let qual2_bytes = &soa_batch2.quals[seq_start2..seq_start2 + seq_len2];
        let qual2 = std::str::from_utf8(qual2_bytes).unwrap_or("");

        // Find primary alignments for mate info
        let primary_idx1 = find_primary_alignment_soa(soa_result1, aln_start1, num_alns1);
        let primary_idx2 = find_primary_alignment_soa(soa_result2, aln_start2, num_alns2);

        // Get mate information from primary alignments
        let (mate2_ref, mate2_pos, mate2_flag, mate2_cigar, mate2_ref_len) =
            get_mate_info_soa(soa_result2, primary_idx2);
        let (mate1_ref, mate1_pos, mate1_flag, mate1_cigar, mate1_ref_len) =
            get_mate_info_soa(soa_result1, primary_idx1);

        // Build mate contexts
        let ctx_for_read1 = PairedFlagContext {
            mate_ref: mate2_ref,
            mate_pos: mate2_pos,
            mate_flag: mate2_flag,
            mate_cigar: mate2_cigar,
            mate_ref_len: mate2_ref_len,
            is_properly_paired: is_properly_paired[read_idx],
        };

        let ctx_for_read2 = PairedFlagContext {
            mate_ref: mate1_ref,
            mate_pos: mate1_pos,
            mate_flag: mate1_flag,
            mate_cigar: mate1_cigar,
            mate_ref_len: mate1_ref_len,
            is_properly_paired: is_properly_paired[read_idx],
        };

        // Handle case where both reads are unmapped
        if num_alns1 == 0 && num_alns2 == 0 {
            // Both unmapped - write unmapped records
            let name = &soa_batch1.names[read_idx];
            let mut unmapped_r1 = create_unmapped_paired(name, seq1_bytes, true);
            let mut unmapped_r2 = create_unmapped_paired(name, seq2_bytes, false);

            // Both unmapped - set mate unmapped flags
            unmapped_r1.flag |= sam_flags::MATE_UNMAPPED;
            unmapped_r2.flag |= sam_flags::MATE_UNMAPPED;

            if let Some(rg) = rg_id {
                unmapped_r1.tags.push(("RG".to_string(), format!("Z:{rg}")));
                unmapped_r2.tags.push(("RG".to_string(), format!("Z:{rg}")));
            }

            write_sam_record(writer, &unmapped_r1, seq1, qual1)?;
            write_sam_record(writer, &unmapped_r2, seq2, qual2)?;
            records_written += 2;
            continue;
        }

        // Select which alignments to output
        let output_indices1 =
            select_output_indices_soa(soa_result1, aln_start1, num_alns1, primary_idx1, opt);
        let output_indices2 =
            select_output_indices_soa(soa_result2, aln_start2, num_alns2, primary_idx2, opt);

        // DEBUG: Check for duplicate indices in output
        if output_indices1.len()
            != output_indices1
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len()
        {
            eprintln!(
                "[SAM_OUTPUT] DUPLICATE indices in output_indices1 for read_idx={}: {:?}",
                read_idx, output_indices1
            );
        }
        if output_indices2.len()
            != output_indices2
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len()
        {
            eprintln!(
                "[SAM_OUTPUT] DUPLICATE indices in output_indices2 for read_idx={}: {:?}",
                read_idx, output_indices2
            );
        }

        // Write R1 alignments
        for &aln_idx in &output_indices1 {
            let is_primary = aln_idx == primary_idx1;
            write_sam_record_paired_soa(
                writer,
                soa_result1,
                aln_idx,
                seq1,
                qual1,
                is_primary,
                true, // is_first_in_pair
                &ctx_for_read1,
                rg_id,
            )?;
            records_written += 1;
        }

        // Write R2 alignments
        for &aln_idx in &output_indices2 {
            let is_primary = aln_idx == primary_idx2;
            write_sam_record_paired_soa(
                writer,
                soa_result2,
                aln_idx,
                seq2,
                qual2,
                is_primary,
                false, // is_first_in_pair
                &ctx_for_read2,
                rg_id,
            )?;
            records_written += 1;
        }
    }

    Ok(records_written)
}

/// Find the primary alignment index in SoA data
///
/// Primary is the alignment NOT marked as SECONDARY with highest score
fn find_primary_alignment_soa(
    soa_result: &SoAAlignmentResult,
    aln_start: usize,
    num_alns: usize,
) -> usize {
    if num_alns == 0 {
        return aln_start; // No alignments (will be handled as unmapped)
    }

    let mut best_score = i32::MIN;
    let mut primary_idx = aln_start;

    for i in 0..num_alns {
        let aln_idx = aln_start + i;
        let is_secondary = soa_result.flags[aln_idx] & sam_flags::SECONDARY != 0;

        if !is_secondary && soa_result.scores[aln_idx] > best_score {
            best_score = soa_result.scores[aln_idx];
            primary_idx = aln_idx;
        }
    }

    primary_idx
}

/// Get mate information from SoA alignment
///
/// Returns (ref_name, pos, flag, cigar_string, ref_len)
fn get_mate_info_soa(
    soa_result: &SoAAlignmentResult,
    aln_idx: usize,
) -> (String, u64, u16, String, i32) {
    if aln_idx >= soa_result.flags.len() {
        // No alignment - unmapped
        return ("*".to_string(), 0, sam_flags::UNMAPPED, "*".to_string(), 0);
    }

    let ref_name = soa_result.ref_names[aln_idx].clone();
    let pos = soa_result.positions[aln_idx];
    let flag = soa_result.flags[aln_idx];

    // Format CIGAR string
    let (cigar_start, cigar_len) = soa_result.cigar_boundaries[aln_idx];
    let cigar_string = if cigar_len == 0 {
        "*".to_string()
    } else {
        format_cigar_soa(
            &soa_result.cigar_ops[cigar_start..cigar_start + cigar_len],
            &soa_result.cigar_lens[cigar_start..cigar_start + cigar_len],
            false, // Don't convert S->H for mate CIGAR (MC tag)
        )
    };

    // Calculate reference length from CIGAR
    let ref_len = calculate_ref_len_from_cigar(
        &soa_result.cigar_ops[cigar_start..cigar_start + cigar_len],
        &soa_result.cigar_lens[cigar_start..cigar_start + cigar_len],
    );

    (ref_name, pos, flag, cigar_string, ref_len)
}

/// Calculate reference length from CIGAR operations
fn calculate_ref_len_from_cigar(ops: &[u8], lens: &[i32]) -> i32 {
    ops.iter()
        .zip(lens.iter())
        .filter(|(op, _)| {
            **op == b'M' || **op == b'D' || **op == b'N' || **op == b'=' || **op == b'X'
        })
        .map(|(_, len)| *len)
        .sum()
}

/// Select which alignments to output for a paired read from SoA data
fn select_output_indices_soa(
    soa_result: &SoAAlignmentResult,
    aln_start: usize,
    num_alns: usize,
    primary_idx: usize,
    opt: &MemOpt,
) -> Vec<usize> {
    let mut indices = Vec::new();

    for i in 0..num_alns {
        let aln_idx = aln_start + i;
        let is_unmapped = soa_result.flags[aln_idx] & sam_flags::UNMAPPED != 0;
        let is_primary = aln_idx == primary_idx;
        let is_supplementary = soa_result.flags[aln_idx] & sam_flags::SUPPLEMENTARY != 0;

        let should_output = if opt.output_all_alignments {
            is_unmapped || soa_result.scores[aln_idx] >= opt.t
        } else {
            (is_primary || is_supplementary) && (is_unmapped || soa_result.scores[aln_idx] >= opt.t)
        };

        if should_output {
            indices.push(aln_idx);
        }
    }

    indices
}

/// Write a single paired-end SAM record from SoA data
///
/// This combines the logic from write_sam_record_from_soa() with paired-end
/// flag and mate coordinate handling from prepare_paired_alignment_read1/read2.
#[inline]
fn write_sam_record_paired_soa<W: Write>(
    writer: &mut W,
    soa_result: &SoAAlignmentResult,
    aln_idx: usize,
    seq: &str,
    qual: &str,
    is_primary: bool,
    is_first_in_pair: bool,
    ctx: &PairedFlagContext,
    rg_id: Option<&str>,
) -> std::io::Result<()> {
    let mut flag = soa_result.flags[aln_idx];
    let is_unmapped = flag & sam_flags::UNMAPPED != 0;
    let is_supplementary = flag & sam_flags::SUPPLEMENTARY != 0;

    // Set paired-end flags
    flag |= sam_flags::PAIRED;
    if is_first_in_pair {
        flag |= sam_flags::FIRST_IN_PAIR;
    } else {
        flag |= sam_flags::SECOND_IN_PAIR;
    }

    // Set proper pair flag
    if !is_unmapped && ctx.is_properly_paired && is_primary {
        flag |= sam_flags::PROPER_PAIR;
    }

    // Set mate unmapped/reverse flags
    let mate_is_unmapped = ctx.mate_ref == "*" || (ctx.mate_flag & sam_flags::UNMAPPED) != 0;
    if mate_is_unmapped {
        flag |= sam_flags::MATE_UNMAPPED;
    }
    if !mate_is_unmapped && (ctx.mate_flag & sam_flags::REVERSE) != 0 {
        flag |= sam_flags::MATE_REVERSE;
    }

    // Adjust primary/secondary/supplementary flags
    if is_primary {
        flag &= !sam_flags::SECONDARY;
        flag &= !sam_flags::SUPPLEMENTARY;
    } else if !is_unmapped && !is_supplementary {
        flag |= sam_flags::SECONDARY;
    }

    // Determine output position and mate position
    let (ref_name, sam_pos, rnext, pnext, tlen) = if !mate_is_unmapped {
        if is_unmapped {
            // This read unmapped, mate mapped: use mate's position
            let rnext = "=".to_string();
            let pnext = ctx.mate_pos + 1;
            (ctx.mate_ref.clone(), ctx.mate_pos, rnext, pnext, 0)
        } else {
            // Both mapped
            let ref_name = soa_result.ref_names[aln_idx].clone();
            let pos = soa_result.positions[aln_idx];
            let rnext = if ref_name == ctx.mate_ref {
                "=".to_string()
            } else {
                ctx.mate_ref.clone()
            };
            let pnext = ctx.mate_pos + 1;

            // Calculate TLEN only if on same reference
            let tlen = if ref_name == ctx.mate_ref {
                calculate_tlen_soa(pos, ctx.mate_pos, ctx.mate_ref_len)
            } else {
                0
            };

            (ref_name, pos, rnext, pnext, tlen)
        }
    } else if is_unmapped {
        // Both unmapped
        ("*".to_string(), 0, "*".to_string(), 0, 0)
    } else {
        // This read mapped, mate unmapped: mate goes to our position
        let ref_name = soa_result.ref_names[aln_idx].clone();
        let pos = soa_result.positions[aln_idx];
        (ref_name, pos, "=".to_string(), pos + 1, 0)
    };

    // Format CIGAR string
    let (cigar_start, cigar_len) = soa_result.cigar_boundaries[aln_idx];
    let cigar_string = format_cigar_soa(
        &soa_result.cigar_ops[cigar_start..cigar_start + cigar_len],
        &soa_result.cigar_lens[cigar_start..cigar_start + cigar_len],
        is_supplementary,
    );

    // Handle reverse complement for SEQ and QUAL
    let (output_seq, output_qual) = if flag & sam_flags::REVERSE != 0 {
        let rev_comp_seq: String = seq
            .chars()
            .rev()
            .map(|c| match c {
                'A' => 'T',
                'T' => 'A',
                'C' => 'G',
                'G' => 'C',
                'N' => 'N',
                _ => c,
            })
            .collect();
        let rev_qual: String = qual.chars().rev().collect();
        (rev_comp_seq, rev_qual)
    } else {
        (seq.to_string(), qual.to_string())
    };

    // Handle hard/soft clips for trimming
    let (trimmed_seq, trimmed_qual) = trim_seq_qual_for_clips(
        &output_seq,
        &output_qual,
        &soa_result.cigar_ops[cigar_start..cigar_start + cigar_len],
        &soa_result.cigar_lens[cigar_start..cigar_start + cigar_len],
        is_supplementary,
    );

    // Write mandatory fields
    write!(
        writer,
        "{}	{}	{}	{}	{}	{}	{}	{}	{}	{}	{}",
        soa_result.query_names[aln_idx],
        flag,
        ref_name,
        if ref_name == "*" { 0 } else { sam_pos + 1 },
        soa_result.mapqs[aln_idx],
        cigar_string,
        rnext,
        pnext,
        tlen,
        trimmed_seq,
        trimmed_qual
    )?;

    // Write tags
    let (tag_start, tag_len) = soa_result.tag_boundaries[aln_idx];
    for i in 0..tag_len {
        write!(
            writer,
            "	{}:{}",
            soa_result.tag_names[tag_start + i],
            soa_result.tag_values[tag_start + i]
        )?;
    }

    // Add MC tag for mate CIGAR (only if mate is mapped)
    if !mate_is_unmapped && !ctx.mate_cigar.is_empty() && ctx.mate_cigar != "*" {
        write!(writer, "	MC:Z:{}", ctx.mate_cigar)?;
    }

    // Add RG tag if specified
    if let Some(rg) = rg_id {
        write!(writer, "	RG:Z:{rg}")?;
    }

    writeln!(writer)?;

    Ok(())
}

/// Calculate template length (TLEN) for paired-end alignment
///
/// Matches BWA-MEM2 logic: signed distance from leftmost to rightmost base
fn calculate_tlen_soa(pos1: u64, pos2: u64, mate_ref_len: i32) -> i32 {
    let pos1 = pos1 as i64;
    let pos2 = pos2 as i64;

    if pos1 <= pos2 {
        // This read is leftmost
        (pos2 + mate_ref_len as i64 - pos1) as i32
    } else {
        // Mate is leftmost
        -(pos1 - pos2) as i32
    }
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
            is_alt: false,
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
