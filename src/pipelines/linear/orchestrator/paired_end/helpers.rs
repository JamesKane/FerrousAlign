//! Helper functions for paired-end orchestrator.
//!
//! Contains mate info setting, AoS pairing, mate rescue, and output writing.

use std::io::Write;

use crate::pipelines::linear::batch_extension::types::SoAAlignmentResult;
use crate::pipelines::linear::finalization::sam_flags;
use crate::pipelines::linear::paired::mate_rescue::mate_rescue_soa;
use crate::pipelines::linear::paired::pairing_aos;
use crate::pipelines::linear::stages::finalization::Alignment;

use super::{OrchestratorError, PairedEndOrchestrator, SoAReadBatch};

impl PairedEndOrchestrator<'_> {
    /// Perform AoS pairing on alignment results.
    ///
    /// CRITICAL: This must be done in AoS format to maintain per-read alignment
    /// boundaries correctly. Pure SoA pairing causes 96% duplicate reads.
    ///
    /// BWA-MEM2 behavior (bwamem_pair.cpp mem_sam_pe):
    /// - Sets PAIRED | FIRST_IN_PAIR (0x41) on ALL R1 alignments
    /// - Sets PAIRED | SECOND_IN_PAIR (0x81) on ALL R2 alignments
    /// - Sets PROPER_PAIR only on the best pairing
    /// - Decides: if paired_score > unpaired_score, output best pair; else output idx=0
    pub(super) fn pair_alignments_aos(
        &self,
        alignments1: &mut Vec<Vec<Alignment>>,
        alignments2: &mut Vec<Vec<Alignment>>,
        batch_start_id: u64,
    ) {
        let l_pac = self.index.bns.packed_sequence_length as i64;

        for read_idx in 0..alignments1.len() {
            let alns1 = &mut alignments1[read_idx];
            let alns2 = &mut alignments2[read_idx];

            // BWA-MEM2: Set PAIRED and FIRST/SECOND_IN_PAIR on ALL alignments for this read pair
            for aln in alns1.iter_mut() {
                aln.flag |= sam_flags::PAIRED | sam_flags::FIRST_IN_PAIR;
            }
            for aln in alns2.iter_mut() {
                aln.flag |= sam_flags::PAIRED | sam_flags::SECOND_IN_PAIR;
            }

            if let Some((idx1, idx2, pair_score, _sub_score)) = pairing_aos::mem_pair(
                &self.insert_stats,
                alns1,
                alns2,
                self.options.a,
                batch_start_id + read_idx as u64,
                l_pac,
            ) {
                // BWA-MEM2 bwamem_pair.cpp:452-473:
                // score_un = a[0].a[0].score + a[1].a[0].score - opt->pen_unpaired
                // if (pair_score > score_un) -> use best pair, set PROPER_PAIR
                // else -> use idx=0, no PROPER_PAIR
                let score_un = alns1[0].score + alns2[0].score - self.options.pen_unpaired;

                let (output_idx1, output_idx2) = if pair_score > score_un {
                    // Paired alignment is preferred - use best pair indices
                    alns1[idx1].flag |= sam_flags::PROPER_PAIR;
                    alns2[idx2].flag |= sam_flags::PROPER_PAIR;
                    (idx1, idx2)
                } else {
                    // Unpaired alignment is preferred - use idx=0 (highest scoring)
                    (0, 0)
                };

                // Set mate information on the alignments we'll output
                let mate1 = alns2[output_idx2].clone();
                let mate2 = alns1[output_idx1].clone();
                Self::set_mate_info(&mut alns1[output_idx1], &mate1);
                Self::set_mate_info(&mut alns2[output_idx2], &mate2);

                // Set TLEN
                Self::set_tlen(&mut alns1[output_idx1], &mate1);
                Self::set_tlen(&mut alns2[output_idx2], &mate2);

                // Swap to ensure output_idx is at position 0 (since write_paired_output outputs idx=0)
                if output_idx1 != 0 && !alns1.is_empty() {
                    // Also swap flags to maintain SECONDARY marking
                    let was_secondary = (alns1[output_idx1].flag & sam_flags::SECONDARY) != 0;
                    alns1[output_idx1].flag &= !sam_flags::SECONDARY;
                    if was_secondary {
                        alns1[0].flag |= sam_flags::SECONDARY;
                    }
                    alns1.swap(0, output_idx1);
                }
                if output_idx2 != 0 && !alns2.is_empty() {
                    let was_secondary = (alns2[output_idx2].flag & sam_flags::SECONDARY) != 0;
                    alns2[output_idx2].flag &= !sam_flags::SECONDARY;
                    if was_secondary {
                        alns2[0].flag |= sam_flags::SECONDARY;
                    }
                    alns2.swap(0, output_idx2);
                }
            } else if !alns1.is_empty() && !alns2.is_empty() {
                // No valid pairing found - both mapped but not properly paired
                let (aln1, aln2) = (alns1[0].clone(), alns2[0].clone());
                Self::set_mate_info(&mut alns1[0], &aln2);
                Self::set_mate_info(&mut alns2[0], &aln1);

                // Set TLEN if on same chromosome
                if alns1[0].ref_name == alns2[0].ref_name {
                    Self::set_tlen(&mut alns1[0], &alns2[0]);
                    Self::set_tlen(&mut alns2[0], &alns1[0]);
                }
            }
        }
    }

    /// Set mate information on an alignment.
    ///
    /// Note: PAIRED and FIRST/SECOND_IN_PAIR flags are already set on all alignments
    /// in pair_alignments_aos(). This function only sets mate-specific info (rnext, pnext,
    /// MATE_REVERSE, MATE_UNMAPPED) on primary alignments.
    pub(super) fn set_mate_info(aln: &mut Alignment, mate: &Alignment) {
        aln.rnext = if mate.ref_name == aln.ref_name {
            "=".to_string()
        } else {
            mate.ref_name.clone()
        };
        aln.pnext = mate.pos;

        if (mate.flag & sam_flags::REVERSE) != 0 {
            aln.flag |= sam_flags::MATE_REVERSE;
        }

        if mate.mapq == 0 && mate.score == 0 {
            aln.flag |= sam_flags::MATE_UNMAPPED;
        }
    }

    /// Set TLEN (template length) for a paired alignment.
    ///
    /// BWA-MEM2 formula (bwamem.cpp:1696-1700):
    /// - p0 = pos + (is_rev ? rlen - 1 : 0)  // rightmost position if reverse
    /// - p1 = mate_pos + (mate_is_rev ? mate_rlen - 1 : 0)
    /// - tlen = -(p0 - p1 + (p0 > p1 ? 1 : p0 < p1 ? -1 : 0))
    ///
    /// The tlen is positive for the leftmost alignment and negative for the rightmost.
    pub(super) fn set_tlen(aln: &mut Alignment, mate: &Alignment) {
        if aln.ref_name != mate.ref_name || mate.ref_name == "*" {
            aln.tlen = 0;
            return;
        }

        let is_rev = (aln.flag & sam_flags::REVERSE) != 0;
        let mate_is_rev = (mate.flag & sam_flags::REVERSE) != 0;

        let aln_rlen = aln.reference_length();
        let mate_rlen = mate.reference_length();

        // Calculate 5' end positions (rightmost if reverse strand)
        let p0 = if is_rev {
            aln.pos as i64 + aln_rlen as i64 - 1
        } else {
            aln.pos as i64
        };

        let p1 = if mate_is_rev {
            mate.pos as i64 + mate_rlen as i64 - 1
        } else {
            mate.pos as i64
        };

        // Calculate TLEN with sign convention
        let sign_adj = if p0 > p1 { 1 } else if p0 < p1 { -1 } else { 0 };
        aln.tlen = -(p0 - p1 + sign_adj) as i32;
    }

    /// Perform mate rescue using SoA format for SIMD batching.
    pub(super) fn perform_mate_rescue(
        &self,
        alignments1: &mut Vec<Vec<Alignment>>,
        alignments2: &mut Vec<Vec<Alignment>>,
        batch1: &SoAReadBatch,
        batch2: &SoAReadBatch,
    ) -> usize {
        // Convert to SoA for SIMD rescue
        let mut soa1 = SoAAlignmentResult::from_aos(alignments1);
        let mut soa2 = SoAAlignmentResult::from_aos(alignments2);

        // Find primary alignments for rescue
        let primary_r1: Vec<usize> = find_primary_alignments(&soa1);
        let primary_r2: Vec<usize> = find_primary_alignments(&soa2);

        let pac = &self.index.bns.pac_data;
        let rescued = mate_rescue_soa(
            &mut soa1,
            &mut soa2,
            batch1,
            batch2,
            &primary_r1,
            &primary_r2,
            pac,
            self.index,
            &self.insert_stats,
            self.options.pen_unpaired,
            self.options.max_matesw,
            Some(self.simd_engine),
        );

        // Convert back to AoS
        *alignments1 = soa1.to_aos();
        *alignments2 = soa2.to_aos();

        // Fix hash values for any alignments that don't have one (e.g., rescued alignments)
        // This matches BWA-MEM2's approach where mem_mark_primary_se() sets hashes AFTER mate rescue
        Self::fix_missing_hashes(alignments1);
        Self::fix_missing_hashes(alignments2);

        rescued
    }

    /// Fix missing hash values for alignments that have hash=0
    /// This happens for rescued alignments and unmapped alignments.
    /// Matches BWA-MEM2's approach where hashes are computed after mate rescue.
    fn fix_missing_hashes(all_alignments: &mut [Vec<Alignment>]) {
        for (read_idx, alignments) in all_alignments.iter_mut().enumerate() {
            for (aln_idx, aln) in alignments.iter_mut().enumerate() {
                if aln.hash == 0 {
                    // Compute hash using same formula as finalization: hash_64(read_id + aln_idx)
                    // Use read_idx as the read_id since we don't have the global batch offset here
                    aln.hash = crate::utils::hash_64(read_idx as u64 + aln_idx as u64);
                }
            }
        }
    }

    /// Write paired-end alignments to output.
    pub(super) fn write_paired_output(
        &self,
        alignments1: &[Vec<Alignment>],
        alignments2: &[Vec<Alignment>],
        batch1: &SoAReadBatch,
        batch2: &SoAReadBatch,
        output: &mut dyn Write,
    ) -> Result<usize, OrchestratorError> {
        let mut records = 0;

        for read_idx in 0..alignments1.len() {
            let alns1 = &alignments1[read_idx];
            let alns2 = &alignments2[read_idx];

            // Get sequences and qualities
            let (seq_start1, seq_len1) = batch1.read_boundaries[read_idx];
            let seq1 =
                std::str::from_utf8(&batch1.seqs[seq_start1..seq_start1 + seq_len1]).unwrap_or("");
            let qual1 =
                std::str::from_utf8(&batch1.quals[seq_start1..seq_start1 + seq_len1]).unwrap_or("");

            let (seq_start2, seq_len2) = batch2.read_boundaries[read_idx];
            let seq2 =
                std::str::from_utf8(&batch2.seqs[seq_start2..seq_start2 + seq_len2]).unwrap_or("");
            let qual2 =
                std::str::from_utf8(&batch2.quals[seq_start2..seq_start2 + seq_len2]).unwrap_or("");

            // Write R1: primary (first) + any supplementary alignments
            // BWA-MEM2 outputs primary and supplementary, but skips secondary
            for (idx, aln) in alns1.iter().enumerate() {
                // Write primary (idx == 0) or supplementary (SUPPLEMENTARY flag set)
                if idx == 0 || (aln.flag & sam_flags::SUPPLEMENTARY) != 0 {
                    writeln!(output, "{}", aln.to_sam_string_with_seq(seq1, qual1))
                        .map_err(|e| OrchestratorError::Io(e))?;
                    records += 1;
                }
            }

            // Write R2: primary (first) + any supplementary alignments
            for (idx, aln) in alns2.iter().enumerate() {
                // Write primary (idx == 0) or supplementary (SUPPLEMENTARY flag set)
                if idx == 0 || (aln.flag & sam_flags::SUPPLEMENTARY) != 0 {
                    writeln!(output, "{}", aln.to_sam_string_with_seq(seq2, qual2))
                        .map_err(|e| OrchestratorError::Io(e))?;
                    records += 1;
                }
            }
        }

        Ok(records)
    }
}

/// Find primary alignment indices for each read in a SoA result.
fn find_primary_alignments(soa: &SoAAlignmentResult) -> Vec<usize> {
    (0..soa.num_reads())
        .map(|read_idx| {
            let (start, count) = soa.read_alignment_boundaries[read_idx];
            (start..start + count)
                .find(|&idx| soa.flags[idx] & sam_flags::PROPER_PAIR != 0)
                .unwrap_or(start)
        })
        .collect()
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_mate_info() {
        // Note: PAIRED and FIRST_IN_PAIR are now set in pair_alignments_aos before calling set_mate_info
        let mut aln = Alignment {
            query_name: "read1".to_string(),
            flag: sam_flags::PAIRED | sam_flags::FIRST_IN_PAIR, // Pre-set as done in pair_alignments_aos
            ref_name: "chr1".to_string(),
            ref_id: 0,
            pos: 100,
            mapq: 60,
            score: 100,
            cigar: vec![(b'M', 100)],
            rnext: "*".to_string(),
            pnext: 0,
            tlen: 0,
            seq: String::new(),
            qual: String::new(),
            tags: Vec::new(),
            query_start: 0,
            query_end: 100,
            seed_coverage: 50,
            hash: 0,
            frac_rep: 0.0,
            is_alt: false,
        };

        let mate = Alignment {
            query_name: "read2".to_string(),
            flag: sam_flags::REVERSE,
            ref_name: "chr1".to_string(),
            ref_id: 0,
            pos: 300,
            mapq: 60,
            score: 100,
            cigar: vec![(b'M', 100)],
            rnext: "*".to_string(),
            pnext: 0,
            tlen: 0,
            seq: String::new(),
            qual: String::new(),
            tags: Vec::new(),
            query_start: 0,
            query_end: 100,
            seed_coverage: 50,
            hash: 0,
            frac_rep: 0.0,
            is_alt: false,
        };

        PairedEndOrchestrator::set_mate_info(&mut aln, &mate);

        assert_eq!(aln.rnext, "="); // Same chromosome
        assert_eq!(aln.pnext, 300);
        assert!(aln.flag & sam_flags::PAIRED != 0);
        assert!(aln.flag & sam_flags::FIRST_IN_PAIR != 0);
        assert!(aln.flag & sam_flags::MATE_REVERSE != 0);
    }
}
