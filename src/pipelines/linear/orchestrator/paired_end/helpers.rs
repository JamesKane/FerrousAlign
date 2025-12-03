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

            if let Some((idx1, idx2, _pair_score, _sub_score)) = pairing_aos::mem_pair(
                &self.insert_stats,
                alns1,
                alns2,
                self.options.a,
                batch_start_id + read_idx as u64,
                l_pac,
            ) {
                // Mark as properly paired
                alns1[idx1].flag |= sam_flags::PROPER_PAIR;
                alns2[idx2].flag |= sam_flags::PROPER_PAIR;

                // Set mate information
                let (aln1, aln2) = (alns1[idx1].clone(), alns2[idx2].clone());
                Self::set_mate_info(&mut alns1[idx1], &aln2, true);
                Self::set_mate_info(&mut alns2[idx2], &aln1, false);
            } else if !alns1.is_empty() && !alns2.is_empty() {
                // Singleton: both mapped but not properly paired
                let (aln1, aln2) = (alns1[0].clone(), alns2[0].clone());
                Self::set_mate_info(&mut alns1[0], &aln2, true);
                Self::set_mate_info(&mut alns2[0], &aln1, false);
            }
        }
    }

    /// Set mate information on an alignment.
    pub(super) fn set_mate_info(aln: &mut Alignment, mate: &Alignment, is_first: bool) {
        aln.rnext = if mate.ref_name == aln.ref_name {
            "=".to_string()
        } else {
            mate.ref_name.clone()
        };
        aln.pnext = mate.pos;
        aln.flag |= sam_flags::PAIRED;

        if is_first {
            aln.flag |= sam_flags::FIRST_IN_PAIR;
        } else {
            aln.flag |= sam_flags::SECOND_IN_PAIR;
        }

        if (mate.flag & sam_flags::REVERSE) != 0 {
            aln.flag |= sam_flags::MATE_REVERSE;
        }

        if mate.mapq == 0 && mate.score == 0 {
            aln.flag |= sam_flags::MATE_UNMAPPED;
        }
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

        rescued
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

            // Write R1 alignments
            if let Some(aln) = alns1.first() {
                writeln!(output, "{}", aln.to_sam_string_with_seq(seq1, qual1))
                    .map_err(|e| OrchestratorError::Io(e))?;
                records += 1;
            }

            // Write R2 alignments
            if let Some(aln) = alns2.first() {
                writeln!(output, "{}", aln.to_sam_string_with_seq(seq2, qual2))
                    .map_err(|e| OrchestratorError::Io(e))?;
                records += 1;
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
        let mut aln = Alignment {
            query_name: "read1".to_string(),
            flag: 0,
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
        };

        PairedEndOrchestrator::set_mate_info(&mut aln, &mate, true);

        assert_eq!(aln.rnext, "="); // Same chromosome
        assert_eq!(aln.pnext, 300);
        assert!(aln.flag & sam_flags::PAIRED != 0);
        assert!(aln.flag & sam_flags::FIRST_IN_PAIR != 0);
        assert!(aln.flag & sam_flags::MATE_REVERSE != 0);
    }
}
