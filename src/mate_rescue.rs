// Mate rescue module
//
// This module handles mate rescue using Smith-Waterman alignment:
// - Region calculation based on insert size distribution
// - Banded Smith-Waterman alignment
// - Rescued alignment creation

use crate::align;
use crate::index::BwaIndex;
use crate::insert_size::{InsertSizeStats, infer_orientation};

/// Mate rescue using Smith-Waterman alignment
/// Equivalent to C++ mem_matesw
/// Returns number of rescued alignments added
pub fn mem_matesw(
    bwa_idx: &BwaIndex,
    pac: &[u8], // Pre-loaded PAC data (passed once, not loaded per call)
    stats: &[InsertSizeStats; 4],
    anchor: &align::Alignment,
    mate_seq: &[u8],
    mate_qual: &str,
    mate_name: &str,
    rescued_alignments: &mut Vec<align::Alignment>,
) -> usize {
    use crate::banded_swa::BandedPairWiseSW;

    let l_pac = bwa_idx.bns.packed_sequence_length as i64;
    let l_ms = mate_seq.len() as i32;
    let min_seed_len = bwa_idx.min_seed_len;

    // Check which orientations to skip (already have good pairs)
    let mut skip = [false; 4];
    for r in 0..4 {
        skip[r] = stats[r].failed;
    }

    // Check existing mate alignments to see if we already have pairs in each orientation
    for aln in rescued_alignments.iter() {
        if aln.ref_name == anchor.ref_name {
            let (dir, dist) = infer_orientation(l_pac, anchor.pos as i64, aln.pos as i64);
            if dist >= stats[dir].low as i64 && dist <= stats[dir].high as i64 {
                skip[dir] = true;
            }
        }
    }

    // If all orientations already have consistent pairs, no need for rescue
    if skip.iter().all(|&x| x) {
        return 0;
    }

    // PAC data is now passed as parameter (loaded once per batch, not per call)
    // This eliminates catastrophic I/O: was reading 740MB × 5000+ times per batch!

    // Setup Smith-Waterman aligner (same parameters as generate_seeds)
    let sw_params = BandedPairWiseSW::new(
        4,   // o_del
        2,   // e_del
        4,   // o_ins
        2,   // e_ins
        100, // zdrop
        0,   // end_bonus
        5,   // pen_clip5 (5' clipping penalty, default=5)
        5,   // pen_clip3 (3' clipping penalty, default=5)
        align::DEFAULT_SCORING_MATRIX,
        2,  // w_match
        -4, // w_mismatch
    );

    let mut n_rescued = 0;

    // Try each orientation
    for r in 0..4 {
        if skip[r] {
            continue;
        }

        let is_rev = (r >> 1) != (r & 1); // Whether to reverse complement the mate
        let is_larger = (r >> 1) == 0; // Whether the mate has larger coordinate

        // Prepare mate sequence (reverse complement if needed)
        let seq: Vec<u8>;
        if is_rev {
            seq = mate_seq
                .iter()
                .rev()
                .map(|&b| if b < 4 { 3 - b } else { 4 })
                .collect();
        } else {
            seq = mate_seq.to_vec();
        }

        // Calculate search region
        let (rb, re) = if !is_rev {
            let rb = if is_larger {
                anchor.pos as i64 + stats[r].low as i64
            } else {
                anchor.pos as i64 - stats[r].high as i64
            };
            let re = if is_larger {
                anchor.pos as i64 + stats[r].high as i64
            } else {
                anchor.pos as i64 - stats[r].low as i64
            } + l_ms as i64;
            (rb.max(0), re.min(l_pac << 1))
        } else {
            let rb = if is_larger {
                anchor.pos as i64 + stats[r].low as i64
            } else {
                anchor.pos as i64 - stats[r].high as i64
            } - l_ms as i64;
            let re = if is_larger {
                anchor.pos as i64 + stats[r].high as i64
            } else {
                anchor.pos as i64 - stats[r].low as i64
            };
            (rb.max(0), re.min(l_pac << 1))
        };

        if rb >= re {
            continue;
        }

        // Fetch reference sequence
        let (ref_seq, adj_rb, adj_re, rid) =
            bwa_idx.bns.bns_fetch_seq(&pac, rb, (rb + re) >> 1, re);

        // Check if on same reference and region is large enough
        if rid as usize != anchor.ref_id || (adj_re - adj_rb) < min_seed_len as i64 {
            continue;
        }

        // Perform Smith-Waterman alignment
        let ref_len = ref_seq.len() as i32;
        let (out_score, cigar, _, _) = sw_params.scalar_banded_swa(
            l_ms, &seq, ref_len, &ref_seq, 100, // w (bandwidth)
            0,   // h0 (initial score)
        );

        // Check if alignment is good enough
        if out_score.score < min_seed_len || cigar.is_empty() {
            continue;
        }

        // Calculate alignment start position from CIGAR
        // The end positions are qle and tle
        // We need to calculate start positions by walking back through CIGAR
        let mut _query_consumed = 0i32;
        let mut ref_consumed = 0i32;

        for &(op, len) in &cigar {
            match op {
                0 => {
                    // Match/Mismatch
                    _query_consumed += len;
                    ref_consumed += len;
                }
                1 => {
                    // Insertion (consumes query)
                    _query_consumed += len;
                }
                2 => {
                    // Deletion (consumes reference)
                    ref_consumed += len;
                }
                _ => {}
            }
        }

        // Calculate alignment position on reference
        let tb = (out_score.target_end_pos - ref_consumed).max(0);

        // Adjust for reverse complement and reference position
        let pos = if is_rev {
            ((l_pac << 1) - (adj_rb + out_score.target_end_pos as i64)).max(0) as u64
        } else {
            (adj_rb + tb as i64).max(0) as u64
        };

        // Create alignment structure
        let mut flag = 0u16;
        if is_rev {
            flag |= align::sam_flags::REVERSE; // Reverse complement
        }

        let rescued_aln = align::Alignment {
            query_name: mate_name.to_string(),
            flag,
            ref_name: anchor.ref_name.clone(),
            ref_id: anchor.ref_id,
            pos,
            mapq: 0, // Will be calculated later
            score: out_score.score,
            cigar,
            rnext: String::from("*"),
            pnext: 0,
            tlen: 0,
            seq: String::from_utf8(
                mate_seq
                    .iter()
                    .map(|&b| match b {
                        0 => b'A',
                        1 => b'C',
                        2 => b'G',
                        3 => b'T',
                        _ => b'N',
                    })
                    .collect(),
            )
            .unwrap(),
            qual: mate_qual.to_string(),
            tags: Vec::new(),
            // Internal fields for alignment selection
            query_start: out_score.query_end_pos - out_score.query_end_pos, // Full query alignment
            query_end: out_score.query_end_pos,
            seed_coverage: 0, // Not applicable here, will be updated later
            hash: 0,          // Will be updated later
            frac_rep: 0.0,
        };

        rescued_alignments.push(rescued_aln);
        n_rescued += 1;
    }

    n_rescued
}
