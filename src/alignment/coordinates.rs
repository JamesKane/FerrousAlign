//! Centralized coordinate conversion between FM-index and chromosome space.
//!
//! This module provides a single, canonical implementation of coordinate conversion
//! to prevent bugs from duplicated logic across the codebase.
//!
//! ## BWA-MEM2 Reference
//!
//! The coordinate conversion follows BWA-MEM2's `bns_depos` and `bns_pos2rid` functions:
//! - `bns_depos` (bntseq.h): Converts FM-index position to forward strand position
//! - `bns_pos2rid` (bntseq.c): Finds which reference sequence contains a position
//!
//! ## Key Concept: Forward vs Reverse Strand
//!
//! The FM-index stores both strands of the reference:
//! - Positions in `[0, l_pac)`: Forward strand
//! - Positions in `[l_pac, 2*l_pac)`: Reverse complement strand
//!
//! For reverse strand alignments, `bns_depos` converts the position to its
//! corresponding forward strand coordinate using: `pos_f = (l_pac << 1) - 1 - pos`
//!
//! ## SAM Output Position
//!
//! SAM format requires the leftmost position on the forward reference strand.
//! For reverse strand alignments, we must use `re - 1` (alignment end) as input
//! to `bns_depos` instead of `rb` (alignment start) to get the correct leftmost position.

use crate::index::index::BwaIndex;

/// Result of coordinate conversion from FM-index to chromosome space
#[derive(Debug, Clone)]
pub struct ChromosomeCoordinates {
    /// Chromosome/contig name (e.g., "chr1")
    pub ref_name: String,
    /// Reference sequence ID (index into annotations array)
    pub ref_id: i32,
    /// 0-based position within the chromosome
    pub chr_pos: u64,
    /// Whether the alignment is on the reverse strand
    pub is_rev: bool,
}

impl Default for ChromosomeCoordinates {
    fn default() -> Self {
        Self {
            ref_name: "*".to_string(),
            ref_id: -1,
            chr_pos: 0,
            is_rev: false,
        }
    }
}

/// Convert FM-index alignment boundaries to chromosome coordinates.
///
/// This is the canonical coordinate conversion function that should be used
/// by all code paths (both STANDARD_CIGAR and DEFERRED_CIGAR pipelines).
///
/// ## Arguments
///
/// * `bwa_idx` - Reference to the BWA index containing sequence annotations
/// * `rb` - FM-index position of alignment start (region begin)
/// * `re` - FM-index position of alignment end (region end, exclusive)
///
/// ## Returns
///
/// `ChromosomeCoordinates` containing the chromosome name, ID, position, and strand.
///
/// ## BWA-MEM2 Reference
///
/// Matches logic from bwamem.cpp:1783-1785:
/// ```c
/// pos = bns_depos(bns, p->rb < bns->l_pac? p->rb : p->re - 1, &is_rev);
/// ```
///
/// For forward strand (`rb < l_pac`): use `rb` as input
/// For reverse strand (`rb >= l_pac`): use `re - 1` as input
pub fn fm_to_chromosome_coords(bwa_idx: &BwaIndex, rb: u64, re: u64) -> ChromosomeCoordinates {
    let l_pac = bwa_idx.bns.packed_sequence_length;

    // Determine strand from rb position
    let is_rev = rb >= l_pac;

    // Select input position for bns_depos
    // Forward strand: use alignment start (rb)
    // Reverse strand: use alignment end - 1 (re - 1) to get leftmost SAM position
    let depos_input = if is_rev { re.saturating_sub(1) } else { rb };

    // Convert to forward strand position
    let (pos_f, depos_is_rev) = bwa_idx.bns.bns_depos(depos_input as i64);

    // Find which reference sequence contains this position
    let rid = bwa_idx.bns.bns_pos2rid(pos_f);

    if rid >= 0 && (rid as usize) < bwa_idx.bns.annotations.len() {
        let ann = &bwa_idx.bns.annotations[rid as usize];
        let offset = ann.offset as i64;
        let chr_pos = (pos_f - offset).max(0) as u64;

        log::debug!(
            "COORD_CONVERT: rb={} re={} l_pac={} is_rev={} depos_input={} pos_f={} rid={} chr_pos={}",
            rb,
            re,
            l_pac,
            is_rev,
            depos_input,
            pos_f,
            rid,
            chr_pos
        );

        ChromosomeCoordinates {
            ref_name: ann.name.clone(),
            ref_id: rid,
            chr_pos,
            is_rev: depos_is_rev,
        }
    } else {
        log::debug!(
            "COORD_CONVERT: rb={} re={} l_pac={} -> unmapped (rid={})",
            rb,
            re,
            l_pac,
            rid
        );
        ChromosomeCoordinates::default()
    }
}

/// Convert FM-index start position and reference length to chromosome coordinates.
///
/// This is a convenience wrapper for `fm_to_chromosome_coords` when you have
/// a start position and length rather than start and end positions.
///
/// ## Arguments
///
/// * `bwa_idx` - Reference to the BWA index
/// * `start_pos` - FM-index position of alignment start
/// * `ref_len` - Length of the aligned reference segment
pub fn fm_to_chromosome_coords_with_len(
    bwa_idx: &BwaIndex,
    start_pos: u64,
    ref_len: u64,
) -> ChromosomeCoordinates {
    fm_to_chromosome_coords(bwa_idx, start_pos, start_pos + ref_len)
}

/// Check if an FM-index position is on the reverse strand.
///
/// ## Arguments
///
/// * `pos` - FM-index position
/// * `l_pac` - Length of packed sequence (half of total FM-index length)
#[inline]
pub fn is_reverse_strand(pos: u64, l_pac: u64) -> bool {
    pos >= l_pac
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_reverse_strand() {
        let l_pac = 1000;
        assert!(!is_reverse_strand(0, l_pac));
        assert!(!is_reverse_strand(500, l_pac));
        assert!(!is_reverse_strand(999, l_pac));
        assert!(is_reverse_strand(1000, l_pac));
        assert!(is_reverse_strand(1500, l_pac));
        assert!(is_reverse_strand(1999, l_pac));
    }

    #[test]
    fn test_chromosome_coordinates_default() {
        let coords = ChromosomeCoordinates::default();
        assert_eq!(coords.ref_name, "*");
        assert_eq!(coords.ref_id, -1);
        assert_eq!(coords.chr_pos, 0);
        assert!(!coords.is_rev);
    }
}
