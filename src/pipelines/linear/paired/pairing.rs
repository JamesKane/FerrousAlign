// Paired-end alignment scoring module
//
// This module handles paired-end alignment scoring based on insert size distribution:
// - Position-based sorting and mate finding
// - Normal distribution scoring
// - Best pair selection with tie-breaking
//
// COORDINATE SYSTEM NOTE:
// BWA-MEM2 uses a "bidirectional" coordinate system where the reference is conceptually
// stored in both orientations: [0, l_pac) for forward strand, [l_pac, 2*l_pac) for reverse.
// - Forward strand alignments: position = leftmost coordinate
// - Reverse strand alignments: position = (2*l_pac - 1) - rightmost coordinate
//
// SAM coordinates always use the leftmost position on the forward strand.
// This module converts SAM coordinates to bidirectional coordinates for distance calculation.

use super::super::finalization::Alignment;
use super::super::finalization::sam_flags;
use super::insert_size::InsertSizeStats;
use super::insert_size::erfc_fn as erfc;
use crate::utils::hash_64;

/// Information about a single alignment for pair scoring.
/// Stores position in BWA-MEM2's bidirectional coordinate space for correct distance calculation.
#[derive(Debug, Clone, Copy)]
struct AlignmentForPairing {
    /// Sort key: (reference_id << 32) | forward_normalized_position
    /// The position is normalized to [0, l_pac) for sorting, regardless of strand.
    sort_key: u64,

    /// Packed alignment metadata:
    /// - bits [63:32]: alignment score
    /// - bits [31:2]:  index in original alignment array
    /// - bit  [1]:     1 if alignment is in reverse half of bidirectional space (bidir_pos >= l_pac)
    /// - bit  [0]:     read number (0 = read1, 1 = read2)
    packed_info: u64,
}

/// Result of scoring a candidate pair
#[derive(Debug, Clone, Copy)]
struct CandidatePairScore {
    read1_alignment_idx: usize,
    read2_alignment_idx: usize,
    combined_score: i32,
    tiebreak_hash: u32,
}

/// Convert SAM position to BWA-MEM2 bidirectional coordinate.
///
/// In BWA-MEM2's bidirectional index:
/// - Forward strand: bidir_pos = sam_pos (leftmost coordinate)
/// - Reverse strand: bidir_pos = (2*l_pac - 1) - rightmost_coordinate
///
/// This ensures that distance calculations work correctly for pairs on different strands.
///
/// # Arguments
/// * `sam_pos` - SAM position (0-based, leftmost on forward strand)
/// * `alignment_length` - Length of alignment on reference (from CIGAR)
/// * `is_reverse_strand` - True if alignment is on reverse strand (SAM flag 0x10)
/// * `l_pac` - Length of packed reference (forward strand only)
///
/// # Returns
/// Position in bidirectional coordinate space [0, 2*l_pac)
#[inline]
fn sam_pos_to_bidirectional(
    sam_pos: u64,
    alignment_length: i32,
    is_reverse_strand: bool,
    l_pac: i64,
) -> i64 {
    if is_reverse_strand {
        // For reverse strand: use rightmost position, then map to [l_pac, 2*l_pac)
        // rightmost = sam_pos + alignment_length - 1
        // bidir_pos = (2*l_pac - 1) - rightmost
        let rightmost_pos = sam_pos as i64 + alignment_length as i64 - 1;
        (l_pac << 1) - 1 - rightmost_pos
    } else {
        // Forward strand: position is already correct
        sam_pos as i64
    }
}

/// Convert bidirectional position to forward-normalized position for sorting.
///
/// This maps positions from [l_pac, 2*l_pac) back to [0, l_pac) so that
/// alignments on opposite strands at the same genomic location sort together.
///
/// # Arguments
/// * `bidir_pos` - Position in bidirectional space [0, 2*l_pac)
/// * `l_pac` - Length of packed reference
///
/// # Returns
/// Forward-normalized position in [0, l_pac)
#[inline]
fn bidirectional_to_forward_normalized(bidir_pos: i64, l_pac: i64) -> i64 {
    if bidir_pos >= l_pac {
        // Map from reverse half [l_pac, 2*l_pac) back to [0, l_pac)
        (l_pac << 1) - 1 - bidir_pos
    } else {
        bidir_pos
    }
}

/// Score paired-end alignments based on insert size distribution.
///
/// This is the Rust equivalent of BWA-MEM2's `mem_pair()` function.
/// It finds the best pair of alignments (one from each read) based on:
/// 1. Compatible orientation (FF, FR, RF, or RR)
/// 2. Insert size within expected distribution
/// 3. Combined alignment score with insert size penalty
///
/// # Arguments
/// * `stats` - Insert size statistics for each orientation [FF, FR, RF, RR]
/// * `alns1` - Alignments for read 1
/// * `alns2` - Alignments for read 2
/// * `match_score` - Match score parameter (opt->a) for log-likelihood calculation
/// * `pair_id` - Unique pair identifier for deterministic tie-breaking
/// * `l_pac` - Length of packed reference sequence (for coordinate conversion)
///
/// # Returns
/// `Some((best_idx1, best_idx2, pair_score, sub_score))` if a valid pair is found,
/// where sub_score is the second-best pair score for MAPQ calculation.
/// Returns `None` if no valid pairs exist.
pub fn mem_pair(
    stats: &[InsertSizeStats; 4],
    alns1: &[Alignment],
    alns2: &[Alignment],
    match_score: i32,
    pair_id: u64,
    l_pac: i64,
) -> Option<(usize, usize, i32, i32)> {
    if alns1.is_empty() || alns2.is_empty() {
        return None;
    }

    // Build sorted array of alignment positions in bidirectional coordinates
    // This matches BWA-MEM2's `v` array in mem_pair()
    let mut alignments_sorted: Vec<AlignmentForPairing> =
        Vec::with_capacity(alns1.len() + alns2.len());

    // Add alignments from read1
    for (alignment_idx, aln) in alns1.iter().enumerate() {
        let is_reverse = (aln.flag & sam_flags::REVERSE) != 0;
        let alignment_length = aln.reference_length();

        // Convert SAM position to bidirectional coordinate
        let bidir_pos = sam_pos_to_bidirectional(aln.pos, alignment_length, is_reverse, l_pac);

        // Forward-normalize for sorting (so opposite-strand pairs at same location sort together)
        let fwd_normalized_pos = bidirectional_to_forward_normalized(bidir_pos, l_pac);

        // Track if this position is in the reverse half of bidirectional space
        let is_in_reverse_half = bidir_pos >= l_pac;

        let sort_key = ((aln.ref_id as u64) << 32) | (fwd_normalized_pos as u64);
        let packed_info = ((aln.score as u64) << 32)
            | ((alignment_idx as u64) << 2)
            | ((is_in_reverse_half as u64) << 1); // 0 = read1

        log::trace!(
            "mem_pair R1[{}]: sam_pos={}, ref_len={}, is_rev={}, bidir_pos={}, fwd_norm={}, ref_id={}",
            alignment_idx,
            aln.pos,
            alignment_length,
            is_reverse,
            bidir_pos,
            fwd_normalized_pos,
            aln.ref_id
        );

        alignments_sorted.push(AlignmentForPairing {
            sort_key,
            packed_info,
        });
    }

    // Add alignments from read2
    for (alignment_idx, aln) in alns2.iter().enumerate() {
        let is_reverse = (aln.flag & sam_flags::REVERSE) != 0;
        let alignment_length = aln.reference_length();

        let bidir_pos = sam_pos_to_bidirectional(aln.pos, alignment_length, is_reverse, l_pac);
        let fwd_normalized_pos = bidirectional_to_forward_normalized(bidir_pos, l_pac);
        let is_in_reverse_half = bidir_pos >= l_pac;

        let sort_key = ((aln.ref_id as u64) << 32) | (fwd_normalized_pos as u64);
        let packed_info = ((aln.score as u64) << 32)
            | ((alignment_idx as u64) << 2)
            | ((is_in_reverse_half as u64) << 1)
            | 1; // 1 = read2

        log::trace!(
            "mem_pair R2[{}]: sam_pos={}, ref_len={}, is_rev={}, bidir_pos={}, fwd_norm={}, ref_id={}",
            alignment_idx,
            aln.pos,
            alignment_length,
            is_reverse,
            bidir_pos,
            fwd_normalized_pos,
            aln.ref_id
        );

        alignments_sorted.push(AlignmentForPairing {
            sort_key,
            packed_info,
        });
    }

    // Sort by position (matches BWA-MEM2's ks_introsort_128)
    alignments_sorted.sort_by_key(|a| a.sort_key);

    // Track last seen alignment index for each (read_number, strand_half) combination
    // Index encoding: (strand_half << 1) | read_number
    // - 0: read1 in forward half
    // - 1: read2 in forward half
    // - 2: read1 in reverse half
    // - 3: read2 in reverse half
    let mut last_seen_idx: [i32; 4] = [-1; 4];

    // Collect valid candidate pairs
    let mut candidate_pairs: Vec<CandidatePairScore> = Vec::new();

    // For each alignment, look backward for compatible mates
    for current_idx in 0..alignments_sorted.len() {
        let current = &alignments_sorted[current_idx];

        // Try both possible mate strand configurations
        for mate_strand_config in 0..2 {
            // Calculate orientation index for insert size stats lookup
            // Orientation: (mate_strand_half << 1) | current_strand_half
            let current_strand_half = (current.packed_info >> 1) & 1;
            let orientation_idx = ((mate_strand_config << 1) | current_strand_half) as usize;

            if stats[orientation_idx].failed {
                continue; // This orientation doesn't have valid statistics
            }

            // Look for mate from the other read with the specified strand configuration
            let current_read_num = current.packed_info & 1;
            let mate_read_num = current_read_num ^ 1;
            let mate_lookup_key = ((mate_strand_config << 1) | mate_read_num) as usize;

            if last_seen_idx[mate_lookup_key] < 0 {
                continue; // No previous alignments from mate read with this strand
            }

            // Search backward through previous alignments for compatible pairs
            let mut search_idx = last_seen_idx[mate_lookup_key] as usize;
            loop {
                if search_idx >= alignments_sorted.len() {
                    break;
                }

                let candidate_mate = &alignments_sorted[search_idx];

                // Verify this is the right read/strand combination
                if (candidate_mate.packed_info & 3) != mate_lookup_key as u64 {
                    if search_idx == 0 {
                        break;
                    }
                    search_idx -= 1;
                    continue;
                }

                // Calculate genomic distance between alignments
                // Since both positions are forward-normalized in sort_key, the distance
                // represents the genomic span between them
                let distance = (current.sort_key as i64) - (candidate_mate.sort_key as i64);

                log::trace!(
                    "mem_pair: Checking pair current_idx={}, search_idx={}, orientation={}, distance={}, bounds=[{}, {}]",
                    current_idx,
                    search_idx,
                    orientation_idx,
                    distance,
                    stats[orientation_idx].low,
                    stats[orientation_idx].high
                );

                // Check distance bounds
                if distance > stats[orientation_idx].high as i64 {
                    log::trace!(
                        "mem_pair: Distance {} exceeds upper bound {}, stopping search",
                        distance,
                        stats[orientation_idx].high
                    );
                    break; // Too far apart, stop searching
                }

                if distance < stats[orientation_idx].low as i64 {
                    log::trace!(
                        "mem_pair: Distance {} below lower bound {}, continuing",
                        distance,
                        stats[orientation_idx].low
                    );
                    if search_idx == 0 {
                        break;
                    }
                    search_idx -= 1;
                    continue; // Too close, try next candidate
                }

                // Valid pair found! Calculate combined score with insert size penalty.
                //
                // Formula from BWA-MEM2 (bwamem_pair.cpp:321):
                // q = score1 + score2 + 0.721 * log(2 * erfc(|ns| / sqrt(2))) * opt->a
                //
                // Where:
                // - ns = (distance - mean) / stddev (normalized insert size)
                // - 0.721 = 1/log(4) converts natural log to base-4
                // - erfc is the complementary error function
                let normalized_insert_size =
                    (distance as f64 - stats[orientation_idx].avg) / stats[orientation_idx].std;

                let insert_size_log_penalty = 0.721
                    * (2.0f64 * erfc(normalized_insert_size.abs() / std::f64::consts::SQRT_2)).ln()
                    * (match_score as f64);

                let current_score = (current.packed_info >> 32) as i32;
                let mate_score = (candidate_mate.packed_info >> 32) as i32;
                let mut combined_score =
                    current_score + mate_score + (insert_size_log_penalty + 0.499) as i32;

                if combined_score < 0 {
                    combined_score = 0;
                }

                // Generate deterministic hash for tie-breaking
                let hash_input = (search_idx as u64) << 32 | current_idx as u64;
                let tiebreak_hash = (hash_64(hash_input ^ (pair_id << 8)) & 0xffffffff) as u32;

                // Extract original alignment indices
                let current_alignment_idx = ((current.packed_info >> 2) & 0x3fffffff) as usize;
                let mate_alignment_idx = ((candidate_mate.packed_info >> 2) & 0x3fffffff) as usize;

                // Determine which is read1 and which is read2
                let (read1_idx, read2_idx) = if (candidate_mate.packed_info & 1) == 0 {
                    (mate_alignment_idx, current_alignment_idx)
                } else {
                    (current_alignment_idx, mate_alignment_idx)
                };

                candidate_pairs.push(CandidatePairScore {
                    read1_alignment_idx: read1_idx,
                    read2_alignment_idx: read2_idx,
                    combined_score,
                    tiebreak_hash,
                });

                log::trace!(
                    "mem_pair: Valid pair found! orientation={orientation_idx}, distance={distance}, combined_score={combined_score}"
                );

                if search_idx == 0 {
                    break;
                }
                search_idx -= 1;
            }
        }

        // Update last seen index for this read/strand combination
        let current_lookup_key = (current.packed_info & 3) as usize;
        last_seen_idx[current_lookup_key] = current_idx as i32;
    }

    if candidate_pairs.is_empty() {
        log::trace!(
            "mem_pair: No valid pairs found. alignments_sorted.len()={}, last_seen={:?}",
            alignments_sorted.len(),
            last_seen_idx
        );
        return None;
    }

    // Sort by score (descending), then by hash for deterministic tie-breaking
    candidate_pairs.sort_by(|a, b| match b.combined_score.cmp(&a.combined_score) {
        std::cmp::Ordering::Equal => b.tiebreak_hash.cmp(&a.tiebreak_hash),
        other => other,
    });

    let best_pair = &candidate_pairs[0];
    let second_best_score = if candidate_pairs.len() > 1 {
        candidate_pairs[1].combined_score
    } else {
        0
    };

    Some((
        best_pair.read1_alignment_idx,
        best_pair.read2_alignment_idx,
        best_pair.combined_score,
        second_best_score,
    ))
}

/// SoA-native paired-end alignment scoring
///
/// This function scores all pairs in a batch without converting to Alignment structs.
/// It modifies the SoA flags in-place to mark primary/secondary alignments.
///
/// # Arguments
/// * `soa_r1` - Mutable SoA alignment results for read 1
/// * `soa_r2` - Mutable SoA alignment results for read 2
/// * `stats` - Insert size statistics for each orientation [FF, FR, RF, RR]
/// * `match_score` - Match score parameter (opt->a) for log-likelihood calculation
/// * `l_pac` - Length of packed reference sequence
///
/// # Returns
/// Vectors of primary alignment indices for R1 and R2 (one per read pair)
pub fn pair_alignments_soa(
    soa_r1: &mut crate::pipelines::linear::batch_extension::types::SoAAlignmentResult,
    soa_r2: &mut crate::pipelines::linear::batch_extension::types::SoAAlignmentResult,
    stats: &[InsertSizeStats; 4],
    match_score: i32,
    l_pac: i64,
) -> (Vec<usize>, Vec<usize>) {
    let num_reads = soa_r1.num_reads();
    let mut primary_r1 = Vec::with_capacity(num_reads);
    let mut primary_r2 = Vec::with_capacity(num_reads);

    for read_idx in 0..num_reads {
        let (r1_start, r1_count) = soa_r1.read_alignment_boundaries[read_idx];
        let (r2_start, r2_count) = soa_r2.read_alignment_boundaries[read_idx];

        if r1_count == 0 || r2_count == 0 {
            // No alignments for one or both reads
            // Mark all as secondary if any exist
            for offset in 0..r1_count {
                let idx = r1_start + offset;
                soa_r1.flags[idx] |= sam_flags::SECONDARY;
            }
            for offset in 0..r2_count {
                let idx = r2_start + offset;
                soa_r2.flags[idx] |= sam_flags::SECONDARY;
            }
            primary_r1.push(r1_start); // Use first alignment as placeholder
            primary_r2.push(r2_start);
            continue;
        }

        // Use read index for deterministic tie-breaking
        let pair_id = read_idx as u64;

        // Find best pair using SoA data
        if let Some((best_r1_offset, best_r2_offset, _, _)) =
            find_best_pair_soa(soa_r1, soa_r2, read_idx, stats, match_score, pair_id, l_pac)
        {
            let best_r1_idx = r1_start + best_r1_offset;
            let best_r2_idx = r2_start + best_r2_offset;

            // Clear secondary/supplementary flags for primary pair
            soa_r1.flags[best_r1_idx] &= !(sam_flags::SECONDARY | sam_flags::SUPPLEMENTARY);
            soa_r2.flags[best_r2_idx] &= !(sam_flags::SECONDARY | sam_flags::SUPPLEMENTARY);

            // Mark all other alignments as secondary
            for offset in 0..r1_count {
                let idx = r1_start + offset;
                if idx != best_r1_idx {
                    soa_r1.flags[idx] |= sam_flags::SECONDARY;
                }
            }
            for offset in 0..r2_count {
                let idx = r2_start + offset;
                if idx != best_r2_idx {
                    soa_r2.flags[idx] |= sam_flags::SECONDARY;
                }
            }

            primary_r1.push(best_r1_idx);
            primary_r2.push(best_r2_idx);
        } else {
            // No valid pairs found - use first alignments as primaries
            soa_r1.flags[r1_start] &= !(sam_flags::SECONDARY | sam_flags::SUPPLEMENTARY);
            soa_r2.flags[r2_start] &= !(sam_flags::SECONDARY | sam_flags::SUPPLEMENTARY);

            // Mark others as secondary
            for offset in 1..r1_count {
                let idx = r1_start + offset;
                soa_r1.flags[idx] |= sam_flags::SECONDARY;
            }
            for offset in 1..r2_count {
                let idx = r2_start + offset;
                soa_r2.flags[idx] |= sam_flags::SECONDARY;
            }

            primary_r1.push(r1_start);
            primary_r2.push(r2_start);
        }
    }

    (primary_r1, primary_r2)
}

/// Find the best pair of alignments for a single read pair in SoA format
///
/// Returns (r1_offset, r2_offset, pair_score, sub_score) if a valid pair is found.
/// Offsets are relative to the read's alignment start in the SoA buffers.
fn find_best_pair_soa(
    soa_r1: &crate::pipelines::linear::batch_extension::types::SoAAlignmentResult,
    soa_r2: &crate::pipelines::linear::batch_extension::types::SoAAlignmentResult,
    read_idx: usize,
    stats: &[InsertSizeStats; 4],
    match_score: i32,
    pair_id: u64,
    l_pac: i64,
) -> Option<(usize, usize, i32, i32)> {
    let (r1_start, r1_count) = soa_r1.read_alignment_boundaries[read_idx];
    let (r2_start, r2_count) = soa_r2.read_alignment_boundaries[read_idx];

    // Build sorted array of alignment positions (same structure as AoS version)
    let mut alignments_sorted: Vec<AlignmentForPairing> = Vec::with_capacity(r1_count + r2_count);

    // Add R1 alignments
    for offset in 0..r1_count {
        let idx = r1_start + offset;
        let is_reverse = (soa_r1.flags[idx] & sam_flags::REVERSE) != 0;
        let alignment_length = reference_length_from_cigar_soa_pairing(soa_r1, idx);

        let bidir_pos =
            sam_pos_to_bidirectional(soa_r1.positions[idx], alignment_length, is_reverse, l_pac);
        let fwd_normalized_pos = bidirectional_to_forward_normalized(bidir_pos, l_pac);
        let is_in_reverse_half = bidir_pos >= l_pac;

        let sort_key = ((soa_r1.ref_ids[idx] as u64) << 32) | (fwd_normalized_pos as u64);
        let packed_info = ((soa_r1.scores[idx] as u64) << 32)
            | ((offset as u64) << 2)
            | ((is_in_reverse_half as u64) << 1); // 0 = read1

        alignments_sorted.push(AlignmentForPairing {
            sort_key,
            packed_info,
        });
    }

    // Add R2 alignments
    for offset in 0..r2_count {
        let idx = r2_start + offset;
        let is_reverse = (soa_r2.flags[idx] & sam_flags::REVERSE) != 0;
        let alignment_length = reference_length_from_cigar_soa_pairing(soa_r2, idx);

        let bidir_pos =
            sam_pos_to_bidirectional(soa_r2.positions[idx], alignment_length, is_reverse, l_pac);
        let fwd_normalized_pos = bidirectional_to_forward_normalized(bidir_pos, l_pac);
        let is_in_reverse_half = bidir_pos >= l_pac;

        let sort_key = ((soa_r2.ref_ids[idx] as u64) << 32) | (fwd_normalized_pos as u64);
        let packed_info = ((soa_r2.scores[idx] as u64) << 32)
            | ((offset as u64) << 2)
            | ((is_in_reverse_half as u64) << 1)
            | 1; // 1 = read2

        alignments_sorted.push(AlignmentForPairing {
            sort_key,
            packed_info,
        });
    }

    // Sort by position
    alignments_sorted.sort_by_key(|a| a.sort_key);

    // Track last seen alignment index for each (read_number, strand_half) combination
    let mut last_seen_idx: [i32; 4] = [-1; 4];

    // Collect valid candidate pairs
    let mut candidate_pairs: Vec<CandidatePairScore> = Vec::new();

    // For each alignment, look backward for compatible mates
    for current_idx in 0..alignments_sorted.len() {
        let current = &alignments_sorted[current_idx];

        // Try both possible mate strand configurations
        for mate_strand_config in 0..2 {
            let current_strand_half = (current.packed_info >> 1) & 1;
            let orientation_idx = ((mate_strand_config << 1) | current_strand_half) as usize;

            if stats[orientation_idx].failed {
                continue;
            }

            let current_read_num = current.packed_info & 1;
            let mate_read_num = current_read_num ^ 1;
            let mate_lookup_key = ((mate_strand_config << 1) | mate_read_num) as usize;

            if last_seen_idx[mate_lookup_key] < 0 {
                continue;
            }

            // Search backward for compatible pairs
            let mut search_idx = last_seen_idx[mate_lookup_key] as usize;
            loop {
                if search_idx >= alignments_sorted.len() {
                    break;
                }

                let candidate_mate = &alignments_sorted[search_idx];

                if (candidate_mate.packed_info & 3) != mate_lookup_key as u64 {
                    if search_idx == 0 {
                        break;
                    }
                    search_idx -= 1;
                    continue;
                }

                let distance = (current.sort_key as i64) - (candidate_mate.sort_key as i64);

                if distance > stats[orientation_idx].high as i64 {
                    break;
                }

                if distance < stats[orientation_idx].low as i64 {
                    if search_idx == 0 {
                        break;
                    }
                    search_idx -= 1;
                    continue;
                }

                // Valid pair found - calculate combined score
                let normalized_insert_size =
                    (distance as f64 - stats[orientation_idx].avg) / stats[orientation_idx].std;

                let insert_size_log_penalty = 0.721
                    * (2.0f64 * erfc(normalized_insert_size.abs() / std::f64::consts::SQRT_2)).ln()
                    * (match_score as f64);

                let current_score = (current.packed_info >> 32) as i32;
                let mate_score = (candidate_mate.packed_info >> 32) as i32;
                let mut combined_score =
                    current_score + mate_score + (insert_size_log_penalty + 0.499) as i32;

                if combined_score < 0 {
                    combined_score = 0;
                }

                let hash_input = (search_idx as u64) << 32 | current_idx as u64;
                let tiebreak_hash = (hash_64(hash_input ^ (pair_id << 8)) & 0xffffffff) as u32;

                let current_alignment_idx = ((current.packed_info >> 2) & 0x3fffffff) as usize;
                let mate_alignment_idx = ((candidate_mate.packed_info >> 2) & 0x3fffffff) as usize;

                let (read1_idx, read2_idx) = if (candidate_mate.packed_info & 1) == 0 {
                    (mate_alignment_idx, current_alignment_idx)
                } else {
                    (current_alignment_idx, mate_alignment_idx)
                };

                candidate_pairs.push(CandidatePairScore {
                    read1_alignment_idx: read1_idx,
                    read2_alignment_idx: read2_idx,
                    combined_score,
                    tiebreak_hash,
                });

                if search_idx == 0 {
                    break;
                }
                search_idx -= 1;
            }
        }

        let current_lookup_key = (current.packed_info & 3) as usize;
        last_seen_idx[current_lookup_key] = current_idx as i32;
    }

    if candidate_pairs.is_empty() {
        return None;
    }

    // Sort by score (descending), then by hash
    candidate_pairs.sort_by(|a, b| match b.combined_score.cmp(&a.combined_score) {
        std::cmp::Ordering::Equal => b.tiebreak_hash.cmp(&a.tiebreak_hash),
        other => other,
    });

    let best_pair = &candidate_pairs[0];
    let second_best_score = if candidate_pairs.len() > 1 {
        candidate_pairs[1].combined_score
    } else {
        0
    };

    Some((
        best_pair.read1_alignment_idx,
        best_pair.read2_alignment_idx,
        best_pair.combined_score,
        second_best_score,
    ))
}

/// Calculate reference length from CIGAR string in SoA format (for pairing module)
///
/// Duplicate of the function in insert_size.rs to avoid circular dependencies.
fn reference_length_from_cigar_soa_pairing(
    soa: &crate::pipelines::linear::batch_extension::types::SoAAlignmentResult,
    aln_idx: usize,
) -> i32 {
    let (cigar_start, cigar_count) = soa.cigar_boundaries[aln_idx];
    let mut ref_len = 0i32;

    for offset in 0..cigar_count {
        let idx = cigar_start + offset;
        let op = soa.cigar_ops[idx];
        let len = soa.cigar_lens[idx];

        // M, D, N, =, X consume reference
        if op == b'M' || op == b'D' || op == b'N' || op == b'=' || op == b'X' {
            ref_len += len;
        }
    }

    ref_len
}
