use crate::alignment::banded_swa::BandedPairWiseSW;
use crate::alignment::banded_swa::scalar::implementation::scalar_banded_swa;
use crate::core::alignment::banded_swa::types::{ExtensionDirection, ExtensionResult};
use crate::core::alignment::banded_swa::utils::{reverse_cigar, reverse_sequence};

/// Directional Smith-Waterman alignment for left/right seed extensions
/// Matches C++ bwa-mem2 separate LEFT/RIGHT extension model (bwamem.cpp:2229-2418)
///
/// # Arguments
/// * `direction` - LEFT (5' → seed, reversed) or RIGHT (seed → 3', forward)
/// * Other args same as scalar_banded_swa
///
/// # Returns
/// ExtensionResult with both local and global scores for clipping penalty decision
pub fn scalar_banded_swa_directional(
    sw: &BandedPairWiseSW, // Reference to BandedPairWiseSW
    direction: ExtensionDirection,
    qlen: i32,
    query: &[u8],
    tlen: i32,
    target: &[u8],
    w: i32,
    h0: i32,
) -> ExtensionResult {
    // For LEFT extension: reverse both query and target (C++ bwamem.cpp:2278)
    let (query_to_align, target_to_align) = if direction == ExtensionDirection::Left {
        (reverse_sequence(query), reverse_sequence(target))
    } else {
        (query.to_vec(), target.to_vec())
    };

    // Run standard Smith-Waterman on potentially reversed sequences
    let (out_score, cigar, ref_aligned, query_aligned) =
        scalar_banded_swa(sw, qlen, &query_to_align, tlen, &target_to_align, w, h0);

    // For LEFT extension: reverse CIGAR back to forward orientation
    let final_cigar = if direction == ExtensionDirection::Left {
        reverse_cigar(&cigar)
    } else {
        cigar.to_vec()
    };

    // Apply clipping penalty decision
    let clipping_penalty = match direction {
        ExtensionDirection::Left => sw.pen_clip5(),
        ExtensionDirection::Right => sw.pen_clip3(),
    };

    // Calculate extension lengths from CIGAR
    let query_ext_len: i32 = final_cigar
        .iter()
        .filter_map(|&(op, len)| match op as char {
            'M' | 'I' | 'S' => Some(len),
            _ => None,
        })
        .sum();

    let target_ext_len: i32 = final_cigar
        .iter()
        .filter_map(|&(op, len)| match op as char {
            'M' | 'D' => Some(len),
            _ => None,
        })
        .sum();

    // Determine if we should clip based on clipping penalty
    // C++ logic (bwamem.cpp:2498, 2715): if (gscore <= 0 || gscore <= score - pen_clip)
    let should_clip = out_score.global_score <= 0
        || out_score.global_score <= (out_score.score - clipping_penalty);

    log::debug!(
        "{:?} extension: local_score={}, global_score={}, pen_clip={}, threshold={}, should_clip={}",
        direction,
        out_score.score,
        out_score.global_score,
        clipping_penalty,
        out_score.score - clipping_penalty,
        should_clip
    );

    ExtensionResult {
        local_score: out_score.score,
        global_score: out_score.global_score,
        query_ext_len,
        target_ext_len,
        global_target_len: out_score.gtarget_end_pos,
        should_clip,
        cigar: final_cigar,
        ref_aligned,
        query_aligned,
    }
}
