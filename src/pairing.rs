// Paired-end alignment scoring module
//
// This module handles paired-end alignment scoring based on insert size distribution:
// - Position-based sorting and mate finding
// - Normal distribution scoring
// - Best pair selection with tie-breaking

use crate::align;
use crate::insert_size::InsertSizeStats;
use crate::insert_size::erfc_fn as erfc;
use crate::utils::hash_64;

// Pair information for mem_pair scoring (equivalent to C++ pair64_t)
#[derive(Debug, Clone, Copy)]
struct AlignmentInfo {
    pos_key: u64, // (ref_id << 32) | forward_position
    info: u64,    // (score << 32) | (index << 2) | (is_rev << 1) | read_number
}

// Paired alignment scoring result
#[derive(Debug, Clone, Copy)]
struct PairScore {
    idx1: usize, // Index in read1 alignments
    idx2: usize, // Index in read2 alignments
    score: i32,  // Paired alignment score
    hash: u32,   // Hash for tie-breaking
}

/// Score paired-end alignments based on insert size distribution (C++ mem_pair equivalent)
/// Returns: Option<(best_idx1, best_idx2, pair_score, sub_score)>
pub fn mem_pair(
    stats: &[InsertSizeStats; 4],
    alns1: &[align::Alignment],
    alns2: &[align::Alignment],
    match_score: i32, // opt->a (match score for log-likelihood calculation)
    pair_id: u64,
) -> Option<(usize, usize, i32, i32)> {
    if alns1.is_empty() || alns2.is_empty() {
        return None;
    }

    // Build sorted array of alignment positions (like C++ v array)
    let mut v: Vec<AlignmentInfo> = Vec::new();

    // Add alignments from read1
    for (i, aln) in alns1.iter().enumerate() {
        // Use forward-strand position directly (aln.pos is always on forward strand)
        let is_rev = (aln.flag & 0x10) != 0;
        let pos = aln.pos as i64;

        let pos_key = ((aln.ref_id as u64) << 32) | (pos as u64);
        let info = ((aln.score as u64) << 32) | ((i as u64) << 2) | ((is_rev as u64) << 1) | 0; // 0 = read1

        // DEBUG: Log positions for first few pairs
        if pair_id < 3 {
            log::debug!(
                "mem_pair: R1[{}]: aln.pos={}, is_rev={}, ref_id={}",
                i,
                aln.pos,
                is_rev,
                aln.ref_id
            );
        }

        v.push(AlignmentInfo { pos_key, info });
    }

    // Add alignments from read2
    for (i, aln) in alns2.iter().enumerate() {
        let is_rev = (aln.flag & 0x10) != 0;
        let pos = aln.pos as i64;

        let pos_key = ((aln.ref_id as u64) << 32) | (pos as u64);
        let info = ((aln.score as u64) << 32) | ((i as u64) << 2) | ((is_rev as u64) << 1) | 1; // 1 = read2

        // DEBUG: Log positions for first few pairs
        if pair_id < 3 {
            log::debug!(
                "mem_pair: R2[{}]: aln.pos={}, is_rev={}, ref_id={}",
                i,
                aln.pos,
                is_rev,
                aln.ref_id
            );
        }

        v.push(AlignmentInfo { pos_key, info });
    }

    // Sort by position (like C++ ks_introsort_128)
    v.sort_by_key(|a| a.pos_key);

    // Track last hit for each orientation combination [read][strand]
    let mut y = [-1i32; 4];

    // Array to store valid pairs (like C++ u array)
    let mut u: Vec<PairScore> = Vec::new();

    // For each alignment, look backward for compatible mates
    for i in 0..v.len() {
        for r in 0..2 {
            // Try both orientations
            let dir = ((r << 1) | ((v[i].info >> 1) & 1)) as usize; // orientation index

            if stats[dir].failed {
                continue; // Invalid orientation
            }

            let which = ((r << 1) | ((v[i].info & 1) ^ 1)) as usize; // Look for mate from other read

            if y[which] < 0 {
                continue; // No previous hits from mate
            }

            // Search backward for compatible pairs
            let mut k = y[which] as usize;
            loop {
                if k >= v.len() {
                    break;
                }

                if (v[k].info & 3) != which as u64 {
                    if k == 0 {
                        break;
                    }
                    k -= 1;
                    continue;
                }

                // Calculate distance
                let dist = (v[i].pos_key - v[k].pos_key) as i64;

                // DEBUG: Log distance checks for first few pairs
                if pair_id < 3 {
                    log::debug!(
                        "mem_pair: Checking pair i={}, k={}, dir={}, dist={}, bounds=[{}, {}]",
                        i,
                        k,
                        dir,
                        dist,
                        stats[dir].low,
                        stats[dir].high
                    );
                }

                if dist > stats[dir].high as i64 {
                    if pair_id < 3 {
                        log::debug!("mem_pair: Distance too far, breaking");
                    }
                    break; // Too far
                }

                if dist < stats[dir].low as i64 {
                    if pair_id < 3 {
                        log::debug!("mem_pair: Distance too close, continuing");
                    }
                    if k == 0 {
                        break;
                    }
                    k -= 1;
                    continue; // Too close
                }

                // Compute pairing score using normal distribution
                // q = score1 + score2 + log_prob(insert_size)
                let ns = (dist as f64 - stats[dir].avg) / stats[dir].std;

                // Log-likelihood penalty: .721 * log(2 * erfc(|ns| / sqrt(2))) * match_score
                // .721 = 1/log(4) converts to base-4 log
                let log_prob = 0.721
                    * ((2.0 * erfc(ns.abs() / std::f64::consts::SQRT_2)).ln())
                    * (match_score as f64);

                let score1 = (v[i].info >> 32) as i32;
                let score2 = (v[k].info >> 32) as i32;
                let mut q = score1 + score2 + (log_prob + 0.499) as i32;

                if q < 0 {
                    q = 0;
                }

                // Hash for tie-breaking
                let hash_input = (k as u64) << 32 | i as u64;
                let hash = (hash_64(hash_input ^ (pair_id << 8)) & 0xffffffff) as u32;

                u.push(PairScore {
                    idx1: if (v[k].info & 1) == 0 {
                        ((v[k].info >> 2) & 0x3fffffff) as usize
                    } else {
                        ((v[i].info >> 2) & 0x3fffffff) as usize
                    },
                    idx2: if (v[k].info & 1) == 1 {
                        ((v[k].info >> 2) & 0x3fffffff) as usize
                    } else {
                        ((v[i].info >> 2) & 0x3fffffff) as usize
                    },
                    score: q,
                    hash,
                });

                // DEBUG: Log when we find a valid pair
                if pair_id < 10 {
                    log::debug!(
                        "mem_pair: Found valid pair! dir={}, dist={}, score={}",
                        dir,
                        dist,
                        q
                    );
                }

                if k == 0 {
                    break;
                }
                k -= 1;
            }
        }

        y[(v[i].info & 3) as usize] = i as i32;
    }

    if u.is_empty() {
        // DEBUG: Log why no pairs were found for first few pairs
        if pair_id < 10 {
            log::debug!(
                "mem_pair: No valid pairs in u array. v.len()={}, y={:?}",
                v.len(),
                y
            );
        }
        return None; // No valid pairs found
    }

    // Sort by score (descending), then by hash
    u.sort_by(|a, b| match b.score.cmp(&a.score) {
        std::cmp::Ordering::Equal => b.hash.cmp(&a.hash),
        other => other,
    });

    // Best pair is first
    let best = &u[0];
    let sub_score = if u.len() > 1 { u[1].score } else { 0 };

    Some((best.idx1, best.idx2, best.score, sub_score))
}
