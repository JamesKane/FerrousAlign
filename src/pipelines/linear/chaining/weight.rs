//! Chain weight calculation and gap computation.
//!
//! Implements C++ mem_chain_weight (bwamem.cpp:429-448) for both AoS and SoA formats.

use crate::pipelines::linear::mem_opt::MemOpt;
use crate::pipelines::linear::seeding::{Seed, SoASeedBatch};

use super::types::{Chain, SoAChainBatch};

/// Calculate chain weight based on seed coverage.
///
/// Implements C++ mem_chain_weight (bwamem.cpp:429-448).
///
/// Weight = minimum of query coverage and reference coverage.
/// This accounts for non-overlapping seed lengths in the chain.
///
/// # Returns
/// Tuple of (weight, l_rep) where l_rep is the length of repetitive seeds.
pub fn calculate_chain_weight(chain: &Chain, seeds: &[Seed], opt: &MemOpt) -> (i32, i32) {
    if chain.seeds.is_empty() {
        return (0, 0);
    }

    let mut query_cov = 0;
    let mut last_qe = -1i32;
    let mut l_rep = 0; // Length of repetitive seeds

    for &seed_idx in &chain.seeds {
        let seed = &seeds[seed_idx];
        let qb = seed.query_pos;
        let qe = seed.query_pos + seed.len;

        if qb > last_qe {
            query_cov += seed.len;
        } else if qe > last_qe {
            query_cov += qe - last_qe;
        }
        last_qe = last_qe.max(qe);

        // Check for repetitive seeds: if interval_size > max_occ
        if seed.interval_size > opt.max_occ as u64 {
            l_rep += seed.len;
        }
    }

    let mut ref_cov = 0;
    let mut last_re = 0u64;

    for &seed_idx in &chain.seeds {
        let seed = &seeds[seed_idx];
        let rb = seed.ref_pos;
        let re = rb + seed.len as u64;

        if rb > last_re {
            ref_cov += seed.len;
        } else if re > last_re {
            ref_cov += (re - last_re) as i32;
        }

        last_re = last_re.max(re);
    }

    (query_cov.min(ref_cov), l_rep)
}

/// Calculate chain weight based on seed coverage (SoA-aware version).
///
/// Implements C++ mem_chain_weight (bwamem.cpp:429-448).
///
/// Weight = minimum of query coverage and reference coverage.
/// This accounts for non-overlapping seed lengths in the chain.
///
/// # Returns
/// Tuple of (weight, l_rep) where l_rep is the length of repetitive seeds.
pub fn calculate_chain_weight_soa(
    chain_global_idx: usize,
    soa_chain_batch: &SoAChainBatch,
    soa_seed_batch: &SoASeedBatch,
    opt: &MemOpt,
) -> (i32, i32) {
    let (chain_seed_start_idx, num_seeds_in_chain) =
        soa_chain_batch.chain_seed_boundaries[chain_global_idx];

    if num_seeds_in_chain == 0 {
        return (0, 0);
    }

    let mut query_cov = 0;
    let mut last_qe = -1i32;
    let mut l_rep = 0; // Length of repetitive seeds

    for i in 0..num_seeds_in_chain {
        let global_seed_idx = soa_chain_batch.seeds_indices[chain_seed_start_idx + i];
        let qb = soa_seed_batch.query_pos[global_seed_idx];
        let qe = qb + soa_seed_batch.len[global_seed_idx];
        let seed_len = soa_seed_batch.len[global_seed_idx];

        if qb > last_qe {
            query_cov += seed_len;
        } else if qe > last_qe {
            query_cov += qe - last_qe;
        }
        last_qe = last_qe.max(qe);

        // Check for repetitive seeds: if interval_size > max_occ
        if soa_seed_batch.interval_size[global_seed_idx] > opt.max_occ as u64 {
            l_rep += seed_len;
        }
    }

    let mut ref_cov = 0;
    let mut last_re = 0u64;

    for i in 0..num_seeds_in_chain {
        let global_seed_idx = soa_chain_batch.seeds_indices[chain_seed_start_idx + i];
        let rb = soa_seed_batch.ref_pos[global_seed_idx];
        let seed_len = soa_seed_batch.len[global_seed_idx];
        let re = rb + seed_len as u64;

        if rb > last_re {
            ref_cov += seed_len;
        } else if re > last_re {
            ref_cov += (re - last_re) as i32;
        }

        last_re = last_re.max(re);
    }

    (query_cov.min(ref_cov), l_rep)
}

/// Calculate maximum gap size for a given query length.
///
/// Matches C++ bwamem.cpp:66 cal_max_gap().
#[inline]
pub fn cal_max_gap(opt: &MemOpt, qlen: i32) -> i32 {
    let l_del = ((qlen * opt.a - opt.o_del) as f64 / opt.e_del as f64 + 1.0) as i32;
    let l_ins = ((qlen * opt.a - opt.o_ins) as f64 / opt.e_ins as f64 + 1.0) as i32;

    let l = if l_del > l_ins { l_del } else { l_ins };
    let l = if l > 1 { l } else { 1 };

    if l < (opt.w << 1) {
        l
    } else {
        opt.w << 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cal_max_gap() {
        let opt = MemOpt::default();
        // For a typical 150bp read
        let gap = cal_max_gap(&opt, 150);
        assert!(gap > 0);
        assert!(gap <= opt.w << 1);
    }

    #[test]
    fn test_cal_max_gap_short_read() {
        let opt = MemOpt::default();
        let gap = cal_max_gap(&opt, 10);
        assert!(gap >= 1);
    }
}
