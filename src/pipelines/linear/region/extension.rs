//! Chain extension to alignment regions.
//!
//! Implements the score-only extension phase matching BWA-MEM2's
//! `mem_chain2aln_across_reads_V2` function.

use super::super::chaining::{cal_max_gap, Chain};
use super::super::index::index::BwaIndex;
use super::super::mem_opt::MemOpt;
use super::super::seeding::Seed;
use super::super::batch_extension::dispatch::execute_batch_simd_scoring;
use super::super::batch_extension::types::{ExtensionDirection, ExtensionJobBatch};
use super::merge::merge_scores_to_regions;
use super::types::{ChainExtensionMapping, ScoreOnlyExtensionResult, SeedExtensionMapping};
use crate::compute::ComputeBackend;
use crate::core::alignment::banded_swa::BandedPairWiseSW;
use crate::core::alignment::banded_swa::OutScore;

/// Extension job with sequence data
///
/// Used for building SIMD batch scoring tuples.
#[derive(Debug, Clone)]
pub(crate) struct ExtensionJob {
    pub query: Vec<u8>,
    pub target: Vec<u8>,
    pub h0: i32,
    #[allow(dead_code)]
    pub chain_idx: usize,
    #[allow(dead_code)]
    pub seed_idx: usize,
    pub direction: ExtensionDirection,
}

/// Extend chains to alignment regions using SIMD batch scoring
///
/// This is the score-only extension phase matching BWA-MEM2's
/// `mem_chain2aln_across_reads_V2` function. CIGAR is NOT generated here.
///
/// ## Heterogeneous Compute
///
/// The `compute_backend` parameter controls hardware dispatch:
/// - `CpuSimd`: SIMD batch scoring (active implementation)
/// - `Gpu`: GPU kernel dispatch (NO-OP, falls back to CpuSimd)
/// - `Npu`: NPU seed filtering (NO-OP, falls back to CpuSimd)
pub fn extend_chains_to_regions(
    bwa_idx: &BwaIndex,
    query_name: &str,
    opt: &MemOpt,
    chains: Vec<Chain>,
    seeds: Vec<Seed>,
    encoded_query: Vec<u8>,
    encoded_query_rc: Vec<u8>,
    compute_backend: ComputeBackend,
) -> ScoreOnlyExtensionResult {
    let sw_params = BandedPairWiseSW::new(
        opt.o_del,
        opt.e_del,
        opt.o_ins,
        opt.e_ins,
        opt.zdrop,
        5, // end_bonus
        opt.pen_clip5,
        opt.pen_clip3,
        opt.mat,
        opt.a as i8,
        -(opt.b as i8),
    );

    let query_len = encoded_query.len() as i32;
    let l_pac = bwa_idx.bns.packed_sequence_length;

    // Build extension job batches for left and right extensions
    let mut left_jobs: Vec<ExtensionJob> = Vec::new();
    let mut right_jobs: Vec<ExtensionJob> = Vec::new();
    let mut chain_mappings: Vec<ChainExtensionMapping> = Vec::new();
    let mut chain_ref_segments: Vec<Option<(u64, u64, Vec<u8>)>> = Vec::new();

    for (chain_idx, chain) in chains.iter().enumerate() {
        if chain.seeds.is_empty() {
            chain_mappings.push(ChainExtensionMapping {
                seed_mappings: Vec::new(),
            });
            chain_ref_segments.push(None);
            continue;
        }

        // Calculate rmax bounds
        let (mut rmax_0, mut rmax_1) = (l_pac << 1, 0u64);

        for &seed_idx in &chain.seeds {
            let seed = &seeds[seed_idx];
            let left_margin = seed.query_pos + cal_max_gap(opt, seed.query_pos);
            let b = seed.ref_pos.saturating_sub(left_margin as u64);
            let remaining_query = query_len - seed.query_pos - seed.len;
            let right_margin = remaining_query + cal_max_gap(opt, remaining_query);
            let e = seed.ref_pos + seed.len as u64 + right_margin as u64;
            rmax_0 = rmax_0.min(b);
            rmax_1 = rmax_1.max(e);
        }

        rmax_1 = rmax_1.min(l_pac << 1);
        if rmax_0 < l_pac && l_pac < rmax_1 {
            if seeds[chain.seeds[0]].ref_pos < l_pac {
                rmax_1 = l_pac;
            } else {
                rmax_0 = l_pac;
            }
        }

        // Fetch reference segment
        let rseq = match bwa_idx.bns.get_reference_segment(rmax_0, rmax_1 - rmax_0) {
            Ok(seq) => seq,
            Err(_) => {
                chain_mappings.push(ChainExtensionMapping {
                    seed_mappings: Vec::new(),
                });
                chain_ref_segments.push(None);
                continue;
            }
        };

        chain_ref_segments.push(Some((rmax_0, rmax_1, rseq.clone())));

        // Build seed mappings and extension jobs
        let mut seed_mappings = Vec::new();

        for &seed_chain_idx in chain.seeds.iter().rev() {
            let seed = &seeds[seed_chain_idx];
            let mut left_job_idx = None;
            let mut right_job_idx = None;

            // Left extension
            if seed.query_pos > 0 {
                let tmp = (seed.ref_pos - rmax_0) as usize;
                if tmp > 0 && tmp <= rseq.len() {
                    left_job_idx = Some(left_jobs.len());

                    if log::log_enabled!(log::Level::Debug) {
                        log::debug!(
                            "EXTENSION_JOB {}: Chain[{}] type=LEFT query=[0..{}] ref=[{}..{}] seed_pos={} seed_len={} rev={}",
                            query_name,
                            chain_idx,
                            seed.query_pos,
                            rmax_0,
                            seed.ref_pos,
                            seed.query_pos,
                            seed.len,
                            seed.is_rev
                        );
                    }

                    let query_seg: Vec<u8> = encoded_query[0..seed.query_pos as usize]
                        .iter()
                        .rev()
                        .copied()
                        .collect();
                    let target_seg: Vec<u8> = rseq[0..tmp].iter().rev().copied().collect();

                    left_jobs.push(ExtensionJob {
                        query: query_seg,
                        target: target_seg,
                        h0: seed.len * opt.a,
                        chain_idx,
                        seed_idx: seed_chain_idx,
                        direction: ExtensionDirection::Left,
                    });
                }
            }

            // Right extension
            let seed_query_end = seed.query_pos + seed.len;
            if seed_query_end < query_len {
                let re = ((seed.ref_pos + seed.len as u64) - rmax_0) as usize;
                if re < rseq.len() {
                    right_job_idx = Some(right_jobs.len());

                    if log::log_enabled!(log::Level::Debug) {
                        log::debug!(
                            "EXTENSION_JOB {}: Chain[{}] type=RIGHT query=[{}..{}] ref=[{}..{}] seed_pos={} seed_len={} rev={}",
                            query_name,
                            chain_idx,
                            seed_query_end,
                            query_len,
                            seed.ref_pos + seed.len as u64,
                            rmax_1,
                            seed.query_pos,
                            seed.len,
                            seed.is_rev
                        );
                    }

                    let query_seg: Vec<u8> = encoded_query[seed_query_end as usize..].to_vec();
                    let target_seg: Vec<u8> = rseq[re..].to_vec();

                    right_jobs.push(ExtensionJob {
                        query: query_seg,
                        target: target_seg,
                        h0: 0,
                        chain_idx,
                        seed_idx: seed_chain_idx,
                        direction: ExtensionDirection::Right,
                    });
                }
            }

            seed_mappings.push(SeedExtensionMapping {
                seed_idx: seed_chain_idx,
                left_job_idx,
                right_job_idx,
            });
        }

        chain_mappings.push(ChainExtensionMapping { seed_mappings });
    }

    log::debug!(
        "DEFERRED_CIGAR: {} left jobs, {} right jobs from {} chains",
        left_jobs.len(),
        right_jobs.len(),
        chains.len()
    );

    if log::log_enabled!(log::Level::Debug) && !left_jobs.is_empty() {
        let left_h0s: Vec<i32> = left_jobs.iter().map(|j| j.h0).collect();
        log::debug!("DEFERRED_CIGAR: left h0 values: {left_h0s:?}");
    }

    // Backend dispatch
    let effective_backend = compute_backend.effective_backend();

    let (left_scores, right_scores): (Vec<OutScore>, Vec<OutScore>) = match effective_backend {
        ComputeBackend::CpuSimd(engine) => {
            let left = if !left_jobs.is_empty() {
                execute_simd_scoring(&sw_params, &left_jobs, opt.w, engine)
            } else {
                Vec::new()
            };

            let right = if !right_jobs.is_empty() {
                execute_simd_scoring(&sw_params, &right_jobs, opt.w, engine)
            } else {
                Vec::new()
            };

            (left, right)
        }

        ComputeBackend::Gpu => {
            log::debug!("GPU backend requested but not implemented, falling back to CPU SIMD");
            let engine = crate::compute::simd_abstraction::simd::detect_optimal_simd_engine();
            let left = if !left_jobs.is_empty() {
                execute_simd_scoring(&sw_params, &left_jobs, opt.w, engine)
            } else {
                Vec::new()
            };
            let right = if !right_jobs.is_empty() {
                execute_simd_scoring(&sw_params, &right_jobs, opt.w, engine)
            } else {
                Vec::new()
            };
            (left, right)
        }

        ComputeBackend::Npu => {
            log::debug!("NPU backend requested but not implemented, falling back to CPU SIMD");
            let engine = crate::compute::simd_abstraction::simd::detect_optimal_simd_engine();
            let left = if !left_jobs.is_empty() {
                execute_simd_scoring(&sw_params, &left_jobs, opt.w, engine)
            } else {
                Vec::new()
            };
            let right = if !right_jobs.is_empty() {
                execute_simd_scoring(&sw_params, &right_jobs, opt.w, engine)
            } else {
                Vec::new()
            };
            (left, right)
        }
    };

    if log::log_enabled!(log::Level::Debug) && (!left_scores.is_empty() || !right_scores.is_empty())
    {
        let left_raw: Vec<i32> = left_scores.iter().map(|s| s.score).collect();
        let right_raw: Vec<i32> = right_scores.iter().map(|s| s.score).collect();
        log::debug!("DEFERRED_CIGAR: raw left scores: {left_raw:?}");
        log::debug!("DEFERRED_CIGAR: raw right scores: {right_raw:?}");
    }

    // Merge scores into AlignmentRegions
    let regions = merge_scores_to_regions(
        bwa_idx,
        opt,
        &chains,
        &seeds,
        &chain_mappings,
        &chain_ref_segments,
        &left_scores,
        &right_scores,
        query_len,
    );

    ScoreOnlyExtensionResult {
        regions,
        chains,
        seeds,
        encoded_query,
        encoded_query_rc,
    }
}

/// Execute SIMD batch scoring (score-only, no CIGAR)
fn execute_simd_scoring(
    sw_params: &BandedPairWiseSW,
    jobs: &[ExtensionJob],
    band_width: i32,
    engine: crate::compute::simd_abstraction::simd::SimdEngineType,
) -> Vec<OutScore> {
    if jobs.is_empty() {
        return Vec::new();
    }

    let mut batch = ExtensionJobBatch::new();
    for job in jobs {
        batch.add_job(
            0,
            job.chain_idx,
            job.seed_idx,
            job.direction,
            &job.query,
            &job.target,
            job.h0,
            band_width,
        );
    }

    let results = execute_batch_simd_scoring(sw_params, &mut batch, engine);

    results
        .into_iter()
        .map(|res| OutScore {
            score: res.score,
            query_end_pos: res.query_end,
            target_end_pos: res.ref_end,
            global_score: res.gscore,
            gtarget_end_pos: res.gref_end,
            max_offset: res.max_off,
        })
        .collect()
}
