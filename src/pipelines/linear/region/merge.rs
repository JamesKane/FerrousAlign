//! Score merging for alignment regions.
//!
//! Merges left/right extension scores into AlignmentRegions, implementing
//! the boundary calculation logic from BWA-MEM2.

use super::super::chaining::Chain;
use super::super::index::index::BwaIndex;
use super::super::mem_opt::MemOpt;
use super::super::seeding::Seed;
use super::types::{AlignmentRegion, ChainExtensionMapping};
use crate::core::alignment::banded_swa::OutScore;

/// Merge extension scores into alignment regions for cross-read batching
///
/// This is a public wrapper around the internal merge function, allowing
/// external code (batch_extension.rs) to perform the score merging after
/// cross-read SIMD scoring.
pub fn merge_extension_scores_to_regions(
    bwa_idx: &BwaIndex,
    opt: &MemOpt,
    chains: &[Chain],
    seeds: &[Seed],
    chain_mappings: &[ChainExtensionMapping],
    chain_ref_segments: &[Option<(u64, u64, Vec<u8>)>],
    left_scores: &[OutScore],
    right_scores: &[OutScore],
    query_len: i32,
) -> Vec<AlignmentRegion> {
    merge_scores_to_regions(
        bwa_idx,
        opt,
        chains,
        seeds,
        chain_mappings,
        chain_ref_segments,
        left_scores,
        right_scores,
        query_len,
    )
}

/// Merge left/right extension scores into AlignmentRegions
///
/// This implements the boundary calculation logic from BWA-MEM2's
/// mem_chain2aln_across_reads_V2 (bwamem.cpp:2486-2520).
pub fn merge_scores_to_regions(
    bwa_idx: &BwaIndex,
    opt: &MemOpt,
    chains: &[Chain],
    seeds: &[Seed],
    chain_mappings: &[ChainExtensionMapping],
    chain_ref_segments: &[Option<(u64, u64, Vec<u8>)>],
    left_scores: &[OutScore],
    right_scores: &[OutScore],
    query_len: i32,
) -> Vec<AlignmentRegion> {
    let mut regions = Vec::new();

    for (chain_idx, chain) in chains.iter().enumerate() {
        if chain.seeds.is_empty() {
            continue;
        }

        let mapping = &chain_mappings[chain_idx];
        let ref_segment = match &chain_ref_segments[chain_idx] {
            Some(seg) => seg,
            None => continue,
        };
        let (_rmax_0, _rmax_1, _rseq) = ref_segment;

        // Find best seed in chain (by score)
        let mut best_score = i32::MIN;
        let mut best_region: Option<AlignmentRegion> = None;

        for seed_mapping in &mapping.seed_mappings {
            let seed = &seeds[seed_mapping.seed_idx];
            let mut region = AlignmentRegion::new(chain_idx, seed_mapping.seed_idx);

            // Initialize boundaries from seed
            region.qb = seed.query_pos;
            region.qe = seed.query_pos + seed.len;
            region.rb = seed.ref_pos;
            region.re = seed.ref_pos + seed.len as u64;
            region.seedlen0 = seed.len;
            region.frac_rep = chain.frac_rep;
            region.w = opt.w;

            let mut total_score = seed.len * opt.a;

            // Debug logging - log everything, filter later
            log::info!(
                "EXTENSION_DEBUG: chain_idx={} seed_idx={} seed: qpos={} len={} rpos={} initial_qb={} initial_qe={}",
                chain_idx, seed_mapping.seed_idx, seed.query_pos, seed.len, seed.ref_pos, region.qb, region.qe
            );

            // Process left extension
            if let Some(left_idx) = seed_mapping.left_job_idx {
                if left_idx < left_scores.len() {
                    let left_score = &left_scores[left_idx];
                    total_score = left_score.score;

                    log::info!(
                        "  LEFT_EXT: score={} global_score={} query_end_pos={} target_end_pos={} gtarget_end_pos={} pen_clip5={}",
                        left_score.score, left_score.global_score, left_score.query_end_pos,
                        left_score.target_end_pos, left_score.gtarget_end_pos, opt.pen_clip5
                    );

                    if left_score.global_score <= 0
                        || left_score.global_score <= left_score.score - opt.pen_clip5
                    {
                        region.qb = seed.query_pos - left_score.query_end_pos;
                        region.rb = seed.ref_pos - left_score.target_end_pos as u64;
                        region.truesc = left_score.score;
                        log::info!("    Using local: qb={} rb={}", region.qb, region.rb);
                    } else {
                        region.qb = 0;
                        region.rb = seed.ref_pos - left_score.gtarget_end_pos as u64;
                        region.truesc = left_score.global_score;
                        log::info!("    Using global: qb={} rb={}", region.qb, region.rb);
                    }
                }
            } else if seed.query_pos == 0 {
                region.truesc = seed.len * opt.a;
            }

            // Process right extension
            if let Some(right_idx) = seed_mapping.right_job_idx {
                if right_idx < right_scores.len() {
                    let right_score = &right_scores[right_idx];
                    total_score += right_score.score;

                    log::info!(
                        "  RIGHT_EXT: score={} global_score={} query_end_pos={} target_end_pos={} gtarget_end_pos={} pen_clip3={}",
                        right_score.score, right_score.global_score, right_score.query_end_pos,
                        right_score.target_end_pos, right_score.gtarget_end_pos, opt.pen_clip3
                    );

                    if right_score.global_score <= 0
                        || right_score.global_score <= right_score.score - opt.pen_clip3
                    {
                        region.qe = seed.query_pos + seed.len + right_score.query_end_pos;
                        region.re =
                            seed.ref_pos + seed.len as u64 + right_score.target_end_pos as u64;
                        log::info!("    Using local: qe={} re={}", region.qe, region.re);
                    } else {
                        region.qe = query_len;
                        region.re =
                            seed.ref_pos + seed.len as u64 + right_score.gtarget_end_pos as u64;
                        log::info!("    Using global: qe={} re={}", region.qe, region.re);
                    }
                }
            }

            region.score = total_score;
            if region.truesc == 0 {
                region.truesc = total_score;
            }

            // Calculate seed coverage
            region.seedcov = 0;
            for &other_seed_idx in &chain.seeds {
                let other_seed = &seeds[other_seed_idx];
                if other_seed.query_pos >= region.qb
                    && other_seed.query_pos + other_seed.len <= region.qe
                    && other_seed.ref_pos >= region.rb
                    && other_seed.ref_pos + other_seed.len as u64 <= region.re
                {
                    region.seedcov += other_seed.len;
                }
            }

            // Convert FM-index position to chromosome coordinates
            let coords =
                super::super::coordinates::fm_to_chromosome_coords(bwa_idx, region.rb, region.re);
            region.rid = coords.ref_id;
            region.ref_name = coords.ref_name;
            region.chr_pos = coords.chr_pos;
            region.is_rev = coords.is_rev;

            log::info!(
                "  FINAL_REGION: qb={} qe={} rb={} re={} score={} truesc={} seedcov={}",
                region.qb, region.qe, region.rb, region.re, region.score, region.truesc, region.seedcov
            );

            if total_score > best_score {
                best_score = total_score;
                best_region = Some(region);
            }
        }

        if let Some(ref region) = best_region {
            log::info!(
                "DEFERRED_REGION: chain_idx={} seed_idx={} qb={} qe={} rb={} re={} score={} chr_pos={} ref={}",
                region.chain_idx,
                region.seed_idx,
                region.qb,
                region.qe,
                region.rb,
                region.re,
                region.score,
                region.chr_pos,
                region.ref_name
            );
            regions.push(region.clone());
        }
    }

    regions
}
