/// SoA-native finalization (PR3)
///
/// Converts extension results to final alignments while working with SoA structures.
/// Extracts per-read data from SoA batches to call existing finalization logic.

use super::types::{ReadExtensionMappings, SoAReadExtensionContext};
use super::super::chaining::Chain;
use super::super::finalization::Alignment;
use super::super::index::index::BwaIndex;
use super::super::mem_opt::MemOpt;
use super::super::region::{merge_extension_scores_to_regions, generate_cigar_from_region};
use super::super::seeding::Seed;
use crate::core::alignment::banded_swa::OutScore;
use rayon::prelude::*;
use std::sync::Arc;

/// Finalize alignments from SoA batch data
///
/// This function:
/// 1. Iterates through each read in the batch using SoA boundaries
/// 2. Extracts per-read chains, seeds, and extension scores
/// 3. Calls merge_extension_scores_to_regions to create alignment regions
/// 4. Generates CIGAR strings for surviving alignments
/// 5. Marks secondary alignments
///
/// # Arguments
/// * `bwa_idx` - Reference genome index
/// * `pac_data` - Packed reference sequence
/// * `opt` - Alignment options
/// * `soa_context` - SoA read extension context with batch data
/// * `mappings` - Per-read mappings from chains/seeds to job indices
/// * `per_read_left_scores` - Left extension scores per read
/// * `per_read_right_scores` - Right extension scores per read
/// * `batch_start_id` - Starting read ID for deterministic hashing
///
/// # Returns
/// Vector of alignments for each read (same order as input)
pub fn finalize_alignments_soa(
    bwa_idx: &Arc<&BwaIndex>,
    pac_data: &Arc<&[u8]>,
    opt: &Arc<&MemOpt>,
    soa_context: &SoAReadExtensionContext,
    mappings: Vec<ReadExtensionMappings>,
    per_read_left_scores: Vec<Vec<OutScore>>,
    per_read_right_scores: Vec<Vec<OutScore>>,
    batch_start_id: u64,
) -> Vec<Vec<Alignment>> {
    let num_reads = soa_context.read_boundaries.len();

    // Process each read in parallel
    let valid_alignments: Vec<(usize, Vec<Alignment>)> = (0..num_reads)
        .into_par_iter()
        .map(|read_idx| {
            let read_id = batch_start_id + read_idx as u64;
            let query_name = &soa_context.query_names[read_idx];
            let query_len = soa_context.query_lengths[read_idx];

            // Extract per-read encoded query
            let (encoded_query_start, encoded_query_len) =
                soa_context.encoded_query_boundaries[read_idx];
            let encoded_query = &soa_context.encoded_queries
                [encoded_query_start..encoded_query_start + encoded_query_len];

            // Extract per-read seeds from SoA
            let (seed_start_idx, num_seeds) = soa_context.soa_seed_batch.read_seed_boundaries[read_idx];
            let mut seeds = Vec::with_capacity(num_seeds);

            // Build mapping from global seed index to local seed index
            let mut global_to_local_seed: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();

            for local_seed_idx in 0..num_seeds {
                let global_seed_idx = seed_start_idx + local_seed_idx;
                global_to_local_seed.insert(global_seed_idx, local_seed_idx);

                seeds.push(Seed {
                    query_pos: soa_context.soa_seed_batch.query_pos[global_seed_idx],
                    ref_pos: soa_context.soa_seed_batch.ref_pos[global_seed_idx],
                    len: soa_context.soa_seed_batch.len[global_seed_idx],
                    is_rev: soa_context.soa_seed_batch.is_rev[global_seed_idx],
                    interval_size: soa_context.soa_seed_batch.interval_size[global_seed_idx],
                    rid: soa_context.soa_seed_batch.rid[global_seed_idx],
                });
            }

            // Extract per-read chains from SoA, remapping seed indices to local
            let (chain_start_idx, num_chains) =
                soa_context.soa_chain_batch.read_chain_boundaries[read_idx];
            let mut chains = Vec::with_capacity(num_chains);

            for local_chain_idx in 0..num_chains {
                let global_chain_idx = chain_start_idx + local_chain_idx;

                // Get seed indices for this chain and remap to local indices
                let (seed_indices_start, num_seeds_in_chain) =
                    soa_context.soa_chain_batch.chain_seed_boundaries[global_chain_idx];
                let global_seed_indices = &soa_context.soa_chain_batch.seeds_indices
                    [seed_indices_start..seed_indices_start + num_seeds_in_chain];

                // Remap global seed indices to local indices
                let local_seed_indices: Vec<usize> = global_seed_indices
                    .iter()
                    .filter_map(|&global_idx| global_to_local_seed.get(&global_idx).copied())
                    .collect();

                chains.push(Chain {
                    score: soa_context.soa_chain_batch.score[global_chain_idx],
                    seeds: local_seed_indices,
                    query_start: soa_context.soa_chain_batch.query_start[global_chain_idx],
                    query_end: soa_context.soa_chain_batch.query_end[global_chain_idx],
                    ref_start: soa_context.soa_chain_batch.ref_start[global_chain_idx],
                    ref_end: soa_context.soa_chain_batch.ref_end[global_chain_idx],
                    is_rev: soa_context.soa_chain_batch.is_rev[global_chain_idx],
                    weight: soa_context.soa_chain_batch.weight[global_chain_idx],
                    kept: soa_context.soa_chain_batch.kept[global_chain_idx],
                    frac_rep: soa_context.soa_chain_batch.frac_rep[global_chain_idx],
                    rid: soa_context.soa_chain_batch.rid[global_chain_idx],
                    pos: soa_context.soa_chain_batch.pos[global_chain_idx],
                    last_qbeg: soa_context.soa_chain_batch.last_qbeg[global_chain_idx],
                    last_rbeg: soa_context.soa_chain_batch.last_rbeg[global_chain_idx],
                    last_len: soa_context.soa_chain_batch.last_len[global_chain_idx],
                });
            }

            // Extract chain_ref_segments with full reference sequences
            let mut chain_ref_segments = Vec::with_capacity(num_chains);
            for local_chain_idx in 0..num_chains {
                let global_chain_idx = chain_start_idx + local_chain_idx;

                let ref_segment = match soa_context.chain_ref_segments[global_chain_idx] {
                    Some((rmax_0, rmax_1)) => {
                        // Re-fetch reference segment for this chain
                        match bwa_idx.bns.get_reference_segment(rmax_0, rmax_1 - rmax_0) {
                            Ok(rseq) => Some((rmax_0, rmax_1, rseq)),
                            Err(_) => None,
                        }
                    }
                    None => None,
                };

                chain_ref_segments.push(ref_segment);
            }

            // Get extension scores for this read
            let left_scores = &per_read_left_scores[read_idx];
            let right_scores = &per_read_right_scores[read_idx];
            let original_mapping = &mappings[read_idx];

            // Remap seed indices in mappings from global to local
            let mut remapped_mapping = super::types::ReadExtensionMappings {
                chain_mappings: Vec::with_capacity(original_mapping.chain_mappings.len()),
            };

            for chain_mapping in &original_mapping.chain_mappings {
                let mut remapped_seed_mappings = Vec::with_capacity(chain_mapping.seed_mappings.len());

                for seed_mapping in &chain_mapping.seed_mappings {
                    let local_seed_idx = global_to_local_seed
                        .get(&seed_mapping.seed_idx)
                        .copied()
                        .unwrap_or(seed_mapping.seed_idx); // Fallback to original if not found

                    remapped_seed_mappings.push(super::super::region::SeedExtensionMapping {
                        seed_idx: local_seed_idx,
                        left_job_idx: seed_mapping.left_job_idx,
                        right_job_idx: seed_mapping.right_job_idx,
                    });
                }

                remapped_mapping.chain_mappings.push(super::super::region::ChainExtensionMapping {
                    seed_mappings: remapped_seed_mappings,
                });
            }

            // Merge extension scores into alignment regions
            let regions = merge_extension_scores_to_regions(
                bwa_idx,
                opt,
                &chains,
                &seeds,
                &remapped_mapping.chain_mappings,
                &chain_ref_segments,
                left_scores,
                right_scores,
                query_len,
            );

            if regions.is_empty() {
                return (
                    read_idx,
                    vec![super::orchestration::create_unmapped_alignment_internal(query_name)],
                );
            }

            // Filter regions by score threshold
            let mut filtered_regions: Vec<_> =
                regions.into_iter().filter(|r| r.score >= opt.t).collect();

            if filtered_regions.is_empty() {
                return (
                    read_idx,
                    vec![super::orchestration::create_unmapped_alignment_internal(query_name)],
                );
            }

            // Sort by score descending
            filtered_regions.sort_by(|a, b| b.score.cmp(&a.score));

            // Generate CIGAR strings and create alignments
            let mut alignments = Vec::new();
            for (idx, region) in filtered_regions.iter().enumerate() {
                let cigar_result =
                    generate_cigar_from_region(bwa_idx, pac_data, encoded_query, region, opt);

                let (cigar, nm, md_tag) = match cigar_result {
                    Some(result) => result,
                    None => continue,
                };

                let flag = if region.is_rev {
                    super::super::finalization::sam_flags::REVERSE
                } else {
                    0
                };

                let hash = crate::utils::hash_64(read_id + idx as u64);

                alignments.push(Alignment {
                    query_name: query_name.clone(),
                    flag,
                    ref_name: region.ref_name.clone(),
                    ref_id: region.rid as usize,
                    pos: region.chr_pos,
                    mapq: 60,
                    score: region.score,
                    cigar,
                    rnext: "*".to_string(),
                    pnext: 0,
                    tlen: 0,
                    seq: String::new(),
                    qual: String::new(),
                    tags: vec![
                        ("AS".to_string(), format!("i:{}", region.score)),
                        ("NM".to_string(), format!("i:{nm}")),
                        ("MD".to_string(), format!("Z:{md_tag}")),
                    ],
                    query_start: region.qb,
                    query_end: region.qe,
                    seed_coverage: region.seedcov,
                    hash,
                    frac_rep: region.frac_rep,
                });
            }

            if alignments.is_empty() {
                return (
                    read_idx,
                    vec![super::orchestration::create_unmapped_alignment_internal(query_name)],
                );
            }

            // Mark secondary alignments
            super::super::finalization::mark_secondary_alignments(&mut alignments, opt);

            (read_idx, alignments)
        })
        .collect();

    // Assemble final results in read order
    let mut all_alignments: Vec<Vec<Alignment>> = soa_context
        .query_names
        .iter()
        .map(|name| vec![super::orchestration::create_unmapped_alignment_internal(name)])
        .collect();

    for (read_idx, alignments) in valid_alignments {
        all_alignments[read_idx] = alignments;
    }

    all_alignments
}
