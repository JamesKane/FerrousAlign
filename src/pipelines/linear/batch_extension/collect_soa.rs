use super::super::chaining::cal_max_gap;
use super::super::index::index::BwaIndex;
use super::super::mem_opt::MemOpt;
use super::super::region::{ChainExtensionMapping, SeedExtensionMapping};
/// SoA-native extension job collection (PR3)
///
/// Collects extension jobs directly from SoA chains and seeds,
/// eliminating per-read AoS intermediate representations.
use super::types::{
    ExtensionDirection, ExtensionJobBatch, REVERSE_BUF, ReadExtensionMappings,
    SoAReadExtensionContext,
};

/// Collect extension jobs from all reads in the SoA batch
///
/// This function processes all reads, chains, and seeds in SoA format,
/// building extension jobs directly without AoS conversions.
///
/// # Arguments
/// * `bwa_idx` - Reference genome index
/// * `opt` - Alignment options
/// * `soa_context` - SoA read extension context (mutable for chain_ref_segments)
/// * `left_batch` - Batch for left extension jobs (output)
/// * `right_batch` - Batch for right extension jobs (output)
///
/// # Returns
/// Per-read mappings from chains/seeds to job indices
pub fn collect_extension_jobs_batch_soa(
    bwa_idx: &BwaIndex,
    opt: &MemOpt,
    soa_context: &mut SoAReadExtensionContext,
    left_batch: &mut ExtensionJobBatch,
    right_batch: &mut ExtensionJobBatch,
) -> Vec<ReadExtensionMappings> {
    let l_pac = bwa_idx.bns.packed_sequence_length;
    let num_reads = soa_context.read_boundaries.len();

    // Pre-allocate chain_ref_segments for all chains across all reads
    let total_chains: usize = soa_context
        .soa_chain_batch
        .read_chain_boundaries
        .iter()
        .map(|(_, count)| *count)
        .sum();
    soa_context.chain_ref_segments = vec![None; total_chains];

    // Per-read mappings
    let mut all_mappings = Vec::with_capacity(num_reads);

    // Process each read
    for read_idx in 0..num_reads {
        let query_len = soa_context.query_lengths[read_idx];
        let (encoded_query_start, encoded_query_len) =
            soa_context.encoded_query_boundaries[read_idx];
        let encoded_query = &soa_context.encoded_queries
            [encoded_query_start..encoded_query_start + encoded_query_len];

        // Get chains for this read
        let (chain_start_idx, num_chains) =
            soa_context.soa_chain_batch.read_chain_boundaries[read_idx];

        let mut mappings = ReadExtensionMappings {
            chain_mappings: Vec::with_capacity(num_chains),
        };

        // Track local (per-read) job indices
        let mut left_local_idx = 0usize;
        let mut right_local_idx = 0usize;

        // Process each chain
        for local_chain_idx in 0..num_chains {
            let global_chain_idx = chain_start_idx + local_chain_idx;

            // Check if chain is kept (filtered)
            if soa_context.soa_chain_batch.kept[global_chain_idx] == 0 {
                mappings.chain_mappings.push(ChainExtensionMapping {
                    seed_mappings: Vec::new(),
                });
                continue;
            }

            // Get seeds for this chain
            let (seed_indices_start, num_seeds) =
                soa_context.soa_chain_batch.chain_seed_boundaries[global_chain_idx];

            if num_seeds == 0 {
                mappings.chain_mappings.push(ChainExtensionMapping {
                    seed_mappings: Vec::new(),
                });
                continue;
            }

            // Get seed indices for this chain
            let seed_indices = &soa_context.soa_chain_batch.seeds_indices
                [seed_indices_start..seed_indices_start + num_seeds];

            // Calculate rmax bounds
            let (mut rmax_0, mut rmax_1) = (l_pac << 1, 0u64);

            for &seed_idx in seed_indices {
                let seed_query_pos = soa_context.soa_seed_batch.query_pos[seed_idx];
                let seed_ref_pos = soa_context.soa_seed_batch.ref_pos[seed_idx];
                let seed_len = soa_context.soa_seed_batch.len[seed_idx];

                let left_margin = seed_query_pos + cal_max_gap(opt, seed_query_pos);
                let b = seed_ref_pos.saturating_sub(left_margin as u64);
                let remaining_query = query_len - seed_query_pos - seed_len;
                let right_margin = remaining_query + cal_max_gap(opt, remaining_query);
                let e = seed_ref_pos + seed_len as u64 + right_margin as u64;
                rmax_0 = rmax_0.min(b);
                rmax_1 = rmax_1.max(e);
            }

            rmax_1 = rmax_1.min(l_pac << 1);
            if rmax_0 < l_pac && l_pac < rmax_1 {
                // Chain spans forward/reverse boundary - adjust bounds
                if soa_context.soa_seed_batch.ref_pos[seed_indices[0]] < l_pac {
                    rmax_1 = l_pac;
                } else {
                    rmax_0 = l_pac;
                }
            }

            // Fetch reference segment
            let rseq = match bwa_idx.bns.get_reference_segment(rmax_0, rmax_1 - rmax_0) {
                Ok(seq) => seq,
                Err(_) => {
                    mappings.chain_mappings.push(ChainExtensionMapping {
                        seed_mappings: Vec::new(),
                    });
                    continue;
                }
            };

            // Store reference segment
            soa_context.chain_ref_segments[global_chain_idx] = Some((rmax_0, rmax_1));

            log::debug!(
                "RMAX_CALC: read_idx={} chain_idx={} rmax_0={} rmax_1={} l_pac={} is_rev={}",
                read_idx, local_chain_idx, rmax_0, rmax_1, l_pac, rmax_0 >= l_pac
            );

            // Build seed mappings and extension jobs
            let mut seed_mappings = Vec::new();

            // Process seeds in reverse order (matching AoS logic)
            for &seed_idx in seed_indices.iter().rev() {
                let seed_query_pos = soa_context.soa_seed_batch.query_pos[seed_idx];
                let seed_ref_pos = soa_context.soa_seed_batch.ref_pos[seed_idx];
                let seed_len = soa_context.soa_seed_batch.len[seed_idx];

                let mut left_job_idx = None;
                let mut right_job_idx = None;

                // Left extension
                if seed_query_pos > 0 {
                    let tmp = (seed_ref_pos - rmax_0) as usize;
                    if tmp > 0 && tmp <= rseq.len() {
                        REVERSE_BUF.with(|buf_cell| {
                            let mut buf = buf_cell.borrow_mut();
                            buf.clear();

                            // Append reversed query segment
                            let query_slice = &encoded_query[0..seed_query_pos as usize];
                            buf.extend(query_slice.iter().rev().copied());
                            let query_seg_len = buf.len();

                            // Append reversed target segment
                            let target_slice = &rseq[0..tmp];
                            buf.extend(target_slice.iter().rev().copied());

                            // Create slices
                            let current_query_seg = &buf[0..query_seg_len];
                            let current_target_seg = &buf[query_seg_len..];

                            // Add job
                            left_job_idx = Some(left_local_idx);
                            left_local_idx += 1;
                            left_batch.add_job(
                                read_idx,
                                local_chain_idx,
                                seed_idx,
                                ExtensionDirection::Left,
                                current_query_seg,
                                current_target_seg,
                                seed_len * opt.a, // h0 = seed_len * match_score
                                opt.w,
                            );
                        });
                    }
                }

                // Right extension
                let seed_query_end = seed_query_pos + seed_len;
                if seed_query_end < query_len {
                    let tmp = (seed_ref_pos + seed_len as u64 - rmax_0) as usize;
                    if tmp < rseq.len() {
                        let current_query_seg =
                            &encoded_query[seed_query_end as usize..query_len as usize];
                        let current_target_seg = &rseq[tmp..];

                        right_job_idx = Some(right_local_idx);
                        right_local_idx += 1;
                        right_batch.add_job(
                            read_idx,
                            local_chain_idx,
                            seed_idx,
                            ExtensionDirection::Right,
                            current_query_seg,
                            current_target_seg,
                            0, // h0 = 0 for right extension
                            opt.w,
                        );
                    }
                }

                seed_mappings.push(SeedExtensionMapping {
                    seed_idx,
                    left_job_idx,
                    right_job_idx,
                });
            }

            mappings
                .chain_mappings
                .push(ChainExtensionMapping { seed_mappings });
        }

        all_mappings.push(mappings);
    }

    all_mappings
}
