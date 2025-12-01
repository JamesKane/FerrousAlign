use super::super::chaining::cal_max_gap;
use super::super::index::index::BwaIndex;
use super::super::mem_opt::MemOpt;
use super::super::region::{ChainExtensionMapping, SeedExtensionMapping};
use super::types::{
    ExtensionDirection, ExtensionJobBatch, REVERSE_BUF, ReadExtensionContext, ReadExtensionMappings,
};

// ============================================================================
// CROSS-READ BATCHING - COLLECTION INFRASTRUCTURE
// ============================================================================
//
// These structures and functions implement BWA-MEM2's cross-read batching strategy:
// 1. Seeding and chaining happen per-read (parallelized)
// 2. Extension jobs are COLLECTED from all reads into a single batch
// 3. SIMD scoring executes on the FULL batch (maximizing lane utilization)
// 4. Results are distributed back to per-read structures
// 5. Finalization happens per-read (parallelized)
//
// ============================================================================

/// Collect extension jobs from a single read's chains
///
/// This extracts the job collection logic from `extend_chains_to_regions()`,
/// storing jobs in the cross-read batches instead of executing immediately.
///
/// **Important**: The job indices stored in SeedExtensionMapping are LOCAL
/// (per-read) indices, not global batch indices. This allows
/// merge_scores_to_regions() to work correctly with per-read score vectors.
pub fn collect_extension_jobs_for_read(
    bwa_idx: &BwaIndex,
    opt: &MemOpt,
    read_idx: usize,
    ctx: &mut ReadExtensionContext,
    left_batch: &mut ExtensionJobBatch,
    right_batch: &mut ExtensionJobBatch,
) -> ReadExtensionMappings {
    let l_pac = bwa_idx.bns.packed_sequence_length;
    let query_len = ctx.query_len;

    let mut mappings = ReadExtensionMappings {
        chain_mappings: Vec::with_capacity(ctx.chains.len()),
    };

    // Track local (per-read) job indices for use by merge_scores_to_regions()
    // These are indices into the per-read score vectors, NOT global batch indices
    let mut left_local_idx = 0usize;
    let mut right_local_idx = 0usize;

    // Pre-allocate chain_ref_segments
    ctx.chain_ref_segments = Vec::with_capacity(ctx.chains.len());

    for (chain_idx, chain) in ctx.chains.iter().enumerate() {
        if chain.seeds.is_empty() {
            mappings.chain_mappings.push(ChainExtensionMapping {
                seed_mappings: Vec::new(),
            });
            ctx.chain_ref_segments.push(None);
            continue;
        }

        // Calculate rmax bounds (same as region.rs)
        let (mut rmax_0, mut rmax_1) = (l_pac << 1, 0u64);

        for &seed_idx in &chain.seeds {
            let seed = &ctx.seeds[seed_idx];
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
            if ctx.seeds[chain.seeds[0]].ref_pos < l_pac {
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
                ctx.chain_ref_segments.push(None);
                continue;
            }
        };

        ctx.chain_ref_segments
            .push(Some((rmax_0, rmax_1, rseq.clone())));

        // Build seed mappings and extension jobs
        let mut seed_mappings = Vec::new();

        for &seed_chain_idx in chain.seeds.iter().rev() {
            let seed = &ctx.seeds[seed_chain_idx];
            let mut left_job_idx = None;
            let mut right_job_idx = None;

            // Left extension
            if seed.query_pos > 0 {
                let tmp = (seed.ref_pos - rmax_0) as usize;
                if tmp > 0 && tmp <= rseq.len() {
                    REVERSE_BUF.with(|buf_cell| {
                        let mut buf = buf_cell.borrow_mut();
                        buf.clear(); // Clear for reuse

                        // Append reversed query segment
                        let query_slice_to_reverse = &ctx.encoded_query[0..seed.query_pos as usize];
                        buf.extend(query_slice_to_reverse.iter().rev().copied());
                        let query_seg_len = buf.len();

                        // Append reversed target segment
                        let target_slice_to_reverse = &rseq[0..tmp];
                        buf.extend(target_slice_to_reverse.iter().rev().copied());
                        // target_seg_len is implicitly buf.len() - query_seg_len here, but not needed

                        // Create slices from the temporary buffer
                        let current_query_seg = &buf[0..query_seg_len];
                        let current_target_seg = &buf[query_seg_len..];

                        // Use local (per-read) index, NOT global batch index
                        left_job_idx = Some(left_local_idx);
                        left_local_idx += 1;
                        left_batch.add_job(
                            read_idx,
                            chain_idx,
                            seed_chain_idx,
                            ExtensionDirection::Left,
                            current_query_seg,
                            current_target_seg,
                            seed.len * opt.a, // h0 = seed_len * match_score
                            opt.w,
                        );
                    });
                }
            }

            // Right extension
            let seed_query_end = seed.query_pos + seed.len;
            if seed_query_end < query_len {
                let re = ((seed.ref_pos + seed.len as u64) - rmax_0) as usize;
                if re < rseq.len() {
                    let query_seg = &ctx.encoded_query[seed_query_end as usize..];
                    let target_seg = &rseq[re..];

                    // Use local (per-read) index, NOT global batch index
                    right_job_idx = Some(right_local_idx);
                    right_local_idx += 1;
                    right_batch.add_job(
                        read_idx,
                        chain_idx,
                        seed_chain_idx,
                        ExtensionDirection::Right,
                        query_seg,
                        target_seg,
                        0, // h0 = 0 for right extension
                        opt.w,
                    );
                }
            }

            seed_mappings.push(SeedExtensionMapping {
                seed_idx: seed_chain_idx,
                left_job_idx,
                right_job_idx,
            });
        }

        mappings
            .chain_mappings
            .push(ChainExtensionMapping { seed_mappings });
    }

    mappings
}

/// Collect extension jobs from ALL reads in a batch
///
/// This is the main entry point for cross-read batching.
/// Returns the filled job batches and per-read mappings.
pub fn collect_extension_jobs_batch(
    bwa_idx: &BwaIndex,
    opt: &MemOpt,
    read_contexts: &mut [ReadExtensionContext],
) -> (
    ExtensionJobBatch,
    ExtensionJobBatch,
    Vec<ReadExtensionMappings>,
) {
    // Estimate capacity based on typical chains per read
    let estimated_jobs = read_contexts.len() * 8; // ~4 chains, 2 directions each
    let estimated_seq_bytes = estimated_jobs * 200; // ~100bp per extension

    let mut left_batch = ExtensionJobBatch::with_capacity(estimated_jobs, estimated_seq_bytes);
    let mut right_batch = ExtensionJobBatch::with_capacity(estimated_jobs, estimated_seq_bytes);
    let mut all_mappings = Vec::with_capacity(read_contexts.len());

    for (read_idx, ctx) in read_contexts.iter_mut().enumerate() {
        let mappings = collect_extension_jobs_for_read(
            bwa_idx,
            opt,
            read_idx,
            ctx,
            &mut left_batch,
            &mut right_batch,
        );
        all_mappings.push(mappings);
    }

    log::debug!(
        "CROSS_READ_BATCH: Collected {} left jobs, {} right jobs from {} reads",
        left_batch.len(),
        right_batch.len(),
        read_contexts.len()
    );

    (left_batch, right_batch, all_mappings)
}
