use super::super::chaining::{chain_seeds_batch, filter_chains_batch};
use super::super::index::index::BwaIndex;
use super::super::mem_opt::MemOpt;
use super::super::seeding::find_seeds_batch;
use super::collect_soa::collect_extension_jobs_batch_soa;
use super::dispatch::execute_batch_simd_scoring;
use super::finalize_soa::finalize_alignments_soa;
use super::types::{
    BatchExtensionResult, ExtensionJobBatch, SoAAlignmentResult, SoAReadExtensionContext,
};
use crate::compute::simd_abstraction::simd::SimdEngineType;
use crate::core::alignment::banded_swa::{BandedPairWiseSW, OutScore};
use crate::core::io::soa_readers::SoAReadBatch;
use std::sync::Arc;

/// Convert batch extension results to per-read OutScore vectors (helper)
fn convert_batch_results_to_outscores(
    results: &[BatchExtensionResult],
    batch: &ExtensionJobBatch,
    num_reads: usize,
) -> Vec<Vec<OutScore>> {
    // Pre-allocate per-read result vectors
    let mut per_read_scores: Vec<Vec<OutScore>> = vec![Vec::new(); num_reads];

    // Determine size of each read's score vector from the batch jobs
    let mut read_job_counts: Vec<usize> = vec![0; num_reads];
    for job in &batch.jobs {
        read_job_counts[job.read_idx] += 1;
    }

    for (read_idx, count) in read_job_counts.iter().enumerate() {
        per_read_scores[read_idx].reserve(*count);
    }

    // Distribute results back to per-read vectors
    // Note: results are in the same order as batch.jobs
    for result in results {
        per_read_scores[result.read_idx].push(OutScore {
            score: result.score,
            query_end_pos: result.query_end,
            target_end_pos: result.ref_end,
            global_score: result.gscore,
            gtarget_end_pos: result.gref_end,
            max_offset: result.max_off,
        });
    }

    per_read_scores
}

/// SoA-native sub-batch processing (PR3)
///
/// This is the end-to-end SoA implementation that eliminates AoS-to-SoA conversion.
/// Data flows: SoAReadBatch → find_seeds_batch → chain_seeds_batch → extension → finalization
///
/// Process a sub-batch using end-to-end SoA pipeline (PR3/PR4)
///
/// This function implements the complete SoA flow:
/// 1. Seeding: find_seeds_batch (SoA) - PR2
/// 2. Chaining: chain_seeds_batch + filter_chains_batch (SoA) - PR2
/// 3. Extension: collect_extension_jobs_batch_soa (NEW) - PR3
/// 4. SIMD scoring: execute_batch_simd_scoring (already SoA)
/// 5. Finalization: finalize_alignments_soa (NEW) - PR3, returns SoAAlignmentResult in PR4
///
/// # Arguments
/// * `bwa_idx` - Reference genome index
/// * `pac_data` - Packed reference sequence
/// * `opt` - Alignment options
/// * `soa_read_batch` - SoA read batch from SoaFastqReader
/// * `batch_start_id` - Starting read ID for deterministic hashing
/// * `engine` - SIMD engine type
///
/// # Returns
/// SoAAlignmentResult with batch-wide alignment data (PR4)
pub fn process_sub_batch_internal_soa(
    bwa_idx: &Arc<&BwaIndex>,
    pac_data: &Arc<&[u8]>,
    opt: &Arc<&MemOpt>,
    soa_read_batch: &SoAReadBatch,
    batch_start_id: u64,
    engine: SimdEngineType,
) -> SoAAlignmentResult {
    let batch_size = soa_read_batch.len();

    if batch_size == 0 {
        return SoAAlignmentResult::new();
    }

    log::debug!("SOA_PIPELINE: Processing {batch_size} reads with end-to-end SoA");

    // Phase 1: SoA Seeding (PR2)
    let (soa_seed_batch, encoded_queries, encoded_queries_rc) =
        find_seeds_batch(bwa_idx, soa_read_batch, opt);

    log::debug!(
        "SOA_PIPELINE: Generated {} seeds across {} reads",
        soa_seed_batch.query_pos.len(),
        soa_seed_batch.read_seed_boundaries.len()
    );

    // Phase 2: SoA Chaining (PR2)
    let l_pac = bwa_idx.bns.packed_sequence_length;
    let mut soa_chain_batch = chain_seeds_batch(&soa_seed_batch, opt, l_pac);

    log::debug!(
        "SOA_PIPELINE: Generated {} chains before filtering",
        soa_chain_batch.score.len()
    );

    // Extract query lengths for filtering
    let query_lengths: Vec<i32> = soa_read_batch
        .read_boundaries
        .iter()
        .map(|(_, len)| *len as i32)
        .collect();

    filter_chains_batch(
        &mut soa_chain_batch,
        &soa_seed_batch,
        opt,
        &query_lengths,
        &soa_read_batch.names,
    );

    let kept_chains = soa_chain_batch.kept.iter().filter(|&&k| k > 0).count();
    log::debug!("SOA_PIPELINE: {kept_chains} chains kept after filtering");

    // Build SoAReadExtensionContext
    let mut soa_context = SoAReadExtensionContext {
        read_boundaries: soa_read_batch.read_boundaries.clone(),
        query_names: soa_read_batch.names.clone(),
        query_lengths: query_lengths.clone(),
        encoded_queries: encoded_queries.encoded_seqs,
        encoded_query_boundaries: encoded_queries.query_boundaries,
        encoded_queries_rc: encoded_queries_rc.encoded_seqs,
        encoded_query_rc_boundaries: encoded_queries_rc.query_boundaries,
        soa_seed_batch,
        soa_chain_batch,
        chain_ref_segments: Vec::new(), // Will be populated during extension job collection
    };

    // Phase 3: Extension job collection (PR3 - SoA native)
    let mut left_batch = ExtensionJobBatch::with_capacity(batch_size * 10, batch_size * 1024);
    let mut right_batch = ExtensionJobBatch::with_capacity(batch_size * 10, batch_size * 1024);

    let mappings = collect_extension_jobs_batch_soa(
        bwa_idx,
        opt,
        &mut soa_context,
        &mut left_batch,
        &mut right_batch,
    );

    log::debug!(
        "SOA_PIPELINE: Collected {} left jobs, {} right jobs",
        left_batch.len(),
        right_batch.len()
    );

    // Phase 4: SIMD scoring (already SoA)
    let sw_params = BandedPairWiseSW::new(
        opt.o_del,
        opt.e_del,
        opt.o_ins,
        opt.e_ins,
        opt.zdrop,
        5,
        opt.pen_clip5,
        opt.pen_clip3,
        opt.mat,
        opt.a as i8,
        -(opt.b as i8),
    );

    let left_results = execute_batch_simd_scoring(&sw_params, &mut left_batch, engine);
    let right_results = execute_batch_simd_scoring(&sw_params, &mut right_batch, engine);

    log::debug!(
        "SOA_PIPELINE: SIMD scoring complete: {} left results, {} right results",
        left_results.len(),
        right_results.len()
    );

    // Phase 4.5: Distribute results
    let per_read_left_scores =
        convert_batch_results_to_outscores(&left_results, &left_batch, batch_size);
    let per_read_right_scores =
        convert_batch_results_to_outscores(&right_results, &right_batch, batch_size);

    // Phase 5: Finalization (PR3/PR4 - SoA native, returns SoAAlignmentResult)
    let alignments = finalize_alignments_soa(
        bwa_idx,
        pac_data,
        opt,
        &soa_context,
        mappings,
        per_read_left_scores,
        per_read_right_scores,
        batch_start_id,
    );

    log::debug!(
        "SOA_PIPELINE: Finalization complete, {} reads with {} total alignments",
        alignments.num_reads(),
        alignments.len()
    );

    alignments
}
