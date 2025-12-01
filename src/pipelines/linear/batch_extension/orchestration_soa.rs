/// SoA-native sub-batch processing (PR3)
///
/// This is the end-to-end SoA implementation that eliminates AoS-to-SoA conversion.
/// Data flows: SoAReadBatch → find_seeds_batch → chain_seeds_batch → extension → finalization

use super::types::{SoAReadExtensionContext, ExtensionJobBatch};
use super::super::index::index::BwaIndex;
use super::super::mem_opt::MemOpt;
use super::super::seeding::find_seeds_batch;
use super::super::chaining::{chain_seeds_batch, filter_chains_batch};
use super::collect_soa::collect_extension_jobs_batch_soa;
use super::dispatch::execute_batch_simd_scoring;
use super::distribute::convert_batch_results_to_outscores;
use super::finalize_soa::finalize_alignments_soa;
use crate::core::io::soa_readers::SoAReadBatch;
use crate::core::alignment::banded_swa::BandedPairWiseSW;
use crate::compute::simd_abstraction::simd::SimdEngineType;
use super::super::finalization::Alignment;
use std::sync::Arc;

/// Process a sub-batch using end-to-end SoA pipeline (PR3)
///
/// This function implements the complete SoA flow:
/// 1. Seeding: find_seeds_batch (SoA) - PR2
/// 2. Chaining: chain_seeds_batch + filter_chains_batch (SoA) - PR2
/// 3. Extension: collect_extension_jobs_batch_soa (NEW) - PR3
/// 4. SIMD scoring: execute_batch_simd_scoring (already SoA)
/// 5. Finalization: finalize_alignments_soa (NEW) - PR3
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
/// Vector of alignments for each read (same order as input)
pub fn process_sub_batch_internal_soa(
    bwa_idx: &Arc<&BwaIndex>,
    pac_data: &Arc<&[u8]>,
    opt: &Arc<&MemOpt>,
    soa_read_batch: &SoAReadBatch,
    batch_start_id: u64,
    engine: SimdEngineType,
) -> Vec<Vec<Alignment>> {
    let batch_size = soa_read_batch.len();

    if batch_size == 0 {
        return Vec::new();
    }

    log::debug!("SOA_PIPELINE: Processing {} reads with end-to-end SoA", batch_size);

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

    filter_chains_batch(&mut soa_chain_batch, &soa_seed_batch, opt, &query_lengths);

    let kept_chains = soa_chain_batch.kept.iter().filter(|&&k| k > 0).count();
    log::debug!(
        "SOA_PIPELINE: {} chains kept after filtering",
        kept_chains
    );

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

    // Phase 5: Finalization (PR3 - SoA native)
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
        "SOA_PIPELINE: Finalization complete, {} reads with alignments",
        alignments.len()
    );

    alignments
}
