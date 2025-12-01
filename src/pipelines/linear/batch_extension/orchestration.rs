use super::super::index::index::BwaIndex;
use super::super::mem_opt::MemOpt;
use super::types::ReadExtensionContext;

use rayon::prelude::*;
use std::sync::Arc;

use crate::compute::simd_abstraction::simd::SimdEngineType;
use crate::core::alignment::banded_swa::{BandedPairWiseSW, OutScore}; // Corrected import // Corrected import

use super::super::finalization::{Alignment, mark_secondary_alignments}; // Corrected import
use super::super::pipeline::{build_and_filter_chains, find_seeds}; // Corrected import
use super::super::region::{generate_cigar_from_region, merge_extension_scores_to_regions}; // Corrected import
use super::collect::collect_extension_jobs_batch;
use super::dispatch::execute_batch_simd_scoring; // Corrected import
use super::distribute::convert_batch_results_to_outscores; // Corrected import // Corrected import

/// Sub-batch size for parallel processing
///
/// On x86_64 with AVX2 (256-bit, batch16), 512 reads provides good SIMD utilization.
/// On aarch64 with NEON (128-bit, batch8), we use 1024 reads to compensate for
/// the smaller batch size and amortize per-batch overhead.
#[cfg(target_arch = "aarch64")]
const SUB_BATCH_SIZE: usize = 1024;

#[cfg(not(target_arch = "aarch64"))]
const SUB_BATCH_SIZE: usize = 512;

/// Process a batch of reads using cross-read SIMD batching
///
/// This is the high-performance alternative to per-read align_read_deferred().
/// Instead of processing each read's extensions independently, we collect
/// extension jobs from ALL reads and process them together with SIMD.
///
/// # Performance
/// - AVX-512: ~2x better SIMD utilization (32 lanes fully used)
/// - AVX2: ~1.5x better SIMD utilization (16 lanes fully used)
///
/// # Arguments
/// * `bwa_idx` - Reference genome index
/// * `pac_data` - Packed reference sequence for CIGAR generation
/// * `opt` - Alignment options
/// * `names` - Read names
/// * `seqs` - Read sequences
/// * `quals` - Quality strings
/// * `batch_start_id` - Starting read ID for deterministic hash tie-breaking
/// * `engine` - SIMD engine type
///
/// # Returns
/// Vector of alignments for each read
pub fn process_batch_cross_read(
    bwa_idx: &BwaIndex,
    pac_data: &[u8],
    opt: &MemOpt,
    names: &[String],
    seqs: &[Vec<u8>],
    _quals: &[String],
    batch_start_id: u64,
    engine: SimdEngineType,
) -> Vec<Vec<Alignment>> {
    let bwa_idx_arc = Arc::new(bwa_idx);
    let pac_data_arc = Arc::new(pac_data);
    let opt_arc = Arc::new(opt);

    process_sub_batch_internal(
        &bwa_idx_arc,
        &pac_data_arc,
        &opt_arc,
        names,
        seqs,
        batch_start_id,
        engine,
    )
}

/// Create an unmapped alignment record (internal helper)
pub fn create_unmapped_alignment_internal(query_name: &str) -> Alignment {
    use super::super::finalization::sam_flags;

    Alignment {
        query_name: query_name.to_string(),
        flag: sam_flags::UNMAPPED,
        ref_name: "*".to_string(),
        ref_id: 0,
        pos: 0,
        mapq: 0,
        score: 0,
        cigar: Vec::new(),
        rnext: "*".to_string(),
        pnext: 0,
        tlen: 0,
        seq: String::new(),
        qual: String::new(),
        tags: vec![
            ("AS".to_string(), "i:0".to_string()),
            ("NM".to_string(), "i:0".to_string()),
        ],
        query_start: 0,
        query_end: 0,
        seed_coverage: 0,
        hash: 0,
        frac_rep: 0.0,
    }
}

/// Process a batch of reads using parallel sub-batches with cross-read SIMD batching
///
/// This is the optimized version that:
/// 1. Splits reads into chunks of SUB_BATCH_SIZE (~512)
/// 2. Processes chunks in parallel using rayon
/// 3. Within each chunk, uses cross-read batching for SIMD scoring
///
/// This approach maintains parallelism while improving SIMD lane utilization.
pub fn process_batch_parallel_subbatch(
    bwa_idx: &BwaIndex,
    pac_data: &[u8],
    opt: &MemOpt,
    names: &[String],
    seqs: &[Vec<u8>],
    _quals: &[String],
    batch_start_id: u64,
    engine: SimdEngineType,
) -> Vec<Vec<Alignment>> {
    let batch_size = names.len();

    if batch_size == 0 {
        return Vec::new();
    }

    // Wrap in Arc for thread-safe sharing
    let bwa_idx = Arc::new(bwa_idx);
    let pac_data = Arc::new(pac_data);
    let opt = Arc::new(opt);

    // For small batches, use single sub-batch (no overhead)
    if batch_size <= SUB_BATCH_SIZE {
        return process_sub_batch_internal(
            &bwa_idx,
            &pac_data,
            &opt,
            names,
            seqs,
            batch_start_id,
            engine,
        );
    }

    // Calculate number of sub-batches
    let num_sub_batches = batch_size.div_ceil(SUB_BATCH_SIZE);

    log::debug!(
        "PARALLEL_SUBBATCH: Processing {batch_size} reads in {num_sub_batches} sub-batches of ~{SUB_BATCH_SIZE} reads each"
    );

    // Process sub-batches in parallel
    let sub_batch_results: Vec<Vec<Vec<Alignment>>> = (0..num_sub_batches)
        .into_par_iter()
        .map(|sub_batch_idx| {
            let start_idx = sub_batch_idx * SUB_BATCH_SIZE;
            let end_idx = (start_idx + SUB_BATCH_SIZE).min(batch_size);
            let sub_batch_start_id = batch_start_id + start_idx as u64;

            // Slice the data for this sub-batch
            let sub_names = &names[start_idx..end_idx];
            let sub_seqs = &seqs[start_idx..end_idx];

            process_sub_batch_internal(
                &bwa_idx,
                &pac_data,
                &opt,
                sub_names,
                sub_seqs,
                sub_batch_start_id,
                engine,
            )
        })
        .collect();

    // Flatten sub-batch results into final result vector
    let mut all_alignments = Vec::with_capacity(batch_size);
    for sub_result in sub_batch_results {
        all_alignments.extend(sub_result);
    }

    all_alignments
}

/// Internal function to process a single sub-batch with cross-read batching
fn process_sub_batch_internal(
    bwa_idx: &Arc<&BwaIndex>,
    pac_data: &Arc<&[u8]>,
    opt: &Arc<&MemOpt>,
    names: &[String],
    seqs: &[Vec<u8>],
    batch_start_id: u64,
    engine: SimdEngineType,
) -> Vec<Vec<Alignment>> {
    let batch_size = names.len();

    if batch_size == 0 {
        return Vec::new();
    }

    // Phase 1: Seeding and chaining (parallel within sub-batch)
    let contexts: Vec<Option<ReadExtensionContext>> = names
        .par_iter()
        .zip(seqs.par_iter())
        .map(|(name, seq)| {
            let (seeds, encoded_query, encoded_query_rc) = find_seeds(bwa_idx, name, seq, opt);

            if seeds.is_empty() {
                return None;
            }

            let (chains, sorted_seeds) = build_and_filter_chains(seeds, opt, seq.len(), name);

            if chains.is_empty() {
                return None;
            }

            Some(ReadExtensionContext {
                query_name: name.clone(),
                encoded_query,
                encoded_query_rc,
                chains,
                seeds: sorted_seeds,
                query_len: seq.len() as i32,
                chain_ref_segments: Vec::new(),
            })
        })
        .collect();

    // Convert Option<Context> to mutable contexts, tracking indices
    let mut read_contexts: Vec<ReadExtensionContext> = Vec::with_capacity(batch_size);
    let mut context_to_read_idx: Vec<usize> = Vec::with_capacity(batch_size);

    for (read_idx, ctx_opt) in contexts.into_iter().enumerate() {
        if let Some(ctx) = ctx_opt {
            read_contexts.push(ctx);
            context_to_read_idx.push(read_idx);
        }
    }

    if read_contexts.is_empty() {
        // All reads unmapped
        return names
            .iter()
            .map(|name| vec![create_unmapped_alignment_internal(name)])
            .collect();
    }

    // Phase 2: Collect extension jobs
    let (mut left_batch, mut right_batch, mappings) =
        collect_extension_jobs_batch(bwa_idx, opt, &mut read_contexts);

    // Phase 3: Execute SIMD scoring
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

    // Phase 4: Distribute results
    let per_read_left_scores =
        convert_batch_results_to_outscores(&left_results, &left_batch, read_contexts.len());
    let per_read_right_scores =
        convert_batch_results_to_outscores(&right_results, &right_batch, read_contexts.len());

    // Phase 5: Finalization
    finalize_alignments(
        bwa_idx,
        pac_data,
        opt,
        names,
        read_contexts,
        context_to_read_idx,
        mappings,
        per_read_left_scores,
        per_read_right_scores,
        batch_start_id,
    )
}

fn finalize_alignments(
    bwa_idx: &Arc<&BwaIndex>,
    pac_data: &Arc<&[u8]>,
    opt: &Arc<&MemOpt>,
    names: &[String],
    read_contexts: Vec<ReadExtensionContext>,
    context_to_read_idx: Vec<usize>,
    mappings: Vec<super::types::ReadExtensionMappings>,
    per_read_left_scores: Vec<Vec<OutScore>>,
    per_read_right_scores: Vec<Vec<OutScore>>,
    batch_start_id: u64,
) -> Vec<Vec<Alignment>> {
    let valid_alignments: Vec<(usize, Vec<Alignment>)> = read_contexts
        .into_par_iter()
        .zip(mappings.par_iter())
        .zip(per_read_left_scores.par_iter())
        .zip(per_read_right_scores.par_iter())
        .enumerate()
        .map(|(ctx_idx, (((ctx, mapping), left_scores), right_scores))| {
            let read_idx = context_to_read_idx[ctx_idx];
            let read_id = batch_start_id + read_idx as u64;

            let regions = merge_extension_scores_to_regions(
                bwa_idx,
                opt,
                &ctx.chains,
                &ctx.seeds,
                &mapping.chain_mappings,
                &ctx.chain_ref_segments,
                left_scores,
                right_scores,
                ctx.query_len,
            );

            if regions.is_empty() {
                return (
                    read_idx,
                    vec![create_unmapped_alignment_internal(&ctx.query_name)],
                );
            }

            let mut filtered_regions: Vec<_> =
                regions.into_iter().filter(|r| r.score >= opt.t).collect();

            if filtered_regions.is_empty() {
                return (
                    read_idx,
                    vec![create_unmapped_alignment_internal(&ctx.query_name)],
                );
            }

            filtered_regions.sort_by(|a, b| b.score.cmp(&a.score));

            let mut alignments = Vec::new();
            for (idx, region) in filtered_regions.iter().enumerate() {
                let cigar_result =
                    generate_cigar_from_region(bwa_idx, pac_data, &ctx.encoded_query, region, opt);

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
                    query_name: ctx.query_name.clone(),
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
                    vec![create_unmapped_alignment_internal(&ctx.query_name)],
                );
            }

            mark_secondary_alignments(&mut alignments, opt);

            (read_idx, alignments)
        })
        .collect();

    // Assemble final results
    let mut all_alignments: Vec<Vec<Alignment>> = names
        .iter()
        .map(|name| vec![create_unmapped_alignment_internal(name)])
        .collect();

    for (read_idx, alignments) in valid_alignments {
        all_alignments[read_idx] = alignments;
    }

    all_alignments
}
