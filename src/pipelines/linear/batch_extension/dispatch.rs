use super::soa::make_batch_soa;
use super::types::{BatchExtensionResult, ExtensionJobBatch};
use crate::compute::simd_abstraction::simd::SimdEngineType;
use crate::core::alignment::banded_swa::BandedPairWiseSW;
use crate::core::alignment::banded_swa::OutScore;
use crate::core::alignment::banded_swa::shared::{SoAInputs, SoAInputs16};
use crate::core::alignment::kswv_batch::KswResult;

#[cfg(target_arch = "x86_64")]
use crate::core::alignment::banded_swa::isa_avx2::{
    simd_banded_swa_batch16_int16_soa, simd_banded_swa_batch32_soa,
};
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
use crate::core::alignment::banded_swa::isa_avx512_int8::simd_banded_swa_batch64_soa;
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
use crate::core::alignment::banded_swa::isa_avx512_int16::simd_banded_swa_batch32_int16_soa;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::core::alignment::banded_swa::isa_sse_neon::{
    simd_banded_swa_batch8_int16_soa, simd_banded_swa_batch16_soa,
};
#[cfg(target_arch = "x86_64")]
use crate::core::alignment::kswv_avx2::kswv_batch32_soa;
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
use crate::core::alignment::kswv_avx512::kswv_batch64_soa;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::core::alignment::kswv_sse_neon::kswv_batch16_soa;

/// Execute batched SIMD scoring for extension jobs.
/// This function converts the batch to a Structure-of-Arrays (SoA) layout
/// and then calls the centralized dispatch function `dispatch_banded_swa_soa`.
pub fn execute_batch_simd_scoring(
    sw_params: &BandedPairWiseSW,
    batch: &mut ExtensionJobBatch,
    engine: SimdEngineType,
) -> Vec<BatchExtensionResult> {
    if batch.is_empty() {
        return Vec::new();
    }

    // Determine max length to decide between i8 and i16 paths.
    // i8 path is faster but supports only seqs <= 128bp.
    let max_len = batch
        .jobs
        .iter()
        .map(|j| j.query_len.max(j.ref_len))
        .max()
        .unwrap_or(0);

    let use_i16 = max_len > 128;

    // Convert batch to SoA layout. The number of lanes depends on the SIMD engine.
    // If using i16 path, we halve the lane count to fit 16-bit scores in the vector registers.
    match engine {
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        SimdEngineType::Engine512 => {
            if use_i16 {
                // AVX512 i16: 32 lanes
                let (q, t, p) = make_batch_soa::<32>(&batch.jobs, &batch.query_seqs, &batch.ref_seqs);
                batch.query_soa = q;
                batch.target_soa = t;
                batch.pos_offsets = p;
                batch.lanes = 32;
            } else {
                // AVX512 i8: 64 lanes
                let (q, t, p) = make_batch_soa::<64>(&batch.jobs, &batch.query_seqs, &batch.ref_seqs);
                batch.query_soa = q;
                batch.target_soa = t;
                batch.pos_offsets = p;
                batch.lanes = 64;
            }
        }
        #[cfg(target_arch = "x86_64")]
        SimdEngineType::Engine256 => {
            if use_i16 {
                // AVX2 i16: 16 lanes
                let (q, t, p) = make_batch_soa::<16>(&batch.jobs, &batch.query_seqs, &batch.ref_seqs);
                batch.query_soa = q;
                batch.target_soa = t;
                batch.pos_offsets = p;
                batch.lanes = 16;
            } else {
                // AVX2 i8: 32 lanes
                let (q, t, p) = make_batch_soa::<32>(&batch.jobs, &batch.query_seqs, &batch.ref_seqs);
                batch.query_soa = q;
                batch.target_soa = t;
                batch.pos_offsets = p;
                batch.lanes = 32;
            }
        }
        SimdEngineType::Engine128 => {
            if use_i16 {
                // SSE/Neon i16: 8 lanes
                let (q, t, p) = make_batch_soa::<8>(&batch.jobs, &batch.query_seqs, &batch.ref_seqs);
                batch.query_soa = q;
                batch.target_soa = t;
                batch.pos_offsets = p;
                batch.lanes = 8;
            } else {
                // SSE/Neon i8: 16 lanes
                let (q, t, p) = make_batch_soa::<16>(&batch.jobs, &batch.query_seqs, &batch.ref_seqs);
                batch.query_soa = q;
                batch.target_soa = t;
                batch.pos_offsets = p;
                batch.lanes = 16;
            }
        }
    }

    let scores = dispatch_banded_swa_soa(sw_params, batch, engine, use_i16);

    // Map scores back to results
    let mut results = Vec::with_capacity(batch.len());
    for (job, score) in batch.jobs.iter().zip(scores.iter()) {
        results.push(BatchExtensionResult {
            read_idx: job.read_idx,
            chain_idx: job.chain_idx,
            seed_idx: job.seed_idx,
            direction: job.direction,
            score: score.score,
            query_end: score.query_end_pos,
            ref_end: score.target_end_pos,
            gscore: score.global_score,
            gref_end: score.gtarget_end_pos,
            max_off: score.max_offset,
        });
    }
    results
}

/// Centralized dispatch for banded Smith-Waterman alignment using SoA layout.
///
/// This function iterates over all chunks created by make_batch_soa and calls
/// the appropriate SIMD kernel for each chunk. The pos_offsets array contains
/// metadata for each chunk: [q_offset, t_offset, max_qlen, max_tlen].
fn dispatch_banded_swa_soa(
    sw_params: &BandedPairWiseSW,
    batch: &ExtensionJobBatch,
    engine: SimdEngineType,
    use_i16: bool,
) -> Vec<OutScore> {
    let mut all_scores = Vec::with_capacity(batch.jobs.len());

    let simd_width = batch.lanes;
    if simd_width == 0 || batch.jobs.is_empty() {
        return all_scores;
    }

    let o_del = sw_params.o_del();
    let e_del = sw_params.e_del();
    let o_ins = sw_params.o_ins();
    let e_ins = sw_params.e_ins();
    let zdrop = sw_params.zdrop();
    let mat = sw_params.scoring_matrix();
    let m = sw_params.alphabet_size();

    // pos_offsets has 4 entries per chunk: [q_offset, t_offset, max_qlen, max_tlen]
    let num_chunks = batch.pos_offsets.len() / 4;

    for chunk_idx in 0..num_chunks {
        let meta_base = chunk_idx * 4;
        let q_offset = batch.pos_offsets[meta_base];
        let t_offset = batch.pos_offsets[meta_base + 1];
        let max_qlen = batch.pos_offsets[meta_base + 2] as i32;
        let max_tlen = batch.pos_offsets[meta_base + 3] as i32;

        // Jobs for this chunk
        let job_start = chunk_idx * simd_width;
        let job_end = (job_start + simd_width).min(batch.jobs.len());
        let chunk_jobs = &batch.jobs[job_start..job_end];
        let actual_lanes = chunk_jobs.len();

        if actual_lanes == 0 {
            continue;
        }

        // SoA slice sizes for this chunk
        let q_soa_size = max_qlen as usize * simd_width;
        let t_soa_size = max_tlen as usize * simd_width;

        // Bounds check
        if q_offset + q_soa_size > batch.query_soa.len() ||
           t_offset + t_soa_size > batch.target_soa.len() {
            log::error!(
                "SoA bounds error: chunk {} q_offset={} q_size={} q_total={} t_offset={} t_size={} t_total={}",
                chunk_idx, q_offset, q_soa_size, batch.query_soa.len(),
                t_offset, t_soa_size, batch.target_soa.len()
            );
            break;
        }

        let chunk_query_soa = &batch.query_soa[q_offset..q_offset + q_soa_size];
        let chunk_target_soa = &batch.target_soa[t_offset..t_offset + t_soa_size];

        // Prepare per-lane metadata
        let mut w_vec = vec![0i8; simd_width];
        for (i, job) in chunk_jobs.iter().enumerate() {
            w_vec[i] = job.band_width as i8;
        }

        let chunk_scores = if use_i16 {
            // i16 path for sequences > 128bp
            let mut qlen_vec = vec![0i16; simd_width];
            let mut tlen_vec = vec![0i16; simd_width];
            let mut h0_vec = vec![0i16; simd_width];

            for (i, job) in chunk_jobs.iter().enumerate() {
                qlen_vec[i] = job.query_len as i16;
                tlen_vec[i] = job.ref_len as i16;
                h0_vec[i] = job.h0 as i16;
            }

            // Convert SoA to i16
            let query_soa_i16: Vec<i16> = chunk_query_soa.iter().map(|&x| x as i16).collect();
            let target_soa_i16: Vec<i16> = chunk_target_soa.iter().map(|&x| x as i16).collect();

            // Note: Kernel reads full SIMD width via loadu, so pass full vectors, not slices
            let inputs = SoAInputs16 {
                query_soa: &query_soa_i16,
                target_soa: &target_soa_i16,
                qlen: &qlen_vec,
                tlen: &tlen_vec,
                h0: &h0_vec,
                w: &w_vec,
                max_qlen,
                max_tlen,
            };

            unsafe {
                match engine {
                    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
                    SimdEngineType::Engine512 => simd_banded_swa_batch32_int16_soa(
                        &inputs, actual_lanes, o_del, e_del, o_ins, e_ins, zdrop, mat, m,
                    ),
                    #[cfg(target_arch = "x86_64")]
                    SimdEngineType::Engine256 => simd_banded_swa_batch16_int16_soa(
                        &inputs, actual_lanes, o_del, e_del, o_ins, e_ins, zdrop, mat, m,
                    ),
                    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
                    SimdEngineType::Engine128 => simd_banded_swa_batch8_int16_soa(
                        &inputs, actual_lanes, o_del, e_del, o_ins, e_ins, zdrop, mat, m,
                    ),
                }
            }
        } else {
            // i8 path for sequences <= 128bp
            let mut qlen_vec = vec![0i8; simd_width];
            let mut tlen_vec = vec![0i8; simd_width];
            let mut h0_vec = vec![0i8; simd_width];

            for (i, job) in chunk_jobs.iter().enumerate() {
                qlen_vec[i] = job.query_len.min(127) as i8;
                tlen_vec[i] = job.ref_len.min(127) as i8;
                h0_vec[i] = job.h0.min(127) as i8;
            }

            // Note: Kernel reads full SIMD width via loadu, so pass full vectors, not slices
            let inputs = SoAInputs {
                query_soa: chunk_query_soa,
                target_soa: chunk_target_soa,
                qlen: &qlen_vec,
                tlen: &tlen_vec,
                h0: &h0_vec,
                w: &w_vec,
                lanes: actual_lanes,
                max_qlen,
                max_tlen,
            };

            unsafe {
                match engine {
                    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
                    SimdEngineType::Engine512 => simd_banded_swa_batch64_soa(
                        &inputs, actual_lanes, o_del, e_del, o_ins, e_ins, zdrop, mat, m,
                    ),
                    #[cfg(target_arch = "x86_64")]
                    SimdEngineType::Engine256 => simd_banded_swa_batch32_soa(
                        &inputs, actual_lanes, o_del, e_del, o_ins, e_ins, zdrop, mat, m,
                    ),
                    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
                    SimdEngineType::Engine128 => simd_banded_swa_batch16_soa(
                        &inputs, actual_lanes, o_del, e_del, o_ins, e_ins, zdrop, mat, m,
                    ),
                }
            }
        };

        all_scores.extend(chunk_scores.into_iter().take(actual_lanes));
    }

    all_scores
}

pub fn dispatch_kswv_soa(
    jobs: &[crate::core::alignment::shared_types::AlignJob],
    ws: &mut crate::core::alignment::workspace::AlignmentWorkspace,
    batch_size: usize,
    count: usize,
    match_score: i8,
    mismatch_penalty: i8,
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    ambig_penalty: i8,
    debug: bool,
    engine: SimdEngineType,
) -> Vec<KswResult> {
    log::debug!(
        "dispatch_kswv_soa: jobs.len()={}, batch_size={}, count={}",
        jobs.len(),
        batch_size,
        count
    );

    // Process jobs in batches of SIMD width
    let mut all_results = Vec::with_capacity(count);

    let mut offset = 0;
    while offset < count {
        let chunk_size = (count - offset).min(batch_size);
        let chunk_jobs = &jobs[offset..offset + chunk_size];

        log::debug!("  Processing chunk offset={offset}, chunk_size={chunk_size}");

        // SAFETY: Get raw pointer to workspace BEFORE any borrows.
        // We'll use this to access kernel buffers while SoA data borrows the workspace.
        // This is safe because SoA buffers and kernel buffers are disjoint fields.
        let ws_ptr: *mut crate::core::alignment::workspace::AlignmentWorkspace = ws as *mut _;

        // Transpose this chunk to SoA layout (borrows ws)
        let inputs = ws.ensure_and_transpose_ksw(chunk_jobs, batch_size);

        // Debug: Check if input lengths are valid
        if chunk_jobs.len() > 0 {
            log::debug!(
                "  First job: qlen={} tlen={} band={} query[0..5]={:?} target[0..5]={:?}",
                chunk_jobs[0].qlen,
                chunk_jobs[0].tlen,
                chunk_jobs[0].band,
                &chunk_jobs[0].query[..chunk_jobs[0].qlen.min(5)],
                &chunk_jobs[0].target[..chunk_jobs[0].tlen.min(5)]
            );
        }

        let chunk_results = unsafe {
            match engine {
                #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
                SimdEngineType::Engine512 => kswv_batch64_soa(
                    &inputs,
                    &mut *ws_ptr,
                    chunk_size,
                    match_score,
                    mismatch_penalty,
                    o_del,
                    e_del,
                    o_ins,
                    e_ins,
                    ambig_penalty,
                    debug,
                ),
                #[cfg(target_arch = "x86_64")]
                SimdEngineType::Engine256 => kswv_batch32_soa(
                    &inputs,
                    &mut *ws_ptr,
                    chunk_size,
                    match_score,
                    mismatch_penalty,
                    o_del,
                    e_del,
                    o_ins,
                    e_ins,
                    ambig_penalty,
                    debug,
                ),
                SimdEngineType::Engine128 => kswv_batch16_soa(
                    &inputs,
                    &mut *ws_ptr,
                    chunk_size,
                    match_score,
                    mismatch_penalty,
                    o_del,
                    e_del,
                    o_ins,
                    e_ins,
                    ambig_penalty,
                    debug,
                ),
            }
        };

        log::debug!("  Chunk returned {} results", chunk_results.len());

        // Debug: Check if any results have non-zero scores
        let non_zero_scores = chunk_results.iter().filter(|r| r.score > 0).count();
        if non_zero_scores == 0 && chunk_results.len() > 0 {
            log::debug!(
                "  WARNING: All {} results have score=0! First result: score={} qe={} te={} qb={} tb={}",
                chunk_results.len(),
                chunk_results[0].score,
                chunk_results[0].qe,
                chunk_results[0].te,
                chunk_results[0].qb,
                chunk_results[0].tb
            );
        } else {
            log::debug!("  {} results have score > 0", non_zero_scores);
        }

        all_results.extend(chunk_results);
        offset += chunk_size;
    }

    log::debug!(
        "dispatch_kswv_soa: returning {} total results",
        all_results.len()
    );
    all_results
}
