use super::soa::make_batch_soa;
use super::types::{BatchExtensionResult, ExtensionJobBatch};
use crate::compute::simd_abstraction::simd::SimdEngineType;
use crate::core::alignment::banded_swa::BandedPairWiseSW;
use crate::core::alignment::banded_swa::OutScore;
use crate::core::alignment::banded_swa::shared::{SoAInputs, SoAInputs16};
use crate::core::alignment::kswv_batch::KswResult;
use crate::core::alignment::shared_types::KswSoA;

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

    // Convert batch to SoA layout. The number of lanes depends on the SIMD engine.
    match engine {
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        SimdEngineType::Engine512 => {
            let (q, t, p) = make_batch_soa::<64>(&batch.jobs, &batch.query_seqs, &batch.ref_seqs);
            batch.query_soa = q;
            batch.target_soa = t;
            batch.pos_offsets = p;
            batch.lanes = 64;
        }
        #[cfg(target_arch = "x86_64")]
        SimdEngineType::Engine256 => {
            let (q, t, p) = make_batch_soa::<32>(&batch.jobs, &batch.query_seqs, &batch.ref_seqs);
            batch.query_soa = q;
            batch.target_soa = t;
            batch.pos_offsets = p;
            batch.lanes = 32;
        }
        SimdEngineType::Engine128 => {
            let (q, t, p) = make_batch_soa::<16>(&batch.jobs, &batch.query_seqs, &batch.ref_seqs);
            batch.query_soa = q;
            batch.target_soa = t;
            batch.pos_offsets = p;
            batch.lanes = 16;
        }
    }

    let scores = dispatch_banded_swa_soa(sw_params, batch, engine);

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
/// This function serves as the single point of truth for dispatch policy. It:
/// 1. Computes the maximum sequence length in the batch.
/// 2. Selects an i8 kernel for lengths <= 128, or an i16 kernel for longer sequences.
/// 3. Selects the SIMD width based on the `SimdEngineType`.
/// 4. Calls the appropriate SoA-based SIMD kernel.
fn dispatch_banded_swa_soa(
    sw_params: &BandedPairWiseSW,
    batch: &ExtensionJobBatch,
    engine: SimdEngineType,
) -> Vec<OutScore> {
    let max_len = batch
        .jobs
        .iter()
        .map(|j| j.query_len.max(j.ref_len))
        .max()
        .unwrap_or(0);

    let use_i8_path = max_len <= 128;

    let num_jobs = batch.len();
    let o_del = sw_params.o_del();
    let e_del = sw_params.e_del();
    let o_ins = sw_params.o_ins();
    let e_ins = sw_params.e_ins();
    let zdrop = sw_params.zdrop();
    let mat = sw_params.scoring_matrix();
    let m = sw_params.alphabet_size();

    if use_i8_path {
        // i8 path for shorter sequences
        let simd_width = batch.lanes;
        let mut qlen_vec = vec![0i8; simd_width];
        let mut tlen_vec = vec![0i8; simd_width];
        let mut h0_vec = vec![0i8; simd_width];
        let mut w_vec = vec![0i8; simd_width];

        for (idx, job) in batch.jobs.iter().enumerate() {
            if idx < simd_width {
                qlen_vec[idx] = job.query_len.min(127) as i8;
                tlen_vec[idx] = job.ref_len.min(127) as i8;
                h0_vec[idx] = job.h0 as i8;
                w_vec[idx] = job.band_width as i8;
            }
        }

        let inputs = SoAInputs {
            query_soa: &batch.query_soa,
            target_soa: &batch.target_soa,
            qlen: &qlen_vec,
            tlen: &tlen_vec,
            h0: &h0_vec,
            w: &w_vec,
            lanes: batch.lanes,
            max_qlen: batch.jobs.iter().map(|j| j.query_len).max().unwrap_or(0),
            max_tlen: batch.jobs.iter().map(|j| j.ref_len).max().unwrap_or(0),
        };

        unsafe {
            match engine {
                #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
                SimdEngineType::Engine512 => simd_banded_swa_batch64_soa(
                    &inputs, num_jobs, o_del, e_del, o_ins, e_ins, zdrop, mat, m,
                ),
                #[cfg(target_arch = "x86_64")]
                SimdEngineType::Engine256 => simd_banded_swa_batch32_soa(
                    &inputs, num_jobs, o_del, e_del, o_ins, e_ins, zdrop, mat, m,
                ),
                #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
                SimdEngineType::Engine128 => simd_banded_swa_batch16_soa(
                    &inputs, num_jobs, o_del, e_del, o_ins, e_ins, zdrop, mat, m,
                ),
            }
        }
    } else {
        // i16 path for longer sequences
        let simd_width = match engine {
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            SimdEngineType::Engine512 => 32,
            #[cfg(target_arch = "x86_64")]
            SimdEngineType::Engine256 => 16,
            SimdEngineType::Engine128 => 8,
        };

        let mut qlen_vec = vec![0i8; simd_width];
        let mut tlen_vec = vec![0i8; simd_width];
        let mut h0_vec = vec![0i16; simd_width];
        let mut w_vec = vec![0i8; simd_width];

        let query_soa_i16: Vec<i16> = batch.query_soa.iter().map(|&x| x as i16).collect();
        let target_soa_i16: Vec<i16> = batch.target_soa.iter().map(|&x| x as i16).collect();

        for (idx, job) in batch.jobs.iter().enumerate() {
            if idx < simd_width {
                qlen_vec[idx] = job.query_len.min(127) as i8;
                tlen_vec[idx] = job.ref_len.min(127) as i8;
                h0_vec[idx] = job.h0 as i16;
                w_vec[idx] = job.band_width as i8;
            }
        }

        let inputs = SoAInputs16 {
            query_soa: &query_soa_i16,
            target_soa: &target_soa_i16,
            qlen: &qlen_vec,
            tlen: &tlen_vec,
            h0: &h0_vec,
            w: &w_vec,
            max_qlen: batch.jobs.iter().map(|j| j.query_len).max().unwrap_or(0),
            max_tlen: batch.jobs.iter().map(|j| j.ref_len).max().unwrap_or(0),
        };

        unsafe {
            match engine {
                #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
                SimdEngineType::Engine128 => simd_banded_swa_batch8_int16_soa(
                    &inputs, num_jobs, o_del, e_del, o_ins, e_ins, zdrop, mat, m,
                ),
                #[cfg(target_arch = "x86_64")]
                SimdEngineType::Engine256 => simd_banded_swa_batch16_int16_soa(
                    &inputs, num_jobs, o_del, e_del, o_ins, e_ins, zdrop, mat, m,
                ),
                #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
                SimdEngineType::Engine512 => simd_banded_swa_batch32_int16_soa(
                    &inputs, num_jobs, o_del, e_del, o_ins, e_ins, zdrop, mat, m,
                ),
            }
        }
    }
}

pub fn dispatch_kswv_soa(
    inputs: &KswSoA,
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
    match engine {
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        SimdEngineType::Engine512 => unsafe {
            kswv_batch64_soa(
                inputs,
                count,
                match_score,
                mismatch_penalty,
                o_del,
                e_del,
                o_ins,
                e_ins,
                ambig_penalty,
                debug,
            )
        },
        #[cfg(target_arch = "x86_64")]
        SimdEngineType::Engine256 => unsafe {
            kswv_batch32_soa(
                inputs,
                count,
                match_score,
                mismatch_penalty,
                o_del,
                e_del,
                o_ins,
                e_ins,
                ambig_penalty,
                debug,
            )
        },
        SimdEngineType::Engine128 => unsafe {
            kswv_batch16_soa(
                inputs,
                count,
                match_score,
                mismatch_penalty,
                o_del,
                e_del,
                o_ins,
                e_ins,
                ambig_penalty,
                debug,
            )
        },
    }
}
