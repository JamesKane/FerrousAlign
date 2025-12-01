use super::types::{BatchExtensionResult, ExtensionJobBatch};
use crate::core::alignment::banded_swa::BandedPairWiseSW;
use crate::core::alignment::banded_swa::OutScore;
use crate::compute::simd_abstraction::simd::SimdEngineType;
use super::soa::make_batch_soa; // Import make_batch_soa

// Imports for SoA dispatch
use crate::core::alignment::banded_swa::shared::{SoAInputs, SoAInputs16};
#[cfg(target_arch = "x86_64")]
use crate::core::alignment::banded_swa::isa_avx2::{simd_banded_swa_batch16_int16_soa, simd_banded_swa_batch32_soa};
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::core::alignment::banded_swa::isa_sse_neon::{simd_banded_swa_batch16_soa, simd_banded_swa_batch8_int16_soa};
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
use crate::core::alignment::banded_swa::isa_avx512_int16::simd_banded_swa_batch32_int16_soa;

/// Execute batched SIMD scoring for extension jobs
///
/// This function is the main entry point for scoring a batch. It dynamically
/// chooses between two paths:
/// - **SoA (Structure-of-Arrays) Path**: For reads up to 128bp, it uses an i8
///   kernel with an interleaved memory layout for maximum SIMD throughput.
/// - **AoS (Array-of-Structures) Path**: For longer reads, it falls back to the
///   original i16 kernel, which operates on chunked, non-interleaved data.
pub fn execute_batch_simd_scoring(
    sw_params: &BandedPairWiseSW,
    batch: &mut ExtensionJobBatch, // Now mutable
    engine: SimdEngineType,
) -> Vec<BatchExtensionResult> {
    if batch.is_empty() {
        return Vec::new();
    }

    // Decide execution strategy based on sequence lengths.
    // The SoA path uses i8 kernels, which are efficient for lengths <= 128.
    let max_len = batch
        .jobs
        .iter()
        .map(|j| j.query_len.max(j.ref_len))
        .max()
        .unwrap_or(0);
    let use_soa_path = max_len <= 128;

    if use_soa_path {
        // SoA Path (i8 kernels)
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

        let scores = dispatch_simd_scoring_soa(sw_params, batch, engine);

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
    } else {
        // SoA Path (i16 kernels for long reads)
        // Need to convert existing u8 query_soa/target_soa to i16 versions
        // and extract other i16-specific parameters from batch.jobs.
        let scores = unsafe { dispatch_simd_scoring_soa_i16(sw_params, batch, engine) };

        // Map scores back to results (same as i8 path)
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
}

/// Dispatch to the appropriate SoA SIMD i16 kernel based on engine type.
unsafe fn dispatch_simd_scoring_soa_i16(
    sw_params: &BandedPairWiseSW,
    batch: &ExtensionJobBatch,
    engine: SimdEngineType,
) -> Vec<OutScore> {
    // Determine SIMD width based on engine
    let simd_width = match engine {
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        SimdEngineType::Engine512 => 32,
        #[cfg(target_arch = "x86_64")]
        SimdEngineType::Engine256 => 16,
        SimdEngineType::Engine128 => 8,
        _ => panic!("Unsupported SIMD engine type for i16 SoA dispatch: {:?}", engine),
    };

    // Prepare SoAInputs16 from the batch
    // These vectors will hold the lane-packed data
    let mut qlen_vec = vec![0i8; simd_width]; // Initialize with zeros and correct size
    let mut tlen_vec = vec![0i8; simd_width];
    let mut h0_vec = vec![0i16; simd_width];
    let mut w_vec = vec![0i8; simd_width];

    // Convert query_soa and target_soa (u8) to i16 for the i16 kernel
    let query_soa_i16: Vec<i16> = batch.query_soa.iter().map(|&x| x as i16).collect();
    let target_soa_i16: Vec<i16> = batch.target_soa.iter().map(|&x| x as i16).collect();

    // Populate the lane-packed data from jobs, up to batch.len()
    for (idx, job) in batch.jobs.iter().enumerate() {
        if idx < simd_width { // Ensure we don't write out of bounds of our vectors
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

    let num_jobs = batch.len();
    let o_del = sw_params.o_del();
    let e_del = sw_params.e_del();
    let o_ins = sw_params.o_ins();
    let e_ins = sw_params.e_ins();
    let zdrop = sw_params.zdrop();
    let mat = sw_params.scoring_matrix();
    let m = sw_params.alphabet_size();

    // Dispatch based on engine type
    match engine {
        SimdEngineType::Engine128 => {
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            {
                simd_banded_swa_batch8_int16_soa(&inputs, num_jobs, o_del, e_del, o_ins, e_ins, zdrop, mat, m)
            }
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            {
                panic!("Engine128 not supported on this architecture without x86_64 or aarch64 features for i16 SoA.");
            }
        },
        #[cfg(target_arch = "x86_64")]
        SimdEngineType::Engine256 => {
            simd_banded_swa_batch16_int16_soa(&inputs, num_jobs, o_del, e_del, o_ins, e_ins, zdrop, mat, m)
        },
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        SimdEngineType::Engine512 => {
            simd_banded_swa_batch32_int16_soa(&inputs, num_jobs, o_del, e_del, o_ins, e_ins, zdrop, mat, m)
        },
        _ => panic!("Unsupported SIMD engine type for i16 SoA dispatch: {:?}", engine),
    }
}

/// AoS (Array-of-Structures) path for batched SIMD scoring.
///
/// Processes jobs in chunks matching the SIMD width for maximum lane utilization.
/// This path is used as a fallback for reads longer than 128bp, where the i16
/// kernel is required.
///
/// # NEON Optimization (Session 32)
/// On 128-bit engines (NEON/SSE), we use "double-pump" batching: process 16 jobs
/// at a time using 2x batch8 calls. This amortizes job collection and result
/// distribution overhead, reducing pipeline overhead from ~44% to ~35%.


/// Dispatch to the appropriate SoA SIMD kernel based on engine type.
fn dispatch_simd_scoring_soa(
    sw_params: &BandedPairWiseSW,
    batch: &ExtensionJobBatch,
    engine: SimdEngineType,
) -> Vec<OutScore> {
    let simd_width = batch.lanes; // Use batch.lanes for the actual width

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
        lanes: batch.lanes, // This is already set in execute_batch_simd_scoring
        max_qlen: batch.jobs.iter().map(|j| j.query_len).max().unwrap_or(0),
        max_tlen: batch.jobs.iter().map(|j| j.ref_len).max().unwrap_or(0),
    };

    let num_jobs = batch.len();
    let o_del = sw_params.o_del();
    let e_del = sw_params.e_del();
    let o_ins = sw_params.o_ins();
    let e_ins = sw_params.e_ins();
    let zdrop = sw_params.zdrop();
    let mat = sw_params.scoring_matrix();
    let m = sw_params.alphabet_size();

    unsafe {
        match engine {
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            SimdEngineType::Engine512 => crate::core::alignment::banded_swa::isa_avx512_int8::simd_banded_swa_batch64_soa(&inputs, num_jobs, o_del, e_del, o_ins, e_ins, zdrop, mat, m),
            #[cfg(target_arch = "x86_64")]
            SimdEngineType::Engine256 => crate::core::alignment::banded_swa::isa_avx2::simd_banded_swa_batch32_soa(&inputs, num_jobs, o_del, e_del, o_ins, e_ins, zdrop, mat, m),
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            SimdEngineType::Engine128 => crate::core::alignment::banded_swa::isa_sse_neon::simd_banded_swa_batch16_soa(&inputs, num_jobs, o_del, e_del, o_ins, e_ins, zdrop, mat, m),
            _ => panic!("Unsupported SIMD engine type for i8 SoA dispatch: {:?}", engine),
        }
    }
}