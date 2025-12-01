use super::types::{BatchExtensionResult, ExtensionJobBatch};
use crate::core::alignment::banded_swa::BandedPairWiseSW;
use crate::core::alignment::banded_swa::OutScore;
use crate::compute::simd_abstraction::simd::SimdEngineType;
use super::soa::make_batch_soa; // Import make_batch_soa

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
        // AoS Path (i16 kernels for long reads)
        unsafe { execute_batch_simd_scoring_aos(sw_params, batch, engine) }
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
pub unsafe fn execute_batch_simd_scoring_aos(
    sw_params: &BandedPairWiseSW,
    batch: &ExtensionJobBatch,
    engine: SimdEngineType,
) -> Vec<BatchExtensionResult> {
    if batch.is_empty() {
        return Vec::new();
    }

    let simd_width = match engine {
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        SimdEngineType::Engine512 => 32,
        #[cfg(target_arch = "x86_64")]
        SimdEngineType::Engine256 => 16,
        SimdEngineType::Engine128 => 8,
    };

    // OPTIMIZATION: On 128-bit engines, process 16 jobs at a time (2x batch8)
    // to amortize job collection and result distribution overhead.
    // This "double-pump" approach reduces pipeline overhead on NEON/SSE.
    let effective_chunk_size = if simd_width == 8 { 16 } else { simd_width };

    let mut results = Vec::with_capacity(batch.len());

    // Process in chunks of effective_chunk_size
    for chunk_start in (0..batch.len()).step_by(effective_chunk_size) {
        let chunk_end = (chunk_start + effective_chunk_size).min(batch.len());
        let chunk_jobs = &batch.jobs[chunk_start..chunk_end];

        // Build tuple batch for all jobs in this chunk
        // Format: (query_len, query_slice, ref_len, ref_slice, band_width, h0)
        let simd_batch: Vec<(i32, &[u8], i32, &[u8], i32, i32)> = chunk_jobs
            .iter()
            .enumerate()
            .map(|(i, job)| {
                let job_idx = chunk_start + i;
                (
                    job.query_len,
                    batch.get_query_seq(job_idx),
                    job.ref_len,
                    batch.get_ref_seq(job_idx),
                    job.band_width,
                    job.h0,
                )
            })
            .collect();

        // For 128-bit engines with >8 jobs, use double-pump (2x batch8)
        let scores: Vec<OutScore> = if simd_width == 8 && simd_batch.len() > 8 {
            let (first_half, second_half) = simd_batch.split_at(8);
            let mut scores1 = unsafe { crate::core::alignment::banded_swa::isa_sse_neon::simd_banded_swa_batch8_int16(first_half, sw_params.o_del(), sw_params.e_del(), sw_params.o_ins(), sw_params.e_ins(), sw_params.zdrop(), sw_params.scoring_matrix(), sw_params.alphabet_size()) };
            let scores2 = unsafe { crate::core::alignment::banded_swa::isa_sse_neon::simd_banded_swa_batch8_int16(second_half, sw_params.o_del(), sw_params.e_del(), sw_params.o_ins(), sw_params.e_ins(), sw_params.zdrop(), sw_params.scoring_matrix(), sw_params.alphabet_size()) };
            scores1.extend(scores2);
            scores1
        } else {
            crate::core::alignment::banded_swa::dispatch::simd_banded_swa_dispatch(sw_params, &simd_batch)
        };

        // Map scores back to results with job metadata (single pass for all jobs)
        for (job, score) in chunk_jobs.iter().zip(scores.iter()) {
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
    }

    results
}

/// Dispatch to the appropriate SoA SIMD kernel based on engine type.
fn dispatch_simd_scoring_soa(
    sw_params: &BandedPairWiseSW,
    batch: &ExtensionJobBatch,
    engine: SimdEngineType,
) -> Vec<OutScore> {
    // These dispatch functions will be implemented in Step 4. They are expected
    // to exist on BandedPairWiseSW and accept an ExtensionJobBatch directly.
    // The generic const `W` corresponds to the i8 SIMD width.
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    {
        match engine {
            SimdEngineType::Engine512 => crate::core::alignment::banded_swa::dispatch::simd_banded_swa_dispatch_soa::<64>(sw_params, batch),
            SimdEngineType::Engine256 => crate::core::alignment::banded_swa::dispatch::simd_banded_swa_dispatch_soa::<32>(sw_params, batch),
            SimdEngineType::Engine128 => crate::core::alignment::banded_swa::dispatch::simd_banded_swa_dispatch_soa::<16>(sw_params, batch),
        }
    }

    #[cfg(all(target_arch = "x86_64", not(feature = "avx512")))]
    {
        match engine {
            SimdEngineType::Engine256 => crate::core::alignment::banded_swa::dispatch::simd_banded_swa_dispatch_soa::<32>(sw_params, batch),
            SimdEngineType::Engine128 => crate::core::alignment::banded_swa::dispatch::simd_banded_swa_dispatch_soa::<16>(sw_params, batch),
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        // aarch64 (NEON) or scalar fallback
        match engine {
            SimdEngineType::Engine128 => crate::core::alignment::banded_swa::dispatch::simd_banded_swa_dispatch_soa::<16>(sw_params, batch),
            _ => crate::core::alignment::banded_swa::dispatch::scalar_dispatch_from_soa(sw_params, batch), // Should not happen if engine is selected correctly
        }
    }
}