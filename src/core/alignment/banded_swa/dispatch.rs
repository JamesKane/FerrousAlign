use super::shared::SoAInputs;
use crate::alignment::banded_swa::BandedPairWiseSW;
use crate::alignment::banded_swa::scalar::implementation::scalar_banded_swa;
use crate::core::alignment::banded_swa::types::{AlignmentResult, ExtensionDirection, OutScore};

use crate::pipelines::linear::batch_extension::ExtensionJobBatch;

/// Dispatch to the appropriate SoA SIMD kernel based on engine type.
///
/// This function orchestrates batched scoring on pre-formatted SoA data.
/// It iterates over chunks defined in the `ExtensionJobBatch`, prepares
/// `SoAInputs` for each, and calls the appropriate ISA-specific kernel.
pub fn simd_banded_swa_dispatch_soa<const W: usize>(
    sw_params: &BandedPairWiseSW,
    batch: &ExtensionJobBatch,
) -> Vec<OutScore> {
    let mut all_results = Vec::with_capacity(batch.len());
    if batch.is_empty() {
        return all_results;
    }

    let num_chunks = batch.jobs.len().div_ceil(W);

    for chunk_idx in 0..num_chunks {
        let job_idx_start = chunk_idx * W;
        let chunk_size = (batch.len() - job_idx_start).min(W);
        if chunk_size == 0 {
            continue;
        }
        let jobs_chunk = &batch.jobs[job_idx_start..job_idx_start + chunk_size];

        // This implementation assumes that `make_batch_soa` has stored
        // chunk metadata (offsets and max lengths) in the `pos_offsets` vector
        // in groups of 4: [q_off, t_off, max_q, max_t, ...].
        let meta_offset = chunk_idx * 4;
        if meta_offset + 3 >= batch.pos_offsets.len() {
            // This indicates a problem with how pos_offsets was populated.
            // For now, we'll skip this chunk to avoid a panic.
            // A proper fix would be in `make_batch_soa`.
            log::error!("Incomplete pos_offsets metadata for chunk {chunk_idx}");
            continue;
        }
        let q_offset = batch.pos_offsets[meta_offset];
        let t_offset = batch.pos_offsets[meta_offset + 1];
        let max_qlen_chunk = batch.pos_offsets[meta_offset + 2];
        let max_tlen_chunk = batch.pos_offsets[meta_offset + 3];

        let q_soa_chunk_len = max_qlen_chunk * W;
        let t_soa_chunk_len = max_tlen_chunk * W;

        let q_soa_chunk = &batch.query_soa[q_offset..q_offset + q_soa_chunk_len];
        let t_soa_chunk = &batch.target_soa[t_offset..t_offset + t_soa_chunk_len];

        let mut qlen = [0i8; W];
        let mut tlen = [0i8; W];
        let mut h0 = [0i8; W];
        let mut w_arr = [0i8; W];

        for i in 0..chunk_size {
            let job = &jobs_chunk[i];
            qlen[i] = job.query_len.min(127) as i8;
            tlen[i] = job.ref_len.min(127) as i8;
            h0[i] = job.h0 as i8;
            w_arr[i] = job.band_width as i8;
        }

        let inputs = SoAInputs {
            query_soa: q_soa_chunk,
            target_soa: t_soa_chunk,
            qlen: &qlen[..chunk_size],
            tlen: &tlen[..chunk_size],
            w: &w_arr[..chunk_size],
            h0: &h0[..chunk_size],
            lanes: chunk_size,
            max_qlen: max_qlen_chunk as i32,
            max_tlen: max_tlen_chunk as i32,
        };

        let chunk_results = unsafe {
            if W == 64 {
                #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
                {
                    super::isa_avx512_int8::simd_banded_swa_batch64_soa(
                        &inputs,
                        chunk_size,
                        sw_params.o_del(),
                        sw_params.e_del(),
                        sw_params.o_ins(),
                        sw_params.e_ins(),
                        sw_params.zdrop(),
                        sw_params.scoring_matrix(),
                        sw_params.alphabet_size(),
                    )
                }
                #[cfg(not(all(target_arch = "x86_64", feature = "avx512")))]
                {
                    unreachable!("AVX-512 support not compiled, but dispatch width was 64");
                }
            } else if W == 32 {
                #[cfg(target_arch = "x86_64")]
                {
                    super::isa_avx2::simd_banded_swa_batch32_soa(
                        &inputs,
                        chunk_size,
                        sw_params.o_del(),
                        sw_params.e_del(),
                        sw_params.o_ins(),
                        sw_params.e_ins(),
                        sw_params.zdrop(),
                        sw_params.scoring_matrix(),
                        sw_params.alphabet_size(),
                    )
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    unreachable!("AVX2 support not compiled, but dispatch width was 32");
                }
            } else if W == 16 {
                super::isa_sse_neon::simd_banded_swa_batch16_soa(
                    &inputs,
                    chunk_size,
                    sw_params.o_del(),
                    sw_params.e_del(),
                    sw_params.o_ins(),
                    sw_params.e_ins(),
                    sw_params.zdrop(),
                    sw_params.scoring_matrix(),
                    sw_params.alphabet_size(),
                )
            } else {
                unreachable!("Invalid SIMD width for SoA dispatch");
            }
        };
        all_results.extend(chunk_results);
    }
    all_results
}

/// Scalar fallback for SoA dispatch path.
pub fn scalar_dispatch_from_soa(
    sw_params: &BandedPairWiseSW,
    batch: &ExtensionJobBatch,
) -> Vec<OutScore> {
    let mut results = Vec::with_capacity(batch.len());
    for i in 0..batch.len() {
        let job = &batch.jobs[i];
        let q_seq = batch.get_query_seq(i);
        let r_seq = batch.get_ref_seq(i);
        let (score, _, _, _) = scalar_banded_swa(
            sw_params,
            job.query_len,
            q_seq,
            job.ref_len,
            r_seq,
            job.band_width,
            job.h0,
        );
        results.push(score);
    }
    results
}

/// Runtime dispatch to optimal SIMD implementation based on CPU features
///
/// **Current Status**:
/// - ✅ SSE/NEON (128-bit, 16-way): Fully implemented
/// - ⏳ AVX2 (256-bit, 32-way): Infrastructure ready, kernel TODO
/// - ⏳ AVX-512 (512-bit, 64-way): Infrastructure ready, kernel TODO
///
/// **Performance Expectations**:
/// - AVX2: ~1.8-2.2x speedup over SSE (memory-bound workload)
/// - AVX-512: ~2.5-3.0x speedup over SSE (on compatible CPUs)
#[deprecated(
    since = "0.7.0",
    note = "Legacy AoS dispatch; will be removed. Benches should be updated to use SoA entry points."
)]
pub fn simd_banded_swa_dispatch(
    _sw_params: &BandedPairWiseSW,
    _batch: &[(i32, &[u8], i32, &[u8], i32, i32)],
) -> Vec<OutScore> {
    panic!(
        "Legacy AoS dispatch is deprecated and will be removed. Benches should be updated to use SoA entry points."
    );
}

/// Runtime dispatch for 16-bit SIMD batch scoring (score-only, no CIGAR)
///
/// Uses i16 arithmetic to handle sequences where max score > 127.
/// For typical 150bp reads with match=1, max score = 150 which overflows i8.
///
/// **Important**: This processes 8 alignments in parallel (vs 16 for 8-bit).
/// Use this function when:
/// - seq_len * match_score >= 127
/// - For 150bp reads with match=1, always use this version
#[deprecated(
    since = "0.7.0",
    note = "Legacy AoS dispatch; will be removed. Benches should be updated to use SoA entry points."
)]
pub fn simd_banded_swa_dispatch_int16(
    _sw_params: &BandedPairWiseSW,
    _batch: &[(i32, &[u8], i32, &[u8], i32, i32)],
) -> Vec<OutScore> {
    panic!(
        "Legacy AoS dispatch is deprecated and will be removed. Benches should be updated to use SoA entry points."
    );
}

/// Runtime dispatch version of batch alignment with CIGAR generation
///
/// **Current Implementation**: Uses scalar Smith-Waterman for both scoring and CIGAR.
/// This matches the proven C++ bwa-mem2 approach where CIGAR generation is done
/// via scalar traceback (not SIMD).
///
/// **Future Optimization (TODO)**:
/// To achieve BWA-MEM2 performance, we need to implement deferred CIGAR generation:
/// 1. Extension phase: SIMD batch scoring only (scores for ALL chains)
/// 2. Finalization phase: Filter chains by score
/// 3. SAM output phase: Generate CIGARs only for surviving alignments
///
/// This would eliminate ~80-90% of CIGAR generation work (which is 46% of CPU time).
/// The 16-bit SIMD batch scoring function (simd_banded_swa_batch8_int16) is ready
/// for this optimization but requires architectural changes to defer CIGAR generation.
#[deprecated(
    since = "0.7.0",
    note = "Legacy scalar dispatch; will be removed. Use the SoA pipeline with deferred CIGAR generation."
)]
pub fn simd_banded_swa_dispatch_with_cigar(
    _sw_params: &BandedPairWiseSW,
    _batch: &[(
        i32,
        Vec<u8>,
        i32,
        Vec<u8>,
        i32,
        i32,
        Option<ExtensionDirection>,
    )],
) -> Vec<AlignmentResult> {
    panic!(
        "Legacy scalar dispatch is deprecated and will be removed. Use the SoA pipeline with deferred CIGAR generation."
    );
}
