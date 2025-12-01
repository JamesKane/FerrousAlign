// bwa-mem2-rust/src/banded_swa_avx2.rs
//
// AVX2 (256-bit) SIMD implementation of banded Smith-Waterman alignment
// Processes 32 alignments in parallel (2x speedup over SSE)
//
// This is a port of C++ bwa-mem2's smithWaterman256_8 function
// Reference: /Users/jkane/Applications/bwa-mem2/src/bandedSWA.cpp:722-1150

#![cfg(target_arch = "x86_64")]

use super::engines::SwEngine256;
use crate::core::alignment::banded_swa::KernelParams;
use crate::core::alignment::banded_swa::OutScore;
use crate::core::alignment::banded_swa::engines16::SwEngine256_16;
use crate::core::alignment::banded_swa::kernel::sw_kernel_with_ws;
use crate::core::alignment::shared_types::AlignJob;
use crate::core::alignment::shared_types::{Banding, GapPenalties, KernelConfig, ScoringMatrix};
use crate::core::alignment::workspace::with_workspace;
use crate::{generate_swa_entry_i16, generate_swa_entry_i16_soa};

/// AVX2-optimized banded Smith-Waterman for batches of up to 32 alignments
/// Uses arena-backed SoA buffers and reusable DP rows (no per-call heap allocs).
#[deprecated(
    since = "0.7.0",
    note = "Legacy AoS entry point; will be removed. Use SoA dispatch functions instead."
)]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn simd_banded_swa_batch32(
    batch: &[(i32, &[u8], i32, &[u8], i32, i32)],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    zdrop: i32,
    mat: &[i8; 25],
    m: i32,
) -> Vec<OutScore> {
    const W: usize = 32;

    // Follow bwa-mem2 policy: dispatch to 16-bit kernel if any sequence length > 127
    let needs_i16 = batch
        .iter()
        .any(|(ql, _q, tl, _t, _w, _h0)| (*ql > 127) || (*tl > 127));
    if needs_i16 {
        return simd_banded_swa_batch16_int16(batch, o_del, e_del, o_ins, e_ins, zdrop, mat, m);
    }

    // Convert legacy AoS tuples into AlignJob slice
    let mut jobs: [AlignJob; W] = [AlignJob {
        query: &[],
        target: &[],
        qlen: 0,
        tlen: 0,
        band: 0,
        h0: 0,
    }; W];
    let lanes = batch.len().min(W);
    for i in 0..lanes {
        let (ql, q, tl, t, w, h0) = batch[i];
        jobs[i] = AlignJob {
            query: q,
            target: t,
            qlen: ql as usize,
            tlen: tl as usize,
            band: w,
            h0,
        };
    }

    let soa = with_workspace(|ws| ws.ensure_and_transpose_banded_owned(&jobs[..lanes], W));

    let cfg = KernelConfig {
        gaps: GapPenalties {
            o_del,
            e_del,
            o_ins,
            e_ins,
        },
        banding: Banding { band: 0, zdrop },
        scoring: ScoringMatrix { mat5x5: mat, m },
    };

    let params = KernelParams {
        batch,
        query_soa: &soa.query_soa,
        target_soa: &soa.target_soa,
        qlen: &soa.qlen,
        tlen: &soa.tlen,
        h0: &soa.h0,
        w: &soa.band,
        max_qlen: soa.max_qlen,
        max_tlen: soa.max_tlen,
        o_del,
        e_del,
        o_ins,
        e_ins,
        zdrop,
        mat,
        m,
        cfg: Some(cfg),
    };

    with_workspace(|ws| {
        // Use workspace-powered kernel variant to avoid per-call row allocations
        sw_kernel_with_ws::<W, SwEngine256>(&params, ws)
    })
}

/// AVX2-optimized banded Smith-Waterman for batches of up to 16 alignments (16-bit scores)
///
/// **SIMD Width**: 16 lanes (256-bit / 16-bit)
/// **Parallelism**: Processes 16 alignments simultaneously
/// **Score Range**: Full i16 range (-32768 to 32767) for sequences > 127bp
///
/// This is the 16-bit precision version optimized for:
/// - Sequences longer than 127bp where 8-bit scores would overflow
/// - Typical 151bp Illumina reads (max score = 151 with match=1)
///
/// **Performance**: 2x parallelism over SSE 8-wide (8 vs 16 lanes)
generate_swa_entry_i16!(
    name = simd_banded_swa_batch16_int16,
    width = 16,
    engine = SwEngine256_16,
    cfg = cfg(target_arch = "x86_64"),
    target_feature = "avx2",
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_banded_swa_batch16_int16_basic() {
        // Basic test for 16-bit AVX2 batch function
        // Kernel expects 2-bit encoded bases: A=0, C=1, G=2, T=3
        let query: [u8; 4] = [0, 1, 2, 3]; // ACGT in 2-bit encoding
        let target: [u8; 4] = [0, 1, 2, 3]; // ACGT in 2-bit encoding
        let batch = vec![(4, &query[..], 4, &target[..], 10, 0)];

        // Default scoring matrix (match=1, mismatch=0)
        let mut mat = [0i8; 25];
        for i in 0..4 {
            mat[i * 5 + i] = 1; // Match score on diagonal
        }

        let results = unsafe { simd_banded_swa_batch16_int16(&batch, 6, 1, 6, 1, 100, &mat, 5) };

        assert_eq!(results.len(), 1);
        assert!(
            results[0].score > 0,
            "Score {} should be > 0",
            results[0].score
        );
    }

    #[test]
    fn test_simd_banded_swa_batch32_skeleton() {
        // Basic test to ensure the function compiles and runs
        let query = b"ACGT";
        let target = b"ACGT";
        let batch = vec![(4, &query[..], 4, &target[..], 10, 0)];

        let results = unsafe {
            simd_banded_swa_batch32(
                &batch, 6,   // o_del
                1,   // e_del
                6,   // o_ins
                1,   // e_ins
                100, // zdrop
                &[0i8; 25], 5,
            )
        };

        assert_eq!(results.len(), 1);
        // TODO: Add proper assertions once implementation is complete
    }
}

use crate::generate_swa_entry_soa;

generate_swa_entry_soa!(
    name = simd_banded_swa_batch32_soa,
    width = 32,
    engine = SwEngine256,
    cfg = cfg(target_arch = "x86_64"),
    target_feature = "avx2",
);

generate_swa_entry_i16_soa!(
    name = simd_banded_swa_batch16_int16_soa,
    width = 16,
    engine = SwEngine256_16,
    cfg = cfg(target_arch = "x86_64"),
    target_feature = "avx2",
);
