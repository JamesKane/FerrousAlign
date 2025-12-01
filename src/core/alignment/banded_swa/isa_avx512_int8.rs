//! AVX‑512 int8 thin wrapper (64 lanes)
#![cfg(target_arch = "x86_64")]

use crate::core::alignment::banded_swa::OutScore;
use crate::generate_swa_entry_soa;
use super::engines::SwEngine512;
use crate::core::alignment::banded_swa::KernelParams;
use crate::core::alignment::banded_swa::kernel::sw_kernel_avx512_with_ws;
use crate::core::alignment::shared_types::{AlignJob};
use crate::core::alignment::workspace::{with_workspace, OwnedSwSoA};
use crate::core::alignment::shared_types::{KernelConfig, GapPenalties, Banding, ScoringMatrix};

/// AVX-512-optimized banded Smith-Waterman for batches of up to 64 alignments
/// Uses arena-backed SoA buffers and reusable DP rows (no per-call heap allocs).
#[deprecated(
    since = "0.7.0",
    note = "Legacy AoS entry point; will be removed. Use SoA dispatch functions instead."
)]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn simd_banded_swa_batch64(
    batch: &[(i32, &[u8], i32, &[u8], i32, i32)],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    zdrop: i32,
    mat: &[i8; 25],
    m: i32,
) -> Vec<OutScore> {
    const W: usize = 64;

    // Convert legacy AoS tuples into AlignJob slice
    let mut jobs: [AlignJob; W] = [AlignJob { query: &[], target: &[], qlen: 0, tlen: 0, band: 0, h0: 0 }; W];
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

    let soa = with_workspace(|ws| {
        ws.ensure_and_transpose_banded_owned(&jobs[..lanes], W)
    });

    let cfg = KernelConfig {
        gaps: GapPenalties { o_del, e_del, o_ins, e_ins },
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
        // Use AVX-512 fast path (falls back to generic kernel internally for now)
        sw_kernel_avx512_with_ws::<W, SwEngine512>(&params, ws)
    })
}

generate_swa_entry_soa!(
    name = simd_banded_swa_batch64_soa,
    width = 64,
    engine = SwEngine512,
    cfg = cfg(all(target_arch = "x86_64", feature = "avx512")),
    target_feature = "avx512bw",
);