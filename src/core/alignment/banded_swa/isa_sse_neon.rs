

use crate::core::alignment::banded_swa::OutScore;
use super::engines::SwEngine128;
use crate::core::alignment::banded_swa::KernelParams;
use crate::core::alignment::banded_swa::kernel::sw_kernel;
use crate::core::alignment::banded_swa::kernel::sw_kernel_with_ws;

use super::engines16::SwEngine128_16;
use crate::generate_swa_entry_i16;
use crate::generate_swa_entry_i16_soa;


use super::shared::{pad_batch, soa_transform}; // Updated path
use crate::core::alignment::shared_types::AlignJob;
use crate::core::alignment::shared_types::SoAProvider;
use crate::core::alignment::workspace::BandedSoAProvider;
use crate::core::alignment::shared_types::WorkspaceArena;
use crate::core::alignment::workspace::with_workspace;



/// SSE/NEON-optimized banded Smith-Waterman for batches of up to 16 alignments
/// Processes 16 alignments in parallel (baseline SIMD for all platforms)
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn simd_banded_swa_batch16(
    batch: &[(i32, &[u8], i32, &[u8], i32, i32)],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    zdrop: i32,
    mat: &[i8; 25],
    m: i32,
) -> Vec<OutScore> {
    const W: usize = 16;

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

    // Build SoA using 64B-aligned provider (no per-call Vec allocations)
    let mut provider = BandedSoAProvider::new();
    let soa = provider.ensure_and_transpose(&jobs[..lanes], W);

    let params = KernelParams {
        batch,
        query_soa: soa.query_soa,
        target_soa: soa.target_soa,
        qlen: soa.qlen,
        tlen: soa.tlen,
        h0: soa.h0,
        w: soa.band,
        max_qlen: soa.max_qlen,
        max_tlen: soa.max_tlen,
        o_del,
        e_del,
        o_ins,
        e_ins,
        zdrop,
        mat,
        m,
    };

    // Use workspace-powered kernel variant to avoid per-call row allocations
    with_workspace(|ws| sw_kernel_with_ws::<W, SwEngine128>(&params, ws))
}



use crate::generate_swa_entry_soa; // This macro is exported at crate root by shared module

#[cfg(target_arch = "x86_64")]
generate_swa_entry_soa!(
    name = simd_banded_swa_batch16_soa,
    width = 16,
    engine = SwEngine128,
    cfg = cfg(target_arch = "x86_64"),
    target_feature = "sse2",
);


generate_swa_entry_i16!(
    name = simd_banded_swa_batch8_int16,
    width = 8,
    engine = SwEngine128_16,
    cfg = cfg(any(target_arch = "x86_64", target_arch = "aarch64")),
    target_feature = "",
);

generate_swa_entry_i16_soa!(
    name = simd_banded_swa_batch8_int16_soa,
    width = 8,
    engine = SwEngine128_16,
    cfg = cfg(any(target_arch = "x86_64", target_arch = "aarch64")),
    target_feature = "",
);

#[cfg(target_arch = "aarch64")]
generate_swa_entry_soa!(
    name = simd_banded_swa_batch16_soa,
    width = 16,
    engine = SwEngine128,
    cfg = cfg(target_arch = "aarch64"),
    target_feature = "neon",
);