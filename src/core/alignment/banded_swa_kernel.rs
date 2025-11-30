//! Generic kernel surface for banded Smith–Waterman.
//!
//! This module defines the minimal trait (`SwSimd`) and the parameter carrier
//! (`KernelParams`) that a shared DP kernel will use. In this stage, the kernel
//! function is a stub to allow incremental adoption by per‑ISA wrappers without
//! changing behavior yet.

use crate::alignment::banded_swa::OutScore;

/// Minimal SIMD engine contract for the shared SW kernel.
///
/// Note: We intentionally keep this trait very small initially. As the shared
/// kernel is implemented, only the actually required ops will be added here,
/// and concrete engines (SSE/NEON/AVX2/AVX‑512/portable) will implement it.
pub trait SwSimd: Copy {
    /// Number of 8‑bit lanes processed in parallel.
    const LANES: usize;
}

/// Input parameters for the shared SW kernel.
///
/// The kernel operates on Structure‑of‑Arrays (SoA) buffers for query and target
/// sequences. Callers are expected to ensure the buffers are sized as
/// `max_len * E::LANES` and padded appropriately.
#[derive(Debug)]
pub struct KernelParams<'a> {
    /// Batch of alignments (AoS metadata). Tuples are
    /// (qlen, query, tlen, target, band_w, h0).
    pub batch: &'a [(i32, &'a [u8], i32, &'a [u8], i32, i32)],

    /// Query sequences in SoA layout: `query_soa[pos * LANES + lane]`.
    pub query_soa: &'a [u8],

    /// Target sequences in SoA layout: `target_soa[pos * LANES + lane]`.
    pub target_soa: &'a [u8],

    /// Per‑lane query lengths (clamped), in 8‑bit lanes.
    pub qlen: &'a [i8],
    /// Per‑lane target lengths (clamped), in 8‑bit lanes.
    pub tlen: &'a [i8],
    /// Per‑lane initial score h0, in 8‑bit lanes.
    pub h0: &'a [i8],
    /// Per‑lane band width, in 8‑bit lanes.
    pub w: &'a [i8],

    /// Maximum clamped query length across lanes.
    pub max_qlen: i32,
    /// Maximum clamped target length across lanes.
    pub max_tlen: i32,

    /// Gap penalties and z‑drop.
    pub o_del: i32,
    pub e_del: i32,
    pub o_ins: i32,
    pub e_ins: i32,
    pub zdrop: i32,

    /// Scoring matrix (5x5: A,C,G,T,N) and its dimension (typically 5).
    pub mat: &'a [i8; 25],
    pub m: i32,
}

/// Shared banded SW kernel (placeholder).
///
/// Safety: The caller must ensure that `query_soa` and `target_soa` are laid
/// out as SoA with length at least `max_len * E::LANES`, padded to avoid
/// out‑of‑bounds loads, and that `qlen/tlen/h0/w` slices have length
/// `E::LANES`.
#[inline]
pub unsafe fn sw_kernel<const W: usize, E: SwSimd>(_params: &KernelParams<'_>) -> Vec<OutScore> {
    // Placeholder implementation to allow incremental adoption without any
    // behavioral changes. ISA entry points will be redirected here in a later
    // step, once the kernel body is implemented.
    Vec::new()
}
