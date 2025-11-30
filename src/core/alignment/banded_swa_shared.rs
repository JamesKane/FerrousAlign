//! Shared, inlineable helpers for banded Smith–Waterman SIMD kernels.
//!
//! These functions are extracted from per-ISA implementations to reduce
//! duplication. They are designed to be aggressively inlined and cheap.

use crate::alignment::banded_swa::OutScore;

/// Pad a batch of jobs to a fixed SIMD width and extract lane-wise parameters.
#[inline(always)]
#[allow(dead_code)]
pub fn pad_batch<'a, const W: usize>(
    batch: &'a [(i32, &'a [u8], i32, &'a [u8], i32, i32)],
) -> (
    [i8; W], // qlen
    [i8; W], // tlen
    [i8; W], // h0
    [i8; W], // w
    i32,     // max_qlen
    i32,     // max_tlen
    [(i32, &'a [u8], i32, &'a [u8], i32, i32); W], // padded batch
) {
    let mut qlen = [0i8; W];
    let mut tlen = [0i8; W];
    let mut h0 = [0i8; W];
    let mut w_arr = [0i8; W];
    let mut max_q = 0i32;
    let mut max_t = 0i32;

    // Pad to width W (truncate extra lanes if provided)
    let mut padded: [(i32, &'a [u8], i32, &'a [u8], i32, i32); W] = [(0, &[][..], 0, &[][..], 0, 0); W];
    for i in 0..W {
        let tup = if i < batch.len() { batch[i] } else { (0, &[][..], 0, &[][..], 0, 0) };
        let (q, _qs, t, _ts, w, h) = tup;
        padded[i] = tup;
        qlen[i] = q.min(127) as i8;
        tlen[i] = t.min(127) as i8;
        h0[i] = h as i8;
        w_arr[i] = w as i8;
        if q > max_q { max_q = q; }
        if t > max_t { max_t = t; }
    }

    (qlen, tlen, h0, w_arr, max_q, max_t, padded)
}

/// Convert sequences to SoA layout for SIMD-friendly access.
#[inline(always)]
#[allow(dead_code)]
pub fn soa_transform<'a, const W: usize, const MAX: usize>(
    padded: &[(i32, &'a [u8], i32, &'a [u8], i32, i32); W],
) -> (Vec<u8>, Vec<u8>) {
    let mut query_soa = vec![0u8; MAX * W];
    let mut target_soa = vec![0u8; MAX * W];

    for i in 0..W {
        let (q_len, query, t_len, target, _w, _h) = padded[i];
        let qn = (q_len as usize).min(MAX);
        let tn = (t_len as usize).min(MAX);
        for j in 0..qn { query_soa[j * W + i] = query[j]; }
        for j in qn..MAX { query_soa[j * W + i] = 0xFF; }
        for j in 0..tn { target_soa[j * W + i] = target[j]; }
        for j in tn..MAX { target_soa[j * W + i] = 0xFF; }
    }
    (query_soa, target_soa)
}

/// Pack lane-wise trackers into `OutScore` results.
#[inline(always)]
#[allow(dead_code)]
pub fn pack_outscores<const W: usize>(
    scores: [i32; W],
    q_end: [i32; W],
    t_end: [i32; W],
    gscore: [i32; W],
    g_t_end: [i32; W],
    max_off: [i32; W],
    lanes: usize,
) -> Vec<OutScore> {
    let n = lanes.min(W);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(OutScore {
            score: scores[i],
            target_end_pos: t_end[i],
            gtarget_end_pos: g_t_end[i],
            query_end_pos: q_end[i],
            global_score: gscore[i],
            max_offset: max_off[i],
        });
    }
    out
}

// ----------------------------------------------------------------------------
// Macro: generate_swa_entry!
// ----------------------------------------------------------------------------
// This macro will be used to emit thin per‑ISA entry points that:
// 1) Pad the batch and extract lane parameters
// 2) Transform sequences to SoA (or no‑op once SoA‑first lands)
// 3) Build KernelParams and call the shared kernel
//
// Note: Defining this macro does not change behavior. It will be expanded and
// used by ISA modules in subsequent steps once the shared kernel is implemented.
#[macro_export]
macro_rules! generate_swa_entry {
    (
        name = $name:ident,
        width = $W:expr,
        engine = $E:ty,
        cfg = $cfg:meta,
    ) => {
        #[$cfg]
        #[allow(unsafe_op_in_unsafe_fn)]
        pub unsafe fn $name(
            batch: &[(i32, &[u8], i32, &[u8], i32, i32)],
            o_del: i32,
            e_del: i32,
            o_ins: i32,
            e_ins: i32,
            zdrop: i32,
            mat: &[i8; 25],
            m: i32,
        ) -> Vec<$crate::alignment::banded_swa::OutScore> {
            const SIMD_WIDTH: usize = $W;
            const MAX_SEQ_LEN: usize = 128;

            let (qlen, tlen, h0, w_arr, max_qlen, max_tlen, padded) =
                $crate::alignment::banded_swa_shared::pad_batch::<SIMD_WIDTH>(batch);
            let (query_soa, target_soa) =
                $crate::alignment::banded_swa_shared::soa_transform::<SIMD_WIDTH, MAX_SEQ_LEN>(&padded);

            let params = $crate::alignment::banded_swa_kernel::KernelParams {
                batch,
                query_soa: &query_soa,
                target_oa: &target_soa,
                qlen: &qlen,
                tlen: &tlen,
                h0: &h0,
                w: &w_arr,
                max_qlen,
                max_tlen,
                o_del,
                e_del,
                o_ins,
                e_ins,
                zdrop,
                mat,
                m,
            };

            // Placeholder call; returns empty Vec until the shared kernel is implemented.
            $crate::alignment::banded_swa_kernel::sw_kernel::<SIMD_WIDTH, $E>(&params)
        }
    };
}
