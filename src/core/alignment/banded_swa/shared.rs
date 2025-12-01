//! Shared, inlineable helpers for banded Smith–Waterman SIMD kernels.
//!
//! These functions are extracted from per-ISA implementations to reduce
//! duplication. They are designed to be aggressively inlined and cheap.

use crate::core::alignment::banded_swa::OutScore;


/// Carrier for pre-formatted Structure-of-Arrays (SoA) data.
///
/// This is used for the SoA-first path where data transformation is skipped.
#[derive(Debug)]
pub struct SoAInputs<'a> {
    pub query_soa: &'a [u8],
    pub target_soa: &'a [u8],
    pub qlen: &'a [i8],
    pub tlen: &'a [i8],
    pub w: &'a [i8],
    pub h0: &'a [i8],
    pub lanes: usize,
    pub max_qlen: i32,
    pub max_tlen: i32,
}

/// Carrier for pre-formatted Structure-of-Arrays (SoA) data for i16 kernel.
#[derive(Debug)]
pub struct SoAInputs16<'a> {
    pub query_soa: &'a [i16],
    pub target_soa: &'a [i16],
    pub qlen: &'a [i8],
    pub tlen: &'a [i8],
    pub h0: &'a [i16], // i16 h0 slice
    pub w: &'a [i8],
    pub max_qlen: i32,
    pub max_tlen: i32,
}

/// Pad a batch of jobs to a fixed SIMD width and extract lane-wise parameters.
#[inline(always)]
#[allow(dead_code)]
pub fn pad_batch<'a, const W: usize>(
    batch: &'a [(i32, &'a [u8], i32, &'a [u8], i32, i32)],
) -> (
    [i8; W],                                       // qlen
    [i8; W],                                       // tlen
    [i8; W],                                       // h0
    [i8; W],                                       // w
    i32,                                           // max_qlen
    i32,                                           // max_tlen
    [(i32, &'a [u8], i32, &'a [u8], i32, i32); W], // padded batch
) {
    let mut qlen = [0i8; W];
    let mut tlen = [0i8; W];
    let mut h0 = [0i8; W];
    let mut w_arr = [0i8; W];
    let mut max_q = 0i32;
    let mut max_t = 0i32;

    // Pad to width W (truncate extra lanes if provided)
    let mut padded: [(i32, &'a [u8], i32, &'a [u8], i32, i32); W] =
        [(0, &[][..], 0, &[][..], 0, 0); W];
    for i in 0..W {
        let tup = if i < batch.len() {
            batch[i]
        } else {
            (0, &[][..], 0, &[][..], 0, 0)
        };
        let (q, _qs, t, _ts, w, h) = tup;
        padded[i] = tup;
        qlen[i] = q.min(127) as i8;
        tlen[i] = t.min(127) as i8;
        h0[i] = h as i8;
        w_arr[i] = w as i8;
        if q > max_q {
            max_q = q;
        }
        if t > max_t {
            max_t = t;
        }
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
        for j in 0..qn {
            query_soa[j * W + i] = query[j];
        }
        for j in qn..MAX {
            query_soa[j * W + i] = 0xFF;
        }
        for j in 0..tn {
            target_soa[j * W + i] = target[j];
        }
        for j in tn..MAX {
            target_soa[j * W + i] = 0xFF;
        }
    }
    (query_soa, target_soa)
}

/// No-op transform for data that is already in SoA format.
#[inline(always)]
#[allow(dead_code)]
pub fn soa_transform_pre_soa<'a>(
    query_soa: &'a [u8],
    target_soa: &'a [u8],
) -> (&'a [u8], &'a [u8]) {
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
        target_feature = $tf:literal,
    ) => {
        #[$cfg]
        #[target_feature(enable = $tf)]
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
                crate::core::alignment::banded_swa::shared::pad_batch::<SIMD_WIDTH>(batch);
            let (query_soa, target_soa) = crate::core::alignment::banded_swa::shared::soa_transform::<
                SIMD_WIDTH,
                MAX_SEQ_LEN,
            >(&padded);

            let params = crate::core::alignment::banded_swa::kernel::KernelParams {
                batch,
                query_soa: &query_soa,
                target_soa: &target_soa,
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
            crate::core::alignment::banded_swa::kernel::sw_kernel::<SIMD_WIDTH, $E>(&params)
        }
    };
}

#[macro_export]
macro_rules! generate_swa_entry_i16 {
    ( name = $name:ident, width = $W:expr, engine = $E:ty, cfg = $cfg:meta, target_feature = $tf:expr, ) => {
        #[$cfg]
        #[allow(unsafe_op_in_unsafe_fn)]
        #[cfg_attr(any(), target_feature(enable = $tf))] // leave empty tf for NEON/SSE if desired
        pub unsafe fn $name(
            batch: &[(i32,&[u8],i32,&[u8],i32,i32)],
            o_del: i32, e_del: i32, o_ins: i32, e_ins: i32,
            zdrop: i32, mat: &[i8; 25], m: i32,
        ) -> Vec<OutScore> {
            const W: usize = $W;
            const MAX: usize = 512; // typical default for i16 path
            let (qlen, tlen, h0_i8, w_arr, max_q, max_t, padded) =
                crate::core::alignment::banded_swa::shared::pad_batch::<W>(batch);
            let (query_soa_u8, target_soa_u8) = crate::core::alignment::banded_swa::shared::soa_transform::<
                W,
                MAX,
            >(&padded);

            // Convert u8 SoA to i16 SoA for the i16 kernel
            let mut query_soa_i16 = Vec::with_capacity(query_soa_u8.len());
            for &val in query_soa_u8.iter() {
                query_soa_i16.push(val as i16);
            }
            let mut target_soa_i16 = Vec::with_capacity(target_soa_u8.len());
            for &val in target_soa_u8.iter() {
                target_soa_i16.push(val as i16);
            }

            let mut h0: [i16; W] = [0; W];
            for i in 0..W { h0[i] = h0_i8[i] as i16; }
            let params = crate::core::alignment::banded_swa::kernel_i16::KernelParams16 {
                batch,
                query_soa: &query_soa_i16,
                target_soa: &target_soa_i16,
                qlen: &qlen,
                tlen: &tlen,
                h0: &h0,
                w: &w_arr,
                max_qlen: max_q,
                max_tlen: max_t,
                o_del, e_del, o_ins, e_ins,
                zdrop,
                mat, m,
            };
            crate::core::alignment::banded_swa::kernel_i16::sw_kernel_i16::<W, $E>(&params)
        }
    };
}

#[macro_export]
macro_rules! generate_swa_entry_soa {
    (
        name = $name:ident,
        width = $W:expr,
        engine = $E:ty,
        cfg = $cfg:meta,
        target_feature = $tf:literal,
    ) => {
        #[$cfg]
        #[target_feature(enable = $tf)]
        #[allow(unsafe_op_in_unsafe_fn)]
        pub unsafe fn $name(
            inputs: &crate::core::alignment::banded_swa::shared::SoAInputs,
            num_jobs: usize,
            o_del: i32,
            e_del: i32,
            o_ins: i32,
            e_ins: i32,
            zdrop: i32,
            mat: &[i8; 25],
            m: i32,
        ) -> Vec<$crate::alignment::banded_swa::OutScore> {
            const SIMD_WIDTH: usize = $W;

            // The kernel expects an AoS batch slice for length info.
            // We create a dummy one since we're on the SoA path.
            let dummy_batch_arr = [(0, &[][..], 0, &[][..], 0, 0); SIMD_WIDTH];
            let dummy_batch = &dummy_batch_arr[0..num_jobs];

            let params = crate::core::alignment::banded_swa::kernel::KernelParams {
                batch: dummy_batch,
                query_soa: inputs.query_soa,
                target_soa: inputs.target_soa,
                qlen: inputs.qlen,
                tlen: inputs.tlen,
                h0: inputs.h0,
                w: inputs.w,
                max_qlen: inputs.max_qlen,
                max_tlen: inputs.max_tlen,
                o_del,
                e_del,
                o_ins,
                e_ins,
                zdrop,
                mat,
                m,
            };

            crate::core::alignment::banded_swa::kernel::sw_kernel::<SIMD_WIDTH, $E>(&params)
        }
    };
}

#[macro_export]
macro_rules! generate_swa_entry_i16_soa {
    (
        name = $name:ident,
        width = $W:expr,
        engine = $E:ty,
        cfg = $cfg:meta,
        target_feature = $tf:literal,
    ) => {
        #[$cfg]
        #[allow(unsafe_op_in_unsafe_fn)]
        #[cfg_attr(any(), target_feature(enable = $tf))]
        pub unsafe fn $name(
            inputs: &crate::core::alignment::banded_swa::shared::SoAInputs16,
            num_jobs: usize,
            o_del: i32, e_del: i32, o_ins: i32, e_ins: i32,
            zdrop: i32, mat: &[i8; 25], m: i32,
        ) -> Vec<$crate::alignment::banded_swa::OutScore> {
            const W: usize = $W;
            // Dummy AoS to satisfy KernelParams16::batch type; lengths come from inputs
            let dummy_batch_arr = [(0, &[][..], 0, &[][..], 0, 0); W];
            let dummy_batch = &dummy_batch_arr[0..num_jobs];

            let params = crate::core::alignment::banded_swa::kernel_i16::KernelParams16 {
                batch: dummy_batch,
                query_soa: inputs.query_soa,
                target_soa: inputs.target_soa,
                qlen: inputs.qlen,
                tlen: inputs.tlen,
                h0: inputs.h0,        // i16 h0 slice
                w: inputs.w,
                max_qlen: inputs.max_qlen,
                max_tlen: inputs.max_tlen,
                o_del, e_del, o_ins, e_ins,
                zdrop, mat, m,
            };
            crate::core::alignment::banded_swa::kernel_i16::sw_kernel_i16::<W, $E>(&params)
        }
    };
}