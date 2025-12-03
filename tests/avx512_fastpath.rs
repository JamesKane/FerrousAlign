// tests/avx512_fastpath.rs
// AVX-512 fast-path smoke/parity tests (feature-gated)

#![cfg(all(target_arch = "x86_64", feature = "avx512"))]

use ferrous_align::core::alignment::banded_swa::isa_avx512_int8::simd_banded_swa_batch64_soa;
use ferrous_align::core::alignment::banded_swa::kernel::{KernelParams, sw_kernel};
use ferrous_align::core::alignment::banded_swa::engines::SwEngine512;
use ferrous_align::core::alignment::banded_swa::shared::{SoAInputs, pad_batch, soa_transform};

#[test]
fn avx512_fastpath_parity_small() {
    if !std::is_x86_feature_detected!("avx512bw") {
        eprintln!("Skipping AVX-512 parity test: CPU lacks avx512bw");
        return;
    }

    // Build a simple batch of 64 identical short reads (len 64 <= 128)
    const W: usize = 64;
    let len = 64usize;
    let q = vec![0u8; len];
    let t = vec![0u8; len];
    let mut batch: Vec<(i32, &[u8], i32, &[u8], i32, i32)> = Vec::with_capacity(W);
    for _ in 0..W {
        batch.push((len as i32, &q[..], len as i32, &t[..], 20, 0));
    }

    // Scoring: match=1, mismatch=0
    let mut mat = [0i8; 25];
    for i in 0..4 {
        mat[i * 5 + i] = 1;
    }

    // Prepare SoA inputs
    let (qlen, tlen, h0, w_arr, max_q, max_t, padded) = pad_batch::<W>(&batch);
    let (qsoa, tsoa) = soa_transform::<W, 512>(&padded);
    let inputs = SoAInputs {
        query_soa: &qsoa,
        target_soa: &tsoa,
        qlen: &qlen,
        tlen: &tlen,
        w: &w_arr,
        h0: &h0,
        lanes: W,
        max_qlen: max_q,
        max_tlen: max_t,
    };

    // AVX-512 fast path via SoA entry
    let fast = unsafe { simd_banded_swa_batch64_soa(&inputs, W, 6, 1, 6, 1, 100, &mat, 5) };

    // Generic shared kernel (engine 512) on same inputs
    let params = KernelParams {
        batch: &batch,
        query_soa: &qsoa,
        target_soa: &tsoa,
        qlen: &qlen,
        tlen: &tlen,
        h0: &h0,
        w: &w_arr,
        max_qlen: max_q,
        max_tlen: max_t,
        o_del: 6,
        e_del: 1,
        o_ins: 6,
        e_ins: 1,
        zdrop: 100,
        mat: &mat,
        m: 5,
        cfg: None,
    };
    let generic = unsafe { sw_kernel::<W, SwEngine512>(&params, W) };

    assert_eq!(fast.len(), generic.len());
    for (a, b) in fast.iter().zip(generic.iter()) {
        assert_eq!(a.score, b.score, "AVX-512 fast path score mismatch");
        assert_eq!(a.query_end_pos, b.query_end_pos, "qe mismatch");
        assert_eq!(a.target_end_pos, b.target_end_pos, "te mismatch");
    }
}
