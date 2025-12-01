// tests/kernel_parity.rs
use ferrous_align::compute::simd_abstraction::simd::detect_optimal_simd_engine;
use ferrous_align::core::alignment::banded_swa::{
    BandedPairWiseSW,
    kernel::{KernelParams, SwEngine256, sw_kernel}, // SwEngine256 from kernel
};
use ferrous_align::pipelines::linear::batch_extension::{
    dispatch::execute_batch_simd_scoring,
    types::{ExtensionDirection, ExtensionJobBatch},
};

#[cfg(target_arch = "x86_64")]
#[test]
fn kernel_sw_vs_avx2_basic_parity() {
    // 2-bit encoded A,C,G,T = 0,1,2,3
    let q: [u8; 8] = [0, 1, 2, 3, 0, 1, 2, 3];
    let t: [u8; 8] = [0, 1, 2, 3, 0, 1, 2, 3];

    let mut batch = ExtensionJobBatch::new();
    batch.add_job(0, 0, 0, ExtensionDirection::Right, &q, &t, 0, 10);

    // Simple scoring: match=1, mismatch=0
    let mut mat = [0i8; 25];
    for i in 0..4 {
        mat[i * 5 + i] = 1;
    }

    let sw_params = BandedPairWiseSW::new(6, 1, 6, 1, 100, 0, 0, 0, mat, 1, -4);
    let engine = detect_optimal_simd_engine();

    let soa_out = execute_batch_simd_scoring(&sw_params, &mut batch, engine);

    // Prepare shared-kernel params (W=32, SwEngine256)
    const W: usize = 32;
    let batch_vec = vec![(8, &q[..], 8, &t[..], 10, 0)];
    let (qlen, tlen, h0, w_arr, max_q, max_t, padded) =
        ferrous_align::core::alignment::banded_swa::shared::pad_batch::<W>(&batch_vec);
    let (query_soa, target_soa) =
        ferrous_align::core::alignment::banded_swa::shared::soa_transform::<W, 128>(&padded);

    let params = KernelParams {
        batch: &[(8, &q[..], 8, &t[..], 10, 0)],
        query_soa: &query_soa,
        target_soa: &target_soa,
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

    let shared_out = unsafe { sw_kernel::<W, SwEngine256>(&params) };

    assert_eq!(shared_out.len(), soa_out.len());
    for (a, b) in shared_out.iter().zip(soa_out.iter()) {
        assert_eq!(
            a.score, b.score,
            "score mismatch: shared={} soa={}",
            a.score, b.score
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn kernel_sw_vs_avx2_mismatch_case() {
    // Deliberate mismatches
    let q: [u8; 8] = [0, 0, 0, 0, 1, 1, 1, 1];
    let t: [u8; 8] = [3, 3, 3, 3, 2, 2, 2, 2];

    let mut batch = ExtensionJobBatch::new();
    batch.add_job(0, 0, 0, ExtensionDirection::Right, &q, &t, 0, 10);

    // Simple scoring: match=1, mismatch=0
    let mut mat = [0i8; 25];
    for i in 0..4 {
        mat[i * 5 + i] = 1;
    }

    let sw_params = BandedPairWiseSW::new(6, 1, 6, 1, 100, 0, 0, 0, mat, 1, -4);
    let engine = detect_optimal_simd_engine();

    let soa_out = execute_batch_simd_scoring(&sw_params, &mut batch, engine);

    const W: usize = 32;
    let batch_vec = vec![(8, &q[..], 8, &t[..], 10, 0)];
    let (qlen, tlen, h0, w_arr, max_q, max_t, padded) =
        ferrous_align::core::alignment::banded_swa::shared::pad_batch::<W>(&batch_vec);
    let (query_soa, target_soa) =
        ferrous_align::core::alignment::banded_swa::shared::soa_transform::<W, 128>(&padded);

    let params = KernelParams {
        batch: &[(8, &q[..], 8, &t[..], 10, 0)],
        query_soa: &query_soa,
        target_soa: &target_soa,
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
    let shared_out = unsafe { sw_kernel::<W, SwEngine256>(&params) };

    assert_eq!(shared_out.len(), soa_out.len());
    for (a, b) in shared_out.iter().zip(soa_out.iter()) {
        assert_eq!(
            a.score, b.score,
            "score mismatch: shared={} soa={}",
            a.score, b.score
        );
    }
}
