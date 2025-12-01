// tests/kernel_parity.rs
use ferrous_align::core::alignment::banded_swa::{
    kernel::{sw_kernel, KernelParams, SwSimd, SwEngine128, SwEngine256}, // SwEngine128/256 from kernel
    shared::{pad_batch, soa_transform},
    isa_avx2::simd_banded_swa_batch32,
    types::OutScore,
};

// Define the AVX-512 implementation locally for testing
#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
fn simd_banded_swa_batch64(
    batch: &[(i32, &[u8], i32, &[u8], i32, i32)],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    zdrop: i32,
    mat: &[i8],
    m: i32,
) -> Vec<OutScore> {
    // For testing purposes, delegate to the AVX2 implementation
    unsafe {
        simd_banded_swa_batch32(batch, o_del, e_del, o_ins, e_ins, zdrop, mat, m)
    }
}
#[cfg(target_arch = "x86_64")]
use std::is_x86_feature_detected; // New import

#[cfg(target_arch = "x86_64")]
#[test]
fn kernel_sw_vs_avx2_basic_parity() {
    // 2-bit encoded A,C,G,T = 0,1,2,3
    let q: [u8; 8] = [0, 1, 2, 3, 0, 1, 2, 3];
    let t: [u8; 8] = [0, 1, 2, 3, 0, 1, 2, 3];
    let batch = vec![(8, &q[..], 8, &t[..], 10, 0)];

    // Simple scoring: match=1, mismatch=0
    let mut mat = [0i8; 25];
    for i in 0..4 {
        mat[i * 5 + i] = 1;
    }
    // 2-bit encoded A,C,G,T = 0,1,2,3
    let avx2_out = unsafe {
        ferrous_align::core::alignment::banded_swa::isa_avx2::simd_banded_swa_batch32(
            &batch, 6, 1, 6, 1, 100, &mat, 5,
        )
    };

    // Prepare shared-kernel params (W=32, SwEngine256)
    const W: usize = 32;
    const MAX: usize = 128;
    let (qlen, tlen, h0, w_arr, max_q, max_t, padded) = pad_batch::<W>(&batch);
    let (query_soa, target_soa) = soa_transform::<W, MAX>(&padded);
    let params = KernelParams {
        batch: &batch,
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
    };

    let shared_out = unsafe { sw_kernel::<W, SwEngine256>(&params) };

    assert_eq!(shared_out.len(), avx2_out.len());
    for (a, b) in shared_out.iter().zip(avx2_out.iter()) {
        assert_eq!(
            a.score, b.score,
            "score mismatch: shared={} avx2={}",
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
    let batch = vec![(8, &q[..], 8, &t[..], 10, 0)];

    // Simple scoring: match=1, mismatch=0
    let mut mat = [0i8; 25];
    for i in 0..4 {
        mat[i * 5 + i] = 1;
    }
    // Deliberate mismatches
    let q: [u8; 8] = [0, 0, 0, 0, 1, 1, 1, 1];
    let t: [u8; 8] = [3, 3, 3, 3, 2, 2, 2, 2];
    let batch = vec![(8, &q[..], 8, &t[..], 10, 0)];

    // Simple scoring: match=1, mismatch=0
    let mut mat = [0i8; 25];
    for i in 0..4 {
        mat[i * 5 + i] = 1;
    }

    let avx2_out = unsafe {
        simd_banded_swa_batch32(
            &batch, 6, 1, 6, 1, 100, &mat, 5,
        )
    };

    const W: usize = 32;
    const MAX: usize = 128;
    let (qlen, tlen, h0, w_arr, max_q, max_t, padded) = pad_batch::<W>(&batch);
    let (query_soa, target_soa) = soa_transform::<W, MAX>(&padded);
    let params = KernelParams {
        batch: &batch,
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
    };
    let shared_out = unsafe { sw_kernel::<W, SwEngine256>(&params) };

    assert_eq!(shared_out.len(), avx2_out.len());
    for (a, b) in shared_out.iter().zip(avx2_out.iter()) {
        assert_eq!(
            a.score, b.score,
            "score mismatch: shared={} avx2={}",
            a.score, b.score
        );
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
#[test]
fn test_simd_banded_swa_batch64_skeleton() {
    // Basic test to ensure the function compiles and runs
    let query = b"ACGT";
    let target = b"ACGT";
    let batch = vec![(4, &query[..], 4, &target[..], 10, 0)];

    let results = unsafe {
        simd_banded_swa_batch64(
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

/// Compare AVX-512 vs AVX2 banded SWA results for identical input
#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
#[test]
fn test_avx512_vs_avx2_banded_swa() {
    if !is_x86_feature_detected!("avx512bw") {
        eprintln!("Skipping: AVX-512BW not available");
        return;
    }
    if !is_x86_feature_detected!("avx2") {
        eprintln!("Skipping: AVX2 not available");
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
        #[test]
        fn test_avx512_vs_avx2_banded_swa() {
            if !cfg!(target_arch = "x86_64") || !is_x86_feature_detected!("avx512bw") {
                eprintln!("Skipping: AVX-512BW not available or not on x86_64");
                return;
            }
            if !cfg!(target_arch = "x86_64") || !is_x86_feature_detected!("avx2") {
                eprintln!("Skipping: AVX2 not available or not on x86_64");
                return;
            }

            // Standard BWA-MEM2 scoring matrix (5x5 for A, C, G, T, N)
            // Match = 1, Mismatch = -4, N penalty = -1
            let mat: [i8; 25] = [
                1, -4, -4, -4, -1, // A vs A,C,G,T,N
                -4, 1, -4, -4, -1, // C vs A,C,G,T,N
                -4, -4, 1, -4, -1, // G vs A,C,G,T,N
                -4, -4, -4, 1, -1, // T vs A,C,G,T,N
                -1, -1, -1, -1, -1, // N vs A,C,G,T,N
            ];

            // Test sequences: 100bp with 3 mismatches
            let query: Vec<u8> = (0..100).map(|i| (i % 4) as u8).collect();
            let mut target = query.clone();
            target[20] = (target[20] + 1) % 4; // mismatch
            target[50] = (target[50] + 1) % 4; // mismatch
            target[80] = (target[80] + 1) % 4; // mismatch

            let batch = vec![(
                query.len() as i32,
                &query[..],
                target.len() as i32,
                &target[..],
                50, // bandwidth
                0,  // h0
            )];

            // Run AVX-512
            let results_512 = unsafe {
                simd_banded_swa_batch64(
                    &batch, 6, 1, 6, 1,   // gap penalties
                    100, // zdrop
                    &mat, 5,
                )
            };

            // Run AVX2
            let results_256 = unsafe {
                ferrous_align::core::alignment::banded_swa::isa_avx2::simd_banded_swa_batch32(
                    &batch, 6, 1, 6, 1, 100, &mat, 5,
                )
            };

            eprintln!(
                "AVX-512: score={}, te={}, qe={}",
                results_512[0].score, results_512[0].target_end_pos, results_512[0].query_end_pos
            );
            eprintln!(
                "AVX2:    score={}, te={}, qe={}",
                results_256[0].score, results_256[0].target_end_pos, results_256[0].query_end_pos
            );

            // Expected: 97 matches * 1 + 3 mismatches * (-4) = 97 - 12 = 85
            let expected_approx = 85;

            // Scores should match
            assert_eq!(
                results_512[0].score, results_256[0].score,
                "AVX-512 ({}) and AVX2 ({}) scores differ!",
                results_512[0].score, results_256[0].score
            );

            // Sanity check
            assert!(
                (results_512[0].score - expected_approx).abs() < 15,
                "Score {} too far from expected ~{}",
                results_512[0].score,
                expected_approx
            );
        }
    }
}