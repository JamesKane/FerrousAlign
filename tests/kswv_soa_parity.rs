// tests/kswv_soa_parity.rs
// Parity tests for kswv SoA adapter vs direct kernel invocation.

use ferrous_align::core::alignment::kswv_batch::{KswResult, SeqPair};
use ferrous_align::core::alignment::workspace::with_workspace;

#[cfg(target_arch = "x86_64")]
#[test]
fn kswv_sse_neon_soa_vs_direct_parity() {
    use ferrous_align::core::alignment::kswv_sse_neon::batch_ksw_align_sse_neon as direct;
    
    use ferrous_align::core::alignment::shared_types::KswSoA;
    use ferrous_align::core::alignment::kswv_sse_neon::kswv_batch16_soa as soa_entry;

    // Build two simple identical sequences across lanes
    let q = vec![0u8; 16 * 16]; // 16 positions * 16 lanes
    let t = vec![0u8; 16 * 16];

    // Setup pairs
    let pairs: Vec<SeqPair> = (0..16)
        .map(|id| SeqPair { ref_len: 16, query_len: 16, id, ..Default::default() })
        .collect();
    let mut results_direct = vec![KswResult::default(); 16];

    // Workspace-backed buffers for the direct call
    let used = with_workspace(|ws| {
        let (h0, h1, f, row_max) = ws.ksw_buffers_for_width(16);
        // SAFETY: invoke the direct kernel with SoA buffers and workspace rows
        unsafe {
            direct(
                t.as_ptr(),
                q.as_ptr(),
                16, // nrow
                16, // ncol
                &pairs,
                results_direct.as_mut_slice(),
                1,    // match
                0,    // mismatch penalty
                6, 1, // o_del, e_del
                6, 1, // o_ins, e_ins
                -1,   // ambig penalty
                0,    // phase
                false,
                Some((h0, h1, f, row_max)),
            )
        }
    });
    assert!(used > 0);

    // Now wrap via the SoA entry (adapter): construct KswSoA view and call
    let soa_scores = {
        let inputs = KswSoA {
            ref_soa: &t,
            query_soa: &q,
            qlen: &vec![16i8; 16],
            tlen: &vec![16i8; 16],
            band: &vec![0i8; 16],
            h0: &vec![0i8; 16],
            lanes: 16,
            max_qlen: 16,
            max_tlen: 16,
        };
        unsafe { soa_entry(&inputs, 16, 1, 0, 6, 1, 6, 1, -1, false) }
    };

    assert!(soa_scores.len() <= results_direct.len());
    let used_len = soa_scores.len();
    for (a, b) in soa_scores.iter().zip(results_direct.iter().take(used_len)) {
        assert_eq!(a.score, b.score, "score parity failed");
        assert_eq!(a.te, b.te, "te parity failed");
        assert_eq!(a.qe, b.qe, "qe parity failed");
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn kswv_avx2_soa_vs_direct_parity() {
    use ferrous_align::core::alignment::kswv_avx2::batch_ksw_align_avx2 as direct;
    use ferrous_align::core::alignment::kswv_avx2::kswv_batch32_soa as soa_entry;
    use ferrous_align::core::alignment::shared_types::KswSoA;

    if !std::is_x86_feature_detected!("avx2") {
        eprintln!("Skipping AVX2 parity test: CPU lacks avx2");
        return;
    }

    let q = vec![0u8; 32 * 16];
    let t = vec![0u8; 32 * 16];

    let pairs: Vec<SeqPair> = (0..32)
        .map(|id| SeqPair { ref_len: 16, query_len: 16, id, ..Default::default() })
        .collect();
    let mut results_direct = vec![KswResult::default(); 32];

    let used = with_workspace(|ws| {
        let (h0, h1, f, row_max) = ws.ksw_buffers_for_width(32);
        unsafe {
            direct(
                t.as_ptr(), q.as_ptr(), 16, 16, &pairs, results_direct.as_mut_slice(),
                1, 0, 6, 1, 6, 1, -1, 0, false, Some((h0, h1, f, row_max)),
            )
        }
    });
    assert!(used > 0);

    let inputs = KswSoA {
        ref_soa: &t,
        query_soa: &q,
        qlen: &vec![16i8; 32],
        tlen: &vec![16i8; 32],
        band: &vec![0i8; 32],
        h0: &vec![0i8; 32],
        lanes: 32,
        max_qlen: 16,
        max_tlen: 16,
    };
    let soa_scores = unsafe { soa_entry(&inputs, 32, 1, 0, 6, 1, 6, 1, -1, false) };

    assert!(soa_scores.len() <= results_direct.len());
    let used_len = soa_scores.len();
    for (a, b) in soa_scores.iter().zip(results_direct.iter().take(used_len)) {
        assert_eq!(a.score, b.score, "score parity failed");
        assert_eq!(a.te, b.te, "te parity failed");
        assert_eq!(a.qe, b.qe, "qe parity failed");
    }
}
