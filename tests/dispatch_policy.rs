// tests/dispatch_policy.rs
// Validate bwa-mem2–style dispatch policy for banded_swa wrappers:
// inputs with any qlen/tlen > 127 should route to the i16 kernel.

use ferrous_align::core::alignment::banded_swa::isa_avx2::{
    simd_banded_swa_batch16_int16, simd_banded_swa_batch32,
};

#[cfg(target_arch = "x86_64")]
#[test]
fn dispatch_routes_long_reads_to_i16_avx2() {
    // Construct a simple batch with lengths > 127 so that the AVX2 i8 wrapper
    // dispatches to the i16 path. We use a trivial scoring matrix with matches = 1.
    let q = vec![0u8; 200];
    let t = vec![0u8; 200];
    let batch = vec![(200, &q[..], 200, &t[..], 10, 0)];

    // Simple scoring: match=1, mismatch=0
    let mut mat = [0i8; 25];
    for i in 0..4 { mat[i * 5 + i] = 1; }

    // Call the AVX2 i8 wrapper; it should internally route to the 16-bit kernel
    let via_dispatch = unsafe { simd_banded_swa_batch32(&batch, 6, 1, 6, 1, 100, &mat, 5) };

    // Directly call the i16 AVX2 entry and compare results
    let direct_i16 = unsafe { simd_banded_swa_batch16_int16(&batch, 6, 1, 6, 1, 100, &mat, 5) };

    assert_eq!(via_dispatch.len(), direct_i16.len());
    for (a, b) in via_dispatch.iter().zip(direct_i16.iter()) {
        assert_eq!(a.score, b.score, "score mismatch: dispatch vs direct i16");
        assert_eq!(a.query_end_pos, b.query_end_pos, "qe mismatch");
        assert_eq!(a.target_end_pos, b.target_end_pos, "te mismatch");
    }
}
