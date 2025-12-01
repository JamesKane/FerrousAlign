// tests/workspace_alloc.rs
// Validate that workspace-backed buffers are reused across calls (no reallocation)
// for constant shapes: DP rows (banded SW) and KSWV SoA buffers.

use ferrous_align::core::alignment::workspace::with_workspace;
use ferrous_align::core::alignment::shared_types::{AlignJob, WorkspaceArena};

#[test]
fn workspace_rows_reuse_no_realloc_for_same_shape() {
    with_workspace(|ws| {
        // Shape parameters
        let lanes = 16usize; // any SIMD width; rows are sized as qmax*lanes
        let qmax = 64usize;
        let tmax = 64usize;

        // First ensure and take pointers
        ws.ensure_rows(lanes, qmax, tmax, core::mem::size_of::<i8>());
        let (h0, e0, f0) = ws.rows_u8().expect("u8 rows available");
        let h_ptr0 = h0.as_ptr();
        let e_ptr0 = e0.as_ptr();
        let f_ptr0 = f0.as_ptr();

        // Second ensure with same shape; expect the same underlying buffers (no reallocation)
        ws.ensure_rows(lanes, qmax, tmax, core::mem::size_of::<i8>());
        let (h1, e1, f1) = ws.rows_u8().expect("u8 rows available");
        let h_ptr1 = h1.as_ptr();
        let e_ptr1 = e1.as_ptr();
        let f_ptr1 = f1.as_ptr();

        assert_eq!(h_ptr0, h_ptr1, "H row pointer changed between identical ensure_rows calls");
        assert_eq!(e_ptr0, e_ptr1, "E row pointer changed between identical ensure_rows calls");
        assert_eq!(f_ptr0, f_ptr1, "F row pointer changed between identical ensure_rows calls");
    });
}

#[test]
fn ksw_soa_buffers_reuse_no_realloc_for_same_shape() {
    use ferrous_align::core::alignment::workspace::AlignmentWorkspace; // for method on ws

    with_workspace(|ws: &mut AlignmentWorkspace| {
        // Build a small job set of size 8 lanes (will be padded as needed)
        let lanes = 16usize;
        let jobs: Vec<AlignJob> = (0..8)
            .map(|_| AlignJob { query: &[0u8; 32][..], target: &[0u8; 32][..], qlen: 32, tlen: 32, band: 0, h0: 0 })
            .collect();

        // First transpose
        let soa1 = ws.ensure_and_transpose_ksw(&jobs, lanes);
        let q_ptr1 = soa1.query_soa.as_ptr();
        let r_ptr1 = soa1.ref_soa.as_ptr();

        // Second transpose with the same shape
        let soa2 = ws.ensure_and_transpose_ksw(&jobs, lanes);
        let q_ptr2 = soa2.query_soa.as_ptr();
        let r_ptr2 = soa2.ref_soa.as_ptr();

        assert_eq!(q_ptr1, q_ptr2, "KSW query_soa pointer changed between identical transposes");
        assert_eq!(r_ptr1, r_ptr2, "KSW ref_soa pointer changed between identical transposes");
    });
}
