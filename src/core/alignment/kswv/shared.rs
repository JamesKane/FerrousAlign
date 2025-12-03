//! Shared macros/adapters for kswv (horizontal SIMD) SoA entry points.
//!
//! These macros generate thin SoA wrappers around the existing per‑ISA kswv
//! batch functions, constructing `SeqPair`/`KswResult` arrays and sourcing
//! aligned workspace buffers from the thread‑local `AlignmentWorkspace`.

#[macro_export]
macro_rules! generate_ksw_entry_soa {
    (
        name = $name:ident,
        callee = $callee:ident,
        width = $W:expr,
        cfg = $cfg:meta,
        target_feature = $tf:literal,
    ) => {
        #[$cfg]
        #[allow(unsafe_op_in_unsafe_fn)]
        #[cfg_attr(any(), target_feature(enable = $tf))]
        pub unsafe fn $name(
            inputs: &$crate::core::alignment::shared_types::KswSoA,
            ws: &mut $crate::core::alignment::workspace::AlignmentWorkspace,
            num_jobs: usize,
            match_score: i8,
            mismatch_penalty: i8,
            o_del: i32,
            e_del: i32,
            o_ins: i32,
            e_ins: i32,
            ambig_penalty: i8,
            debug: bool,
        ) -> Vec<$crate::core::alignment::kswv_batch::KswResult> {
            use $crate::core::alignment::kswv_batch::{KswResult, SeqPair};

            const W: usize = $W;
            let lanes = num_jobs.min(W);

            // Build vectors for pairs/results (pad inactive lanes with zeros)
            let mut pairs: Vec<SeqPair> = (0..W).map(|_| SeqPair::default()).collect();
            let mut results: Vec<KswResult> = vec![KswResult::default(); W];

            // Initialize active lanes from KswSoA scalars
            for lane in 0..lanes {
                pairs[lane].id = lane as usize;
                pairs[lane].ref_len = inputs.tlen[lane] as i32;
                pairs[lane].query_len = inputs.qlen[lane] as i32;
                pairs[lane].h0 = inputs.h0[lane] as i32;
            }

            // Provide ISA‑appropriate workspace buffers for the batch kernel
            // SAFETY: We use unsafe pointer casting here because `inputs` already borrows
            // from `ws`, but the KSW buffers are disjoint from the SoA buffers. The Rust
            // borrow checker can't prove this disjoint access, so we use raw pointers.
            let wb = unsafe {
                let ws_ptr = ws as *mut $crate::core::alignment::workspace::AlignmentWorkspace;
                Some((*ws_ptr).ksw_buffers_for_width(W))
            };

            // nrow = max_tlen, ncol = max_qlen
            let nrow = inputs.max_tlen as i16;
            let ncol = inputs.max_qlen as i16;

            // SAFETY: Call into the per‑ISA batch kernel provided by the module
            let _used = unsafe {
                $callee(
                    inputs.ref_soa.as_ptr(),
                    inputs.query_soa.as_ptr(),
                    nrow,
                    ncol,
                    &pairs,
                    results.as_mut_slice(),
                    match_score,
                    mismatch_penalty,
                    o_del,
                    e_del,
                    o_ins,
                    e_ins,
                    ambig_penalty,
                    0, // phase (unused)
                    debug,
                    wb,
                )
            };

            // Return exactly num_jobs results (what the caller expects)
            // The kernel fills results[0..lanes], we return num_jobs of them
            results[..num_jobs].to_vec()
        }
    };
}
