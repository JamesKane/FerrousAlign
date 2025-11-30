// bwa-mem2-rust/src/banded_swa_sse_neon.rs
//
// SSE/NEON (128-bit) SIMD implementation of banded Smith-Waterman alignment
// Processes 16 alignments in parallel (baseline SIMD for all platforms)
//
// This is a port of the AVX2 version adapted for 128-bit SIMD width
// Works on both x86_64 (SSE2+) and aarch64 (NEON)

use crate::alignment::banded_swa::OutScore;
use crate::alignment::banded_swa_kernel::{KernelParams, SwEngine128, sw_kernel};
use crate::alignment::banded_swa_shared::{pad_batch, soa_transform};

/// SSE/NEON-optimized banded Smith-Waterman for batches of up to 16 alignments
///
/// **SIMD Width**: 16 lanes (baseline for SSE2/NEON)
/// **Parallelism**: Processes 16 alignments simultaneously
/// **Platform**: Works on x86_64 (SSE2+) and aarch64 (NEON)
///
/// **Algorithm**:
/// - Uses Structure-of-Arrays (SoA) layout for SIMD-friendly access
/// - Implements standard Smith-Waterman DP recurrence
/// - Adaptive banding: Only compute cells within [i-w, i+w+1]
/// - Z-drop early termination: Stop lanes when score drops > zdrop
///
/// **Memory Layout**:
/// - Query/target sequences: `seq[position][lane]` (interleaved)
/// - DP matrices (H, E, F): `matrix[position * 16 + lane]`
/// - Query profiles: `profile[target_base][query_pos * 16 + lane]`
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn simd_banded_swa_batch16(
    batch: &[(i32, &[u8], i32, &[u8], i32, i32)],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    zdrop: i32,
    mat: &[i8; 25],
    m: i32,
) -> Vec<OutScore> {
    const W: usize = 16;
    const MAX_SEQ_LEN: usize = 128; // keep i8 limits aligned with AVX2

    let (qlen, tlen, h0, w_arr, max_qlen, max_tlen, padded) = pad_batch::<W>(batch);
    let (query_soa, target_soa) = soa_transform::<W, MAX_SEQ_LEN>(&padded);

    let params = KernelParams {
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

    sw_kernel::<W, SwEngine128>(&params)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_banded_swa_batch16_basic() {
        // Basic test to ensure the function compiles and runs
        // Kernel expects 2-bit encoded bases: A=0, C=1, G=2, T=3
        let query: [u8; 4] = [0, 1, 2, 3]; // ACGT in 2-bit encoding
        let target: [u8; 4] = [0, 1, 2, 3]; // ACGT in 2-bit encoding
        let batch = vec![(4, &query[..], 4, &target[..], 10, 0)];

        // Default scoring matrix (match=1, mismatch=0)
        let mut mat = [0i8; 25];
        for i in 0..4 {
            mat[i * 5 + i] = 1; // Match score on diagonal
        }

        let results = unsafe {
            simd_banded_swa_batch16(
                &batch, 6,   // o_del
                1,   // e_del
                6,   // o_ins
                1,   // e_ins
                100, // zdrop
                &mat, 5,
            )
        };

        assert_eq!(results.len(), 1);
        // Perfect match should have score >= 4
        assert!(
            results[0].score >= 4,
            "Expected score >= 4 for perfect match, got {}",
            results[0].score
        );
    }

    #[test]
    fn test_simd_banded_swa_batch16_multiple() {
        // Test with multiple alignments in batch
        // Kernel expects 2-bit encoded bases: A=0, C=1, G=2, T=3
        let q1: [u8; 4] = [0, 1, 2, 3]; // ACGT in 2-bit encoding
        let t1: [u8; 4] = [0, 1, 2, 3]; // ACGT in 2-bit encoding
        let q2: [u8; 4] = [0, 0, 0, 0]; // AAAA in 2-bit encoding
        let t2: [u8; 4] = [3, 3, 3, 3]; // TTTT in 2-bit encoding

        let batch = vec![
            (4, &q1[..], 4, &t1[..], 10, 0),
            (4, &q2[..], 4, &t2[..], 10, 0),
        ];

        let mut mat = [0i8; 25];
        for i in 0..4 {
            mat[i * 5 + i] = 1; // Match
        }

        let results = unsafe { simd_banded_swa_batch16(&batch, 6, 1, 6, 1, 100, &mat, 5) };

        assert_eq!(results.len(), 2);
        // First alignment (perfect match) should score higher
        assert!(
            results[0].score >= results[1].score,
            "Perfect match should score >= mismatch: {} vs {}",
            results[0].score,
            results[1].score
        );
    }
}

use crate::generate_swa_entry_soa;

#[cfg(target_arch = "x86_64")]
generate_swa_entry_soa!(
    name = simd_banded_swa_batch16_soa,
    width = 16,
    engine = SwEngine128,
    cfg = cfg(target_arch = "x86_64"),
    target_feature = "sse2",
);

#[cfg(target_arch = "aarch64")]
generate_swa_entry_soa!(
    name = simd_banded_swa_batch16_soa,
    width = 16,
    engine = SwEngine128,
    cfg = cfg(target_arch = "aarch64"),
    target_feature = "neon",
);
