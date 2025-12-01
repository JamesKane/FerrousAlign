//! AVX‑512 int16 path (placeholder/manual for now)
// Keep minimal until an i16 shared kernel lands

#![cfg(target_arch = "x86_64")]

use crate::alignment::banded_swa::engines16::SwEngine512_16;
use crate::alignment::workspace::with_workspace;
use crate::core::alignment::banded_swa::OutScore;
use crate::generate_swa_entry_i16; // Import the macro
use crate::generate_swa_entry_i16_soa; // Import the macro
use std::arch::x86_64::*; // For raw AVX-512 intrinsics // Import SwEngine512_16

/// AVX-512-optimized banded Smith-Waterman for batches of up to 32 alignments (16-bit scores)
///
/// **SIMD Width**: 32 lanes (512-bit / 16-bit)
/// **Parallelism**: Processes 32 alignments simultaneously
/// **Score Range**: Full i16 range (-32768 to 32767) for sequences > 127bp
///
/// This is the 16-bit precision version optimized for:
/// - Sequences longer than 127bp where 8-bit scores would overflow
/// - Typical 151bp Illumina reads (max score = 151 with match=1)
///
/// **Performance**: 4x parallelism over SSE 8-wide (8 vs 32 lanes)
generate_swa_entry_i16!(
    name = simd_banded_swa_batch32_int16,
    width = 32,
    engine = SwEngine512_16,
    cfg = cfg(all(target_arch = "x86_64", feature = "avx512")),
    target_feature = "avx512bw,avx512f",
);

generate_swa_entry_i16_soa!(
    name = simd_banded_swa_batch32_int16_soa,
    width = 32,
    engine = SwEngine512_16,
    cfg = cfg(all(target_arch = "x86_64", feature = "avx512")),
    target_feature = "avx512bw,avx512f",
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_banded_swa_batch32_int16_basic() {
        // Basic test for 16-bit AVX-512 batch function
        // Kernel expects 2-bit encoded bases: A=0, C=1, G=2, T=3
        let query: [u8; 4] = [0, 1, 2, 3]; // ACGT in 2-bit encoding
        let target: [u8; 4] = [0, 1, 2, 3]; // ACGT in 2-bit encoding
        let batch = vec![(4, &query[..], 4, &target[..], 10, 0)];

        // Default scoring matrix (match=1, mismatch=0)
        let mut mat = [0i8; 25];
        for i in 0..4 {
            mat[i * 5 + i] = 1; // Match score on diagonal
        }

        let results = unsafe { simd_banded_swa_batch32_int16(&batch, 6, 1, 6, 1, 100, &mat, 5) };

        assert_eq!(results.len(), 1);
        assert!(
            results[0].score > 0,
            "Score {} should be > 0",
            results[0].score
        );
    }
}
