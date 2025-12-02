// bwa-mem2-rust/src/banded_swa_avx2.rs
//
// AVX2 (256-bit) SIMD implementation of banded Smith-Waterman alignment
// Processes 32 alignments in parallel (2x speedup over SSE)
//
// This is a port of C++ bwa-mem2's smithWaterman256_8 function
// Reference: /Users/jkane/Applications/bwa-mem2/src/bandedSWA.cpp:722-1150

#![cfg(target_arch = "x86_64")]

use super::engines::SwEngine256;
use crate::core::alignment::banded_swa::engines16::SwEngine256_16;
use crate::core::alignment::banded_swa::OutScore;
use crate::{generate_swa_entry_i16, generate_swa_entry_i16_soa};


/// AVX2-optimized banded Smith-Waterman for batches of up to 16 alignments (16-bit scores)
///
/// **SIMD Width**: 16 lanes (256-bit / 16-bit)
/// **Parallelism**: Processes 16 alignments simultaneously
/// **Score Range**: Full i16 range (-32768 to 32767) for sequences > 127bp
///
/// This is the 16-bit precision version optimized for:
/// - Sequences longer than 127bp where 8-bit scores would overflow
/// - Typical 151bp Illumina reads (max score = 151 with match=1)
///
/// **Performance**: 2x parallelism over SSE 8-wide (8 vs 16 lanes)
generate_swa_entry_i16!(
    name = simd_banded_swa_batch16_int16,
    width = 16,
    engine = SwEngine256_16,
    cfg = cfg(target_arch = "x86_64"),
    target_feature = "avx2",
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_banded_swa_batch16_int16_basic() {
        // Basic test for 16-bit AVX2 batch function
        // Kernel expects 2-bit encoded bases: A=0, C=1, G=2, T=3
        let query: [u8; 4] = [0, 1, 2, 3]; // ACGT in 2-bit encoding
        let target: [u8; 4] = [0, 1, 2, 3]; // ACGT in 2-bit encoding
        let batch = vec![(4, &query[..], 4, &target[..], 10, 0)];

        // Default scoring matrix (match=1, mismatch=0)
        let mut mat = [0i8; 25];
        for i in 0..4 {
            mat[i * 5 + i] = 1; // Match score on diagonal
        }

        let results = unsafe { simd_banded_swa_batch16_int16(&batch, 6, 1, 6, 1, 100, &mat, 5) };

        assert_eq!(results.len(), 1);
        assert!(
            results[0].score > 0,
            "Score {} should be > 0",
            results[0].score
        );
    }

}

use crate::generate_swa_entry_soa;

generate_swa_entry_soa!(
    name = simd_banded_swa_batch32_soa,
    width = 32,
    engine = SwEngine256,
    cfg = cfg(target_arch = "x86_64"),
    target_feature = "avx2",
);

generate_swa_entry_i16_soa!(
    name = simd_banded_swa_batch16_int16_soa,
    width = 16,
    engine = SwEngine256_16,
    cfg = cfg(target_arch = "x86_64"),
    target_feature = "avx2",
);
