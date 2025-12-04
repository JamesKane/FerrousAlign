//! Test to reproduce memory corruption crash in batch extension dispatch.
//!
//! The crash occurs with `free(): invalid size` during multi-chunk batch processing.
//! This test creates a batch larger than SIMD width to trigger multi-chunk processing.

use ferrous_align::compute::simd_abstraction::simd::SimdEngineType;
use ferrous_align::core::alignment::banded_swa::BandedPairWiseSW;
use ferrous_align::pipelines::linear::batch_extension::dispatch::execute_batch_simd_scoring;
use ferrous_align::pipelines::linear::batch_extension::types::{
    ExtensionDirection, ExtensionJobBatch,
};

/// Create a simple test sequence of given length
fn make_test_seq(len: usize, base: u8) -> Vec<u8> {
    // Use simple repeating pattern: base, base+1, base+2, base+3, ...
    (0..len).map(|i| ((base as usize + i) % 4) as u8).collect()
}

/// Standard scoring matrix for tests
fn make_sw_params() -> BandedPairWiseSW {
    // Standard BWA-MEM2 scoring matrix
    let mat = [
        1, -4, -4, -4, -1,  // A vs A,C,G,T,N
        -4, 1, -4, -4, -1,  // C vs ...
        -4, -4, 1, -4, -1,  // G vs ...
        -4, -4, -4, 1, -1,  // T vs ...
        -1, -1, -1, -1, -1, // N vs ...
    ];
    BandedPairWiseSW::new(6, 1, 6, 1, 100, 0, 5, 5, mat, 1, -4)
}

/// Test basic single-chunk batch (should work)
#[test]
fn test_single_chunk_batch() {
    let sw_params = make_sw_params();
    let mut batch = ExtensionJobBatch::new();

    // Add 8 jobs (fits in single chunk for all SIMD widths)
    for i in 0..8 {
        let query = make_test_seq(100, 0);
        let ref_seq = make_test_seq(100, 0);
        batch.add_job(
            i,      // read_idx
            0,      // chain_idx
            0,      // seed_idx
            ExtensionDirection::Right,
            &query,
            &ref_seq,
            0,      // h0
            50,     // band_width
        );
    }

    let engine = SimdEngineType::Engine256;
    let results = execute_batch_simd_scoring(&sw_params, &mut batch, engine);

    assert_eq!(results.len(), 8, "Should get 8 results for 8 jobs");
    for (i, r) in results.iter().enumerate() {
        assert_eq!(r.read_idx, i, "Result {} should have read_idx {}", i, i);
        assert!(r.score > 0, "Score should be positive for matching sequences");
    }
}

/// Test multi-chunk batch (triggers the bug)
#[test]
fn test_multi_chunk_batch_avx2() {
    let sw_params = make_sw_params();
    let mut batch = ExtensionJobBatch::new();

    // Add 64 jobs - enough to span multiple chunks for AVX2 (32 lanes)
    // This triggers the multi-chunk loop in dispatch_banded_swa_soa
    for i in 0..64 {
        let query = make_test_seq(100, 0);
        let ref_seq = make_test_seq(100, 0);
        batch.add_job(
            i,      // read_idx
            0,      // chain_idx
            0,      // seed_idx
            ExtensionDirection::Right,
            &query,
            &ref_seq,
            0,      // h0
            50,     // band_width
        );
    }

    let engine = SimdEngineType::Engine256;
    let results = execute_batch_simd_scoring(&sw_params, &mut batch, engine);

    assert_eq!(results.len(), 64, "Should get 64 results for 64 jobs");
    for (i, r) in results.iter().enumerate() {
        assert_eq!(r.read_idx, i, "Result {} should have read_idx {}", i, i);
    }
}

/// Test large batch (100+ jobs, like real workload)
#[test]
fn test_large_batch_realistic() {
    let sw_params = make_sw_params();
    let mut batch = ExtensionJobBatch::new();

    // Add 100 jobs with varying lengths (realistic scenario)
    for i in 0..100 {
        // Vary lengths between 50-150bp
        let qlen = 50 + (i % 100);
        let rlen = 50 + ((i * 7) % 100);

        let query = make_test_seq(qlen, 0);
        let ref_seq = make_test_seq(rlen, 0);
        batch.add_job(
            i,      // read_idx
            0,      // chain_idx
            0,      // seed_idx
            ExtensionDirection::Right,
            &query,
            &ref_seq,
            0,      // h0
            50,     // band_width
        );
    }

    let engine = SimdEngineType::Engine256;
    let results = execute_batch_simd_scoring(&sw_params, &mut batch, engine);

    assert_eq!(results.len(), 100, "Should get 100 results for 100 jobs");
}

/// Test with sequences longer than 128bp (triggers i16 path)
#[test]
fn test_long_sequences_i16_path() {
    let sw_params = make_sw_params();
    let mut batch = ExtensionJobBatch::new();

    // Add 32 jobs with 150bp sequences (forces i16 path)
    for i in 0..32 {
        let query = make_test_seq(150, 0);
        let ref_seq = make_test_seq(150, 0);
        batch.add_job(
            i,
            0,
            0,
            ExtensionDirection::Right,
            &query,
            &ref_seq,
            0,
            50,
        );
    }

    let engine = SimdEngineType::Engine256;
    let results = execute_batch_simd_scoring(&sw_params, &mut batch, engine);

    assert_eq!(results.len(), 32, "Should get 32 results for 32 jobs");
}

/// Test mixed short and long sequences
#[test]
fn test_mixed_length_sequences() {
    let sw_params = make_sw_params();
    let mut batch = ExtensionJobBatch::new();

    // Mix of short (< 128bp) and long (> 128bp) sequences
    for i in 0..50 {
        let qlen = if i % 2 == 0 { 100 } else { 150 };
        let rlen = if i % 3 == 0 { 90 } else { 160 };

        let query = make_test_seq(qlen, 0);
        let ref_seq = make_test_seq(rlen, 0);
        batch.add_job(
            i,
            0,
            0,
            ExtensionDirection::Right,
            &query,
            &ref_seq,
            0,
            50,
        );
    }

    let engine = SimdEngineType::Engine256;
    let results = execute_batch_simd_scoring(&sw_params, &mut batch, engine);

    assert_eq!(results.len(), 50, "Should get 50 results for 50 jobs");
}

/// Test SSE/NEON path (Engine128)
#[test]
fn test_engine128_multi_chunk() {
    let sw_params = make_sw_params();
    let mut batch = ExtensionJobBatch::new();

    // 32 jobs for Engine128 (16 lanes) = 2 chunks
    for i in 0..32 {
        let query = make_test_seq(100, 0);
        let ref_seq = make_test_seq(100, 0);
        batch.add_job(
            i,
            0,
            0,
            ExtensionDirection::Right,
            &query,
            &ref_seq,
            0,
            50,
        );
    }

    let engine = SimdEngineType::Engine128;
    let results = execute_batch_simd_scoring(&sw_params, &mut batch, engine);

    assert_eq!(results.len(), 32, "Should get 32 results for 32 jobs");
}
