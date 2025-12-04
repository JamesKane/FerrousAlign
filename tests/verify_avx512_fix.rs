use ferrous_align::core::alignment::banded_swa::BandedPairWiseSW;
use ferrous_align::pipelines::linear::batch_extension::types::{ExtensionJobBatch, ExtensionDirection};
use ferrous_align::pipelines::linear::batch_extension::dispatch::execute_batch_simd_scoring;
use ferrous_align::compute::simd_abstraction::simd::SimdEngineType;

#[test]
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
fn test_avx512_i16_long_read_fix() {
    // 1. Setup scoring params
    let sw_params = BandedPairWiseSW::new(
        6, 1, 6, 1, 100, 0, 5, 5,
        [1, -4, -4, -4, -1, -4, 1, -4, -4, -1, -4, -4, 1, -4, -1, -4, -4, -4, 1, -1, -1, -1, -1, -1, -1],
        1, -4
    );

    // 2. Create a batch with > 128bp sequences to trigger i16 path
    // AVX512 i16 width is 32. We want to ensure we fill a chunk.
    let mut batch = ExtensionJobBatch::new();
    let num_jobs = 64; 
    
    // Create a 150bp perfect match scenario
    // If the bug persists (band stuck), score will be very low.
    // If fixed, score should be ~150.
    let seq_len = 150;
    let seq = vec![0u8; seq_len]; // All 'A's
    
    for i in 0..num_jobs {
        batch.add_job(
            0, 0, i, ExtensionDirection::Right,
            &seq, &seq, 10, 10
        );
    }
    
    // 3. Execute with AVX512 engine
    println!("Executing with Engine512...");
    let engine = SimdEngineType::Engine512;
    let results = execute_batch_simd_scoring(&sw_params, &mut batch, engine);
    
    println!("Jobs: {}, Results: {}", num_jobs, results.len());
    assert_eq!(results.len(), num_jobs, "AVX512 Batch truncation check");
    
    // 4. Verify scores
    for (i, res) in results.iter().enumerate() {
        let expected_min = 140; // Allow some margin, but should be close to 150
        assert!(res.score > expected_min, "Job {} score {} too low (expected > {}). Fix failed for AVX512 i16.", i, res.score, expected_min);
    }
    
    println!("AVX512 i16 test passed successfully.");
}
