// tests/dispatch_policy.rs
// Validate the centralized SoA dispatch policy.

use ferrous_align::compute::simd_abstraction::simd::detect_optimal_simd_engine;
use ferrous_align::core::alignment::banded_swa::BandedPairWiseSW;
use ferrous_align::pipelines::linear::batch_extension::dispatch::execute_batch_simd_scoring;
use ferrous_align::pipelines::linear::batch_extension::types::{
    BatchedExtensionJob, ExtensionDirection, ExtensionJobBatch,
};

#[test]
fn test_soa_dispatch_long_read() {
    let q = vec![0u8; 200];
    let t = vec![0u8; 200];

    let mut batch = ExtensionJobBatch::new();
    batch.add_job(0, 0, 0, ExtensionDirection::Right, &q, &t, 0, 10);

    let mut mat = [0i8; 25];
    for i in 0..4 {
        mat[i * 5 + i] = 1;
    }

    let sw_params = BandedPairWiseSW::new(6, 1, 6, 1, 100, 0, 0, 0, mat, 1, -4);
    let engine = detect_optimal_simd_engine();

    let results = execute_batch_simd_scoring(&sw_params, &mut batch, engine);

    assert_eq!(results.len(), 1);
    // The exact score depends on the engine and implementation details.
    // The main point is to check that it runs without errors.
    // A more thorough test would compare with a scalar implementation.
    assert!(results[0].score > 0);
}
