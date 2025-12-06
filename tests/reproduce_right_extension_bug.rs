use ferrous_align::compute::simd_abstraction::simd::SimdEngineType;
use ferrous_align::core::alignment::banded_swa::BandedPairWiseSW;
use ferrous_align::pipelines::linear::batch_extension::dispatch::execute_batch_simd_scoring;
use ferrous_align::pipelines::linear::batch_extension::types::{
    ExtensionDirection, ExtensionJobBatch,
};

#[test]
#[cfg(target_arch = "x86_64")]
fn test_reproduce_batch_truncation_bug() {
    // 1. Setup scoring params
    let sw_params = BandedPairWiseSW::new(
        6,
        1,
        6,
        1,
        100,
        0,
        5,
        5,
        [
            1, -4, -4, -4, -1, -4, 1, -4, -4, -1, -4, -4, 1, -4, -1, -4, -4, -4, 1, -1, -1, -1, -1,
            -1, -1,
        ],
        1,
        -4,
    );

    // 2. Create a batch with > 32 jobs (assuming AVX2 width is 32 for i8)
    let mut batch = ExtensionJobBatch::new();
    let num_jobs = 33;

    let q_seq = b"ACGT";
    let t_seq = b"ACGT";

    for i in 0..num_jobs {
        batch.add_job(0, 0, i, ExtensionDirection::Right, q_seq, t_seq, 10, 10);
    }

    // 3. Execute batch scoring
    let engine = SimdEngineType::Engine256;
    let results = execute_batch_simd_scoring(&sw_params, &mut batch, engine);

    println!("Jobs: {}, Results: {}", num_jobs, results.len());
    assert_eq!(
        results.len(),
        num_jobs,
        "Batch truncation bug: expected {} results, got {}",
        num_jobs,
        results.len()
    );
}

#[test]
#[cfg(target_arch = "x86_64")]
#[ignore = "Known bug: i16 SIMD kernel stride issue for sequences >127bp"]
fn test_reproduce_long_read_stride_bug() {
    // 1. Setup scoring params
    let sw_params = BandedPairWiseSW::new(
        6,
        1,
        6,
        1,
        100,
        0,
        5,
        5,
        [
            1, -4, -4, -4, -1, -4, 1, -4, -4, -1, -4, -4, 1, -4, -1, -4, -4, -4, 1, -1, -1, -1, -1,
            -1, -1,
        ],
        1,
        -4,
    );

    // 2. Create a batch with > 128bp sequences to trigger i16 path
    let mut batch = ExtensionJobBatch::new();
    let num_jobs = 32; // Full batch

    // Job 0..15: All A
    // Job 16..31: All C
    let seq_a = vec![0u8; 150]; // A (0)
    let seq_c = vec![1u8; 150]; // C (1)

    for i in 0..num_jobs {
        let seq = if i < 16 { &seq_a } else { &seq_c };
        batch.add_job(0, 0, i, ExtensionDirection::Right, seq, seq, 10, 10);
    }

    let engine = SimdEngineType::Engine256;
    let results = execute_batch_simd_scoring(&sw_params, &mut batch, engine);

    println!("Jobs: {}, Results: {}", num_jobs, results.len());

    // Check count (Bug 2)
    assert_eq!(
        results.len(),
        num_jobs,
        "Batch truncation bug (i16 path): expected {} results, got {}",
        num_jobs,
        results.len()
    );

    // Check score of Job 0 (Bug 1)
    // If stride bug exists, Job 0 sees Mix of A and C.
    // A (0) vs C (1) is mismatch (-4).
    // If proper, Job 0 sees A vs A (match +1).
    // Score should be high.
    let score_0 = results[0].score;
    println!("Job 0 Score: {}", score_0);

    // Approx score: 150 * 1 = 150.
    // If mixed: 50% match, 50% mismatch -> 75 - 300 = negative.
    assert!(
        score_0 > 100,
        "Stride bug detected! Score {} is too low for perfect match.",
        score_0
    );
}
