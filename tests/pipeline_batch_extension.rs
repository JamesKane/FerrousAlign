// tests/pipeline_batch_extension.rs
use ferrous_align::compute::simd_abstraction::simd::SimdEngineType;
use ferrous_align::core::alignment::banded_swa::BandedPairWiseSW; // Also need OutScore for BatchExtensionResult
use ferrous_align::pipelines::linear::batch_extension::soa::make_batch_soa;
use ferrous_align::pipelines::linear::batch_extension::{
    BatchExtensionResult, ChainExtensionScores, ExtensionDirection, ExtensionJobBatch,
    execute_batch_simd_scoring,
};

#[test]
fn test_extension_job_batch_add_and_retrieve() {
    let mut batch = ExtensionJobBatch::new();

    // Add a job
    let query = vec![0u8, 1, 2, 3]; // ACGT
    let ref_seq = vec![0u8, 1, 2, 3, 0, 1];
    batch.add_job(0, 0, 0, ExtensionDirection::Left, &query, &ref_seq, 10, 50);

    assert_eq!(batch.len(), 1);
    assert_eq!(batch.get_query_seq(0), &query);
    assert_eq!(batch.get_ref_seq(0), &ref_seq);
    assert_eq!(batch.jobs[0].h0, 10);
}

#[test]
fn test_extension_job_batch_multiple_jobs() {
    let mut batch = ExtensionJobBatch::new();

    // Add multiple jobs
    batch.add_job(
        0,
        0,
        0,
        ExtensionDirection::Left,
        &[0, 1, 2],
        &[0, 1],
        5,
        50,
    );
    batch.add_job(
        0,
        1,
        0,
        ExtensionDirection::Right,
        &[3, 2, 1],
        &[3, 2, 1, 0],
        0,
        50,
    );
    batch.add_job(
        1,
        0,
        0,
        ExtensionDirection::Left,
        &[0, 0, 1, 1],
        &[2, 2],
        8,
        50,
    );

    assert_eq!(batch.len(), 3);

    // Verify each job's sequences
    assert_eq!(batch.get_query_seq(0), &[0, 1, 2]);
    assert_eq!(batch.get_ref_seq(0), &[0, 1]);

    assert_eq!(batch.get_query_seq(1), &[3, 2, 1]);
    assert_eq!(batch.get_ref_seq(1), &[3, 2, 1, 0]);

    assert_eq!(batch.get_query_seq(2), &[0, 0, 1, 1]);
    assert_eq!(batch.get_ref_seq(2), &[2, 2]);
}

#[test]
fn test_distribute_extension_results() {
    // This test relies on `distribute_extension_results`
    // which needs to be explicitly imported or fully qualified from the new module path.
    use ferrous_align::pipelines::linear::batch_extension::distribute::distribute_extension_results;

    let results = vec![
        BatchExtensionResult {
            read_idx: 0,
            chain_idx: 0,
            seed_idx: 0,
            direction: ExtensionDirection::Left,
            score: 100,
            query_end: 10,
            ref_end: 15,
            gscore: 95,
            gref_end: 14,
            max_off: 2,
        },
        BatchExtensionResult {
            read_idx: 0,
            chain_idx: 0,
            seed_idx: 0,
            direction: ExtensionDirection::Right,
            score: 80,
            query_end: 20,
            ref_end: 25,
            gscore: 75,
            gref_end: 24,
            max_off: 1,
        },
        BatchExtensionResult {
            read_idx: 1,
            chain_idx: 0,
            seed_idx: 0,
            direction: ExtensionDirection::Left,
            score: 50,
            query_end: 5,
            ref_end: 8,
            gscore: 48,
            gref_end: 7,
            max_off: 0,
        },
    ];

    let mut all_chain_scores: Vec<Vec<ChainExtensionScores>> = vec![vec![], vec![]];

    distribute_extension_results(&results, &mut all_chain_scores);

    // Check read 0, chain 0
    assert_eq!(all_chain_scores[0][0].left_score, Some(100));
    assert_eq!(all_chain_scores[0][0].right_score, Some(80));

    // Check read 1, chain 0
    assert_eq!(all_chain_scores[1][0].left_score, Some(50));
    assert_eq!(all_chain_scores[1][0].right_score, None);
}

#[test]
fn test_make_batch_soa() {
    let mut batch = ExtensionJobBatch::new();
    let q1 = vec![0, 1, 2, 3];
    let r1 = vec![0, 1, 2, 3, 0];
    let q2 = vec![1, 2, 3];
    let r2 = vec![1, 2, 3, 1, 2];

    batch.add_job(0, 0, 0, ExtensionDirection::Left, &q1, &r1, 0, 0);
    batch.add_job(1, 0, 0, ExtensionDirection::Left, &q2, &r2, 0, 0);

    const W: usize = 16;
    let (query_soa, target_soa, pos_offsets) =
        make_batch_soa::<W>(&batch.jobs, &batch.query_seqs, &batch.ref_seqs);

    batch.query_soa = query_soa;
    batch.target_soa = target_soa;
    batch.pos_offsets = pos_offsets;
    batch.lanes = W;

    assert_eq!(batch.lanes, W);

    // pos_offsets stores [q_off, t_off, max_q, max_t]
    assert_eq!(batch.pos_offsets.len(), 4);
    assert_eq!(batch.pos_offsets[0], 0); // q_offset
    assert_eq!(batch.pos_offsets[1], 0); // t_offset
    let max_q = batch.pos_offsets[2];
    let max_t = batch.pos_offsets[3];
    assert_eq!(max_q, 4); // q1 is longer
    assert_eq!(max_t, 5); // r1 and r2 have same length

    let q_soa_len = max_q * W;
    let t_soa_len = max_t * W;
    assert_eq!(batch.query_soa.len(), q_soa_len);
    assert_eq!(batch.target_soa.len(), t_soa_len);

    // Check lane 0 (job 0)
    for pos in 0..q1.len() {
        assert_eq!(batch.query_soa[pos * W + 0], q1[pos]);
    }
    for pos in q1.len()..max_q {
        assert_eq!(batch.query_soa[pos * W + 0], 0xFF);
    }
    for pos in 0..r1.len() {
        assert_eq!(batch.target_soa[pos * W + 0], r1[pos]);
    }
    for pos in r1.len()..max_t {
        assert_eq!(batch.target_soa[pos * W + 0], 0xFF);
    }

    // Check lane 1 (job 1)
    for pos in 0..q2.len() {
        assert_eq!(batch.query_soa[pos * W + 1], q2[pos]);
    }
    for pos in q2.len()..max_q {
        assert_eq!(batch.query_soa[pos * W + 1], 0xFF);
    }
    for pos in 0..r2.len() {
        assert_eq!(batch.target_soa[pos * W + 1], r2[pos]);
    }
    for pos in r2.len()..max_t {
        assert_eq!(batch.target_soa[pos * W + 1], 0xFF);
    }

    // Check other lanes are padded
    for lane in 2..W {
        for pos in 0..max_q {
            assert_eq!(batch.query_soa[pos * W + lane], 0xFF);
        }
        for pos in 0..max_t {
            assert_eq!(batch.target_soa[pos * W + lane], 0xFF);
        }
    }
}

#[test]
fn test_execute_batch_simd_scoring_i8_soa_path() {
    // Setup BandedPairWiseSW with dummy parameters
    let mut mat = [0i8; 25];
    // Set match scores (A=0, C=1, G=2, T=3, N=4)
    mat[0 * 5 + 0] = 1; // A-A
    mat[1 * 5 + 1] = 1; // C-C
    mat[2 * 5 + 2] = 1; // G-G
    mat[3 * 5 + 3] = 1; // T-T
    // Set mismatch scores to -2
    for i in 0..4 {
        for j in 0..4 {
            if i != j {
                mat[i * 5 + j] = -2;
            }
        }
    }
    // N (ambiguous) score
    for i in 0..5 {
        mat[i * 5 + 4] = -1; // N vs any
        mat[4 * 5 + i] = -1; // Any vs N
    }
    mat[4 * 5 + 4] = 0; // N vs N

    let sw_params = BandedPairWiseSW::new(6, 1, 6, 1, 100, 1, -2, 5, mat, 1, -2);

    let mut batch = ExtensionJobBatch::new();
    // Shorter sequences to trigger i8 path (max_len <= 128)
    let q1 = vec![0, 1, 2, 3]; // ACGT
    let r1 = vec![0, 1, 2, 3]; // ACGT
    let q2 = vec![1, 1, 1, 1]; // CCCC
    let r2 = vec![1, 2, 1, 2]; // C G C G

    batch.add_job(0, 0, 0, ExtensionDirection::Left, &q1, &r1, 0, 10);
    batch.add_job(0, 0, 1, ExtensionDirection::Right, &q2, &r2, 0, 10);

    // Prepare SoA data (mock what execute_batch_simd_scoring does internally)
    const W: usize = 16; // Assuming Engine128 (SSE/NEON) for basic test
    let (query_soa, target_soa, pos_offsets) =
        make_batch_soa::<W>(&batch.jobs, &batch.query_seqs, &batch.ref_seqs);
    batch.query_soa = query_soa;
    batch.target_soa = target_soa;
    batch.pos_offsets = pos_offsets;
    batch.lanes = W;

    let results = execute_batch_simd_scoring(&sw_params, &mut batch, SimdEngineType::Engine128);

    assert_eq!(results.len(), 2);
    // Basic assertions for scores. Exact scores depend on parameters,
    // so just checking they are positive/reasonable.
    assert!(results[0].score > 0);
    assert!(results[1].score > 0);
    // Check end positions are within sequence lengths
    assert!(results[0].query_end <= q1.len() as i32);
    assert!(results[0].ref_end <= r1.len() as i32);
    assert!(results[1].query_end <= q2.len() as i32);
    assert!(results[1].ref_end <= r2.len() as i32);
}

#[test]
fn test_execute_batch_simd_scoring_i16_soa_path() {
    // Setup BandedPairWiseSW with dummy parameters
    let mut mat = [0i8; 25];
    // Set match scores (A=0, C=1, G=2, T=3, N=4)
    mat[0 * 5 + 0] = 1; // A-A
    mat[1 * 5 + 1] = 1; // C-C
    mat[2 * 5 + 2] = 1; // G-G
    mat[3 * 5 + 3] = 1; // T-T
    // Set mismatch scores to -2
    for i in 0..4 {
        for j in 0..4 {
            if i != j {
                mat[i * 5 + j] = -2;
            }
        }
    }
    // N (ambiguous) score
    for i in 0..5 {
        mat[i * 5 + 4] = -1; // N vs any
        mat[4 * 5 + i] = -1; // Any vs N
    }
    mat[4 * 5 + 4] = 0; // N vs N

    let sw_params = BandedPairWiseSW::new(6, 1, 6, 1, 100, 1, -2, 5, mat, 1, -2);

    let mut batch = ExtensionJobBatch::new();
    // Longer sequences to trigger i16 path (max_len > 128)
    let q_long = vec![0; 150]; // 150 'A's
    let r_long = vec![0; 150]; // 150 'A's
    let q_medium = vec![1; 130]; // 130 'C's
    let r_medium = vec![1; 130]; // 130 'C's

    batch.add_job(0, 0, 0, ExtensionDirection::Left, &q_long, &r_long, 0, 20);
    batch.add_job(
        0,
        0,
        1,
        ExtensionDirection::Right,
        &q_medium,
        &r_medium,
        0,
        20,
    );

    // Prepare SoA data (mock what execute_batch_simd_scoring does internally)
    // The i16 path currently uses W=8 for Engine128
    const W: usize = 8;
    let (query_soa, target_soa, pos_offsets) =
        make_batch_soa::<W>(&batch.jobs, &batch.query_seqs, &batch.ref_seqs);
    batch.query_soa = query_soa;
    batch.target_soa = target_soa;
    batch.pos_offsets = pos_offsets;
    batch.lanes = W;

    let results = execute_batch_simd_scoring(&sw_params, &mut batch, SimdEngineType::Engine128);

    assert_eq!(results.len(), 2);
    assert!(results[0].score > 0);
    assert!(results[1].score > 0);
    assert!(results[0].query_end <= q_long.len() as i32);
    assert!(results[0].ref_end <= r_long.len() as i32);
    assert!(results[1].query_end <= q_medium.len() as i32);
    assert!(results[1].ref_end <= r_medium.len() as i32);
}
