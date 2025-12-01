// tests/pipeline_batch_extension.rs
use ferrous_align::pipelines::linear::batch_extension::{
    BatchedExtensionJob, BatchExtensionResult, ChainExtensionScores, ExtensionDirection, ExtensionJobBatch
};
use ferrous_align::pipelines::linear::batch_extension::soa::make_batch_soa;
use ferrous_align::core::alignment::banded_swa::OutScore; // Also need OutScore for BatchExtensionResult

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
    let (query_soa, target_soa, pos_offsets) = make_batch_soa::<W>(&batch.jobs, &batch.query_seqs, &batch.ref_seqs);
    
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
