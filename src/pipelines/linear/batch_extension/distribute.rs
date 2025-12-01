use super::types::{BatchExtensionResult, ChainExtensionScores, ExtensionJobBatch};
use crate::core::alignment::banded_swa::OutScore;

/// Distribute extension results back to per-read chain scores
pub fn distribute_extension_results(
    results: &[BatchExtensionResult],
    all_chain_scores: &mut [Vec<ChainExtensionScores>],
) {
    for result in results {
        // Ensure the chain_scores vector is large enough
        let chain_scores = &mut all_chain_scores[result.read_idx];
        if chain_scores.len() <= result.chain_idx {
            chain_scores.resize(result.chain_idx + 1, ChainExtensionScores::default());
        }

        let scores = &mut chain_scores[result.chain_idx];
        match result.direction {
            super::types::ExtensionDirection::Left => {
                scores.left_score = Some(result.score);
                scores.left_query_end = Some(result.query_end);
                scores.left_ref_end = Some(result.ref_end);
                scores.left_gscore = Some(result.gscore);
                scores.left_gref_end = Some(result.gref_end);
            }
            super::types::ExtensionDirection::Right => {
                scores.right_score = Some(result.score);
                scores.right_query_end = Some(result.query_end);
                scores.right_ref_end = Some(result.ref_end);
                scores.right_gscore = Some(result.gscore);
                scores.right_gref_end = Some(result.gref_end);
            }
        }
    }
}

/// Convert batch extension results back to per-read OutScore vectors
///
/// This allows us to use the existing `merge_scores_to_regions()` infrastructure
/// after cross-read SIMD scoring.
pub fn convert_batch_results_to_outscores(
    results: &[BatchExtensionResult],
    batch: &ExtensionJobBatch,
    num_reads: usize,
) -> Vec<Vec<OutScore>> {
    // Pre-allocate per-read result vectors
    let mut per_read_scores: Vec<Vec<OutScore>> = vec![Vec::new(); num_reads];

    // Determine size of each read's score vector from the batch jobs
    let mut read_job_counts: Vec<usize> = vec![0; num_reads];
    for job in &batch.jobs {
        read_job_counts[job.read_idx] += 1;
    }

    for (read_idx, count) in read_job_counts.iter().enumerate() {
        per_read_scores[read_idx].reserve(*count);
    }

    // Distribute results back to per-read vectors
    // Note: results are in the same order as batch.jobs
    for result in results {
        per_read_scores[result.read_idx].push(OutScore {
            score: result.score,
            query_end_pos: result.query_end,
            target_end_pos: result.ref_end,
            global_score: result.gscore,
            gtarget_end_pos: result.gref_end,
            max_offset: result.max_off,
        });
    }

    per_read_scores
}
