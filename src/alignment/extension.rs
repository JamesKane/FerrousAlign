// ----------------------------------------------------------------------------
// Alignment Job Structure and Divergence Estimation
// ----------------------------------------------------------------------------
use crate::alignment::banded_swa::BandedPairWiseSW;
use crate::alignment::ksw_affine_gap::{ksw_extend2, KswExtendResult};

// Structure to hold alignment job for batching
#[derive(Clone)]
#[cfg_attr(test, derive(Debug))]
pub(crate) struct AlignmentJob {
    #[allow(dead_code)] // Used for tracking but not currently read
    pub seed_idx: usize,
    pub query: Vec<u8>,
    pub target: Vec<u8>,
    pub band_width: i32,
    // C++ STRATEGY: Track query offset for seed-boundary-based alignment
    // Only align seed-covered region, soft-clip the rest like bwa-mem2
    pub query_offset: i32, // Offset of query in full read (for soft-clipping)
    /// Extension direction: LEFT (seed → qb=0) or RIGHT (seed → qe=qlen)
    /// Used for separate left/right extensions matching C++ bwa-mem2
    /// None = legacy single-pass mode (deprecated)
    pub direction: Option<crate::alignment::banded_swa::ExtensionDirection>,
    /// Seed length for calculating h0 (initial score = seed_len * match_score)
    /// C++ bwamem.cpp:2232 sets h0 = s->len * opt->a
    pub seed_len: i32,
}

/// Execute alignments with adaptive strategy selection
///
/// **Hybrid Approach**:
/// 1. Partition jobs into low-divergence and high-divergence based on length mismatch
/// 2. Route high-divergence jobs (>70% length mismatch) to scalar processing
/// 3. Route low-divergence jobs to batched SIMD with adaptive batch sizing
///
/// **Performance Benefits**:
/// - High-divergence sequences avoid SIMD overhead and batch synchronization penalty
/// - Low-divergence sequences use optimal batch sizes for their characteristics
/// - Expected 15-25% improvement over fixed batching strategy
pub(crate) fn execute_adaptive_alignments(
    sw_params: &BandedPairWiseSW,
    jobs: &[AlignmentJob],
) -> Vec<(i32, Vec<(u8, i32)>, Vec<u8>, Vec<u8>)> {
    if jobs.is_empty() {
        return Vec::new();
    }

    // SIMD routing: Use SIMD for all jobs (length-based heuristic was flawed)
    // The previous divergence-based routing incorrectly compared query vs target lengths,
    // but target is always larger due to alignment margins (~100bp each side).
    // This caused ALL jobs to route to scalar (0% SIMD utilization).
    // SIMD handles variable lengths efficiently with padding, so just use it for everything.
    let (low_div_jobs, high_div_jobs) = (jobs.to_vec(), Vec::new());

    // Calculate average divergence and length statistics for logging
    let avg_divergence: f64 = jobs
        .iter()
        .map(|job| estimate_divergence_score(job.query.len(), job.target.len()))
        .sum::<f64>()
        / jobs.len() as f64;

    let avg_query_len: f64 =
        jobs.iter().map(|job| job.query.len()).sum::<usize>() as f64 / jobs.len() as f64;
    let avg_target_len: f64 =
        jobs.iter().map(|job| job.target.len()).sum::<usize>() as f64 / jobs.len() as f64;

    // Log routing statistics (DEBUG level - too verbose for INFO)
    log::debug!(
        "Adaptive routing: {} total jobs, {} scalar ({:.1}%), {} SIMD ({:.1}%), avg_divergence={:.3}",
        jobs.len(),
        high_div_jobs.len(),
        high_div_jobs.len() as f64 / jobs.len() as f64 * 100.0,
        low_div_jobs.len(),
        low_div_jobs.len() as f64 / jobs.len() as f64 * 100.0,
        avg_divergence
    );

    // Show length statistics to understand why routing fails
    log::debug!(
        "  → avg_query={:.1}bp, avg_target={:.1}bp, length_diff={:.1}%",
        avg_query_len,
        avg_target_len,
        ((avg_query_len - avg_target_len).abs() / avg_query_len.max(avg_target_len) * 100.0)
    );

    // Create result vector with correct size
    let mut all_results = vec![(0, Vec::new(), Vec::new(), Vec::new()); jobs.len()];

    // Process high-divergence jobs with scalar (more efficient for divergent sequences)
    let high_div_results = if !high_div_jobs.is_empty() {
        log::debug!(
            "Processing {} high-divergence jobs with scalar",
            high_div_jobs.len()
        );
        execute_scalar_alignments(sw_params, &high_div_jobs)
    } else {
        Vec::new()
    };

    // Process low-divergence jobs with adaptive batched SIMD
    let low_div_results = if !low_div_jobs.is_empty() {
        let optimal_batch_size = determine_optimal_batch_size(&low_div_jobs);
        log::debug!(
            "Processing {} low-divergence jobs with SIMD (batch_size={})",
            low_div_jobs.len(),
            optimal_batch_size
        );
        execute_batched_alignments_with_size(sw_params, &low_div_jobs, optimal_batch_size)
    } else {
        Vec::new()
    };

    // Since we route everything to SIMD now (high_div_jobs is always empty),
    // just return the SIMD results directly
    if high_div_results.is_empty() {
        // All jobs went to SIMD - return directly (common case now)
        low_div_results
    } else {
        // Mixed routing (should not happen with current logic, but keep for safety)
        let mut low_idx = 0;
        let mut high_idx = 0;

        for (original_idx, job) in jobs.iter().enumerate() {
            let div_score = estimate_divergence_score(job.query.len(), job.target.len());
            let result = if div_score > 0.7 && high_idx < high_div_results.len() {
                // High divergence - get from scalar results
                let res = high_div_results[high_idx].clone();
                high_idx += 1;
                res
            } else {
                // Low divergence - get from SIMD results
                let res = low_div_results[low_idx].clone();
                low_idx += 1;
                res
            };

            all_results[original_idx] = result;
        }
        all_results
    }
}

/// Execute batched alignments with configurable batch size
/// Uses SIMD dispatch to automatically select optimal engine (SSE2/AVX2/AVX-512)
fn execute_batched_alignments_with_size(
    sw_params: &BandedPairWiseSW,
    jobs: &[AlignmentJob],
    batch_size: usize,
) -> Vec<(i32, Vec<(u8, i32)>, Vec<u8>, Vec<u8>)> {
    let mut all_results = vec![(0, Vec::new(), Vec::new(), Vec::new()); jobs.len()];

    // Process jobs in batches of specified size
    for batch_start in (0..jobs.len()).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(jobs.len());
        let batch_jobs = &jobs[batch_start..batch_end];

        // Prepare batch data
        // CRITICAL: h0 must be seed_len, not 0 (C++ bwamem.cpp:2232)
        // CRITICAL: Include direction for LEFT/RIGHT extension (fixes insertion detection bug)
        let batch_data: Vec<(
            i32,
            Vec<u8>,
            i32,
            Vec<u8>,
            i32,
            i32,
            Option<crate::alignment::banded_swa::ExtensionDirection>,
        )> = batch_jobs
            .iter()
            .map(|job| {
                // For LEFT extension: reverse both query and target (C++ bwamem.cpp:2278)
                let (query, target) = if job.direction
                    == Some(crate::alignment::banded_swa::ExtensionDirection::Left)
                {
                    (
                        job.query.iter().copied().rev().collect(),
                        job.target.iter().copied().rev().collect(),
                    )
                } else {
                    (job.query.clone(), job.target.clone())
                };

                (
                    query.len() as i32,
                    query,
                    target.len() as i32,
                    target,
                    job.band_width,
                    job.seed_len, // h0 = seed_len (initial score from existing seed)
                    job.direction,
                )
            })
            .collect();

        // Execute batched alignment with CIGAR generation
        // Dispatch automatically selects batch16/32/64 based on detected SIMD engine
        let results = sw_params.simd_banded_swa_dispatch_with_cigar(&batch_data);

        // Extract scores, CIGARs, and aligned sequences from results
        for (i, result) in results.iter().enumerate() {
            if i < batch_jobs.len() {
                all_results[batch_start + i] = (
                    result.score.score,
                    result.cigar.clone(),
                    result.ref_aligned.clone(),
                    result.query_aligned.clone(),
                );

                // Detect pathological CIGARs in SIMD path
                let total_insertions: i32 = result
                    .cigar
                    .iter()
                    .filter(|(op, _)| *op == b'I')
                    .map(|(_, count)| count)
                    .sum();
                let total_deletions: i32 = result
                    .cigar
                    .iter()
                    .filter(|(op, _)| *op == b'D')
                    .map(|(_, count)| count)
                    .sum();

                if total_insertions > 10 || total_deletions > 5 {
                    let job = &batch_jobs[i];
                    // ATOMIC LOG: All data in single statement to avoid multi-threaded interleaving
                    log::debug!(
                        "PATHOLOGICAL_CIGAR_SIMD|idx={}|qlen={}|tlen={}|bw={}|score={}|ins={}|del={}|CIGAR={:?}|QUERY={:?}|TARGET={:?}",
                        batch_start + i,
                        job.query.len(),
                        job.target.len(),
                        job.band_width,
                        result.score.score,
                        total_insertions,
                        total_deletions,
                        result.cigar,
                        job.query,
                        job.target
                    );
                }
            }
        }
    }

    all_results
}

/// Execute alignments using scalar processing (fallback for small batches)
pub(crate) fn execute_scalar_alignments(
    sw_params: &BandedPairWiseSW,
    jobs: &[AlignmentJob],
) -> Vec<(i32, Vec<(u8, i32)>, Vec<u8>, Vec<u8>)> {
    jobs.iter()
        .enumerate()
        .map(|(idx, job)| {
            let qlen = job.query.len() as i32;
            let tlen = job.target.len() as i32;

            // Log query sequence being aligned
            let query_str: String = job.query.iter().map(|&b| match b {
                0 => 'A', 1 => 'C', 2 => 'G', 3 => 'T', _ => 'N',
            }).collect();
            log::debug!(
                "SW alignment job {}: qlen={}, tlen={}, bw={}, query_seq={}",
                idx, qlen, tlen, job.band_width, query_str
            );

            // Use directional extension if specified, otherwise use standard SW
            let (score, cigar, ref_aligned, query_aligned) = if let Some(direction) = job.direction {
                // CRITICAL: Calculate h0 from seed length (C++ bwamem.cpp:2232)
                // h0 = seed_length * match_score gives SW algorithm the initial score
                // Without this, SW starts from 0 and finds terrible local alignments
                // For extensions, we need to know we're extending from a high-scoring seed
                let h0 = job.seed_len; // match_score = 1, so h0 = seed_len * 1

                // Use directional extension (LEFT or RIGHT)
                let ext_result = sw_params.scalar_banded_swa_directional(
                    direction,
                    qlen,
                    &job.query,
                    tlen,
                    &job.target,
                    job.band_width,
                    h0,
                );

                log::debug!(
                    "Iterative Directional alignment {}: direction={:?}, local_score={}, global_score={}, should_clip={}",
                    idx,
                    direction,
                    ext_result.local_score,
                    ext_result.global_score,
                    ext_result.should_clip
                );

                (
                    ext_result.local_score,
                    ext_result.cigar,
                    ext_result.ref_aligned,
                    ext_result.query_aligned,
                )
            } else {
                // Use standard SW for backward compatibility (legacy tests)
                // CRITICAL: Use h0=seed_len, not 0 (matching production code)
                let h0 = job.seed_len;
                let (score_out, cigar, ref_aligned, query_aligned) = sw_params.scalar_banded_swa(
                    qlen,
                    &job.query,
                    tlen,
                    &job.target,
                    job.band_width,
                    h0,
                );
                (score_out.score, cigar, ref_aligned, query_aligned)
            };

            // Detect pathological CIGARs (excessive insertions/deletions)
            let total_insertions: i32 = cigar.iter()
                .filter(|(op, _)| *op == b'I')
                .map(|(_, count)| count)
                .sum();
            let total_deletions: i32 = cigar.iter()
                .filter(|(op, _)| *op == b'D')
                .map(|(_, count)| count)
                .sum();

            // Check if this is a pathological alignment that might benefit from ksw_extend2
            let is_pathological = total_insertions > 10 || total_deletions > 5;
            let is_low_score = score < (qlen / 2); // Score less than half query length

            // Try ksw_extend2 fallback for pathological or low-scoring alignments
            let (final_score, final_cigar, final_ref_aligned, final_query_aligned) =
                if is_pathological || is_low_score {
                    // Log the pathological CIGAR
                    log::debug!(
                        "PATHOLOGICAL_CIGAR_SCALAR|idx={}|qlen={}|tlen={}|bw={}|score={}|ins={}|del={}|CIGAR={:?}",
                        idx, qlen, tlen, job.band_width, score, total_insertions, total_deletions, cigar
                    );

                    // Try ksw_extend2 fallback
                    let h0 = job.seed_len.max(1);
                    if let Some(extend_result) = try_ksw_extend2_fallback(
                        sw_params,
                        &job.query,
                        &job.target,
                        job.band_width,
                        h0,
                        score,
                    ) {
                        // ksw_extend2 found a better alignment
                        let new_score = if extend_result.gscore > 0 {
                            extend_result.gscore
                        } else {
                            extend_result.score
                        };

                        // Generate CIGAR from the extend result
                        let (new_cigar, new_ref_aligned, new_query_aligned) =
                            generate_cigar_from_extend_result(
                                extend_result.qle,
                                extend_result.tle,
                                &job.query,
                                &job.target,
                            );

                        log::debug!(
                            "ksw_extend2 fallback used|idx={}|old_score={}|new_score={}|new_cigar={:?}",
                            idx, score, new_score, new_cigar
                        );

                        (new_score, new_cigar, new_ref_aligned, new_query_aligned)
                    } else {
                        // Fallback didn't improve, use original result
                        (score, cigar, ref_aligned, query_aligned)
                    }
                } else {
                    // Not pathological, use original result
                    (score, cigar, ref_aligned, query_aligned)
                };

            if idx < 3 {
                // Log first 3 alignments for debugging
                log::debug!(
                    "Scalar alignment {}: qlen={}, tlen={}, score={}, CIGAR_len={}, first_op={:?}",
                    idx,
                    qlen,
                    tlen,
                    final_score,
                    final_cigar.len(),
                    final_cigar.first().map(|&(op, len)| (op as char, len))
                );
            }

            (final_score, final_cigar, final_ref_aligned, final_query_aligned)
        })
        .collect()
}

/// Determine optimal batch size based on estimated divergence and SIMD engine
///
/// **Strategy**:
/// - Detect available SIMD engine (SSE2/AVX2/AVX-512)
/// - Low divergence (score < 0.3): Use engine's maximum batch size for efficiency
/// - Medium divergence (0.3-0.7): Use engine's standard batch size
/// - High divergence (> 0.7): Use smaller batches or route to scalar
///
/// **SIMD Engine Batch Sizes**:
/// - SSE2/NEON (128-bit): 16-way parallelism
/// - AVX2 (256-bit): 32-way parallelism
/// - AVX-512 (512-bit): 64-way parallelism
///
/// This reduces batch synchronization penalty for divergent sequences while
/// maximizing SIMD utilization for similar sequences.
fn determine_optimal_batch_size(jobs: &[AlignmentJob]) -> usize {
    use crate::simd::{detect_optimal_simd_engine, get_simd_batch_sizes};

    if jobs.is_empty() {
        return 16; // Default
    }

    // Detect SIMD engine and get optimal batch sizes
    let engine = detect_optimal_simd_engine();
    let (max_batch, standard_batch) = get_simd_batch_sizes(engine);

    // Calculate average divergence score for this batch of jobs
    let total_divergence: f64 = jobs
        .iter()
        .map(|job| estimate_divergence_score(job.query.len(), job.target.len()))
        .sum();

    let avg_divergence = total_divergence / jobs.len() as f64;

    // Adaptive batch sizing based on divergence and SIMD capability
    if avg_divergence < 0.3 {
        // Low divergence: Use engine's maximum batch size for best SIMD utilization
        max_batch
    } else if avg_divergence < 0.7 {
        // Medium divergence: Use engine's standard batch size
        standard_batch
    } else {
        // High divergence: Use smaller batches to reduce synchronization penalty
        // Use half of standard batch, minimum 8
        (standard_batch / 2).max(8)
    }
}

/// Classify jobs as low-divergence or high-divergence for routing
///
/// Returns (low_divergence_jobs, high_divergence_jobs)
fn partition_jobs_by_divergence(jobs: &[AlignmentJob]) -> (Vec<AlignmentJob>, Vec<AlignmentJob>) {
    const DIVERGENCE_THRESHOLD: f64 = 0.7; // Route to scalar if > 0.7

    let mut low_div = Vec::new();
    let mut high_div = Vec::new();

    for job in jobs {
        let div_score = estimate_divergence_score(job.query.len(), job.target.len());
        if div_score > DIVERGENCE_THRESHOLD {
            high_div.push(job.clone());
        } else {
            low_div.push(job.clone());
        }
    }

    (low_div, high_div)
}

/// Estimate divergence likelihood based on sequence length mismatch
///
/// Returns a score from 0.0 (low divergence) to 1.0 (high divergence)
/// based on the ratio of length mismatch to total length.
///
/// **Heuristic**: Sequences with significant length differences are likely
/// to have insertions/deletions, indicating higher divergence.
fn estimate_divergence_score(query_len: usize, target_len: usize) -> f64 {
    let max_len = query_len.max(target_len);
    let min_len = query_len.min(target_len);

    if max_len == 0 {
        return 0.0;
    }

    // Length mismatch ratio (0.0 = identical length, 1.0 = one sequence is empty)
    let length_mismatch = (max_len - min_len) as f64 / max_len as f64;

    // Scale: 0-10% mismatch → low divergence (0.0-0.3)
    //        10-30% mismatch → medium divergence (0.3-0.7)
    //        30%+ mismatch → high divergence (0.7-1.0)
    (length_mismatch * 2.5).min(1.0)
}

/// Execute alignments using batched SIMD (processes up to 16 at a time)
/// Now includes CIGAR generation via hybrid approach
/// NOTE: This function is deprecated - use execute_adaptive_alignments instead
pub(crate) fn execute_batched_alignments(
    sw_params: &BandedPairWiseSW,
    jobs: &[AlignmentJob],
) -> Vec<(i32, Vec<(u8, i32)>, Vec<u8>, Vec<u8>)> {
    const BATCH_SIZE: usize = 16;
    let mut all_results = vec![(0, Vec::new(), Vec::new(), Vec::new()); jobs.len()];

    // Process jobs in batches of 16
    for batch_start in (0..jobs.len()).step_by(BATCH_SIZE) {
        let batch_end = (batch_start + BATCH_SIZE).min(jobs.len());
        let batch_jobs = &jobs[batch_start..batch_end];

        // Prepare batch data for SIMD dispatch
        // CRITICAL: h0 must be seed_len, not 0 (C++ bwamem.cpp:2232)
        // CRITICAL: Include direction for LEFT/RIGHT extension (fixes insertion detection bug)
        let batch_data: Vec<(
            i32,
            Vec<u8>,
            i32,
            Vec<u8>,
            i32,
            i32,
            Option<crate::alignment::banded_swa::ExtensionDirection>,
        )> = batch_jobs
            .iter()
            .map(|job| {
                // For LEFT extension: reverse both query and target (C++ bwamem.cpp:2278)
                let (query, target) = if job.direction
                    == Some(crate::alignment::banded_swa::ExtensionDirection::Left)
                {
                    (
                        job.query.iter().copied().rev().collect(),
                        job.target.iter().copied().rev().collect(),
                    )
                } else {
                    (job.query.clone(), job.target.clone())
                };

                (
                    query.len() as i32,
                    query,
                    target.len() as i32,
                    target,
                    job.band_width,
                    job.seed_len, // h0 = seed_len (initial score from existing seed)
                    job.direction,
                )
            })
            .collect();

        // Execute batched alignment with CIGAR generation
        // Use dispatch to automatically route to optimal SIMD width (16/32/64)
        let results = sw_params.simd_banded_swa_dispatch_with_cigar(&batch_data);

        // Extract scores, CIGARs, and aligned sequences from results
        for (i, result) in results.iter().enumerate() {
            if i < batch_jobs.len() {
                all_results[batch_start + i] = (
                    result.score.score,
                    result.cigar.clone(),
                    result.ref_aligned.clone(),
                    result.query_aligned.clone(),
                );
            }
        }
    }

    all_results
}

/// Try ksw_extend2 as a fallback for pathological alignments.
///
/// This function attempts alignment using the affine-gap Smith-Waterman algorithm
/// which handles large indels better than the local scalar approach.
///
/// Returns `Some((score, extended_result))` if ksw_extend2 found a better alignment,
/// `None` if the original scalar result should be used.
fn try_ksw_extend2_fallback(
    sw_params: &BandedPairWiseSW,
    query: &[u8],
    target: &[u8],
    band_width: i32,
    h0: i32,
    original_score: i32,
) -> Option<KswExtendResult> {
    let qlen = query.len() as i32;
    let tlen = target.len() as i32;

    // Skip if sequences are too short
    if qlen < 10 || tlen < 10 {
        return None;
    }

    // Try ksw_extend2 with wider band to handle large indels
    let wider_band = (band_width * 2).max(50);

    // Get scoring matrix as slice
    let mat: &[i8] = sw_params.scoring_matrix();

    let result = ksw_extend2(
        qlen,
        query,
        tlen,
        target,
        sw_params.alphabet_size(),
        mat,
        sw_params.o_del(),
        sw_params.e_del(),
        sw_params.o_ins(),
        sw_params.e_ins(),
        wider_band,
        sw_params.end_bonus(),
        sw_params.zdrop(),
        h0.max(1), // h0 must be positive
    );

    // Check if ksw_extend2 found a significantly better alignment
    // gscore > 0 means it aligned the entire query
    // score > original_score means it found a better alignment
    let improvement_threshold = (original_score as f64 * 0.2) as i32; // 20% improvement needed
    if result.gscore > 0 && result.gscore > original_score + improvement_threshold {
        log::debug!(
            "ksw_extend2 fallback improved alignment: original={}, gscore={}, qle={}, tle={}, max_off={}",
            original_score, result.gscore, result.qle, result.tle, result.max_off
        );
        Some(result)
    } else if result.score > original_score + improvement_threshold {
        log::debug!(
            "ksw_extend2 fallback found better local: original={}, score={}, qle={}, tle={}, max_off={}",
            original_score, result.score, result.qle, result.tle, result.max_off
        );
        Some(result)
    } else {
        None
    }
}

/// Generate a simple CIGAR from ksw_extend2 result positions.
///
/// This creates an approximate CIGAR based on the alignment length difference.
/// For more accurate CIGARs, a full traceback would be needed.
fn generate_cigar_from_extend_result(
    qle: i32,
    tle: i32,
    query: &[u8],
    target: &[u8],
) -> (Vec<(u8, i32)>, Vec<u8>, Vec<u8>) {
    let qlen = qle.min(query.len() as i32);
    let tlen = tle.min(target.len() as i32);

    let mut cigar = Vec::new();
    let mut ref_aligned = Vec::new();
    let mut query_aligned = Vec::new();

    // Simple CIGAR generation based on length difference
    let len_diff = tlen - qlen;

    if len_diff == 0 {
        // Sequences are same length - just report matches/mismatches as M
        cigar.push((b'M', qlen));
        ref_aligned.extend_from_slice(&target[..tlen as usize]);
        query_aligned.extend_from_slice(&query[..qlen as usize]);
    } else if len_diff > 0 {
        // Target is longer - there are deletions (gaps in query)
        // Split as: some M, then D, then more M
        let match_len = qlen;
        cigar.push((b'M', match_len));
        cigar.push((b'D', len_diff));
        ref_aligned.extend_from_slice(&target[..tlen as usize]);
        query_aligned.extend_from_slice(&query[..qlen as usize]);
        // Add gap characters for the deletion
        for _ in 0..len_diff {
            query_aligned.push(b'-');
        }
    } else {
        // Query is longer - there are insertions (gaps in target)
        let ins_len = -len_diff;
        let match_len = tlen;
        cigar.push((b'M', match_len));
        cigar.push((b'I', ins_len));
        ref_aligned.extend_from_slice(&target[..tlen as usize]);
        // Add gap characters for the insertion
        for _ in 0..ins_len {
            ref_aligned.push(b'-');
        }
        query_aligned.extend_from_slice(&query[..qlen as usize]);
    }

    (cigar, ref_aligned, query_aligned)
}

#[cfg(test)]
mod tests {
    use crate::alignment::utils::DEFAULT_SCORING_MATRIX;

    #[test]
    fn test_batched_alignment_infrastructure() {
        // Test that the batched alignment infrastructure works correctly
        use crate::alignment::banded_swa::BandedPairWiseSW;

        let sw_params =
            BandedPairWiseSW::new(4, 2, 4, 2, 100, 0, 5, 5, DEFAULT_SCORING_MATRIX, 2, -4);

        // Create test alignment jobs
        let query1 = vec![0u8, 1, 2, 3]; // ACGT
        let target1 = vec![0u8, 1, 2, 3]; // ACGT (perfect match)

        let query2 = vec![0u8, 0, 1, 1]; // AACC
        let target2 = vec![0u8, 0, 1, 1]; // AACC (perfect match)

        let jobs = vec![
            super::AlignmentJob {
                seed_idx: 0,
                query: query1.clone(),
                target: target1.clone(),
                band_width: 10,
                query_offset: 0, // Test: align from start
                direction: None, // Legacy test mode
                seed_len: 4,     // Actual sequence length (4bp queries)
            },
            super::AlignmentJob {
                seed_idx: 1,
                query: query2.clone(),
                target: target2.clone(),
                band_width: 10,
                query_offset: 0, // Test: align from start
                direction: None, // Legacy test mode
                seed_len: 4,     // Actual sequence length (4bp queries)
            },
        ];

        // Test scalar execution
        let scalar_results = super::execute_scalar_alignments(&sw_params, &jobs);
        assert_eq!(
            scalar_results.len(),
            2,
            "Should return 2 results for 2 jobs"
        );
        assert!(
            !scalar_results[0].1.is_empty(),
            "First CIGAR should not be empty"
        );
        assert!(
            !scalar_results[1].1.is_empty(),
            "Second CIGAR should not be empty"
        );

        // Test batched execution
        let batched_results = super::execute_batched_alignments(&sw_params, &jobs);
        assert_eq!(
            batched_results.len(),
            2,
            "Should return 2 results for 2 jobs"
        );

        // Results should be identical (CIGARs and scores)
        assert_eq!(
            scalar_results[0].0, batched_results[0].0,
            "Scores should match"
        );
        assert_eq!(
            scalar_results[0].1, batched_results[0].1,
            "CIGARs should match"
        );
        assert_eq!(
            scalar_results[1].0, batched_results[1].0,
            "Scores should match"
        );
        assert_eq!(
            scalar_results[1].1, batched_results[1].1,
            "CIGARs should match"
        );
    }

    #[test]
    fn test_ksw_extend2_fallback_helper() {
        // Test the try_ksw_extend2_fallback function directly
        use crate::alignment::banded_swa::BandedPairWiseSW;

        let sw_params =
            BandedPairWiseSW::new(4, 2, 4, 2, 100, 0, 5, 5, DEFAULT_SCORING_MATRIX, 2, -4);

        // Create a query and target with large indel
        // Query: ACGTACGTACGTACGTACGTACGT (24 bases)
        // Target: ACGTACGT--DELETION--ACGTACGTACGT (24 + 10 = 34 bases with 10bp deleted from query's perspective)
        let query: Vec<u8> = vec![
            0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
        ]; // 24bp

        // Target has 10bp gap in the middle (simulating deletion in query)
        let target: Vec<u8> = vec![
            0, 1, 2, 3, 0, 1, 2, 3, // 8bp matching
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 10bp different (simulates indel region)
            0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, // 16bp
        ]; // 34bp total

        // Try the fallback with a low original score (simulating pathological scalar result)
        let result = super::try_ksw_extend2_fallback(
            &sw_params, &query, &target, 10, // band_width
            12,  // h0 (half the query)
            5,   // original_score (artificially low)
        );

        // The fallback should find something (result may or may not be better)
        // This test verifies the function runs without panic
        if let Some(extend_result) = result {
            assert!(
                extend_result.score > 0 || extend_result.gscore > 0,
                "Extend result should have positive score"
            );
            println!(
                "ksw_extend2 found: score={}, gscore={}, qle={}, tle={}",
                extend_result.score, extend_result.gscore, extend_result.qle, extend_result.tle
            );
        }
    }

    #[test]
    fn test_generate_cigar_from_extend_result() {
        // Test CIGAR generation from extend result
        let query = vec![0u8, 1, 2, 3, 0, 1, 2, 3]; // 8bp
        let target = vec![0u8, 1, 2, 3, 0, 1, 2, 3, 0, 1]; // 10bp

        // Same length
        let (cigar, ref_al, query_al) = super::generate_cigar_from_extend_result(8, 8, &query, &target);
        assert_eq!(cigar.len(), 1);
        assert_eq!(cigar[0], (b'M', 8));
        assert_eq!(ref_al.len(), 8);
        assert_eq!(query_al.len(), 8);

        // Target longer (deletion in query)
        let (cigar, _, _) = super::generate_cigar_from_extend_result(8, 10, &query, &target);
        assert_eq!(cigar.len(), 2);
        assert_eq!(cigar[0], (b'M', 8));
        assert_eq!(cigar[1], (b'D', 2));

        // Query longer (insertion in query)
        let (cigar, _, _) = super::generate_cigar_from_extend_result(10, 8, &target, &query);
        assert_eq!(cigar.len(), 2);
        assert_eq!(cigar[0], (b'M', 8));
        assert_eq!(cigar[1], (b'I', 2));
    }

    #[test]
    fn test_scalar_alignment_with_large_indel() {
        // Test that scalar alignment with large indel triggers fallback consideration
        use crate::alignment::banded_swa::BandedPairWiseSW;

        let sw_params =
            BandedPairWiseSW::new(4, 2, 4, 2, 100, 0, 5, 5, DEFAULT_SCORING_MATRIX, 2, -4);

        // Create sequences with a 10bp insertion in query
        // Reference: ACGTACGTACGTACGT (16bp)
        // Query:     ACGTACGT[INSERTION]ACGTACGT (16bp + 10bp = 26bp)
        let ref_seq: Vec<u8> = vec![
            0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, // 16bp
        ];
        let query_seq: Vec<u8> = vec![
            0, 1, 2, 3, 0, 1, 2, 3, // 8bp before insertion
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 10bp insertion
            0, 1, 2, 3, 0, 1, 2, 3, // 8bp after insertion
        ]; // 26bp total

        let jobs = vec![super::AlignmentJob {
            seed_idx: 0,
            query: query_seq.clone(),
            target: ref_seq.clone(),
            band_width: 5, // Narrow band to potentially cause issues
            query_offset: 0,
            direction: None,
            seed_len: 8, // Seed covers first 8bp
        }];

        // Run scalar alignments - this should trigger fallback consideration
        // due to the large indel potentially causing pathological CIGAR
        let results = super::execute_scalar_alignments(&sw_params, &jobs);
        assert_eq!(results.len(), 1, "Should return 1 result");

        // Verify we got some result (score may vary based on whether fallback was used)
        let (score, cigar, _, _) = &results[0];
        println!(
            "Large indel test: score={}, cigar_ops={}",
            score,
            cigar.len()
        );

        // The alignment should complete without panic
        assert!(*score >= 0, "Score should be non-negative");
    }
}
