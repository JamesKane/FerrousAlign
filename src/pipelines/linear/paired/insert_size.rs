use super::super::finalization::Alignment;
use super::super::finalization::sam_flags;

// Insert size statistics module
//
// This module handles paired-end insert size distribution analysis:
// - Orientation detection (FF, FR, RF, RR)
// - Statistical analysis (mean, std, outlier removal)
// - Bootstrap estimation from first batch
// - Proper pair bounds calculation

// Paired-end insert size constants (from C++ bwamem_pair.cpp)
const MIN_DIR_CNT: usize = 10; // Minimum pairs for orientation
const MIN_DIR_RATIO: f64 = 0.05; // Minimum ratio for orientation
const OUTLIER_BOUND: f64 = 2.0; // IQR multiplier for outliers
const MAPPING_BOUND: f64 = 3.0; // IQR multiplier for mapping
const MAX_STDDEV: f64 = 4.0; // Max standard deviations for boundaries
const MIN_RATIO: f64 = 0.8; // Minimum ratio for unique alignments (BWA-MEM2 bwamem_pair.cpp:49)
const MAX_INS: i64 = 10000; // Maximum insert size to consider (default opt->max_ins)

// Insert size statistics for one orientation
#[derive(Debug, Clone)]
pub struct InsertSizeStats {
    pub avg: f64,     // Mean insert size
    pub std: f64,     // Standard deviation
    pub low: i32,     // Lower bound for proper pairs
    pub high: i32,    // Upper bound for proper pairs
    pub failed: bool, // Whether this orientation has enough data
}

/// Infer orientation of paired reads (from C++ mem_infer_dir)
/// Returns: (orientation, insert_size)
/// Orientation: 0=FF, 1=FR, 2=RF, 3=RR
pub fn infer_orientation(l_pac: i64, pos1: i64, pos2: i64) -> (usize, i64) {
    let r1 = if pos1 >= l_pac { 1 } else { 0 };
    let r2 = if pos2 >= l_pac { 1 } else { 0 };

    // p2 is the coordinate of read2 on the read1 strand
    let p2 = if r1 == r2 {
        pos2
    } else {
        (l_pac << 1) - 1 - pos2
    };

    let dist = if p2 > pos1 { p2 - pos1 } else { pos1 - p2 };

    // Calculate orientation
    // (r1 == r2 ? 0 : 1) ^ (p2 > pos1 ? 0 : 3)
    let orientation = if r1 == r2 { 0 } else { 1 } ^ if p2 > pos1 { 0 } else { 3 };

    (orientation, dist)
}

/// Complementary error function (approximation)
pub(crate) fn erfc(x: f64) -> f64 {
    // Use standard library if available, otherwise approximation
    // For now, using a simple approximation
    let t = 1.0 / (1.0 + 0.5 * x.abs());
    let tau = t
        * (-x * x - 1.26551223
            + t * (1.00002368
                + t * (0.37409196
                    + t * (0.09678418
                        + t * (-0.18628806
                            + t * (0.27886807
                                + t * (-1.13520398
                                    + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    if x >= 0.0 { tau } else { 2.0 - tau }
}

/// Calculate insert size statistics for all 4 orientations
/// Returns: [InsertSizeStats; 4] for orientations FF, FR, RF, RR
pub fn calculate_insert_size_stats(
    l_pac: i64,
    pairs: &[(i64, i64)], // (pos1, pos2) for each pair
) -> [InsertSizeStats; 4] {
    // Collect insert sizes for each orientation
    let mut insert_sizes: [Vec<i64>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];

    for &(pos1, pos2) in pairs {
        let (orientation, dist) = infer_orientation(l_pac, pos1, pos2);
        // BWA-MEM2: if (is && is <= opt->max_ins) kv_push(...)
        if dist > 0 && dist <= MAX_INS {
            insert_sizes[orientation].push(dist);
        }
    }

    // BWA-MEM2 style logging
    log::info!(
        "[PE] # candidate unique pairs for (FF, FR, RF, RR): ({}, {}, {}, {})",
        insert_sizes[0].len(),
        insert_sizes[1].len(),
        insert_sizes[2].len(),
        insert_sizes[3].len()
    );

    let mut stats = [
        InsertSizeStats {
            avg: 0.0,
            std: 0.0,
            low: 0,
            high: 0,
            failed: true,
        },
        InsertSizeStats {
            avg: 0.0,
            std: 0.0,
            low: 0,
            high: 0,
            failed: true,
        },
        InsertSizeStats {
            avg: 0.0,
            std: 0.0,
            low: 0,
            high: 0,
            failed: true,
        },
        InsertSizeStats {
            avg: 0.0,
            std: 0.0,
            low: 0,
            high: 0,
            failed: true,
        },
    ];

    let orientation_names = ["FF", "FR", "RF", "RR"];

    // Calculate statistics for each orientation
    for d in 0..4 {
        let sizes = &mut insert_sizes[d];
        let dir_name = format!("{}{}", &"FR"[d >> 1 & 1..=d >> 1 & 1], &"FR"[d & 1..=d & 1]);

        if sizes.len() < MIN_DIR_CNT {
            log::info!("[PE] skip orientation {dir_name} as there are not enough pairs");
            continue;
        }

        log::info!("[PE] analyzing insert size distribution for orientation {dir_name}...");

        // Sort insert sizes (BWA-MEM2: ks_introsort_64)
        sizes.sort_unstable();

        // Calculate percentiles (BWA-MEM2 exact formula)
        let p25 = sizes[(0.25 * sizes.len() as f64 + 0.499) as usize];
        let p50 = sizes[(0.50 * sizes.len() as f64 + 0.499) as usize];
        let p75 = sizes[(0.75 * sizes.len() as f64 + 0.499) as usize];

        let iqr = p75 - p25;

        // Calculate initial bounds for mean/std calculation (outlier removal)
        // BWA-MEM2 bwamem_pair.cpp:118-120
        let mut low = ((p25 as f64 - OUTLIER_BOUND * iqr as f64) + 0.499) as i32;
        if low < 1 {
            low = 1;
        }
        let high_outlier = ((p75 as f64 + OUTLIER_BOUND * iqr as f64) + 0.499) as i32;

        log::info!("[PE] (25, 50, 75) percentile: ({p25}, {p50}, {p75})");
        log::info!(
            "[PE] low and high boundaries for computing mean and std.dev: ({low}, {high_outlier})"
        );

        // Calculate mean (excluding outliers) - BWA-MEM2 bwamem_pair.cpp:123-126
        let mut sum = 0i64;
        let mut count = 0usize;
        for &size in sizes.iter() {
            if size >= low as i64 && size <= high_outlier as i64 {
                sum += size;
                count += 1;
            }
        }

        if count == 0 {
            log::warn!("[PE] no valid samples for orientation {dir_name} within bounds");
            continue;
        }

        let avg = sum as f64 / count as f64;

        // Calculate standard deviation (excluding outliers) - BWA-MEM2 bwamem_pair.cpp:128-131
        let mut sum_sq = 0.0;
        for &size in sizes.iter() {
            if size >= low as i64 && size <= high_outlier as i64 {
                let diff = size as f64 - avg;
                sum_sq += diff * diff;
            }
        }
        let std = (sum_sq / count as f64).sqrt();

        log::info!("[PE] mean and std.dev: ({avg:.2}, {std:.2})");

        // Calculate final bounds for proper pairs (mapping bounds)
        // BWA-MEM2 bwamem_pair.cpp:133-137
        low = ((p25 as f64 - MAPPING_BOUND * iqr as f64) + 0.499) as i32;
        let mut high = ((p75 as f64 + MAPPING_BOUND * iqr as f64) + 0.499) as i32;

        // Adjust using standard deviation - use WIDER bounds
        // BWA-MEM2: if (r->low > r->avg - MAX_STDDEV * r->std) r->low = ...
        let low_stddev = (avg - MAX_STDDEV * std + 0.499) as i32;
        let high_stddev = (avg + MAX_STDDEV * std + 0.499) as i32;

        if low > low_stddev {
            low = low_stddev;
        }
        if high < high_stddev {
            high = high_stddev;
        }
        if low < 1 {
            low = 1;
        }

        log::info!("[PE] low and high boundaries for proper pairs: ({low}, {high})");

        stats[d] = InsertSizeStats {
            avg,
            std,
            low,
            high,
            failed: false,
        };
    }

    // Find max count across all orientations
    let max_count = insert_sizes.iter().map(|v| v.len()).max().unwrap_or(0);

    // Mark orientations with too few pairs as failed
    for d in 0..4 {
        if !stats[d].failed && insert_sizes[d].len() < (max_count as f64 * MIN_DIR_RATIO) as usize {
            stats[d].failed = true;
            log::debug!(
                "Skipping orientation {} (insufficient ratio)",
                orientation_names[d]
            );
        }
    }

    stats
}

/// Calculate the sub-optimal score for alignment uniqueness check
/// BWA-MEM2 bwamem_pair.cpp:67-79 (cal_sub)
/// Returns the best secondary alignment score that has significant overlap with primary
fn calculate_sub_score(alignments: &[Alignment]) -> i32 {
    if alignments.len() <= 1 {
        return 0;
    }

    let primary = &alignments[0];
    let mask_level = 0.5; // Default opt->mask_level

    for secondary in alignments.iter().skip(1) {
        // Check for significant overlap with primary
        let b_max = secondary.query_start.max(primary.query_start);
        let e_min = secondary.query_end.min(primary.query_end);

        if e_min > b_max {
            // Have overlap
            let min_len = (secondary.query_end - secondary.query_start)
                .min(primary.query_end - primary.query_start);
            if (e_min - b_max) as f64 >= min_len as f64 * mask_level {
                // Significant overlap - return this secondary's score
                return secondary.score;
            }
        }
    }

    // No overlapping secondary found - use minimum seed length * match score as default
    // This matches BWA-MEM2: opt->min_seed_len * opt->a (typically 19 * 1 = 19)
    19 // Default fallback
}

/// Bootstrap insert size statistics from first batch only
/// This allows streaming subsequent batches without buffering all alignments
pub fn bootstrap_insert_size_stats(
    first_batch_alignments: &[(Vec<Alignment>, Vec<Alignment>)],
    l_pac: i64,
) -> [InsertSizeStats; 4] {
    log::info!(
        "[PE] Inferring insert size distribution from data (n={})",
        first_batch_alignments.len() * 2
    );

    // Extract position pairs from first batch
    let mut position_pairs: Vec<(i64, i64)> = Vec::new();

    for (alns1, alns2) in first_batch_alignments {
        // Use best alignment from each read for statistics
        if let (Some(aln1), Some(aln2)) = (alns1.first(), alns2.first()) {
            // BWA-MEM2 bwamem_pair.cpp:97-98: Skip non-unique alignments
            // if (cal_sub(opt, r[0]) > MIN_RATIO * r[0]->a[0].score) continue;
            let sub1 = calculate_sub_score(alns1);
            let sub2 = calculate_sub_score(alns2);

            if sub1 as f64 > MIN_RATIO * aln1.score as f64 {
                continue; // Read 1 is not unique enough
            }
            if sub2 as f64 > MIN_RATIO * aln2.score as f64 {
                continue; // Read 2 is not unique enough
            }

            // Only use pairs on same reference (BWA-MEM2 bwamem_pair.cpp:99)
            if aln1.ref_name == aln2.ref_name {
                // Convert positions to bidirectional coordinate space [0, 2*l_pac)
                // This matches BWA-MEM2's convention where:
                // - Forward strand: rb = leftmost position
                // - Reverse strand: rb = (2*l_pac) - 1 - (rightmost position)
                //   (i.e., uses END of alignment, not beginning)
                let is_rev1 = (aln1.flag & sam_flags::REVERSE) != 0;
                let is_rev2 = (aln2.flag & sam_flags::REVERSE) != 0;

                // For reverse strand, use rightmost position (end of alignment)
                // pos + reference_length - 1 = rightmost coordinate
                let ref_len1 = aln1.reference_length() as i64;
                let ref_len2 = aln2.reference_length() as i64;

                let pos1 = if is_rev1 {
                    // BWA-MEM2: rb = (l_pac<<1) - (rb + aln.te + 1)
                    // aln.te is the end position, so rb + aln.te = rightmost position
                    let rightmost = aln1.pos as i64 + ref_len1 - 1;
                    (l_pac << 1) - 1 - rightmost
                } else {
                    aln1.pos as i64
                };

                let pos2 = if is_rev2 {
                    let rightmost = aln2.pos as i64 + ref_len2 - 1;
                    (l_pac << 1) - 1 - rightmost
                } else {
                    aln2.pos as i64
                };

                position_pairs.push((pos1, pos2));
            }
        }
    }

    log::debug!(
        "Extracted {} concordant position pairs from first batch",
        position_pairs.len()
    );

    // Use existing calculation logic
    if !position_pairs.is_empty() {
        calculate_insert_size_stats(l_pac, &position_pairs)
    } else {
        // Return failed stats if no pairs found
        [
            InsertSizeStats {
                avg: 0.0,
                std: 0.0,
                low: 0,
                high: 0,
                failed: true,
            },
            InsertSizeStats {
                avg: 0.0,
                std: 0.0,
                low: 0,
                high: 0,
                failed: true,
            },
            InsertSizeStats {
                avg: 0.0,
                std: 0.0,
                low: 0,
                high: 0,
                failed: true,
            },
            InsertSizeStats {
                avg: 0.0,
                std: 0.0,
                low: 0,
                high: 0,
                failed: true,
            },
        ]
    }
}

/// Bootstrap insert size statistics from SoA alignment results
///
/// SoA-native version that operates directly on SoAAlignmentResult without
/// converting to Alignment structs. This is the primary entry point for the
/// pure SoA pipeline.
///
/// # Arguments
/// * `soa_r1` - SoA alignment results for read 1
/// * `soa_r2` - SoA alignment results for read 2
/// * `l_pac` - Packed reference length (for bidirectional coordinate space)
///
/// # Returns
/// Array of InsertSizeStats for 4 orientations: FF, FR, RF, RR
pub fn bootstrap_insert_size_stats_soa(
    soa_r1: &crate::pipelines::linear::batch_extension::types::SoAAlignmentResult,
    soa_r2: &crate::pipelines::linear::batch_extension::types::SoAAlignmentResult,
    l_pac: i64,
) -> [InsertSizeStats; 4] {
    let num_reads = soa_r1.num_reads();

    log::info!(
        "[PE] Inferring insert size distribution from data (n={})",
        num_reads * 2
    );

    // Extract position pairs from SoA results
    let mut position_pairs: Vec<(i64, i64)> = Vec::new();

    for read_idx in 0..num_reads {
        let (r1_start, r1_count) = soa_r1.read_alignment_boundaries[read_idx];
        let (r2_start, r2_count) = soa_r2.read_alignment_boundaries[read_idx];

        if r1_count == 0 || r2_count == 0 {
            continue; // No alignments for this read pair
        }

        // Get best alignment for R1 (highest score)
        let best_r1_idx = (r1_start..r1_start + r1_count).max_by_key(|&i| soa_r1.scores[i]);

        // Get best alignment for R2 (highest score)
        let best_r2_idx = (r2_start..r2_start + r2_count).max_by_key(|&i| soa_r2.scores[i]);

        if let (Some(r1_idx), Some(r2_idx)) = (best_r1_idx, best_r2_idx) {
            // Calculate sub-optimal score for uniqueness check (BWA-MEM2 bwamem_pair.cpp:97-98)
            let sub1 = calculate_sub_score_soa(soa_r1, read_idx);
            let sub2 = calculate_sub_score_soa(soa_r2, read_idx);

            if sub1 as f64 > MIN_RATIO * soa_r1.scores[r1_idx] as f64 {
                continue; // Read 1 is not unique enough
            }
            if sub2 as f64 > MIN_RATIO * soa_r2.scores[r2_idx] as f64 {
                continue; // Read 2 is not unique enough
            }

            // Check if on same chromosome (BWA-MEM2 bwamem_pair.cpp:99)
            if soa_r1.ref_names[r1_idx] == soa_r2.ref_names[r2_idx] {
                let is_rev1 = (soa_r1.flags[r1_idx] & sam_flags::REVERSE) != 0;
                let is_rev2 = (soa_r2.flags[r2_idx] & sam_flags::REVERSE) != 0;

                // Calculate reference length from CIGAR for each alignment
                let ref_len1 = reference_length_from_cigar_soa(soa_r1, r1_idx) as i64;
                let ref_len2 = reference_length_from_cigar_soa(soa_r2, r2_idx) as i64;

                // Convert positions to bidirectional coordinate space [0, 2*l_pac)
                // For reverse strand, use rightmost position (end of alignment)
                let pos1 = if is_rev1 {
                    let rightmost = soa_r1.positions[r1_idx] as i64 + ref_len1 - 1;
                    (l_pac << 1) - 1 - rightmost
                } else {
                    soa_r1.positions[r1_idx] as i64
                };

                let pos2 = if is_rev2 {
                    let rightmost = soa_r2.positions[r2_idx] as i64 + ref_len2 - 1;
                    (l_pac << 1) - 1 - rightmost
                } else {
                    soa_r2.positions[r2_idx] as i64
                };

                position_pairs.push((pos1, pos2));
            }
        }
    }

    log::debug!(
        "Extracted {} concordant position pairs from first batch",
        position_pairs.len()
    );

    // Use existing calculation logic
    if !position_pairs.is_empty() {
        calculate_insert_size_stats(l_pac, &position_pairs)
    } else {
        // Return failed stats if no pairs found
        [
            InsertSizeStats {
                avg: 0.0,
                std: 0.0,
                low: 0,
                high: 0,
                failed: true,
            },
            InsertSizeStats {
                avg: 0.0,
                std: 0.0,
                low: 0,
                high: 0,
                failed: true,
            },
            InsertSizeStats {
                avg: 0.0,
                std: 0.0,
                low: 0,
                high: 0,
                failed: true,
            },
            InsertSizeStats {
                avg: 0.0,
                std: 0.0,
                low: 0,
                high: 0,
                failed: true,
            },
        ]
    }
}

/// Calculate sub-optimal score for a read's alignments in SoA format
///
/// Returns the best secondary alignment score that has significant overlap with primary.
/// This is used for uniqueness filtering during insert size bootstrapping.
fn calculate_sub_score_soa(
    soa: &crate::pipelines::linear::batch_extension::types::SoAAlignmentResult,
    read_idx: usize,
) -> i32 {
    let (start, count) = soa.read_alignment_boundaries[read_idx];

    if count <= 1 {
        return 0;
    }

    // Primary is the first (best scoring) alignment for this read
    let primary_idx = start;
    let mask_level = 0.5; // Default opt->mask_level

    // Check secondary alignments for overlap with primary
    for offset in 1..count {
        let secondary_idx = start + offset;

        // Check for significant overlap with primary
        let b_max = soa.query_starts[secondary_idx].max(soa.query_starts[primary_idx]);
        let e_min = soa.query_ends[secondary_idx].min(soa.query_ends[primary_idx]);

        if e_min > b_max {
            // Have overlap
            let min_len = (soa.query_ends[secondary_idx] - soa.query_starts[secondary_idx])
                .min(soa.query_ends[primary_idx] - soa.query_starts[primary_idx]);
            if (e_min - b_max) as f64 >= min_len as f64 * mask_level {
                // Significant overlap - return this secondary's score
                return soa.scores[secondary_idx];
            }
        }
    }

    // No overlapping secondary found - use minimum seed length * match score as default
    19 // Default fallback (matches BWA-MEM2)
}

/// Calculate reference length from CIGAR string in SoA format
///
/// Sums M, D, N, =, X operations (those that consume reference bases).
fn reference_length_from_cigar_soa(
    soa: &crate::pipelines::linear::batch_extension::types::SoAAlignmentResult,
    aln_idx: usize,
) -> i32 {
    let (cigar_start, cigar_count) = soa.cigar_boundaries[aln_idx];
    let mut ref_len = 0i32;

    for offset in 0..cigar_count {
        let idx = cigar_start + offset;
        let op = soa.cigar_ops[idx];
        let len = soa.cigar_lens[idx];

        // M, D, N, =, X consume reference
        if op == b'M' || op == b'D' || op == b'N' || op == b'=' || op == b'X' {
            ref_len += len;
        }
    }

    ref_len
}

// Re-export erfc for pairing module
pub(crate) use erfc as erfc_fn;
