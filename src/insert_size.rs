// Insert size statistics module
//
// This module handles paired-end insert size distribution analysis:
// - Orientation detection (FF, FR, RF, RR)
// - Statistical analysis (mean, std, outlier removal)
// - Bootstrap estimation from first batch
// - Proper pair bounds calculation

use crate::align;

// Paired-end insert size constants (from C++ bwamem_pair.cpp)
const MIN_DIR_CNT: usize = 10; // Minimum pairs for orientation
const MIN_DIR_RATIO: f64 = 0.05; // Minimum ratio for orientation
const OUTLIER_BOUND: f64 = 2.0; // IQR multiplier for outliers
const MAPPING_BOUND: f64 = 3.0; // IQR multiplier for mapping
const MAX_STDDEV: f64 = 4.0; // Max standard deviations for boundaries

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
        if dist > 0 {
            insert_sizes[orientation].push(dist);
        }
    }

    log::info!(
        "Paired-end: {} candidate pairs (FF={}, FR={}, RF={}, RR={})",
        insert_sizes.iter().map(|v| v.len()).sum::<usize>(),
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

        if sizes.len() < MIN_DIR_CNT {
            log::debug!(
                "Skipping orientation {} (insufficient pairs: {} < {})",
                orientation_names[d],
                sizes.len(),
                MIN_DIR_CNT
            );
            continue;
        }

        log::info!(
            "Analyzing insert size for orientation {} ({} pairs)",
            orientation_names[d],
            sizes.len()
        );

        // Sort insert sizes
        sizes.sort_unstable();

        // Calculate percentiles
        let p25 = sizes[(0.25 * sizes.len() as f64 + 0.499) as usize];
        let p50 = sizes[(0.50 * sizes.len() as f64 + 0.499) as usize];
        let p75 = sizes[(0.75 * sizes.len() as f64 + 0.499) as usize];

        let iqr = p75 - p25;

        // Calculate initial bounds for mean/std calculation (outlier removal)
        let mut low = ((p25 as f64 - OUTLIER_BOUND * iqr as f64) + 0.499) as i32;
        if low < 1 {
            low = 1;
        }
        let mut high = ((p75 as f64 + OUTLIER_BOUND * iqr as f64) + 0.499) as i32;

        log::debug!("  Percentiles (25/50/75): {}/{}/{}", p25, p50, p75);
        log::debug!("  Outlier bounds: {} - {}", low, high);

        // Calculate mean (excluding outliers)
        let mut sum = 0i64;
        let mut count = 0usize;
        for &size in sizes.iter() {
            if size >= low as i64 && size <= high as i64 {
                sum += size;
                count += 1;
            }
        }

        if count == 0 {
            log::warn!(
                "No valid samples for orientation {} within bounds",
                orientation_names[d]
            );
            continue;
        }

        let avg = sum as f64 / count as f64;

        // Calculate standard deviation (excluding outliers)
        let mut sum_sq = 0.0;
        for &size in sizes.iter() {
            if size >= low as i64 && size <= high as i64 {
                let diff = size as f64 - avg;
                sum_sq += diff * diff;
            }
        }
        let std = (sum_sq / count as f64).sqrt();

        log::info!("  Insert size: mean={:.1}, std={:.1}", avg, std);

        // Calculate final bounds for proper pairs (mapping bounds)
        low = ((p25 as f64 - MAPPING_BOUND * iqr as f64) + 0.499) as i32;
        high = ((p75 as f64 + MAPPING_BOUND * iqr as f64) + 0.499) as i32;

        // Adjust using standard deviation
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

        log::info!("  Proper pair bounds: {} - {}", low, high);

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

/// Bootstrap insert size statistics from first batch only
/// This allows streaming subsequent batches without buffering all alignments
pub fn bootstrap_insert_size_stats(
    first_batch_alignments: &[(Vec<align::Alignment>, Vec<align::Alignment>)],
    l_pac: i64,
) -> [InsertSizeStats; 4] {
    log::info!(
        "Bootstrapping insert size statistics from first batch ({} pairs)",
        first_batch_alignments.len()
    );

    // Extract position pairs from first batch
    let mut position_pairs: Vec<(i64, i64)> = Vec::new();

    for (alns1, alns2) in first_batch_alignments {
        // Use best alignment from each read for statistics
        if let (Some(aln1), Some(aln2)) = (alns1.first(), alns2.first()) {
            // Only use pairs on same reference
            if aln1.ref_name == aln2.ref_name {
                // Convert positions to bidirectional coordinate space [0, 2*l_pac)
                // Forward strand: [0, l_pac), Reverse strand: [l_pac, 2*l_pac)
                let is_rev1 = (aln1.flag & 0x10) != 0;
                let is_rev2 = (aln2.flag & 0x10) != 0;

                let pos1 = if is_rev1 {
                    (l_pac << 1) - 1 - (aln1.pos as i64)
                } else {
                    aln1.pos as i64
                };

                let pos2 = if is_rev2 {
                    (l_pac << 1) - 1 - (aln2.pos as i64)
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

// Re-export erfc for pairing module
pub(crate) use erfc as erfc_fn;
