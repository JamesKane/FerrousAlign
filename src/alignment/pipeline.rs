use crate::alignment::banded_swa::BandedPairWiseSW;
use crate::alignment::banded_swa::merge_cigar_operations;
use crate::alignment::chaining::Chain;
use crate::alignment::chaining::cal_max_gap;
use crate::alignment::chaining::chain_seeds;
use crate::alignment::chaining::filter_chains;
use crate::alignment::extension::AlignmentJob;
use crate::alignment::extension::execute_adaptive_alignments;
use crate::alignment::extension::execute_scalar_alignments;
use crate::alignment::finalization::Alignment;
use crate::alignment::finalization::mark_secondary_alignments;
use crate::alignment::finalization::remove_redundant_alignments;
use crate::alignment::finalization::sam_flags;
use crate::alignment::seeding::SMEM;
use crate::alignment::seeding::Seed;
use crate::alignment::seeding::forward_only_seed_strategy;
use crate::alignment::seeding::generate_smems_for_strand;
use crate::alignment::seeding::generate_smems_from_position;
use crate::alignment::seeding::get_sa_entries;
use crate::alignment::utils::base_to_code;
use crate::alignment::utils::reverse_complement_code;
use crate::compute::ComputeBackend;
use crate::index::BwaIndex;
use crate::mem_opt::MemOpt;
use crate::utils::hash_64;

// ============================================================================
// SEED GENERATION (SMEM EXTRACTION)
// ============================================================================
//
// This section contains the main seed generation pipeline:
// - SMEM (Supermaximal Exact Match) extraction using FM-Index
// - Bidirectional search (forward and reverse complement)
// - Seed extension and filtering
// ============================================================================

// ============================================================================
// HETEROGENEOUS COMPUTE INTEGRATION POINT - SEED GENERATION
// ============================================================================
//
// This is the main entry point for per-read alignment. The compute_backend
// parameter controls which hardware accelerator is used for extension.
//
// To add GPU/NPU acceleration:
// 1. Pass appropriate ComputeBackend variant
// 2. In extend_chains_to_alignments(), route based on backend
// 3. For NPU: Use ONE-HOT encoding in find_seeds() (see compute::encoding)
//
// ============================================================================
pub fn generate_seeds(
    bwa_idx: &BwaIndex,
    pac_data: &[u8], // Pre-loaded PAC data for MD tag generation
    query_name: &str,
    query_seq: &[u8],
    query_qual: &str,
    opt: &MemOpt,
    read_id: u64, // Global read ID for deterministic hash tie-breaking (matches C++ bwa-mem2)
) -> Vec<Alignment> {
    // Default to CPU SIMD with auto-detected engine
    // TODO: Accept ComputeBackend parameter when full integration is implemented
    let compute_backend = crate::compute::detect_optimal_backend();
    align_read(
        bwa_idx, pac_data, query_name, query_seq, query_qual, opt, compute_backend, read_id,
    )
}

// ============================================================================
// REFACTORED PIPELINE
// ============================================================================

// Data structures for refactoring
#[derive(Debug, Clone)]
struct SeedJobMapping {
    seed_idx: usize,              // Index into chain.seeds
    left_job_idx: Option<usize>,  // LEFT extension job index
    right_job_idx: Option<usize>, // RIGHT extension job index
}

#[derive(Debug, Clone)]
struct ChainJobMapping {
    seed_jobs: Vec<SeedJobMapping>, // Multiple seeds per chain
}

type RawAlignment = (i32, Vec<(u8, i32)>, Vec<u8>, Vec<u8>);

/// Result of CIGAR merging for a single chain
///
/// Contains the pre-merged CIGAR and metadata needed for candidate building.
/// This struct is produced in the extension stage after combining left + seed + right.
///
/// Phase 3 refactoring: Coordinate conversion is now done in extension stage,
/// so this struct contains chromosome coordinates directly (ref_name, ref_id, chr_pos).
#[derive(Debug, Clone)]
struct MergedChainResult {
    chain_idx: usize,
    cigar: Vec<(u8, i32)>,     // Pre-merged and normalized CIGAR
    score: i32,
    // Reference location (Phase 3 - coordinate conversion in extension)
    ref_name: String,
    ref_id: usize,
    chr_pos: u64,              // 0-based chromosome position
    fm_index_pos: u64,         // Original FM-index position (for MD tag generation)
    fm_is_rev: bool,           // Whether FM-index pos was in reverse complement region
    // Query bounds (from CIGAR analysis)
    query_start: i32,          // Sum of leading clips
    query_end: i32,            // query_len - sum of trailing clips
}

struct ExtensionResult {
    /// Raw extension results (job-level) - kept for debugging/future use
    #[allow(dead_code)]
    extended_cigars: Vec<RawAlignment>,
    /// Pre-merged chain results (Phase 2 - CIGAR merge moved to extension)
    merged_chain_results: Vec<MergedChainResult>,
    filtered_chains: Vec<Chain>,
    sorted_seeds: Vec<Seed>,
    encoded_query: Vec<u8>,
    encoded_query_rc: Vec<u8>,
}

// ============================================================================
// CIGAR MERGING (Phase 2 - Moved from finalization to extension)
// ============================================================================

/// Merge CIGARs for all chains from extension job results
///
/// This function combines left extension + seed + right extension CIGARs
/// and picks the best-scoring seed per chain. Previously this logic was
/// in build_candidate_alignments(), but moving it here centralizes all
/// CIGAR manipulation in the extension stage.
///
/// Phase 3 refactoring: Also does coordinate conversion (FM-index pos -> chromosome pos).
fn merge_cigars_for_chains(
    bwa_idx: &BwaIndex,
    query_name: &str,
    chains: &[Chain],
    seeds: &[Seed],
    chain_to_job_map: &[ChainJobMapping],
    alignment_scores: &[i32],
    alignment_cigars: &[Vec<(u8, i32)>],
    query_len: i32,
) -> Vec<MergedChainResult> {
    let mut results = Vec::new();

    for (chain_idx, chain) in chains.iter().enumerate() {
        if chain.seeds.is_empty() {
            continue;
        }

        let mapping = &chain_to_job_map[chain_idx];
        let mut best_score = 0;
        let mut best_data: Option<(Vec<(u8, i32)>, i32, u64)> = None;

        for seed_job in &mapping.seed_jobs {
            let seed = &seeds[seed_job.seed_idx];
            let mut combined_cigar = Vec::new();
            let mut combined_score = 0;
            let mut alignment_start_pos = seed.ref_pos;

            // Track which extensions exist for proper score calculation
            let has_left = seed_job.left_job_idx.is_some();
            let has_right = seed_job.right_job_idx.is_some();

            // Add left extension CIGAR or soft clip
            if let Some(left_idx) = seed_job.left_job_idx {
                let left_cigar = &alignment_cigars[left_idx];
                combined_cigar.extend(left_cigar.iter().cloned());
                combined_score += alignment_scores[left_idx];
                let left_ref_len: i32 = left_cigar
                    .iter()
                    .filter_map(|&(op, len)| {
                        if op == b'M' || op == b'D' {
                            Some(len)
                        } else {
                            None
                        }
                    })
                    .sum();
                alignment_start_pos = alignment_start_pos.saturating_sub(left_ref_len as u64);
            } else if seed.query_pos > 0 {
                combined_cigar.push((b'S', seed.query_pos));
            }

            // Add seed match
            combined_cigar.push((b'M', seed.len));

            // Score calculation: Each extension score includes h0 = seed.len
            // - Both extensions: left_score + right_score - seed.len (avoid double-counting h0)
            // - One extension only: score already includes seed via h0
            // - No extensions: just seed.len
            if !has_left && !has_right {
                combined_score += seed.len;
            } else if has_left && has_right {
                combined_score -= seed.len;
            }

            // Add right extension CIGAR or soft clip
            let seed_end = seed.query_pos + seed.len;
            if let Some(right_idx) = seed_job.right_job_idx {
                combined_cigar.extend(alignment_cigars[right_idx].iter().cloned());
                combined_score += alignment_scores[right_idx];
            } else if seed_end < query_len {
                combined_cigar.push((b'S', query_len - seed_end));
            }

            // Merge adjacent operations and normalize CIGAR
            let merged_cigar = merge_cigar_operations(combined_cigar);

            if combined_score > best_score {
                best_score = combined_score;
                best_data = Some((merged_cigar, combined_score, alignment_start_pos));
            }
        }

        if let Some((mut cigar, score, start_pos)) = best_data {
            // Ensure CIGAR query length matches actual query length (invariant check)
            let cigar_len: i32 = cigar
                .iter()
                .filter_map(|&(op, len)| {
                    if matches!(op as char, 'M' | 'I' | 'S' | '=' | 'X') {
                        Some(len)
                    } else {
                        None
                    }
                })
                .sum();
            if cigar_len != query_len {
                if let Some(op) = cigar
                    .iter_mut()
                    .rev()
                    .find(|op| matches!(op.0 as char, 'S' | 'M'))
                {
                    op.1 += query_len - cigar_len;
                }
            }

            // Calculate query bounds from CIGAR
            let mut query_start = 0i32;
            let mut query_end = query_len;

            // Sum leading clips
            for &(op, len) in cigar.iter() {
                if op == b'S' || op == b'H' {
                    query_start += len;
                } else {
                    break;
                }
            }

            // Sum trailing clips
            for &(op, len) in cigar.iter().rev() {
                if op == b'S' || op == b'H' {
                    query_end -= len;
                } else {
                    break;
                }
            }

            // Coordinate conversion: FM-index position -> chromosome position
            // Matching BWA-MEM2 bwamem.cpp:1770:
            //   pos = bns_depos(bns, rb < bns->l_pac? rb : re - 1, &is_rev);
            // For reverse strand (rb >= l_pac), use alignment END position, not START
            let l_pac = bwa_idx.bns.packed_sequence_length;
            let ref_len: i64 = cigar
                .iter()
                .filter_map(|&(op, len)| {
                    if matches!(op as char, 'M' | 'D') {
                        Some(len as i64)
                    } else {
                        None
                    }
                })
                .sum();

            // For reverse strand alignments, use end position for bns_depos
            let depos_input = if start_pos >= l_pac {
                start_pos + ref_len as u64 - 1  // Alignment end (re - 1)
            } else {
                start_pos  // Alignment start (rb)
            };
            let (pos_f, fm_is_rev) = bwa_idx.bns.bns_depos(depos_input as i64);
            let rid = bwa_idx.bns.bns_pos2rid(pos_f);

            log::debug!(
                "{}: COORD_TRACE: start_pos={}, l_pac={}, depos_input={}, pos_f={}, rid={}, fm_is_rev={}",
                query_name, start_pos, l_pac, depos_input, pos_f, rid, fm_is_rev
            );

            let (ref_name, ref_id, chr_pos) =
                if rid >= 0 && (rid as usize) < bwa_idx.bns.annotations.len() {
                    let ann = &bwa_idx.bns.annotations[rid as usize];
                    log::debug!(
                        "{}: COORD_TRACE: rid={}, ann.name={}, ann.offset={}, chr_pos={}",
                        query_name, rid, ann.name, ann.offset, pos_f - ann.offset as i64
                    );
                    (
                        ann.name.clone(),
                        rid as usize,
                        (pos_f - ann.offset as i64) as u64,
                    )
                } else {
                    ("unknown_ref".to_string(), 0, 0)
                };

            results.push(MergedChainResult {
                chain_idx,
                cigar,
                score,
                ref_name,
                ref_id,
                chr_pos,
                fm_index_pos: start_pos,
                fm_is_rev,
                query_start,
                query_end,
            });
        }
    }

    results
}

// ============================================================================
// CANDIDATE ALIGNMENT (Phase 1 - Pipeline Front-Loading)
// ============================================================================
//
// CandidateAlignment is an intermediate type that flows from extension to
// finalization. It provides a clear contract between pipeline stages:
//
// - Extension stage: produces CandidateAlignment with all core alignment data
// - Finalization stage: handles ranking, dedup, secondary/supplementary, MAPQ
//
// This separation makes the code more testable and reduces coupling.
// ============================================================================

/// Intermediate alignment candidate between extension and finalization
///
/// Contains all the data needed to make alignment decisions (scoring, dedup,
/// secondary marking) without depending on FM-index access.
#[derive(Debug, Clone)]
struct CandidateAlignment {
    // Reference location
    ref_id: usize,
    ref_name: String,
    pos: u64, // 0-based chromosome position
    strand_rev: bool,

    // Alignment data (merged CIGAR)
    cigar: Vec<(u8, i32)>, // Normalized, merged CIGAR
    score: i32,

    // Query bounds (computed from CIGAR)
    query_start: i32, // Sum of leading S/H
    query_end: i32,   // query_len - sum of trailing S/H

    // Chain provenance (for MAPQ calculation)
    seed_coverage: i32,
    frac_rep: f32,
    hash: u64,

    // MD/NM tags (computed from reference comparison)
    md_tag: String,
    nm: i32,

    // Debug/provenance tracking
    #[allow(dead_code)]
    chain_id: usize,
}

impl CandidateAlignment {
    /// Convert CandidateAlignment to final Alignment struct
    ///
    /// This is the final step in finalization - after all scoring, dedup,
    /// and secondary/supplementary marking decisions have been made.
    fn to_alignment(&self, query_name: &str) -> Alignment {
        use crate::alignment::finalization::sam_flags;

        let flag = if self.strand_rev { sam_flags::REVERSE } else { 0 };

        Alignment {
            query_name: query_name.to_string(),
            flag,
            ref_name: self.ref_name.clone(),
            ref_id: self.ref_id,
            pos: self.pos,
            mapq: 60, // Will be recalculated during secondary marking
            score: self.score,
            cigar: self.cigar.clone(),
            rnext: "*".to_string(),
            pnext: 0,
            tlen: 0,
            // Don't store seq/qual here - they're passed at output time for memory efficiency
            seq: String::new(),
            qual: String::new(),
            tags: vec![
                ("AS".to_string(), format!("i:{}", self.score)),
                ("NM".to_string(), format!("i:{}", self.nm)),
                ("MD".to_string(), format!("Z:{}", self.md_tag)),
            ],
            query_start: self.query_start,
            query_end: self.query_end,
            seed_coverage: self.seed_coverage,
            hash: self.hash,
            frac_rep: self.frac_rep,
        }
    }
}

// ============================================================================
// HETEROGENEOUS COMPUTE INTEGRATION POINT - MAIN ALIGNMENT PIPELINE
// ============================================================================
//
// This is the core alignment pipeline. The compute_backend parameter is
// propagated to Stage 3 (Extension) where the heavy Smith-Waterman work
// is performed on the selected hardware backend.
//
// Pipeline stages:
//   Stage 1: Seeding (FM-Index, always CPU)
//   Stage 2: Chaining (O(n²) DP, always CPU)
//   Stage 3: Extension (Smith-Waterman, routes to SIMD/GPU/NPU)  ← KEY DISPATCH
//   Stage 4: Finalization (scoring, always CPU)
//
// To add GPU/NPU acceleration:
// 1. Stage 3 dispatches based on compute_backend
// 2. GPU: Batch alignment jobs → GPU kernel → collect results
// 3. NPU: Pre-filter seeds before extension (see compute::encoding)
//
// ============================================================================
pub fn align_read(
    bwa_idx: &BwaIndex,
    pac_data: &[u8],
    query_name: &str,
    query_seq: &[u8],
    query_qual: &str,
    opt: &MemOpt,
    compute_backend: ComputeBackend,
    read_id: u64, // Global read ID for deterministic hash tie-breaking (matches C++ bwa-mem2)
) -> Vec<Alignment> {
    // 1. SEEDING: Find maximal exact matches (SMEMs)
    // NOTE: For NPU integration, this is where ONE-HOT encoding would be applied
    // See compute::encoding::EncodingStrategy for the encoding abstraction
    let (seeds, encoded_query, encoded_query_rc) = find_seeds(bwa_idx, query_name, query_seq, opt);

    // 2. CHAINING: Group seeds into potential alignment chains
    let (chains, sorted_seeds) = build_and_filter_chains(seeds, opt, query_seq.len());

    // 3. EXTENSION: Perform banded Smith-Waterman on the chains
    // ========================================================================
    // HETEROGENEOUS COMPUTE DISPATCH POINT
    // ========================================================================
    // This is where alignment work is routed to the selected backend.
    // Currently: CPU SIMD (SSE/AVX2/AVX-512/NEON)
    // Future: GPU (Metal/CUDA), NPU (ANE/ONNX)
    let extension_result = extend_chains_to_alignments(
        bwa_idx,
        query_name,
        opt,
        chains,
        sorted_seeds,
        encoded_query,
        encoded_query_rc,
        compute_backend,
    );

    // 4. FINALIZATION: Score, clean up, and format the final alignments
    let final_alignments = finalize_alignments(
        extension_result,
        bwa_idx,
        pac_data,
        query_name,
        query_seq,
        query_qual,
        opt,
        read_id,
    );

    final_alignments
}

/// Stage 1: Seed Finding
fn find_seeds(
    bwa_idx: &BwaIndex,
    query_name: &str,
    query_seq: &[u8],
    opt: &MemOpt,
) -> (Vec<Seed>, Vec<u8>, Vec<u8>) {
    let query_len = query_seq.len();
    if query_len == 0 {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    #[cfg(feature = "debug-logging")]
    let is_debug_read = query_name.contains("1150:14380");

    #[cfg(feature = "debug-logging")]
    if is_debug_read {
        log::debug!("[DEBUG_READ] Generating seeds for: {}", query_name);
        log::debug!("[DEBUG_READ] Query length: {}", query_len);
    }

    // Create encoded versions of the query sequence
    let mut encoded_query = Vec::with_capacity(query_len);
    let mut encoded_query_rc = Vec::with_capacity(query_len); // Reverse complement
    for &base in query_seq {
        let code = base_to_code(base);
        encoded_query.push(code);
        encoded_query_rc.push(reverse_complement_code(code));
    }
    encoded_query_rc.reverse();

    let mut all_smems: Vec<SMEM> = Vec::new();
    let min_seed_len = opt.min_seed_len;
    let min_intv = 1u64;

    log::debug!(
        "{}: Starting SMEM generation: min_seed_len={}, min_intv={}, query_len={}",
        query_name,
        min_seed_len,
        min_intv,
        query_len
    );

    let mut max_smem_count = 0usize;

    // Process forward and reverse complement strands
    generate_smems_for_strand(
        bwa_idx,
        query_name,
        query_len,
        &encoded_query,
        false,
        min_seed_len,
        min_intv,
        &mut all_smems,
        &mut max_smem_count,
    );
    generate_smems_for_strand(
        bwa_idx,
        query_name,
        query_len,
        &encoded_query_rc,
        true,
        min_seed_len,
        min_intv,
        &mut all_smems,
        &mut max_smem_count,
    );

    // Re-seeding pass: For long unique SMEMs, re-seed from middle to find split alignments
    // This matches C++ bwamem.cpp:695-714
    // C++ uses: (int)(min_seed_len * split_factor + 0.499) which rounds to nearest
    let split_len = (opt.min_seed_len as f32 * opt.split_factor + 0.499) as i32;
    let split_width = opt.split_width as u64;

    // Collect re-seeding candidates from initial SMEMs
    let mut reseed_candidates: Vec<(usize, u64, bool)> = Vec::new(); // (middle_pos, min_intv, is_rc)

    for smem in all_smems.iter() {
        let smem_len = smem.query_end - smem.query_start + 1;
        // Re-seed if: length >= split_len AND interval_size <= split_width
        if smem_len >= split_len && smem.interval_size <= split_width {
            // Calculate middle position: (start + end + 1) >> 1 to match C++
            let middle_pos = ((smem.query_start + smem.query_end + 1) >> 1) as usize;
            let new_min_intv = smem.interval_size + 1;

            log::debug!(
                "{}: Re-seed candidate: smem m={}, n={}, len={}, s={}, middle_pos={}, new_min_intv={}, is_rc={}",
                query_name,
                smem.query_start,
                smem.query_end,
                smem_len,
                smem.interval_size,
                middle_pos,
                new_min_intv,
                smem.is_reverse_complement
            );

            reseed_candidates.push((middle_pos, new_min_intv, smem.is_reverse_complement));
        }
    }

    // Execute re-seeding for each candidate
    let initial_smem_count = all_smems.len();
    for (middle_pos, new_min_intv, is_rc) in reseed_candidates {
        let encoded = if is_rc {
            &encoded_query_rc
        } else {
            &encoded_query
        };

        generate_smems_from_position(
            bwa_idx,
            query_name,
            query_len,
            encoded,
            is_rc,
            min_seed_len,
            new_min_intv,
            middle_pos,
            &mut all_smems,
        );
    }

    if all_smems.len() > initial_smem_count {
        log::debug!(
            "{}: Re-seeding added {} new SMEMs (total: {})",
            query_name,
            all_smems.len() - initial_smem_count,
            all_smems.len()
        );
    }

    // 3rd round seeding: Additional seeding pass with forward-only strategy
    // BWA-MEM2 runs this unconditionally when max_mem_intv > 0 (default 20)
    // Uses min_seed_len + 1 as minimum length and max_mem_intv as the interval threshold
    // This finds seeds that might be missed by the supermaximal SMEM algorithm
    let smems_before_3rd_round = all_smems.len();
    let mut used_3rd_round_seeding = false;

    // Match BWA-MEM2: run 3rd round seeding unconditionally when max_mem_intv > 0
    // (Previously required all SMEMs to exceed max_occ, which was incorrect)
    if opt.max_mem_intv > 0 {
        used_3rd_round_seeding = true;
        log::debug!(
            "{}: Running 3rd round seeding (max_mem_intv={}) with {} existing SMEMs",
            query_name, opt.max_mem_intv, all_smems.len()
        );

        // Use forward-only seed strategy matching BWA-MEM2's bwtSeedStrategyAllPosOneThread
        // This iterates through ALL positions, doing forward extension only,
        // and outputs seeds when interval drops BELOW max_mem_intv
        forward_only_seed_strategy(
            bwa_idx,
            query_name,
            query_len,
            &encoded_query,
            false,
            min_seed_len,
            opt.max_mem_intv,
            &mut all_smems,
        );
        forward_only_seed_strategy(
            bwa_idx,
            query_name,
            query_len,
            &encoded_query_rc,
            true,
            min_seed_len,
            opt.max_mem_intv,
            &mut all_smems,
        );

        if all_smems.len() > smems_before_3rd_round {
            log::debug!(
                "{}: 3rd round seeding added {} new SMEMs (total: {})",
                query_name, all_smems.len() - smems_before_3rd_round, all_smems.len()
            );
        }
    }

    // Filter SMEMs
    let mut unique_filtered_smems: Vec<SMEM> = Vec::new();
    all_smems.sort_by_key(|smem| {
        (
            smem.query_start,
            smem.query_end,
            smem.bwt_interval_start,
            smem.is_reverse_complement,
        )
    });

    // NOTE: split_factor and split_width control RE-SEEDING for chimeric detection,
    // NOT seed filtering. The basic filter (min_seed_len + max_occ) is sufficient.
    // The previous "chimeric filter" was incorrectly discarding valid seeds.
    // See C++ bwamem.cpp:639-695 - split logic is for creating additional sub-seeds,
    // not for removing seeds that pass the basic quality checks.

    // For 3rd round seeding: if all SMEMs still exceed max_occ, use a much higher threshold
    // to allow some seeds through. This is the fallback for highly repetitive regions.
    // BWA-MEM2 uses seed_occurrence_3rd parameter for this purpose.
    let effective_max_occ = if used_3rd_round_seeding {
        // Find the minimum occurrence among all SMEMs and use that as the threshold
        // This ensures at least some seeds pass through
        let min_occ = all_smems.iter().map(|s| s.interval_size).min().unwrap_or(opt.max_occ as u64);
        // Use min_occ + 1 to ensure seeds pass
        let relaxed_threshold = (min_occ + 1).max(opt.max_occ as u64);
        log::debug!(
            "{}: 3rd round seeding used, relaxing max_occ filter from {} to {} (min_occ={})",
            query_name, opt.max_occ, relaxed_threshold, min_occ
        );
        relaxed_threshold
    } else {
        opt.max_occ as u64
    };

    if let Some(mut prev_smem) = all_smems.first().cloned() {
        for i in 0..all_smems.len() {
            let smem = all_smems[i];
            if i > 0 && smem == prev_smem {
                continue;
            }
            let seed_len = smem.query_end - smem.query_start + 1;
            let occurrences = smem.interval_size;
            // Keep seeds that pass basic quality filter (min_seed_len AND max_occ)
            if seed_len >= opt.min_seed_len && occurrences <= effective_max_occ {
                unique_filtered_smems.push(smem);
            }
            prev_smem = smem;
        }
    }

    log::debug!(
        "{}: Generated {} SMEMs, filtered to {} unique",
        query_name,
        all_smems.len(),
        unique_filtered_smems.len()
    );

    let mut sorted_smems = unique_filtered_smems;
    sorted_smems.sort_by_key(|smem| -(smem.query_end - smem.query_start + 1));

    // Match C++ SEEDS_PER_READ limit (see bwa-mem2/src/macro.h)
    const SEEDS_PER_READ: usize = 500;

    let mut seeds = Vec::new();
    for smem in sorted_smems.iter() {
        // Use the new get_sa_entries function to get multiple reference positions
        let ref_positions = crate::alignment::seeding::get_sa_entries(
            bwa_idx,
            smem.bwt_interval_start,
            smem.interval_size,
            opt.max_occ as u32,
        );

        let seed_len = smem.query_end - smem.query_start;
        for ref_pos in ref_positions {
            // Compute rid (chromosome ID) - skip seeds that span chromosome boundaries
            // Matches C++ bwamem.cpp:911-914
            let rid = bwa_idx.bns.pos_to_rid(ref_pos, ref_pos + seed_len as u64);
            if rid < 0 {
                // Seed spans multiple chromosomes or forward-reverse boundary - skip
                continue;
            }

            let seed = Seed {
                query_pos: smem.query_start,
                ref_pos,
                len: seed_len,
                is_rev: smem.is_reverse_complement,
                interval_size: smem.interval_size,
                rid,
            };
            seeds.push(seed);

            // Hard limit on seeds per read to prevent memory explosion
            if seeds.len() >= SEEDS_PER_READ {
                log::debug!(
                    "{}: Hit SEEDS_PER_READ limit ({}), truncating",
                    query_name,
                    SEEDS_PER_READ
                );
                break;
            }
        }

        if seeds.len() >= SEEDS_PER_READ {
            break;
        }
    }

    if max_smem_count > query_len {
        log::debug!(
            "{}: SMEM buffer grew beyond initial capacity! max_smem_count={} > query_len={}",
            query_name,
            max_smem_count,
            query_len
        );
    }

    log::debug!(
        "{}: Created {} seeds from {} SMEMs",
        query_name,
        seeds.len(),
        sorted_smems.len()
    );
    (seeds, encoded_query, encoded_query_rc)
}

/// Stage 2: Chaining
fn build_and_filter_chains(
    seeds: Vec<Seed>,
    opt: &MemOpt,
    query_len: usize,
) -> (Vec<Chain>, Vec<Seed>) {
    // Chain seeds together and then filter them
    let (mut chained_results, sorted_seeds) = chain_seeds(seeds, opt);
    log::debug!("Chaining produced {} chains", chained_results.len());

    let filtered_chains = filter_chains(&mut chained_results, &sorted_seeds, opt, query_len as i32);
    log::debug!(
        "Chain filtering kept {} chains (from {} total)",
        filtered_chains.len(),
        chained_results.len()
    );

    (filtered_chains, sorted_seeds)
}

// ============================================================================
// HETEROGENEOUS COMPUTE INTEGRATION POINT - STAGE 3: EXTENSION
// ============================================================================
//
// This is the primary compute dispatch point. The compute_backend parameter
// determines which hardware accelerator performs Smith-Waterman alignment.
//
// Current implementation:
//   ComputeBackend::CpuSimd → execute_adaptive_alignments() (SIMD batch)
//   ComputeBackend::Gpu     → NO-OP: falls back to CpuSimd
//   ComputeBackend::Npu     → NO-OP: falls back to CpuSimd
//
// To implement GPU acceleration:
// 1. Collect alignment_jobs into GPU-friendly format
// 2. Transfer to GPU memory (zero-copy preferred)
// 3. Execute GPU kernel
// 4. Collect results back
//
// To implement NPU seed pre-filtering:
// 1. Before building alignment_jobs, filter seeds using NPU classifier
// 2. NPU uses ONE-HOT encoding (see compute::encoding module)
// 3. Only surviving seeds become alignment jobs
//
// ============================================================================
fn extend_chains_to_alignments(
    bwa_idx: &BwaIndex,
    _query_name: &str,
    opt: &MemOpt,
    chains: Vec<Chain>,
    seeds: Vec<Seed>,
    encoded_query: Vec<u8>,
    encoded_query_rc: Vec<u8>,
    compute_backend: ComputeBackend,
) -> ExtensionResult {
    let sw_params = BandedPairWiseSW::new(
        opt.o_del,
        opt.e_del,
        opt.o_ins,
        opt.e_ins,
        opt.zdrop,
        5,
        opt.pen_clip5,
        opt.pen_clip3,
        opt.mat,
        opt.a as i8,
        -(opt.b as i8),
    );

    let mut alignment_jobs = Vec::new();
    let mut chain_to_job_map: Vec<ChainJobMapping> = Vec::new();

    for (chain_idx, chain) in chains.iter().enumerate() {
        if chain.seeds.is_empty() {
            chain_to_job_map.push(ChainJobMapping {
                seed_jobs: Vec::new(),
            });
            continue;
        }

        let full_query = if chain.is_rev {
            &encoded_query_rc
        } else {
            &encoded_query
        };
        let query_len = full_query.len() as i32;

        let l_pac = bwa_idx.bns.packed_sequence_length;
        let (mut rmax_0, mut rmax_1) = (l_pac << 1, 0u64);

        for &seed_idx in &chain.seeds {
            let seed = &seeds[seed_idx];
            let left_margin = seed.query_pos + cal_max_gap(opt, seed.query_pos);
            let b = seed.ref_pos.saturating_sub(left_margin as u64);
            let remaining_query = query_len - seed.query_pos - seed.len;
            let right_margin = remaining_query + cal_max_gap(opt, remaining_query);
            let e = seed.ref_pos + seed.len as u64 + right_margin as u64;
            rmax_0 = rmax_0.min(b);
            rmax_1 = rmax_1.max(e);
        }
        rmax_1 = rmax_1.min(l_pac << 1);
        if rmax_0 < l_pac && l_pac < rmax_1 {
            if seeds[chain.seeds[0]].ref_pos < l_pac {
                rmax_1 = l_pac;
            } else {
                rmax_0 = l_pac;
            }
        }

        let rseq = match bwa_idx.bns.get_reference_segment(rmax_0, rmax_1 - rmax_0) {
            Ok(seq) => seq,
            Err(_) => {
                chain_to_job_map.push(ChainJobMapping {
                    seed_jobs: Vec::new(),
                });
                continue;
            }
        };

        let mut seed_job_mappings = Vec::new();
        for &seed_chain_idx in chain.seeds.iter().rev() {
            let seed = &seeds[seed_chain_idx];
            let mut left_job_idx = None;
            let mut right_job_idx = None;

            if seed.query_pos > 0 {
                let tmp = (seed.ref_pos - rmax_0) as usize;
                if tmp > 0 && tmp <= rseq.len() {
                    left_job_idx = Some(alignment_jobs.len());
                    alignment_jobs.push(AlignmentJob {
                        seed_idx: chain_idx,
                        query: full_query[0..seed.query_pos as usize].to_vec(),
                        target: rseq[0..tmp].to_vec(),
                        band_width: opt.w,
                        query_offset: 0,
                        direction: Some(crate::alignment::banded_swa::ExtensionDirection::Left),
                        seed_len: seed.len,
                    });
                }
            }

            let seed_query_end = seed.query_pos + seed.len;
            if seed_query_end < query_len {
                let re = ((seed.ref_pos + seed.len as u64) - rmax_0) as usize;
                if re < rseq.len() {
                    right_job_idx = Some(alignment_jobs.len());
                    alignment_jobs.push(AlignmentJob {
                        seed_idx: chain_idx,
                        query: full_query[seed_query_end as usize..].to_vec(),
                        target: rseq[re..].to_vec(),
                        band_width: opt.w,
                        query_offset: seed_query_end,
                        direction: Some(crate::alignment::banded_swa::ExtensionDirection::Right),
                        seed_len: seed.len,
                    });
                }
            }
            seed_job_mappings.push(SeedJobMapping {
                seed_idx: seed_chain_idx,
                left_job_idx,
                right_job_idx,
            });
        }
        chain_to_job_map.push(ChainJobMapping {
            seed_jobs: seed_job_mappings,
        });
    }

    // ========================================================================
    // HETEROGENEOUS COMPUTE BACKEND DISPATCH
    // ========================================================================
    //
    // Route alignment work to the appropriate hardware backend.
    //
    // Current behavior:
    //   CpuSimd: Use SIMD-accelerated batched alignment (SSE/AVX2/AVX-512/NEON)
    //   Gpu:     NO-OP - falls back to CpuSimd (placeholder for Metal/CUDA)
    //   Npu:     NO-OP - falls back to CpuSimd (placeholder for ANE/ONNX)
    //
    // To add a new backend:
    //   1. Add match arm for the new ComputeBackend variant
    //   2. Implement execute_<backend>_alignments() function
    //   3. Handle result format conversion if needed
    //
    // ========================================================================
    let effective_backend = compute_backend.effective_backend();
    let extended_cigars: Vec<RawAlignment> = match effective_backend {
        // CPU SIMD backend (active implementation)
        ComputeBackend::CpuSimd(_) => {
            if alignment_jobs.len() >= 8 {
                execute_adaptive_alignments(&sw_params, &alignment_jobs)
            } else {
                execute_scalar_alignments(&sw_params, &alignment_jobs)
            }
        }
        // GPU backend placeholder (falls back to CpuSimd via effective_backend())
        ComputeBackend::Gpu => {
            // TODO: When GPU is implemented:
            // execute_gpu_alignments(&gpu_context, &sw_params, &alignment_jobs)
            log::debug!("GPU backend not implemented, using CPU SIMD");
            execute_adaptive_alignments(&sw_params, &alignment_jobs)
        }
        // NPU backend placeholder (falls back to CpuSimd via effective_backend())
        ComputeBackend::Npu => {
            // TODO: When NPU is implemented:
            // Seeds should already be filtered by NPU pre-filter in find_seeds()
            // execute_adaptive_alignments() handles the remaining alignment work
            log::debug!("NPU backend not implemented, using CPU SIMD");
            execute_adaptive_alignments(&sw_params, &alignment_jobs)
        }
    }
    .into_iter()
    .map(|(s, c, ra, qa)| (s, c, ra, qa))
    .collect();

    // Extract scores and CIGARs for merging
    let alignment_scores: Vec<i32> = extended_cigars
        .iter()
        .map(|(score, _, _, _)| *score)
        .collect();
    let alignment_cigars: Vec<Vec<(u8, i32)>> = extended_cigars
        .iter()
        .map(|(_, cigar, _, _)| cigar.clone())
        .collect();

    // Determine query length for merging
    let query_len = if !encoded_query.is_empty() {
        encoded_query.len() as i32
    } else {
        0
    };

    // Merge CIGARs for all chains (Phase 2+3 - centralize CIGAR + coordinate logic)
    let merged_chain_results = merge_cigars_for_chains(
        bwa_idx,
        _query_name,
        &chains,
        &seeds,
        &chain_to_job_map,
        &alignment_scores,
        &alignment_cigars,
        query_len,
    );

    ExtensionResult {
        extended_cigars,
        merged_chain_results,
        filtered_chains: chains,
        sorted_seeds: seeds,
        encoded_query,
        encoded_query_rc,
    }
}

/// Stage 4: Finalization
///
/// This stage converts extension results into final Alignment structs.
/// The flow is:
/// 1. Build CandidateAlignment from extension results (core alignment data)
/// 2. Filter by score threshold
/// 3. Remove redundant alignments
/// 4. Mark secondary/supplementary alignments and calculate MAPQ
/// 5. Generate XA/SA tags
/// 6. Convert to final Alignment structs
fn finalize_alignments(
    extension_result: ExtensionResult,
    bwa_idx: &BwaIndex,
    pac_data: &[u8],
    query_name: &str,
    query_seq: &[u8],
    query_qual: &str,
    opt: &MemOpt,
    read_id: u64,
) -> Vec<Alignment> {
    // Step 1: Build CandidateAlignment from extension results
    let candidates = build_candidate_alignments(
        &extension_result,
        bwa_idx,
        pac_data,
        query_name,
        read_id,
    );

    // Step 2-5: Convert to Alignments and apply post-processing
    finalize_candidates(candidates, query_name, opt)
}

/// Build CandidateAlignment structs from extension results
///
/// This extracts all core alignment data (MD/NM tags) from the pre-merged
/// extension results into a clean intermediate representation.
///
/// Phase 2+3 refactoring: CIGAR merging and coordinate conversion are now
/// done in extend_chains_to_alignments(), so this function only generates
/// MD/NM tags and builds the final CandidateAlignment structs.
fn build_candidate_alignments(
    extension_result: &ExtensionResult,
    bwa_idx: &BwaIndex,
    pac_data: &[u8],
    _query_name: &str,
    read_id: u64,
) -> Vec<CandidateAlignment> {
    let ExtensionResult {
        merged_chain_results,
        filtered_chains,
        encoded_query,
        encoded_query_rc,
        ..
    } = extension_result;

    let mut candidates = Vec::new();

    for merged in merged_chain_results {
        let chain = &filtered_chains[merged.chain_idx];
        let full_query = if chain.is_rev {
            encoded_query_rc
        } else {
            encoded_query
        };

        // Generate MD tag and calculate NM
        // NOTE: We use fm_index_pos (not chr_pos) for MD tag generation because
        // bns_get_seq expects FM-index coordinates
        let md_tag = if !pac_data.is_empty() {
            let ref_len: i32 = merged.cigar
                .iter()
                .filter_map(|&(op, len)| {
                    if matches!(op as char, 'M' | 'D') {
                        Some(len)
                    } else {
                        None
                    }
                })
                .sum();
            let ref_aligned = bwa_idx.bns.bns_get_seq(
                pac_data,
                merged.fm_index_pos as i64,
                merged.fm_index_pos as i64 + ref_len as i64,
            );
            Alignment::generate_md_tag(
                &ref_aligned,
                &full_query[merged.query_start as usize..merged.query_end as usize],
                &merged.cigar,
            )
        } else {
            merged.cigar
                .iter()
                .filter_map(|&(op, len)| if op == b'M' { Some(len) } else { None })
                .sum::<i32>()
                .to_string()
        };

        let nm = Alignment::calculate_exact_nm(&md_tag, &merged.cigar);

        // Compute final strand: XOR of query strand (is_rev) and FM-index strand (fm_is_rev)
        // - is_rev=false, fm_is_rev=false: forward query on forward ref → forward alignment
        // - is_rev=false, fm_is_rev=true: forward query on revcomp ref → reverse alignment
        // - is_rev=true, fm_is_rev=false: revcomp query on forward ref → reverse alignment
        // - is_rev=true, fm_is_rev=true: revcomp query on revcomp ref → forward alignment
        let final_strand_rev = chain.is_rev != merged.fm_is_rev;

        candidates.push(CandidateAlignment {
            ref_id: merged.ref_id,
            ref_name: merged.ref_name.clone(),
            pos: merged.chr_pos,
            strand_rev: final_strand_rev,
            cigar: merged.cigar.clone(),
            score: merged.score,
            query_start: merged.query_start,
            query_end: merged.query_end,
            seed_coverage: chain.weight,
            frac_rep: chain.frac_rep,
            // Hash assigned as 0 initially - will be reassigned after sorting
            hash: 0,
            md_tag,
            nm,
            chain_id: merged.chain_idx,
        });
    }

    // BWA-MEM2 sorting behavior (bwamem.cpp:342, alnreg_slt comparator):
    // Sort by (score DESC, ref_pos ASC, query_pos ASC) BEFORE hash assignment
    // This ensures that for equal scores, lower-position alignments get lower hashes
    // and thus become primary (since hash is used as tie-breaker in alnreg_hlt)
    candidates.sort_by(|a, b| {
        b.score.cmp(&a.score)  // score descending
            .then_with(|| a.pos.cmp(&b.pos))  // ref_pos ascending
            .then_with(|| a.query_start.cmp(&b.query_start))  // query_start ascending
    });

    // Now assign hash based on sorted order (matching C++ bwamem.cpp:1428)
    for (i, candidate) in candidates.iter_mut().enumerate() {
        candidate.hash = hash_64(read_id + i as u64);
    }

    candidates
}

/// Convert CandidateAlignment structs to final Alignments with post-processing
///
/// Handles: score filtering, redundant removal, secondary/supplementary marking,
/// MAPQ calculation, and XA/SA tag generation.
///
/// Phase 4 refactoring: XA/SA tag generation is now consolidated into
/// mark_secondary_alignments(), simplifying this function.
fn finalize_candidates(
    candidates: Vec<CandidateAlignment>,
    query_name: &str,
    opt: &MemOpt,
) -> Vec<Alignment> {
    // Convert candidates to alignments
    let mut alignments: Vec<Alignment> = candidates
        .iter()
        .map(|c| c.to_alignment(query_name))
        .collect();

    if !alignments.is_empty() {
        // Filter alignments by minimum score threshold (matches C++ bwamem.cpp:1538)
        let before_filter = alignments.len();
        alignments.retain(|a| a.score >= opt.t);
        if before_filter != alignments.len() {
            log::debug!(
                "{}: Filtered {} alignments below score threshold {} (before: {}, after: {})",
                query_name,
                before_filter - alignments.len(),
                opt.t,
                before_filter,
                alignments.len()
            );
        }

        // Remove redundant alignments (>95% overlap on both ref and query)
        // This matches C++ mem_sort_dedup_patch (bwamem.cpp:292-351)
        let before_dedup = alignments.len();
        remove_redundant_alignments(&mut alignments, opt);
        if alignments.len() != before_dedup {
            log::debug!(
                "{}: remove_redundant_alignments: {} -> {} alignments",
                query_name, before_dedup, alignments.len()
            );
        }

        // Mark secondary/supplementary, calculate MAPQ, attach XA/SA tags
        // (Phase 4 consolidation: all secondary-related processing in one call)
        alignments.sort_by(|a, b| b.score.cmp(&a.score).then_with(|| a.hash.cmp(&b.hash)));
        mark_secondary_alignments(&mut alignments, opt);
    }

    // If all alignments were filtered/removed, output unmapped record
    // This ensures SAM spec compliance: all input reads must appear in output
    log::debug!("{}: Final alignment count: {}", query_name, alignments.len());
    if alignments.is_empty() {
        log::debug!("{}: No alignments remaining, creating unmapped record", query_name);
        alignments.push(Alignment {
            query_name: query_name.to_string(),
            flag: sam_flags::UNMAPPED,
            ref_name: "*".to_string(),
            ref_id: 0,
            pos: 0,
            mapq: 0,
            score: 0,
            cigar: Vec::new(),
            rnext: "*".to_string(),
            pnext: 0,
            tlen: 0,
            // Don't store seq/qual here - they're passed at output time for memory efficiency
            seq: String::new(),
            qual: String::new(),
            tags: vec![
                ("AS".to_string(), "i:0".to_string()),
                ("NM".to_string(), "i:0".to_string()),
            ],
            query_start: 0,
            query_end: 0,
            seed_coverage: 0,
            hash: 0,
            frac_rep: 0.0,
        });
    }

    alignments
}

#[cfg(test)]
mod tests {
    use crate::alignment::pipeline::sam_flags;
    use crate::index::BwaIndex;

    #[test]
    fn test_generate_seeds_basic() {
        use crate::mem_opt::MemOpt;
        use std::path::Path;

        let prefix = Path::new("test_data/test_ref.fa");
        if !prefix.exists() {
            eprintln!("Skipping test_generate_seeds_basic - test data not found");
            return;
        }

        let bwa_idx = match BwaIndex::bwa_idx_load(&prefix) {
            Ok(idx) => idx,
            Err(_) => {
                eprintln!("Skipping test_generate_seeds_basic - could not load index");
                return;
            }
        };

        let mut opt = MemOpt::default();
        opt.min_seed_len = 10; // Ensure our seed is long enough

        let query_name = "test_query";
        let query_seq = b"ACGTACGTACGT"; // 12bp
        let query_qual = "IIIIIIIIIIII";

        // Test doesn't need real MD tags, use empty pac_data
        let pac_data: &[u8] = &[];
        let read_id = 0u64; // Test read ID
        let alignments =
            super::generate_seeds(&bwa_idx, pac_data, query_name, query_seq, query_qual, &opt, read_id);

        assert!(
            !alignments.is_empty(),
            "Expected at least one alignment for a matching query"
        );

        let primary_alignment = alignments
            .iter()
            .find(|a| a.flag & sam_flags::SECONDARY == 0);
        assert!(primary_alignment.is_some(), "Expected a primary alignment");

        let pa = primary_alignment.unwrap();
        assert_eq!(pa.ref_name, "test_sequence");
        assert!(
            pa.score > 0,
            "Expected a positive score for a good match, got {}",
            pa.score
        );
        assert!(pa.pos < 60, "Position should be within reference length");
        assert_eq!(pa.cigar_string(), "12M", "Expected a perfect match CIGAR");
    }
}
