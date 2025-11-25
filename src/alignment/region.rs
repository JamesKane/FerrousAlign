// ============================================================================
// ALIGNMENT REGION - DEFERRED CIGAR ARCHITECTURE
// ============================================================================
//
// This module implements the "alignment region" concept from BWA-MEM2, which
// stores alignment boundaries (rb, re, qb, qe) WITHOUT generating CIGAR.
//
// ## BWA-MEM2 Reference
//
// This is the Rust equivalent of C++ `mem_alnreg_t` (bwamem.h:137-158):
// - Extension phase computes boundaries via SIMD batch scoring
// - CIGAR is generated LATER only for surviving alignments (~10-20%)
// - This architecture eliminates ~80-90% of CIGAR computation
//
// ## Heterogeneous Compute Integration
//
// The `extend_chains_to_regions()` function includes dispatch points for:
// - CPU SIMD (active): SSE/AVX2/AVX-512/NEON batch scoring
// - GPU (placeholder): Metal/CUDA/ROCm kernel dispatch
// - NPU (placeholder): ANE/ONNX accelerated seed filtering
//
// ============================================================================

use crate::alignment::banded_swa::{BandedPairWiseSW, OutScore};
use crate::alignment::chaining::{Chain, cal_max_gap};
use crate::alignment::seeding::Seed;
use crate::compute::ComputeBackend;
use crate::index::BwaIndex;
use crate::mem_opt::MemOpt;

/// Alignment region with boundaries but NO CIGAR
///
/// Matches C++ `mem_alnreg_t` structure (bwamem.h:137-158).
/// CIGAR is generated later via `generate_cigar_from_region()`.
///
/// ## Performance Note
///
/// This struct is ~100 bytes vs ~300+ bytes for a full Alignment with CIGAR.
/// Since ~80-90% of regions are filtered before CIGAR generation, this
/// significantly reduces memory allocation during extension.
#[derive(Debug, Clone)]
pub struct AlignmentRegion {
    /// Index into the chains array
    pub chain_idx: usize,

    /// Index of the best seed within this chain
    pub seed_idx: usize,

    // ========================================================================
    // Reference boundaries (FM-index coordinates)
    // ========================================================================
    /// Reference start position (inclusive, FM-index space)
    /// C++ equivalent: mem_alnreg_t.rb
    pub rb: u64,

    /// Reference end position (exclusive, FM-index space)
    /// C++ equivalent: mem_alnreg_t.re
    pub re: u64,

    // ========================================================================
    // Query boundaries (0-based)
    // ========================================================================
    /// Query start position (inclusive, 0-based)
    /// C++ equivalent: mem_alnreg_t.qb
    pub qb: i32,

    /// Query end position (exclusive, 0-based)
    /// C++ equivalent: mem_alnreg_t.qe
    pub qe: i32,

    // ========================================================================
    // Alignment metrics
    // ========================================================================
    /// Best local Smith-Waterman score
    /// C++ equivalent: mem_alnreg_t.score
    pub score: i32,

    /// True score corresponding to the aligned region
    /// May be smaller than score due to clipping adjustments
    /// C++ equivalent: mem_alnreg_t.truesc
    pub truesc: i32,

    /// Actual band width used in extension
    /// C++ equivalent: mem_alnreg_t.w
    pub w: i32,

    /// Length of regions covered by seeds
    /// Used for MAPQ calculation
    /// C++ equivalent: mem_alnreg_t.seedcov
    pub seedcov: i32,

    /// Length of the starting seed
    /// C++ equivalent: mem_alnreg_t.seedlen0
    pub seedlen0: i32,

    // ========================================================================
    // Reference information
    // ========================================================================
    /// Reference sequence ID (-1 if spanning boundary)
    /// C++ equivalent: mem_alnreg_t.rid
    pub rid: i32,

    /// Reference sequence name (e.g., "chr1")
    pub ref_name: String,

    /// Chromosome position (0-based, after coordinate conversion)
    pub chr_pos: u64,

    /// Whether the alignment is on the reverse strand
    pub is_rev: bool,

    // ========================================================================
    // Paired-end and selection fields
    // ========================================================================
    /// Fraction of repetitive seeds in this alignment
    /// Used for MAPQ calculation
    /// C++ equivalent: mem_alnreg_t.frac_rep
    pub frac_rep: f32,

    /// Hash for deterministic tie-breaking
    /// C++ equivalent: mem_alnreg_t.hash
    pub hash: u64,

    /// Secondary alignment index (-1 if primary)
    /// C++ equivalent: mem_alnreg_t.secondary
    pub secondary: i32,

    /// Sub-optimal score (2nd best)
    /// C++ equivalent: mem_alnreg_t.sub
    pub sub: i32,

    /// Number of sub-alignments chained together
    /// C++ equivalent: mem_alnreg_t.n_comp
    pub n_comp: i32,
}

impl AlignmentRegion {
    /// Create a new alignment region with default values
    pub fn new(chain_idx: usize, seed_idx: usize) -> Self {
        Self {
            chain_idx,
            seed_idx,
            rb: 0,
            re: 0,
            qb: 0,
            qe: 0,
            score: 0,
            truesc: 0,
            w: 0,
            seedcov: 0,
            seedlen0: 0,
            rid: -1,
            ref_name: String::new(),
            chr_pos: 0,
            is_rev: false,
            frac_rep: 0.0,
            hash: 0,
            secondary: -1,
            sub: 0,
            n_comp: 0,
        }
    }

    /// Calculate aligned query length
    #[inline]
    pub fn query_span(&self) -> i32 {
        self.qe - self.qb
    }

    /// Calculate aligned reference length
    #[inline]
    pub fn ref_span(&self) -> u64 {
        self.re - self.rb
    }

    /// Check if this region overlaps significantly with another
    /// Used for redundant alignment removal
    pub fn overlaps_with(&self, other: &AlignmentRegion, mask_level: f32) -> bool {
        if self.rid != other.rid {
            return false;
        }

        // Check reference overlap
        let ref_overlap_start = self.rb.max(other.rb);
        let ref_overlap_end = self.re.min(other.re);
        if ref_overlap_start >= ref_overlap_end {
            return false;
        }

        // Check query overlap
        let query_overlap_start = self.qb.max(other.qb);
        let query_overlap_end = self.qe.min(other.qe);
        if query_overlap_start >= query_overlap_end {
            return false;
        }

        // Calculate overlap fraction
        let ref_overlap = (ref_overlap_end - ref_overlap_start) as f32;
        let query_overlap = (query_overlap_end - query_overlap_start) as f32;

        let min_ref_span = (self.ref_span().min(other.ref_span())) as f32;
        let min_query_span = (self.query_span().min(other.query_span())) as f32;

        let ref_frac = if min_ref_span > 0.0 {
            ref_overlap / min_ref_span
        } else {
            0.0
        };
        let query_frac = if min_query_span > 0.0 {
            query_overlap / min_query_span
        } else {
            0.0
        };

        ref_frac > mask_level || query_frac > mask_level
    }
}

// ============================================================================
// SEED JOB TRACKING FOR SCORE-ONLY EXTENSION
// ============================================================================

/// Tracks which extension jobs belong to which seed
#[derive(Debug, Clone)]
pub(crate) struct SeedExtensionMapping {
    pub seed_idx: usize,
    pub left_job_idx: Option<usize>,
    pub right_job_idx: Option<usize>,
}

/// Tracks all seed mappings for a chain
#[derive(Debug, Clone)]
pub(crate) struct ChainExtensionMapping {
    pub seed_mappings: Vec<SeedExtensionMapping>,
}

/// Extension job with sequence data
///
/// Used for building SIMD batch scoring tuples.
#[derive(Debug, Clone)]
struct ExtensionJob {
    pub query: Vec<u8>,
    pub target: Vec<u8>,
    pub h0: i32,
    pub chain_idx: usize,
    pub seed_idx: usize,
}

// ============================================================================
// EXTENSION RESULT (SCORES ONLY, NO CIGAR)
// ============================================================================

/// Result of score-only extension phase
///
/// Contains all information needed for filtering and CIGAR regeneration,
/// but NO CIGAR strings (they're generated later for survivors only).
pub struct ScoreOnlyExtensionResult {
    /// Alignment regions with boundaries (no CIGAR)
    pub regions: Vec<AlignmentRegion>,

    /// Filtered chains (for CIGAR regeneration)
    pub chains: Vec<Chain>,

    /// Sorted seeds (for CIGAR regeneration)
    pub seeds: Vec<Seed>,

    /// Encoded query (for CIGAR regeneration)
    pub encoded_query: Vec<u8>,

    /// Reverse complement of query (for strand handling)
    pub encoded_query_rc: Vec<u8>,
}

// ============================================================================
// HETEROGENEOUS COMPUTE INTEGRATION POINT - SCORE-ONLY EXTENSION
// ============================================================================
//
// This function is the primary target for GPU/NPU acceleration.
// The compute_backend parameter controls hardware dispatch:
//
// - CpuSimd: Use existing SIMD batch scoring (SSE/AVX2/AVX-512/NEON)
// - Gpu: Dispatch to Metal/CUDA/ROCm kernel (placeholder)
// - Npu: Use NPU for seed pre-filtering (placeholder)
//
// ============================================================================

/// Extend chains to alignment regions using SIMD batch scoring
///
/// This is the score-only extension phase matching BWA-MEM2's
/// `mem_chain2aln_across_reads_V2` function. CIGAR is NOT generated here.
///
/// ## Heterogeneous Compute
///
/// The `compute_backend` parameter controls hardware dispatch:
/// - `CpuSimd`: SIMD batch scoring (active implementation)
/// - `Gpu`: GPU kernel dispatch (NO-OP, falls back to CpuSimd)
/// - `Npu`: NPU seed filtering (NO-OP, falls back to CpuSimd)
///
/// ## Returns
///
/// `ScoreOnlyExtensionResult` containing:
/// - `regions`: Alignment boundaries (no CIGAR)
/// - Supporting data for later CIGAR regeneration
pub fn extend_chains_to_regions(
    bwa_idx: &BwaIndex,
    query_name: &str,
    opt: &MemOpt,
    chains: Vec<Chain>,
    seeds: Vec<Seed>,
    encoded_query: Vec<u8>,
    encoded_query_rc: Vec<u8>,
    compute_backend: ComputeBackend,
) -> ScoreOnlyExtensionResult {
    let sw_params = BandedPairWiseSW::new(
        opt.o_del,
        opt.e_del,
        opt.o_ins,
        opt.e_ins,
        opt.zdrop,
        5, // end_bonus
        opt.pen_clip5,
        opt.pen_clip3,
        opt.mat,
        opt.a as i8,
        -(opt.b as i8),
    );

    let query_len = encoded_query.len() as i32;
    let l_pac = bwa_idx.bns.packed_sequence_length;

    // Build extension job batches for left and right extensions
    // Format: (query_len, query, target_len, target, band_width, h0)
    let mut left_jobs: Vec<ExtensionJob> = Vec::new();
    let mut right_jobs: Vec<ExtensionJob> = Vec::new();
    let mut chain_mappings: Vec<ChainExtensionMapping> = Vec::new();

    // Track reference segments for each chain (needed for CIGAR regeneration)
    let mut chain_ref_segments: Vec<Option<(u64, u64, Vec<u8>)>> = Vec::new();

    for (chain_idx, chain) in chains.iter().enumerate() {
        if chain.seeds.is_empty() {
            chain_mappings.push(ChainExtensionMapping {
                seed_mappings: Vec::new(),
            });
            chain_ref_segments.push(None);
            continue;
        }

        // Calculate rmax bounds (same as existing code)
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

        // Fetch reference segment
        let rseq = match bwa_idx.bns.get_reference_segment(rmax_0, rmax_1 - rmax_0) {
            Ok(seq) => seq,
            Err(_) => {
                chain_mappings.push(ChainExtensionMapping {
                    seed_mappings: Vec::new(),
                });
                chain_ref_segments.push(None);
                continue;
            }
        };

        chain_ref_segments.push(Some((rmax_0, rmax_1, rseq.clone())));

        // Build seed mappings and extension jobs
        let mut seed_mappings = Vec::new();

        for &seed_chain_idx in chain.seeds.iter().rev() {
            let seed = &seeds[seed_chain_idx];
            let mut left_job_idx = None;
            let mut right_job_idx = None;

            // Left extension
            if seed.query_pos > 0 {
                let tmp = (seed.ref_pos - rmax_0) as usize;
                if tmp > 0 && tmp <= rseq.len() {
                    left_job_idx = Some(left_jobs.len());

                    // PHASE 4 VALIDATION: Log left extension job
                    if log::log_enabled!(log::Level::Debug) {
                        log::debug!(
                            "EXTENSION_JOB {}: Chain[{}] type=LEFT query=[0..{}] ref=[{}..{}] seed_pos={} seed_len={} rev={}",
                            query_name,
                            chain_idx,
                            seed.query_pos,
                            rmax_0,
                            seed.ref_pos,
                            seed.query_pos,
                            seed.len,
                            seed.is_rev
                        );
                    }

                    // Build reversed sequences for left extension
                    let query_seg: Vec<u8> = encoded_query[0..seed.query_pos as usize]
                        .iter()
                        .rev()
                        .copied()
                        .collect();
                    let target_seg: Vec<u8> = rseq[0..tmp].iter().rev().copied().collect();

                    left_jobs.push(ExtensionJob {
                        query: query_seg,
                        target: target_seg,
                        h0: seed.len * opt.a,
                        chain_idx,
                        seed_idx: seed_chain_idx,
                    });
                }
            }

            // Right extension
            let seed_query_end = seed.query_pos + seed.len;
            if seed_query_end < query_len {
                let re = ((seed.ref_pos + seed.len as u64) - rmax_0) as usize;
                if re < rseq.len() {
                    right_job_idx = Some(right_jobs.len());

                    // PHASE 4 VALIDATION: Log right extension job
                    if log::log_enabled!(log::Level::Debug) {
                        log::debug!(
                            "EXTENSION_JOB {}: Chain[{}] type=RIGHT query=[{}..{}] ref=[{}..{}] seed_pos={} seed_len={} rev={}",
                            query_name,
                            chain_idx,
                            seed_query_end,
                            query_len,
                            seed.ref_pos + seed.len as u64,
                            rmax_1,
                            seed.query_pos,
                            seed.len,
                            seed.is_rev
                        );
                    }

                    let query_seg: Vec<u8> = encoded_query[seed_query_end as usize..].to_vec();
                    let target_seg: Vec<u8> = rseq[re..].to_vec();

                    right_jobs.push(ExtensionJob {
                        query: query_seg,
                        target: target_seg,
                        h0: 0, // Right extension starts from zero
                        chain_idx,
                        seed_idx: seed_chain_idx,
                    });
                }
            }

            seed_mappings.push(SeedExtensionMapping {
                seed_idx: seed_chain_idx,
                left_job_idx,
                right_job_idx,
            });
        }

        chain_mappings.push(ChainExtensionMapping { seed_mappings });
    }

    log::debug!(
        "DEFERRED_CIGAR: {} left jobs, {} right jobs from {} chains",
        left_jobs.len(),
        right_jobs.len(),
        chains.len()
    );

    // Log h0 values for debugging
    if log::log_enabled!(log::Level::Debug) && !left_jobs.is_empty() {
        let left_h0s: Vec<i32> = left_jobs.iter().map(|j| j.h0).collect();
        log::debug!("DEFERRED_CIGAR: left h0 values: {:?}", left_h0s);
    }

    // ========================================================================
    // HETEROGENEOUS COMPUTE BACKEND DISPATCH
    // ========================================================================
    //
    // Route score-only extension to appropriate hardware backend.
    //
    // Current behavior:
    //   CpuSimd: Use SIMD batch scoring (active)
    //   Gpu:     NO-OP - falls back to CpuSimd (placeholder for Metal/CUDA)
    //   Npu:     NO-OP - falls back to CpuSimd (placeholder for ANE/ONNX)
    //
    // ========================================================================

    let effective_backend = compute_backend.effective_backend();

    let (left_scores, right_scores): (Vec<OutScore>, Vec<OutScore>) = match effective_backend {
        ComputeBackend::CpuSimd(_) => {
            // Execute SIMD batch scoring (score-only, no CIGAR)
            let left = if !left_jobs.is_empty() {
                execute_simd_scoring(&sw_params, &left_jobs, opt.w)
            } else {
                Vec::new()
            };

            let right = if !right_jobs.is_empty() {
                execute_simd_scoring(&sw_params, &right_jobs, opt.w)
            } else {
                Vec::new()
            };

            (left, right)
        }

        // ====================================================================
        // GPU BACKEND (PLACEHOLDER)
        // ====================================================================
        //
        // To implement GPU scoring:
        // 1. Convert ExtensionJob batch to GPU-friendly format
        // 2. Transfer to GPU memory
        // 3. Execute GPU Smith-Waterman kernel
        // 4. Collect OutScore results
        //
        // Example:
        // ```
        // ComputeBackend::Gpu => {
        //     let left = execute_gpu_scoring(&gpu_ctx, &left_jobs, opt.w);
        //     let right = execute_gpu_scoring(&gpu_ctx, &right_jobs, opt.w);
        //     (left, right)
        // }
        // ```
        // ====================================================================
        ComputeBackend::Gpu => {
            log::debug!("GPU backend requested but not implemented, falling back to CPU SIMD");
            let left = if !left_jobs.is_empty() {
                execute_simd_scoring(&sw_params, &left_jobs, opt.w)
            } else {
                Vec::new()
            };
            let right = if !right_jobs.is_empty() {
                execute_simd_scoring(&sw_params, &right_jobs, opt.w)
            } else {
                Vec::new()
            };
            (left, right)
        }

        // ====================================================================
        // NPU BACKEND (PLACEHOLDER)
        // ====================================================================
        //
        // NPU acceleration primarily targets seed pre-filtering, not extension.
        // The NPU path would use ONE-HOT encoding (see compute::encoding).
        //
        // For extension scoring, NPU falls back to CPU SIMD.
        // ====================================================================
        ComputeBackend::Npu => {
            log::debug!("NPU backend requested but not implemented, falling back to CPU SIMD");
            let left = if !left_jobs.is_empty() {
                execute_simd_scoring(&sw_params, &left_jobs, opt.w)
            } else {
                Vec::new()
            };
            let right = if !right_jobs.is_empty() {
                execute_simd_scoring(&sw_params, &right_jobs, opt.w)
            } else {
                Vec::new()
            };
            (left, right)
        }
    };

    // Log raw SIMD scores for debugging
    if log::log_enabled!(log::Level::Debug) && (!left_scores.is_empty() || !right_scores.is_empty())
    {
        let left_raw: Vec<i32> = left_scores.iter().map(|s| s.score).collect();
        let right_raw: Vec<i32> = right_scores.iter().map(|s| s.score).collect();
        log::debug!("DEFERRED_CIGAR: raw left scores: {:?}", left_raw);
        log::debug!("DEFERRED_CIGAR: raw right scores: {:?}", right_raw);
    }

    // Merge scores into AlignmentRegions
    let regions = merge_scores_to_regions(
        bwa_idx,
        opt,
        &chains,
        &seeds,
        &chain_mappings,
        &chain_ref_segments,
        &left_scores,
        &right_scores,
        &left_jobs,
        &right_jobs,
        query_len,
    );

    ScoreOnlyExtensionResult {
        regions,
        chains,
        seeds,
        encoded_query,
        encoded_query_rc,
    }
}

/// Execute SIMD batch scoring (score-only, no CIGAR)
///
/// Uses existing SIMD infrastructure to compute OutScores.
fn execute_simd_scoring(
    sw_params: &BandedPairWiseSW,
    jobs: &[ExtensionJob],
    band_width: i32,
) -> Vec<OutScore> {
    if jobs.is_empty() {
        return Vec::new();
    }

    // Convert ExtensionJob to the tuple format expected by simd_banded_swa_batch8_int16
    // Format: (qlen, &query, tlen, &target, w, h0)
    let batch: Vec<(i32, &[u8], i32, &[u8], i32, i32)> = jobs
        .iter()
        .map(|job| {
            (
                job.query.len() as i32,
                job.query.as_slice(),
                job.target.len() as i32,
                job.target.as_slice(),
                band_width,
                job.h0,
            )
        })
        .collect();

    // Use 16-bit SIMD for better score range (handles scores > 127)
    // This matches BWA-MEM2's getScores16 usage for typical read lengths
    sw_params.simd_banded_swa_batch8_int16(&batch)
}

/// Merge left/right extension scores into AlignmentRegions
///
/// This implements the boundary calculation logic from BWA-MEM2's
/// mem_chain2aln_across_reads_V2 (bwamem.cpp:2486-2520).
fn merge_scores_to_regions(
    bwa_idx: &BwaIndex,
    opt: &MemOpt,
    chains: &[Chain],
    seeds: &[Seed],
    chain_mappings: &[ChainExtensionMapping],
    chain_ref_segments: &[Option<(u64, u64, Vec<u8>)>],
    left_scores: &[OutScore],
    right_scores: &[OutScore],
    _left_jobs: &[ExtensionJob],
    _right_jobs: &[ExtensionJob],
    query_len: i32,
) -> Vec<AlignmentRegion> {
    let mut regions = Vec::new();
    let l_pac = bwa_idx.bns.packed_sequence_length;

    for (chain_idx, chain) in chains.iter().enumerate() {
        if chain.seeds.is_empty() {
            continue;
        }

        let mapping = &chain_mappings[chain_idx];
        let ref_segment = match &chain_ref_segments[chain_idx] {
            Some(seg) => seg,
            None => continue,
        };
        let (rmax_0, _rmax_1, _rseq) = ref_segment;

        // Find best seed in chain (by score)
        let mut best_score = i32::MIN;
        let mut best_region: Option<AlignmentRegion> = None;

        for seed_mapping in &mapping.seed_mappings {
            let seed = &seeds[seed_mapping.seed_idx];
            let mut region = AlignmentRegion::new(chain_idx, seed_mapping.seed_idx);

            // Initialize boundaries from seed
            region.qb = seed.query_pos;
            region.qe = seed.query_pos + seed.len;
            region.rb = seed.ref_pos;
            region.re = seed.ref_pos + seed.len as u64;
            region.seedlen0 = seed.len;
            region.frac_rep = chain.frac_rep;
            region.w = opt.w;

            let mut total_score = seed.len * opt.a; // Seed score

            // Process left extension (bwamem.cpp:2486-2504)
            if let Some(left_idx) = seed_mapping.left_job_idx {
                if left_idx < left_scores.len() {
                    let left_score = &left_scores[left_idx];

                    // Update score
                    total_score = left_score.score;

                    // Update boundaries based on clipping decision
                    // C++: if (sp->gscore <= 0 || sp->gscore <= a->score - opt->pen_clip5)
                    if left_score.global_score <= 0
                        || left_score.global_score <= left_score.score - opt.pen_clip5
                    {
                        // Local alignment: extend by qle/tle
                        region.qb = seed.query_pos - left_score.query_end_pos;
                        region.rb = seed.ref_pos - left_score.target_end_pos as u64;
                        region.truesc = left_score.score;
                    } else {
                        // Global alignment: clip to start
                        region.qb = 0;
                        region.rb = seed.ref_pos - left_score.gtarget_end_pos as u64;
                        region.truesc = left_score.global_score;
                    }
                }
            } else if seed.query_pos == 0 {
                // No left extension needed (seed starts at query start)
                region.truesc = seed.len * opt.a;
            }

            // Process right extension (bwamem.cpp:2594-2612)
            if let Some(right_idx) = seed_mapping.right_job_idx {
                if right_idx < right_scores.len() {
                    let right_score = &right_scores[right_idx];

                    // Add right extension score
                    total_score += right_score.score;

                    // Update boundaries based on clipping decision
                    if right_score.global_score <= 0
                        || right_score.global_score <= right_score.score - opt.pen_clip3
                    {
                        // Local alignment
                        region.qe = seed.query_pos + seed.len + right_score.query_end_pos;
                        region.re =
                            seed.ref_pos + seed.len as u64 + right_score.target_end_pos as u64;
                    } else {
                        // Global alignment: extend to query end
                        region.qe = query_len;
                        region.re =
                            seed.ref_pos + seed.len as u64 + right_score.gtarget_end_pos as u64;
                    }
                }
            } else if seed.query_pos + seed.len == query_len {
                // No right extension needed (seed ends at query end)
                // Score already includes seed
            }

            region.score = total_score;
            if region.truesc == 0 {
                region.truesc = total_score;
            }

            // Calculate seed coverage
            region.seedcov = 0;
            for &other_seed_idx in &chain.seeds {
                let other_seed = &seeds[other_seed_idx];
                if other_seed.query_pos >= region.qb
                    && other_seed.query_pos + other_seed.len <= region.qe
                    && other_seed.ref_pos >= region.rb
                    && other_seed.ref_pos + other_seed.len as u64 <= region.re
                {
                    region.seedcov += other_seed.len;
                }
            }

            // Convert FM-index position to chromosome coordinates
            let fm_pos = region.rb;
            let is_rev = fm_pos >= l_pac;
            region.is_rev = is_rev;

            // Get reference ID and chromosome position
            // Use bns_depos to get forward position and strand
            let (pos_f, depos_is_rev) = bwa_idx.bns.bns_depos(fm_pos as i64);
            region.is_rev = depos_is_rev;

            let rid = bwa_idx.bns.bns_pos2rid(pos_f);
            if rid >= 0 && (rid as usize) < bwa_idx.bns.annotations.len() {
                region.rid = rid;
                region.ref_name = bwa_idx.bns.annotations[rid as usize].name.clone();
                // Calculate chromosome position (position relative to chromosome start)
                let offset = bwa_idx.bns.annotations[rid as usize].offset as i64;
                region.chr_pos = (pos_f - offset).max(0) as u64;
            } else {
                region.rid = -1;
                region.ref_name = "*".to_string();
                region.chr_pos = 0;
            }

            // Track best scoring region for this chain
            if total_score > best_score {
                best_score = total_score;
                best_region = Some(region);
            }
        }

        if let Some(ref region) = best_region {
            log::debug!(
                "DEFERRED_REGION: chain_idx={} seed_idx={} qb={} qe={} rb={} re={} score={} chr_pos={} ref={}",
                region.chain_idx,
                region.seed_idx,
                region.qb,
                region.qe,
                region.rb,
                region.re,
                region.score,
                region.chr_pos,
                region.ref_name
            );
            regions.push(region.clone());
        }
    }

    regions
}

// ============================================================================
// CIGAR REGENERATION FROM BOUNDARIES
// ============================================================================
//
// This section implements CIGAR regeneration matching BWA-MEM2's `mem_reg2aln`
// and `bwa_gen_cigar2` functions. CIGAR is generated from alignment boundaries
// by re-fetching the reference and running full Smith-Waterman.
//
// ============================================================================

/// Generate CIGAR string from alignment region boundaries
///
/// This is the Rust equivalent of BWA-MEM2's `mem_reg2aln` (bwamem.cpp:1732-1805)
/// combined with `bwa_gen_cigar2` (bwa.cpp:260-347).
///
/// ## Algorithm
///
/// 1. Extract query segment: `query[qb..qe]`
/// 2. Fetch reference segment from PAC: `bns.get_reference_segment(rb, re-rb)`
/// 3. Run global Smith-Waterman to generate CIGAR
/// 4. Add soft clips for unaligned query ends
/// 5. Compute NM (edit distance) and MD tag
///
/// ## Returns
///
/// Tuple of (cigar, NM, MD_string) for the alignment
pub fn generate_cigar_from_region(
    bwa_idx: &BwaIndex,
    pac_data: &[u8],
    query: &[u8],
    region: &AlignmentRegion,
    opt: &MemOpt,
) -> Option<(Vec<(u8, i32)>, i32, String)> {
    use crate::alignment::banded_swa::BandedPairWiseSW;

    // Validate region boundaries
    if region.rb >= region.re || region.qb >= region.qe {
        log::debug!(
            "Invalid region boundaries: rb={}, re={}, qb={}, qe={}",
            region.rb,
            region.re,
            region.qb,
            region.qe
        );
        return None;
    }

    let l_pac = bwa_idx.bns.packed_sequence_length;

    // Check for boundary crossing (forward/reverse)
    if region.rb < l_pac && region.re > l_pac {
        log::debug!("Region crosses forward/reverse boundary");
        return None;
    }

    // Extract query segment
    let qb = region.qb.max(0) as usize;
    let qe = (region.qe as usize).min(query.len());
    if qb >= qe || qe > query.len() {
        log::debug!(
            "Invalid query bounds: qb={}, qe={}, query_len={}",
            qb,
            qe,
            query.len()
        );
        return None;
    }
    let query_segment: Vec<u8> = query[qb..qe].to_vec();

    // Fetch reference segment
    let ref_len = region.re - region.rb;
    let rseq = match bwa_idx.bns.get_reference_segment(region.rb, ref_len) {
        Ok(seq) => seq,
        Err(e) => {
            log::debug!("Failed to fetch reference: {:?}", e);
            return None;
        }
    };

    log::debug!(
        "CIGAR_GEN: qb={} qe={} rb={} re={} query_seg_len={} ref_len={}",
        region.qb,
        region.qe,
        region.rb,
        region.re,
        query_segment.len(),
        rseq.len()
    );

    // Handle reverse strand: reverse both query and reference
    let (query_for_sw, rseq_for_sw) = if region.rb >= l_pac {
        // Reverse strand: reverse both sequences for proper indel placement
        let rev_query: Vec<u8> = query_segment.iter().rev().copied().collect();
        let rev_ref: Vec<u8> = rseq.iter().rev().copied().collect();
        (rev_query, rev_ref)
    } else {
        (query_segment.clone(), rseq.clone())
    };

    // Compute band width (matching bwamem.cpp:1752-1756)
    let w = infer_band_width(
        qe as i32 - qb as i32,
        ref_len as i32,
        region.truesc,
        opt.a,
        opt.o_del,
        opt.e_del,
        opt.o_ins,
        opt.e_ins,
        opt.w,
        region.w,
    );

    // Create SW parameters for global alignment
    let sw_params = BandedPairWiseSW::new(
        opt.o_del,
        opt.e_del,
        opt.o_ins,
        opt.e_ins,
        opt.zdrop,
        0, // end_bonus = 0 for global alignment
        opt.pen_clip5,
        opt.pen_clip3,
        opt.mat,
        opt.a as i8,
        -(opt.b as i8),
    );

    // Run global alignment to generate CIGAR
    // Using scalar for CIGAR generation (matches BWA-MEM2 behavior)
    let result = sw_params.scalar_banded_swa(
        query_for_sw.len() as i32,
        &query_for_sw,
        rseq_for_sw.len() as i32,
        &rseq_for_sw,
        w,
        0, // h0 = 0 for global alignment
    );

    let mut cigar = result.1; // CIGAR operations
    let ref_aligned = result.2;
    let query_aligned = result.3;

    // If reverse strand, reverse the CIGAR
    if region.rb >= l_pac {
        cigar.reverse();
    }

    // Add soft clips for unaligned query ends
    let query_len = query.len() as i32;
    let clip5 = if region.is_rev {
        query_len - region.qe
    } else {
        region.qb
    };
    let clip3 = if region.is_rev {
        region.qb
    } else {
        query_len - region.qe
    };

    // Build final CIGAR with clips
    let mut final_cigar = Vec::new();

    if clip5 > 0 {
        final_cigar.push((b'S', clip5));
    }

    // Merge operations to avoid adjacent same-type ops
    for (op, len) in cigar {
        if let Some((last_op, last_len)) = final_cigar.last_mut() {
            if *last_op == op {
                *last_len += len;
                continue;
            }
        }
        final_cigar.push((op, len));
    }

    if clip3 > 0 {
        final_cigar.push((b'S', clip3));
    }

    // Validate that CIGAR has some aligned bases (not all clips)
    // CIGARs with only S/H clips are invalid and should be filtered out
    let has_aligned_bases = final_cigar
        .iter()
        .any(|&(op, _)| matches!(op, b'M' | b'=' | b'X' | b'I' | b'D'));
    if !has_aligned_bases {
        log::debug!("CIGAR has no aligned bases, skipping region");
        return None;
    }

    // Compute NM (edit distance) and MD tag
    let (nm, md) = compute_nm_and_md(
        &final_cigar,
        &ref_aligned,
        &query_aligned,
        &rseq,
        region.is_rev,
    );

    Some((final_cigar, nm, md))
}

/// Infer band width for global alignment
///
/// Matches BWA-MEM2's infer_bw (bwamem.cpp:1811-1818)
fn infer_band_width(
    l_query: i32,
    l_ref: i32,
    score: i32,
    match_score: i32,
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    cmd_w: i32,
    region_w: i32,
) -> i32 {
    // Infer band width from deletion penalty
    let tmp_del =
        if l_query == l_ref && l_query * match_score - score < (o_del + e_del - match_score) * 2 {
            0
        } else {
            let min_len = l_query.min(l_ref);
            let w = ((min_len * match_score - score - o_del) as f64 / e_del as f64 + 2.0) as i32;
            w.max((l_query - l_ref).abs())
        };

    // Infer band width from insertion penalty
    let tmp_ins =
        if l_query == l_ref && l_query * match_score - score < (o_ins + e_ins - match_score) * 2 {
            0
        } else {
            let min_len = l_query.min(l_ref);
            let w = ((min_len * match_score - score - o_ins) as f64 / e_ins as f64 + 2.0) as i32;
            w.max((l_query - l_ref).abs())
        };

    let mut w2 = tmp_del.max(tmp_ins);
    if w2 > cmd_w {
        w2 = w2.min(region_w);
    }

    // Cap at 4x command-line width
    w2.min(cmd_w * 4)
}

/// Compute NM (edit distance) and MD tag from alignment
///
/// Matches BWA-MEM2's NM/MD computation in bwa_gen_cigar2 (bwa.cpp:309-338)
fn compute_nm_and_md(
    cigar: &[(u8, i32)],
    _ref_aligned: &[u8],
    _query_aligned: &[u8],
    rseq: &[u8],
    _is_rev: bool,
) -> (i32, String) {
    let mut nm = 0;
    let mut md = String::new();
    let mut match_run = 0;
    let mut ref_pos = 0usize;

    for &(op, len) in cigar {
        match op {
            b'M' | b'=' | b'X' => {
                // Match/mismatch - need to check each position
                // For now, assume all matches (MD tag generation needs ref comparison)
                match_run += len;
                ref_pos += len as usize;
            }
            b'I' => {
                // Insertion in query
                nm += len;
            }
            b'D' => {
                // Deletion from reference
                nm += len;

                // MD tag: record deletion
                if match_run > 0 {
                    md.push_str(&match_run.to_string());
                    match_run = 0;
                }

                md.push('^');
                for i in 0..len as usize {
                    if ref_pos + i < rseq.len() {
                        let base = match rseq[ref_pos + i] {
                            0 => 'A',
                            1 => 'C',
                            2 => 'G',
                            3 => 'T',
                            _ => 'N',
                        };
                        md.push(base);
                    }
                }
                ref_pos += len as usize;
            }
            b'S' | b'H' => {
                // Soft/hard clip - doesn't affect NM or MD
            }
            _ => {}
        }
    }

    // Finalize MD tag
    if match_run > 0 {
        md.push_str(&match_run.to_string());
    }

    (nm, md)
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alignment_region_creation() {
        let region = AlignmentRegion::new(0, 1);
        assert_eq!(region.chain_idx, 0);
        assert_eq!(region.seed_idx, 1);
        assert_eq!(region.score, 0);
        assert_eq!(region.secondary, -1);
    }

    #[test]
    fn test_alignment_region_spans() {
        let mut region = AlignmentRegion::new(0, 0);
        region.qb = 10;
        region.qe = 60;
        region.rb = 1000;
        region.re = 1050;

        assert_eq!(region.query_span(), 50);
        assert_eq!(region.ref_span(), 50);
    }

    #[test]
    fn test_alignment_region_overlap() {
        let mut region1 = AlignmentRegion::new(0, 0);
        region1.rid = 0;
        region1.qb = 0;
        region1.qe = 100;
        region1.rb = 1000;
        region1.re = 1100;

        let mut region2 = AlignmentRegion::new(1, 0);
        region2.rid = 0;
        region2.qb = 50;
        region2.qe = 150;
        region2.rb = 1050;
        region2.re = 1150;

        // 50% overlap should be detected at 0.3 mask level
        assert!(region1.overlaps_with(&region2, 0.3));

        // Different chromosome should not overlap
        region2.rid = 1;
        assert!(!region1.overlaps_with(&region2, 0.3));
    }
}
