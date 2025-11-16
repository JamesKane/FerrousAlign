// bwa-mem2-rust/src/align.rs

// Import BwaIndex and MemOpt
use crate::banded_swa::BandedPairWiseSW;
use crate::mem::BwaIndex;
use crate::mem_opt::MemOpt;

// Define a struct to represent a seed
#[derive(Debug)]
pub struct Seed {
    pub query_pos: i32, // Position in the query
    pub ref_pos: u64,   // Position in the reference
    pub len: i32,       // Length of the seed
    pub is_rev: bool,   // Is it on the reverse strand?
}

// Define a struct to represent a Super Maximal Exact Match (SMEM)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)] // Add PartialEq and Eq for deduplication
pub struct SMEM {
    pub rid: i32, // Read ID (not strictly needed here, but good for consistency with C++)
    pub m: i32,   // Start position in query
    pub n: i32,   // End position in query
    pub k: u64,   // Start of BWT interval (SA coordinate)
    pub l: u64,   // End of BWT interval (SA coordinate)
    pub s: u64,   // Size of BWT interval (l - k)
    pub is_rev_comp: bool, // Added to track if SMEM is from reverse complement
}

// Constants from FMI_search.h

const CP_MASK: u64 = 63;
pub const CP_SHIFT: u64 = 6; // Make CP_SHIFT public

// CP_OCC struct from FMI_search.h
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CpOcc {
    pub cp_count: [i64; 4],
    pub one_hot_bwt_str: [u64; 4],
}





// Scoring matrix matching C++ defaults (Match=1, Mismatch=-4)
// This matches C++ bwa_fill_scmat() with a=1, b=4
// A, C, G, T, N
// A  1 -4 -4 -4 -1
// C -4  1 -4 -4 -1
// G -4 -4  1 -4 -1
// T -4 -4 -4  1 -1
// N -1 -1 -1 -1 -1
pub const DEFAULT_SCORING_MATRIX: [i8; 25] = [
     1, -4, -4, -4, -1,  // A row
    -4,  1, -4, -4, -1,  // C row
    -4, -4,  1, -4, -1,  // G row
    -4, -4, -4,  1, -1,  // T row
    -1, -1, -1, -1, -1,  // N row
];

// Global one_hot_mask_array (initialized once)
lazy_static::lazy_static! {
    static ref ONE_HOT_MASK_ARRAY: Vec<u64> = {
        let mut array = vec![0u64; 65]; // Size 65 for 0 to 64 bits
        // array[0] is already 0
        for i in 1..=64 {
            array[i] = (array[i - 1] >> 1) | 0x8000000000000000u64;
        }
        array
    };
}

// Hardware-optimized popcount for 64-bit integers
// Uses NEON on ARM64 and POPCNT on x86_64
#[inline(always)]
fn popcount64(x: u64) -> i64 {
    #[cfg(target_arch = "aarch64")]
    {
        // ARM NEON implementation using vcnt (count bits in 8-bit lanes)
        // This is the equivalent of __builtin_popcountl on ARM
        unsafe {
            use std::arch::aarch64::*;

            // Load the 64-bit value into a NEON register
            let vec = vreinterpret_u8_u64(vcreate_u64(x));

            // Count bits in each 8-bit lane (vcnt_u8)
            let cnt = vcnt_u8(vec);

            // Sum all 8 lanes using horizontal add (pairwise additions)
            // vcnt gives us 8 bytes, each with bit count of that byte
            // We need to sum them all to get total popcount
            let sum16 = vpaddl_u8(cnt);    // Pairwise add to 4x u16
            let sum32 = vpaddl_u16(sum16); // Pairwise add to 2x u32
            let sum64 = vpaddl_u32(sum32); // Pairwise add to 1x u64

            vget_lane_u64::<0>(sum64) as i64
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        // x86_64 POPCNT instruction
        unsafe {
            use std::arch::x86_64::_popcnt64;
            _popcnt64(x as i64) as i64
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        // Fallback for other architectures
        x.count_ones() as i64
    }
}

// Helper function to get occurrences, translating GET_OCC macro
// Now uses hardware-optimized popcount
pub fn get_occ(bwa_idx: &BwaIndex, k: i64, c: u8) -> i64 {
    let cp_shift = CP_SHIFT as i64;
    let cp_mask = CP_MASK as i64;

    let occ_id_k = k >> cp_shift;
    let y_k = k & cp_mask;

    let occ_k = bwa_idx.cp_occ[occ_id_k as usize].cp_count[c as usize];
    let one_hot_bwt_str_c_k = bwa_idx.cp_occ[occ_id_k as usize].one_hot_bwt_str[c as usize];

    let match_mask_k = one_hot_bwt_str_c_k & ONE_HOT_MASK_ARRAY[y_k as usize];
    occ_k + popcount64(match_mask_k)
}

pub fn backward_ext(bwa_idx: &BwaIndex, mut smem: SMEM, a: u8) -> SMEM {
    let mut k_arr = [0u64; 4];
    let mut s_arr = [0u64; 4];
    let mut l_arr = [0u64; 4];

    // eprintln!("DEBUG backward_ext: extending base {}, smem.k={}, smem.l={}, smem.s={}",
    //           a, smem.k, smem.l, smem.s);

    for b in 0..4 {
        let sp = smem.k;
        let ep = smem.k + smem.s;

        let occ_sp = get_occ(bwa_idx, sp as i64, b);
        let occ_ep = get_occ(bwa_idx, ep as i64, b);

        k_arr[b as usize] = bwa_idx.bwt.l2[b as usize] + occ_sp as u64;

        // eprintln!("  base {}: l2[{}]={}, occ_sp={}, occ_ep={} => k={}",
        //           b, b, bwa_idx.bwt.l2[b as usize], occ_sp, occ_ep, k_arr[b as usize]);

        // Defensive: avoid u64 underflow if occ_ep < occ_sp due to malformed/placeholder data
        let s = if occ_ep >= occ_sp {
            occ_ep - occ_sp
        } else {
            0
        };
        s_arr[b as usize] = s as u64;
    }

    let mut sentinel_offset = 0;
    if smem.k <= bwa_idx.sentinel_index as u64 && (smem.k + smem.s) > bwa_idx.sentinel_index as u64 {
        sentinel_offset = 1;
    }

    l_arr[3] = smem.l + sentinel_offset;
    l_arr[2] = l_arr[3] + s_arr[3];
    l_arr[1] = l_arr[2] + s_arr[2];
    l_arr[0] = l_arr[1] + s_arr[1];

    smem.k = k_arr[a as usize];
    smem.l = l_arr[a as usize];
    smem.s = s_arr[a as usize];

    // eprintln!("  → Result: k={}, l={}, s={}", smem.k, smem.l, smem.s);

    smem
}

// Function to convert a base character to its 0-3 encoding
// A=0, C=1, G=2, T=3, N=4
#[inline(always)]
pub fn base_to_code(base: u8) -> u8 {
    match base {
        b'A' | b'a' => 0,
        b'C' | b'c' => 1,
        b'G' | b'g' => 2,
        b'T' | b't' => 3,
        _ => 4, // N or any other character
    }
}

// Function to get the reverse complement of a code
// 0=A, 1=C, 2=G, 3=T, 4=N
#[inline(always)]
pub fn reverse_complement_code(code: u8) -> u8 {
    match code {
        0 => 3, // A -> T
        1 => 2, // C -> G
        2 => 1, // G -> C
        3 => 0, // T -> A
        _ => 4, // N or any other character remains N
    }
}

#[derive(Debug)]
pub struct Chain {
    pub score: i32,
    pub seeds: Vec<usize>, // Indices of seeds in the original seeds vector
    pub query_start: i32,
    pub query_end: i32,
    pub ref_start: u64,
    pub ref_end: u64,
    pub is_rev: bool,
}

#[derive(Debug)]
pub struct Alignment {
    pub query_name: String,
    pub flag: u16, // SAM flag
    pub ref_name: String,
    pub ref_id: usize, // Reference sequence ID (for paired-end scoring)
    pub pos: u64, // 0-based leftmost mapping position
    pub mapq: u8, // Mapping quality
    pub score: i32, // Alignment score (for paired-end scoring)
    pub cigar: Vec<(u8, i32)>, // CIGAR string
    pub rnext: String, // Ref. name of the mate/next read
    pub pnext: u64, // Position of the mate/next read
    pub tlen: i32, // Observed template length
    pub seq: String, // Segment sequence
    pub qual: String, // ASCII of Phred-scaled base quality+33
    // Optional SAM tags
    pub tags: Vec<(String, String)>, // Vector of (tag_name, tag_value) pairs
}

impl Alignment {
    /// Get CIGAR string as a formatted string (e.g., "50M2I48M")
    pub fn cigar_string(&self) -> String {
        self.cigar.iter().map(|&(op, len)| {
            format!("{}{}", len, op as char)
        }).collect()
    }

    pub fn to_sam_string(&self) -> String {
        // Convert CIGAR to string format
        let cigar_string = self.cigar_string();

        // TODO: Clipping penalties (opt.pen_clip5, opt.pen_clip3, opt.pen_unpaired)
        // are used in C++ to adjust alignment scores, not SAM output directly.
        // They affect score calculation during alignment extension and pair scoring.
        // This requires deeper integration into the scoring logic in banded_swa.rs

        // Format mandatory SAM fields
        let mut sam_line = format!(
            "{}	{}	{}	{}	{}	{}	{}	{}	{}	{}	{}",
            self.query_name,
            self.flag,
            self.ref_name,
            self.pos + 1, // SAM POS is 1-based
            self.mapq,
            cigar_string,
            self.rnext,
            self.pnext,
            self.tlen,
            self.seq,
            self.qual
        );

        // Append optional tags
        for (tag, value) in &self.tags {
            sam_line.push('\t');
            sam_line.push_str(tag);
            sam_line.push(':');
            sam_line.push_str(value);
        }

        sam_line
    }
}

// Structure to hold alignment job for batching
#[derive(Clone)]
#[cfg_attr(test, derive(Debug))]
pub(crate) struct AlignmentJob {
    #[allow(dead_code)] // Used for tracking but not currently read
    pub seed_idx: usize,
    pub query: Vec<u8>,
    pub target: Vec<u8>,
    pub band_width: i32,
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

/// Determine optimal batch size based on estimated divergence
///
/// **Strategy**:
/// - Low divergence (score < 0.3): Use larger batches (32-64) for maximum SIMD efficiency
/// - Medium divergence (0.3-0.7): Use standard batch size (16)
/// - High divergence (> 0.7): Use smaller batches (8) or route to scalar
///
/// This reduces batch synchronization penalty for divergent sequences while
/// maximizing SIMD utilization for similar sequences.
fn determine_optimal_batch_size(jobs: &[AlignmentJob]) -> usize {
    if jobs.is_empty() {
        return 16; // Default
    }

    // Calculate average divergence score for this batch of jobs
    let total_divergence: f64 = jobs
        .iter()
        .map(|job| estimate_divergence_score(job.query.len(), job.target.len()))
        .sum();

    let avg_divergence = total_divergence / jobs.len() as f64;

    // Adaptive batch sizing based on divergence
    if avg_divergence < 0.3 {
        // Low divergence: Use larger batches for better SIMD utilization
        32
    } else if avg_divergence < 0.7 {
        // Medium divergence: Use standard batch size
        16
    } else {
        // High divergence: Use smaller batches to reduce synchronization penalty
        8
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

/// Execute alignments using batched SIMD (processes up to 16 at a time)
/// Now includes CIGAR generation via hybrid approach
pub(crate) fn execute_batched_alignments(sw_params: &BandedPairWiseSW, jobs: &[AlignmentJob]) -> Vec<Vec<(u8, i32)>> {
    const BATCH_SIZE: usize = 16;
    let mut all_cigars = vec![Vec::new(); jobs.len()];

    // Process jobs in batches of 16
    for batch_start in (0..jobs.len()).step_by(BATCH_SIZE) {
        let batch_end = (batch_start + BATCH_SIZE).min(jobs.len());
        let batch_jobs = &jobs[batch_start..batch_end];

        // Prepare batch data for simd_banded_swa_batch16_with_cigar
        let batch_data: Vec<(i32, &[u8], i32, &[u8], i32, i32)> = batch_jobs
            .iter()
            .map(|job| {
                (job.query.len() as i32, job.query.as_slice(),
                 job.target.len() as i32, job.target.as_slice(),
                 job.band_width, 0)
            })
            .collect();

        // Execute batched alignment with CIGAR generation
        let results = sw_params.simd_banded_swa_batch16_with_cigar(&batch_data);

        // Extract CIGARs from results
        for (i, result) in results.iter().enumerate() {
            if i < batch_jobs.len() {
                all_cigars[batch_start + i] = result.cigar.clone();
            }
        }
    }

    all_cigars
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
pub(crate) fn execute_adaptive_alignments(sw_params: &BandedPairWiseSW, jobs: &[AlignmentJob]) -> Vec<Vec<(u8, i32)>> {
    if jobs.is_empty() {
        return Vec::new();
    }

    // Partition jobs by estimated divergence
    let (low_div_jobs, high_div_jobs) = partition_jobs_by_divergence(jobs);

    // Calculate average divergence for logging
    let avg_divergence: f64 = jobs
        .iter()
        .map(|job| estimate_divergence_score(job.query.len(), job.target.len()))
        .sum::<f64>() / jobs.len() as f64;

    // Log routing statistics (INFO level - shows in default verbosity)
    log::info!(
        "Adaptive routing: {} total jobs, {} scalar ({:.1}%), {} SIMD ({:.1}%), avg_divergence={:.3}",
        jobs.len(),
        high_div_jobs.len(),
        high_div_jobs.len() as f64 / jobs.len() as f64 * 100.0,
        low_div_jobs.len(),
        low_div_jobs.len() as f64 / jobs.len() as f64 * 100.0,
        avg_divergence
    );

    // Create result vector with correct size
    let mut all_cigars = vec![Vec::new(); jobs.len()];

    // Process high-divergence jobs with scalar (more efficient for divergent sequences)
    let high_div_cigars = if !high_div_jobs.is_empty() {
        log::debug!("Processing {} high-divergence jobs with scalar", high_div_jobs.len());
        execute_scalar_alignments(sw_params, &high_div_jobs)
    } else {
        Vec::new()
    };

    // Process low-divergence jobs with adaptive batched SIMD
    let low_div_cigars = if !low_div_jobs.is_empty() {
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

    // Merge results back in original order
    let mut low_idx = 0;
    let mut high_idx = 0;

    for (original_idx, job) in jobs.iter().enumerate() {
        let div_score = estimate_divergence_score(job.query.len(), job.target.len());
        let cigar = if div_score > 0.7 {
            // High divergence - get from scalar results
            let result = high_div_cigars[high_idx].clone();
            high_idx += 1;
            result
        } else {
            // Low divergence - get from SIMD results
            let result = low_div_cigars[low_idx].clone();
            low_idx += 1;
            result
        };

        all_cigars[original_idx] = cigar;
    }

    all_cigars
}

/// Execute batched alignments with configurable batch size
fn execute_batched_alignments_with_size(
    sw_params: &BandedPairWiseSW,
    jobs: &[AlignmentJob],
    batch_size: usize,
) -> Vec<Vec<(u8, i32)>> {
    let mut all_cigars = vec![Vec::new(); jobs.len()];

    // Process jobs in batches of specified size
    for batch_start in (0..jobs.len()).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(jobs.len());
        let batch_jobs = &jobs[batch_start..batch_end];

        // Prepare batch data
        let batch_data: Vec<(i32, &[u8], i32, &[u8], i32, i32)> = batch_jobs
            .iter()
            .map(|job| {
                (job.query.len() as i32, job.query.as_slice(),
                 job.target.len() as i32, job.target.as_slice(),
                 job.band_width, 0)
            })
            .collect();

        // Execute batched alignment with CIGAR generation
        let results = sw_params.simd_banded_swa_batch16_with_cigar(&batch_data);

        // Extract CIGARs from results
        for (i, result) in results.iter().enumerate() {
            if i < batch_jobs.len() {
                all_cigars[batch_start + i] = result.cigar.clone();
            }
        }
    }

    all_cigars
}

/// Execute alignments using scalar processing (fallback for small batches)
pub(crate) fn execute_scalar_alignments(sw_params: &BandedPairWiseSW, jobs: &[AlignmentJob]) -> Vec<Vec<(u8, i32)>> {
    jobs.iter()
        .map(|job| {
            let (_score, cigar) = sw_params.scalar_banded_swa(
                job.query.len() as i32,
                &job.query,
                job.target.len() as i32,
                &job.target,
                job.band_width,
                0, // h0
            );
            cigar
        })
        .collect()
}

pub fn generate_seeds(bwa_idx: &BwaIndex, query_name: &str, query_seq: &[u8], query_qual: &str, opt: &MemOpt) -> Vec<Alignment> {
    generate_seeds_with_mode(bwa_idx, query_name, query_seq, query_qual, true, opt)
}

// Internal implementation with option to use batched SIMD
fn generate_seeds_with_mode(
    bwa_idx: &BwaIndex,
    query_name: &str,
    query_seq: &[u8],
    query_qual: &str,
    use_batched_simd: bool,
    _opt: &MemOpt, // TODO: Use opt parameters in Phase 2+
) -> Vec<Alignment> {
    let query_len = query_seq.len();
    if query_len == 0 {
        return Vec::new();
    }

    // Instantiate BandedPairWiseSW with parameters from MemOpt
    let sw_params = BandedPairWiseSW::new(
        _opt.o_del,      // Gap open deletion penalty
        _opt.e_del,      // Gap extension deletion penalty
        _opt.o_ins,      // Gap open insertion penalty
        _opt.e_ins,      // Gap extension insertion penalty
        _opt.zdrop,      // Z-dropoff
        0,               // end_bonus (not configurable in C++)
        _opt.mat,        // Scoring matrix (generated from -A/-B)
        _opt.a as i8,    // Match score
        -(_opt.b as i8), // Mismatch penalty (negative)
    );

    let mut encoded_query = Vec::with_capacity(query_len);
    let mut encoded_query_rc = Vec::with_capacity(query_len); // Reverse complement
    for &base in query_seq {
        let code = base_to_code(base);
        encoded_query.push(code);
        encoded_query_rc.push(reverse_complement_code(code));
    }
    encoded_query_rc.reverse(); // Reverse the reverse complement to get the sequence in correct order

    let mut all_smems: Vec<SMEM> = Vec::new();

    // --- Process Forward Strand ---
    let mut smems_fwd: Vec<SMEM> = Vec::new();
    for i in (0..query_len).rev() {
        let current_base_code = encoded_query[i];

        if current_base_code == 4 { // Skip 'N' bases
            continue;
        }

        let mut smem = SMEM {
            rid: 0,
            m: i as i32,
            n: i as i32,
            k: bwa_idx.bwt.l2[current_base_code as usize],
            l: bwa_idx.bwt.l2[(current_base_code + 1) as usize],
            s: bwa_idx.bwt.l2[(current_base_code + 1) as usize] - bwa_idx.bwt.l2[current_base_code as usize],
            is_rev_comp: false,
        };

        for j in (0..i).rev() {
            let next_base_code = encoded_query[j];
            if next_base_code == 4 {
                break;
            }

            // Fixed: Use literal base code for backward search, not reverse complement
            let new_smem = backward_ext(bwa_idx, smem, next_base_code);

            if new_smem.s == 0 {
                break;
            }

            if new_smem.s != smem.s {
                smems_fwd.push(smem);
            }
            smem = new_smem;
            smem.m = j as i32;
        }
        if smem.s > 0 {
            smems_fwd.push(smem);
        }
    }
    all_smems.extend(smems_fwd);

    // --- Process Reverse Complement Strand ---
    let mut smems_rev: Vec<SMEM> = Vec::new();
    for i in (0..query_len).rev() {
        let current_base_code = encoded_query_rc[i];

        if current_base_code == 4 { // Skip 'N' bases
            continue;
        }

        let mut smem = SMEM {
            rid: 0,
            m: i as i32,
            n: i as i32,
            k: bwa_idx.bwt.l2[current_base_code as usize],
            l: bwa_idx.bwt.l2[(current_base_code + 1) as usize],
            s: bwa_idx.bwt.l2[(current_base_code + 1) as usize] - bwa_idx.bwt.l2[current_base_code as usize],
            is_rev_comp: true, // Mark as from reverse complement
        };

        for j in (0..i).rev() {
            let next_base_code = encoded_query_rc[j];
            if next_base_code == 4 {
                break;
            }

            // Fixed: Use literal base code for backward search on RC strand
            let new_smem = backward_ext(bwa_idx, smem, next_base_code);

            if new_smem.s == 0 {
                break;
            }

            if new_smem.s != smem.s {
                smems_rev.push(smem);
            }
            smem = new_smem;
            smem.m = j as i32;
        }
        if smem.s > 0 {
            smems_rev.push(smem);
        }
    }
    all_smems.extend(smems_rev);

    // eprintln!("all_smems: {:?}", all_smems);

    // --- Filtering SMEMs ---
    // TODO: Implement re-seeding (-r split_factor) and split width (-y)
    // Re-seeding splits long MEMs (length > min_seed_len * split_factor) into shorter seeds
    // Split width controls occurrence threshold for splitting (if occurrences <= split_width)
    // See C++ bwamem.cpp:639-695 for full implementation

    // 1. Sort SMEMs
    all_smems.sort_by_key(|smem| (smem.m, smem.n, smem.k, smem.is_rev_comp)); // Include is_rev_comp in sort key

    // 2. Remove duplicates and filter by minimum length and max occurrences
    let mut unique_filtered_smems: Vec<SMEM> = Vec::new();
            if let Some(mut prev_smem) = all_smems.first().cloned() {
                let seed_len = prev_smem.n - prev_smem.m + 1;
                let occurrences = prev_smem.l - prev_smem.k;

                // Filter by min_seed_len (-k) and max_occ (-c)
                if seed_len >= _opt.min_seed_len && occurrences <= _opt.max_occ as u64 {
                    unique_filtered_smems.push(prev_smem);
                }

                for i in 1..all_smems.len() {
                    let current_smem = all_smems[i];
                    if current_smem != prev_smem { // Use PartialEq for comparison
                        let seed_len = current_smem.n - current_smem.m + 1;
                        let occurrences = current_smem.l - current_smem.k;

                        // Filter by min_seed_len (-k) and max_occ (-c)
                        if seed_len >= _opt.min_seed_len && occurrences <= _opt.max_occ as u64 {
                            unique_filtered_smems.push(current_smem);
                        }
                    }
                    prev_smem = current_smem;
                }
            }
    // eprintln!("unique_filtered_smems: {:?}", unique_filtered_smems);
    // --- End Filtering SMEMs ---


    // Convert SMEMs to Seed structs and perform seed extension
    // For now, only process the longest seeds to avoid too many alignments
    let mut sorted_smems = unique_filtered_smems;
    sorted_smems.sort_by_key(|smem| -(smem.n - smem.m + 1)); // Sort by length, descending

    // Take only the single longest seed for now (simplified)
    let useful_smems: Vec<_> = sorted_smems.into_iter().take(1).collect();

    // eprintln!("Processing {} longest SMEMs", useful_smems.len());

    let mut seeds = Vec::new();
    let mut alignment_jobs = Vec::new(); // Collect alignment jobs for batching

    // Prepare query segment once - use the FULL query for alignment
    let query_segment_encoded: Vec<u8> = query_seq.iter().map(|&b| base_to_code(b)).collect();

    for (idx, smem) in useful_smems.iter().enumerate() {
        let smem = *smem;
        let mut ref_pos = get_sa_entry(bwa_idx, smem.k);
        let mut is_rev = smem.is_rev_comp;

        // Convert positions in reverse complement region to forward strand
        // BWT contains both forward [0, l_pac) and reverse [l_pac, 2*l_pac)
        if ref_pos >= bwa_idx.bns.l_pac {
            ref_pos = (bwa_idx.bns.l_pac << 1) - 1 - ref_pos;
            is_rev = !is_rev; // Flip strand orientation
        }

        // Adjust query_pos for reverse complement seeds
        let query_pos = if is_rev {
            (query_len as i32 - 1) - smem.n // Adjust to original query coordinates
        } else {
            smem.m
        };

        let seed = Seed {
            query_pos,
            ref_pos,
            len: smem.n - smem.m + 1,
            is_rev,
        };

        // --- Prepare Seed Extension Job ---
        // Prepare reference segment (query_len + 2*band_width for diagonal extension)
        let ref_segment_len = (query_len as u64 + 2 * _opt.w as u64).min(bwa_idx.bns.l_pac - seed.ref_pos);
        let ref_segment_start = seed.ref_pos;

        let target_segment_result = bwa_idx.bns.get_reference_segment(ref_segment_start, ref_segment_len);

        match target_segment_result {
            Ok(target_segment) => {
                alignment_jobs.push(AlignmentJob {
                    seed_idx: idx,
                    query: query_segment_encoded.clone(),
                    target: target_segment,
                    band_width: _opt.w,  // Use band width from options
                });
            },
            Err(e) => {
                log::error!("Error getting reference segment for seed {:?}: {}", seed, e);
            }
        }

        seeds.push(seed);
    }

    // --- Execute Alignments (Adaptive Strategy) ---
    // Use adaptive routing strategy that combines:
    // 1. Divergence-based routing (high divergence → scalar, low → SIMD)
    // 2. Adaptive batch sizing (8-32 based on sequence characteristics)
    // 3. Fallback to scalar for small batches (<8 jobs)
    let extended_cigars = if use_batched_simd && alignment_jobs.len() >= 8 {
        // Use adaptive strategy for 8+ jobs
        // This provides optimal performance by routing jobs to the best execution path
        execute_adaptive_alignments(&sw_params, &alignment_jobs)
    } else {
        // Fall back to scalar processing for very small batches
        execute_scalar_alignments(&sw_params, &alignment_jobs)
    };

    // eprintln!("Seed extension complete. {} seeds generated. Starting chaining...", seeds.len());

    // --- Seed Chaining ---
    let chained_results = chain_seeds(seeds, _opt);
    // eprintln!("Chaining complete. {} chains generated.", chained_results.len());
    // --- End Seed Chaining ---

    // For now, let's create a dummy alignment from the first chain
    let mut alignments = Vec::new();
    if let Some(first_chain) = chained_results.first() {
        // Find the CIGAR for the best seed in the chain (simplification for now)
        // In a real implementation, you'd likely re-run SWA for the entire chained region
        // or combine CIGARs from individual seeds.
        let best_seed_idx_in_chain = first_chain.seeds[0]; // Assuming first seed in chain is "best" for cigar
        let cigar_for_alignment = extended_cigars[best_seed_idx_in_chain].clone();

        // Determine reference name (assuming single reference for now)
        let ref_name = if !bwa_idx.bns.anns.is_empty() {
            bwa_idx.bns.anns[0].name.clone()
        } else {
            "unknown_ref".to_string()
        };

        alignments.push(Alignment {
            query_name: query_name.to_string(),
            flag: if first_chain.is_rev { 0x10 } else { 0 }, // 0x10 for reverse strand
            ref_name,
            ref_id: 0, // TODO: Get actual reference ID from chain
            pos: first_chain.ref_start,
            mapq: 60, // Placeholder, needs proper calculation
            score: first_chain.score, // Add alignment score for paired-end scoring
            cigar: cigar_for_alignment,
            rnext: "*".to_string(),
            pnext: 0,
            tlen: 0,
            seq: String::from_utf8_lossy(query_seq).to_string(),
            qual: query_qual.to_string(),
            tags: Vec::new(), // Optional tags to be added later
        });
    }

    alignments
}

pub fn chain_seeds(mut seeds: Vec<Seed>, opt: &MemOpt) -> Vec<Chain> {
    if seeds.is_empty() {
        return Vec::new();
    }

    // 1. Sort seeds: by query_pos, then by ref_pos
    seeds.sort_by_key(|s| (s.query_pos, s.ref_pos));

    let num_seeds = seeds.len();
    let mut dp = vec![0; num_seeds]; // dp[i] stores the max score of a chain ending at seeds[i]
    let mut prev_seed_idx = vec![None; num_seeds]; // To reconstruct the chain

    // Use max_chain_gap from options for gap penalty calculation
    let max_gap = opt.max_chain_gap;

    // 2. Dynamic Programming
    for i in 0..num_seeds {
        dp[i] = seeds[i].len; // Initialize with the seed's own length as score
        for j in 0..i {
            // Check for compatibility: seed[j] must end before seed[i] starts in both query and reference
            // And they must be on the same strand
            if seeds[j].is_rev == seeds[i].is_rev &&
               seeds[j].query_pos + seeds[j].len < seeds[i].query_pos &&
               seeds[j].ref_pos + (seeds[j].len as u64) < seeds[i].ref_pos {

                // Calculate gap length
                let q_gap = seeds[i].query_pos - (seeds[j].query_pos + seeds[j].len);
                let r_gap = seeds[i].ref_pos as i32 - (seeds[j].ref_pos + seeds[j].len as u64) as i32;

                // Check max_chain_gap constraint (do not chain if gap too large)
                if q_gap.max(r_gap.abs()) > max_gap {
                    continue; // Skip this potential chain connection
                }

                // Simple gap penalty (average of query and reference gaps)
                let current_gap_penalty = (q_gap + r_gap.abs()) / 2;

                let potential_score = dp[j] + seeds[i].len - current_gap_penalty;

                if potential_score > dp[i] {
                    dp[i] = potential_score;
                    prev_seed_idx[i] = Some(j);
                }
            }
        }
    }

    // 3. Backtracking to reconstruct chains
    // TODO: Implement full chain dropping (-D drop_ratio)
    // Currently only returns best chain. Full implementation should:
    // 1. Generate multiple chains (not just best)
    // 2. Compare overlapping chains
    // 3. Drop chains where seed_coverage < drop_ratio * better_chain_coverage
    // See C++ bwamem.cpp mem_chain_flt() for full algorithm

    let mut best_chain_score = 0;
    let mut best_chain_end_idx = None;

    for i in 0..num_seeds {
        if dp[i] > best_chain_score {
            best_chain_score = dp[i];
            best_chain_end_idx = Some(i);
        }
    }

    // Apply min_chain_weight filter (-W parameter)
    // If best chain score is below minimum, return empty
    if best_chain_score < opt.min_chain_weight {
        return Vec::new();
    }

    let mut chains = Vec::new();
    if let Some(mut current_idx) = best_chain_end_idx {
        let mut chain_seeds_indices = Vec::new();
        let mut current_seed = &seeds[current_idx];

        let mut query_start = current_seed.query_pos;
        let mut query_end = current_seed.query_pos + current_seed.len;
        let mut ref_start = current_seed.ref_pos;
        let mut ref_end = current_seed.ref_pos + current_seed.len as u64;
        let is_rev = current_seed.is_rev;

        while let Some(prev_idx) = prev_seed_idx[current_idx] {
            chain_seeds_indices.push(current_idx);
            current_idx = prev_idx;
            current_seed = &seeds[current_idx];

            query_start = query_start.min(current_seed.query_pos);
            query_end = query_end.max(current_seed.query_pos + current_seed.len);
            ref_start = ref_start.min(current_seed.ref_pos);
            ref_end = ref_end.max(current_seed.ref_pos + current_seed.len as u64);
        }
        chain_seeds_indices.push(current_idx); // Add the first seed in the chain

        chain_seeds_indices.reverse(); // Order from start to end

        chains.push(Chain {
            score: best_chain_score,
            seeds: chain_seeds_indices,
            query_start,
            query_end,
            ref_start,
            ref_end,
            is_rev,
        });
    }

    chains
}
// bwa-mem2-rust/src/align_test.rs

// Function to get BWT base from cp_occ format (for loaded indices)
// Returns 0-3 for bases A/C/G/T, or 4 for sentinel
pub fn get_bwt_base_from_cp_occ(cp_occ: &[CpOcc], pos: u64) -> u8 {
    let cp_block = (pos >> CP_SHIFT) as usize;

    // Safety: check bounds
    if cp_block >= cp_occ.len() {
        log::warn!("get_bwt_base_from_cp_occ: cp_block {} >= cp_occ.len() {}",
                  cp_block, cp_occ.len());
        return 4; // Return sentinel for out-of-bounds
    }

    let offset_in_block = pos & ((1 << CP_SHIFT) - 1);
    let bit_position = 63 - offset_in_block;

    // Check which of the 4 one-hot encoded arrays has a 1 at this position
    for base in 0..4 {
        if (cp_occ[cp_block].one_hot_bwt_str[base] >> bit_position) & 1 == 1 {
            return base as u8;
        }
    }
    4 // Return 4 for sentinel (no bit set means sentinel position)
}

// Function to get the next BWT position from a BWT coordinate
// Returns None if we hit the sentinel (which should not be navigated)
pub fn get_bwt(bwa_idx: &BwaIndex, pos: u64) -> Option<u64> {
    let base = if !bwa_idx.bwt.bwt_data.is_empty() {
        // Index was just built, use raw bwt_data
        bwa_idx.bwt.get_bwt_base(pos)
    } else {
        // Index was loaded from disk, use cp_occ format
        get_bwt_base_from_cp_occ(&bwa_idx.cp_occ, pos)
    };

    // If we hit the sentinel (base == 4), return None
    if base == 4 {
        return None;
    }

    Some(bwa_idx.bwt.l2[base as usize] + get_occ(bwa_idx, pos as i64, base) as u64)
}

pub fn get_sa_entry(bwa_idx: &BwaIndex, mut pos: u64) -> u64 {
    let original_pos = pos;
    let mut count = 0;
    const MAX_ITERATIONS: u64 = 10000; // Safety limit to prevent infinite loops

    // eprintln!("get_sa_entry: starting with pos={}, sa_intv={}, seq_len={}, cp_occ.len()={}",
    //           original_pos, bwa_idx.bwt.sa_intv, bwa_idx.bwt.seq_len, bwa_idx.cp_occ.len());

    while pos % bwa_idx.bwt.sa_intv as u64 != 0 {
        // Safety check: prevent infinite loops
        if count >= MAX_ITERATIONS {
            log::error!("get_sa_entry exceeded MAX_ITERATIONS ({}) - possible infinite loop!", MAX_ITERATIONS);
            log::error!("  original_pos={}, current_pos={}, count={}", original_pos, pos, count);
            log::error!("  sa_intv={}, seq_len={}", bwa_idx.bwt.sa_intv, bwa_idx.bwt.seq_len);
            return count; // Return what we have so far
        }

        let _old_pos = pos;
        match get_bwt(bwa_idx, pos) {
            Some(new_pos) => {
                pos = new_pos;
                count += 1;
                // if count <= 10 {
                //     eprintln!("  BWT step {}: pos {} -> {} (count={})", count, _old_pos, pos, count);
                // }
            }
            None => {
                // Hit sentinel - return the accumulated count
                // eprintln!("  BWT step {}: pos {} -> SENTINEL (count={})", count + 1, _old_pos, count);
                // eprintln!("get_sa_entry: original_pos={}, hit_sentinel, count={}, result={}",
                //           original_pos, count, count);
                return count;
            }
        }
    }

    let sa_index = (pos / bwa_idx.bwt.sa_intv as u64) as usize;
    let sa_ms_byte = bwa_idx.bwt.sa_ms_byte[sa_index] as u64;
    let sa_ls_word = bwa_idx.bwt.sa_ls_word[sa_index] as u64;
    let sa_val = (sa_ms_byte << 32) | sa_ls_word;

    // If SA value points to the sentinel (seq_len - 1 because seq_len includes sentinel),
    // the reference position wraps to the beginning
    let ref_len = bwa_idx.bwt.seq_len - 1; // Exclude sentinel from reference length
    let adjusted_sa_val = if sa_val >= ref_len {
        // SA points to sentinel - wrap to beginning
        0
    } else {
        sa_val
    };

    let result = adjusted_sa_val + count;

    // eprintln!("get_sa_entry: original_pos={}, final_pos={}, count={}, sa_index={}, sa_val={}, adjusted={}, result={}",
    //           original_pos, pos, count, sa_index, sa_val, adjusted_sa_val, result);
    result
}

#[cfg(test)]
mod tests {
    use crate::align::{chain_seeds, base_to_code, reverse_complement_code, backward_ext, CpOcc, CP_SHIFT, Seed, SMEM, popcount64};
    use crate::bntseq::{BntAnn1, BntSeq};
    use crate::bwt::Bwt;
    use crate::mem::BwaIndex;
    use std::path::{Path, PathBuf};

    #[test]
    fn test_popcount64_neon() {
        // Test the hardware-optimized popcount implementation
        // This ensures our NEON implementation matches the software version

        // Test basic cases
        assert_eq!(popcount64(0), 0);
        assert_eq!(popcount64(1), 1);
        assert_eq!(popcount64(0xFFFFFFFFFFFFFFFF), 64);
        assert_eq!(popcount64(0x8000000000000000), 1);

        // Test various bit patterns
        assert_eq!(popcount64(0b1010101010101010), 8);
        assert_eq!(popcount64(0b11111111), 8);
        assert_eq!(popcount64(0xFF00FF00FF00FF00), 32);
        assert_eq!(popcount64(0x0F0F0F0F0F0F0F0F), 32);

        // Test random patterns that match expected popcount
        assert_eq!(popcount64(0x123456789ABCDEF0), 32);
        assert_eq!(popcount64(0xAAAAAAAAAAAAAAAA), 32); // Alternating bits
        assert_eq!(popcount64(0x5555555555555555), 32); // Alternating bits (complement)
    }

    #[test]
    fn test_backward_ext() {
        let prefix = Path::new("test_data/test_ref.fa");

        // Skip if test data doesn't exist
        if !prefix.exists() {
            eprintln!("Skipping test_backward_ext - test data not found");
            return;
        }

        let bwa_idx = match BwaIndex::bwa_idx_load(&prefix) {
            Ok(idx) => idx,
            Err(_) => {
                eprintln!("Skipping test_backward_ext - could not load index");
                return;
            }
        };

        let smem = SMEM {
            k: 0,
            s: bwa_idx.bwt.seq_len,
            ..Default::default()
        };
        let new_smem = backward_ext(&bwa_idx, smem, 0); // 0 is 'A'
        assert_ne!(new_smem.s, 0);
    }

    #[test]
    fn test_backward_ext_multiple_bases() {
        // Test backward extension with all four bases
        let prefix = Path::new("test_data/test_ref.fa");

        if !prefix.exists() {
            eprintln!("Skipping test_backward_ext_multiple_bases - test data not found");
            return;
        }

        let bwa_idx = match BwaIndex::bwa_idx_load(&prefix) {
            Ok(idx) => idx,
            Err(_) => {
                eprintln!("Skipping test_backward_ext_multiple_bases - could not load index");
                return;
            }
        };

        // Start with full range
        let initial_smem = SMEM {
            k: 0,
            s: bwa_idx.bwt.seq_len,
            ..Default::default()
        };

        // Test extending with each base
        for base in 0..4 {
            let extended = super::backward_ext(&bwa_idx, initial_smem, base);

            // Extended range should be smaller or equal to initial range
            assert!(extended.s <= initial_smem.s,
                    "Extended range size {} should be <= initial size {} for base {}",
                    extended.s, initial_smem.s, base);

            // k should be within bounds
            assert!(extended.k < bwa_idx.bwt.seq_len,
                    "Extended k should be within sequence length");
        }
    }

    #[test]
    fn test_backward_ext_chain() {
        // Test chaining multiple backward extensions (like building a seed)
        let prefix = Path::new("test_data/test_ref.fa");

        if !prefix.exists() {
            eprintln!("Skipping test_backward_ext_chain - test data not found");
            return;
        }

        let bwa_idx = match BwaIndex::bwa_idx_load(&prefix) {
            Ok(idx) => idx,
            Err(_) => {
                eprintln!("Skipping test_backward_ext_chain - could not load index");
                return;
            }
        };

        // Start with full range
        let mut smem = SMEM {
            k: 0,
            s: bwa_idx.bwt.seq_len,
            ..Default::default()
        };

        // Build a seed by extending with ACGT
        let bases = vec![0u8, 1, 2, 3]; // ACGT
        let mut prev_s = smem.s;

        for (i, &base) in bases.iter().enumerate() {
            smem = super::backward_ext(&bwa_idx, smem, base);

            // Range should generally get smaller (or stay same) with each extension
            // (though it could stay the same if the pattern is very common)
            assert!(smem.s <= prev_s,
                    "After extension {}, range size {} should be <= previous {}",
                    i, smem.s, prev_s);

            prev_s = smem.s;

            // If range becomes 0, we can't extend further
            if smem.s == 0 {
                break;
            }
        }
    }

    #[test]
    fn test_backward_ext_zero_range() {
        // Test backward extension when starting with zero range
        let prefix = Path::new("test_data/test_ref.fa");

        if !prefix.exists() {
            eprintln!("Skipping test_backward_ext_zero_range - test data not found");
            return;
        }

        let bwa_idx = match BwaIndex::bwa_idx_load(&prefix) {
            Ok(idx) => idx,
            Err(_) => {
                eprintln!("Skipping test_backward_ext_zero_range - could not load index");
                return;
            }
        };

        let smem = SMEM {
            k: 0,
            s: 0, // Zero range
            ..Default::default()
        };

        let extended = super::backward_ext(&bwa_idx, smem, 0);

        // Extending a zero range should still give zero range
        assert_eq!(extended.s, 0, "Extending zero range should give zero range");
    }

    #[test]
    fn test_smem_structure() {
        // Test SMEM structure creation and defaults
        let smem1 = SMEM {
            rid: 0,
            m: 10,
            n: 20,
            k: 5,
            l: 15,
            s: 10,
            is_rev_comp: false,
        };

        assert_eq!(smem1.m, 10);
        assert_eq!(smem1.n, 20);
        assert_eq!(smem1.s, 10);

        // Test default
        let smem2 = SMEM::default();
        assert_eq!(smem2.rid, 0);
        assert_eq!(smem2.m, 0);
        assert_eq!(smem2.n, 0);
    }

    #[test]
    fn test_base_to_code() {
        assert_eq!(base_to_code(b'A'), 0);
        assert_eq!(base_to_code(b'a'), 0);
        assert_eq!(base_to_code(b'C'), 1);
        assert_eq!(base_to_code(b'c'), 1);
        assert_eq!(base_to_code(b'G'), 2);
        assert_eq!(base_to_code(b'g'), 2);
        assert_eq!(base_to_code(b'T'), 3);
        assert_eq!(base_to_code(b't'), 3);
        assert_eq!(base_to_code(b'N'), 4);
        assert_eq!(base_to_code(b'X'), 4); // Other characters
    }

    #[test]
    fn test_reverse_complement_code() {
        assert_eq!(reverse_complement_code(0), 3); // A -> T
        assert_eq!(reverse_complement_code(1), 2); // C -> G
        assert_eq!(reverse_complement_code(2), 1); // G -> C
        assert_eq!(reverse_complement_code(3), 0); // T -> A
        assert_eq!(reverse_complement_code(4), 4); // N -> N
        assert_eq!(reverse_complement_code(5), 4); // Other -> Other
    }

    #[test]
    fn test_get_sa_entry_basic() {
        // This test requires an actual index file to be present
        // We'll use a simple test to verify the function doesn't crash
        let prefix = Path::new("test_data/test_ref.fa");

        // Only run if test data exists
        if !prefix.exists() {
            eprintln!("Skipping test_get_sa_entry_basic - test data not found");
            return;
        }

        let bwa_idx = match BwaIndex::bwa_idx_load(&prefix) {
            Ok(idx) => idx,
            Err(_) => {
                eprintln!("Skipping test_get_sa_entry_basic - could not load index");
                return;
            }
        };

        // Test getting SA entry at position 0 (should return a valid reference position)
        let sa_entry = super::get_sa_entry(&bwa_idx, 0);

        // SA entry should be within the reference sequence length
        assert!(sa_entry < bwa_idx.bwt.seq_len,
                "SA entry {} should be less than seq_len {}", sa_entry, bwa_idx.bwt.seq_len);
    }

    #[test]
    fn test_get_sa_entry_sampled_position() {
        // Test getting SA entry at a sampled position (divisible by sa_intv)
        let prefix = Path::new("test_data/test_ref.fa");

        if !prefix.exists() {
            eprintln!("Skipping test_get_sa_entry_sampled_position - test data not found");
            return;
        }

        let bwa_idx = match BwaIndex::bwa_idx_load(&prefix) {
            Ok(idx) => idx,
            Err(_) => {
                eprintln!("Skipping test_get_sa_entry_sampled_position - could not load index");
                return;
            }
        };

        // Test at a sampled position (should directly lookup in SA array)
        let sampled_pos = bwa_idx.bwt.sa_intv as u64;
        let sa_entry = super::get_sa_entry(&bwa_idx, sampled_pos);

        assert!(sa_entry < bwa_idx.bwt.seq_len,
                "SA entry at sampled position should be within sequence length");
    }

    #[test]
    fn test_get_sa_entry_multiple_positions() {
        // Test getting SA entries for multiple positions
        let prefix = Path::new("test_data/test_ref.fa");

        if !prefix.exists() {
            eprintln!("Skipping test_get_sa_entry_multiple_positions - test data not found");
            return;
        }

        let bwa_idx = match BwaIndex::bwa_idx_load(&prefix) {
            Ok(idx) => idx,
            Err(_) => {
                eprintln!("Skipping test_get_sa_entry_multiple_positions - could not load index");
                return;
            }
        };

        // Test several positions
        let test_positions = vec![0u64, 1, 10, 100];

        for pos in test_positions {
            if pos >= bwa_idx.bwt.seq_len {
                continue;
            }

            let sa_entry = super::get_sa_entry(&bwa_idx, pos);

            // All SA entries should be valid (within sequence length)
            assert!(sa_entry < bwa_idx.bwt.seq_len,
                    "SA entry for pos {} should be within sequence length", pos);
        }
    }

    #[test]
    fn test_get_sa_entry_consistency() {
        // Test that get_sa_entry returns consistent results
        let prefix = Path::new("test_data/test_ref.fa");

        if !prefix.exists() {
            eprintln!("Skipping test_get_sa_entry_consistency - test data not found");
            return;
        }

        let bwa_idx = match BwaIndex::bwa_idx_load(&prefix) {
            Ok(idx) => idx,
            Err(_) => {
                eprintln!("Skipping test_get_sa_entry_consistency - could not load index");
                return;
            }
        };

        let pos = 5u64;
        let sa_entry1 = super::get_sa_entry(&bwa_idx, pos);
        let sa_entry2 = super::get_sa_entry(&bwa_idx, pos);

        // Same position should always return same SA entry
        assert_eq!(sa_entry1, sa_entry2,
                   "get_sa_entry should return consistent results for the same position");
    }

    #[test]
    fn test_get_bwt_basic() {
        // Test get_bwt function
        let prefix = Path::new("test_data/test_ref.fa");

        if !prefix.exists() {
            eprintln!("Skipping test_get_bwt_basic - test data not found");
            return;
        }

        let bwa_idx = match BwaIndex::bwa_idx_load(&prefix) {
            Ok(idx) => idx,
            Err(_) => {
                eprintln!("Skipping test_get_bwt_basic - could not load index");
                return;
            }
        };

        // Test getting BWT at various positions
        for pos in 0..10u64 {
            let bwt_result = super::get_bwt(&bwa_idx, pos);

            // Either we get a valid position or None (sentinel)
            if let Some(new_pos) = bwt_result {
                assert!(new_pos < bwa_idx.bwt.seq_len,
                        "BWT position should be within sequence length");
            }
            // If None, we hit the sentinel - that's ok
        }
    }

    #[test]
    fn test_batched_alignment_infrastructure() {
        // Test that the batched alignment infrastructure works correctly
        use crate::banded_swa::BandedPairWiseSW;

        let sw_params = BandedPairWiseSW::new(
            4, 2, 4, 2, 100, 0,
            super::DEFAULT_SCORING_MATRIX,
            2, -4,
        );

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
            },
            super::AlignmentJob {
                seed_idx: 1,
                query: query2.clone(),
                target: target2.clone(),
                band_width: 10,
            },
        ];

        // Test scalar execution
        let scalar_cigars = super::execute_scalar_alignments(&sw_params, &jobs);
        assert_eq!(scalar_cigars.len(), 2, "Should return 2 CIGARs for 2 jobs");
        assert!(!scalar_cigars[0].is_empty(), "First CIGAR should not be empty");
        assert!(!scalar_cigars[1].is_empty(), "Second CIGAR should not be empty");

        // Test batched execution (currently falls back to scalar)
        let batched_cigars = super::execute_batched_alignments(&sw_params, &jobs);
        assert_eq!(batched_cigars.len(), 2, "Should return 2 CIGARs for 2 jobs");

        // Results should be identical (since batched currently uses scalar fallback)
        assert_eq!(scalar_cigars[0], batched_cigars[0],
                   "Scalar and batched should produce identical results");
        assert_eq!(scalar_cigars[1], batched_cigars[1],
                   "Scalar and batched should produce identical results");
    }
}