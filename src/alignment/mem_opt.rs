use clap::Args;
use std::path::PathBuf;

// bwa-mem2-rust/src/mem_opt.rs
//
// Alignment options structure matching C++ mem_opt_t from bwamem.h

/// Alignment options matching C++ mem_opt_t structure
/// See: src/bwamem.h:129-166 in C++ bwa-mem2
#[derive(Debug, Clone)]
pub struct MemOpt {
    // Scoring parameters
    pub a: i32,            // Match score
    pub b: i32,            // Mismatch penalty
    pub o_del: i32,        // Gap open penalty (deletions)
    pub e_del: i32,        // Gap extension penalty (deletions)
    pub o_ins: i32,        // Gap open penalty (insertions)
    pub e_ins: i32,        // Gap extension penalty (insertions)
    pub pen_unpaired: i32, // Phred-scaled penalty for unpaired reads
    pub pen_clip5: i32,    // 5' clipping penalty
    pub pen_clip3: i32,    // 3' clipping penalty

    // Alignment parameters
    pub w: i32,     // Band width for banded alignment
    pub zdrop: i32, // Z-dropoff (off-diagonal X-dropoff)

    // Seeding parameters
    pub max_mem_intv: u64, // Maximum MEM interval
    pub min_seed_len: i32, // Minimum seed length
    pub split_factor: f32, // Split into a seed if MEM is longer than min_seed_len*split_factor
    pub split_width: i32,  // Split into a seed if its occurrence is smaller than this value
    pub max_occ: i32,      // Skip a seed if its occurrence is larger than this value

    // Chaining parameters
    pub min_chain_weight: i32, // Minimum chain weight
    pub max_chain_extend: i32, // Maximum chain extension
    pub max_chain_gap: i32,    // Do not chain seed if it is max_chain_gap-bp away from closest seed

    // Filtering parameters
    pub mask_level: f32, // Regard hit as redundant if overlap with better hit is over mask_level * min length
    pub drop_ratio: f32, // Drop chain if seed coverage is below drop_ratio * seed coverage of better chain
    pub xa_drop_ratio: f32, // When counting hits for XA tag, ignore alignments with score < xa_drop_ratio * max_score
    pub mask_level_redun: f32, // Mask level for redundant hits

    // Output parameters
    pub t: i32,               // Minimum score threshold to output
    pub max_xa_hits: i32,     // Maximum XA hits to output
    pub max_xa_hits_alt: i32, // Maximum XA hits for ALT contigs

    // Paired-end parameters
    pub max_ins: i32, // Maximum insert size (when estimating distribution, skip pairs with insert > this)
    pub max_matesw: i32, // Perform maximally max_matesw rounds of mate-SW for each end

    // Processing parameters
    pub n_threads: i32,  // Number of threads
    pub chunk_size: i64, // Process chunk_size-bp sequences in a batch

    // Mapping quality parameters
    pub mapq_coef_len: f32, // Coefficient length for mapQ calculation
    pub mapq_coef_fac: i32, // Coefficient factor for mapQ calculation (log of mapq_coef_len)

    // Flags (using bitfield in C++, separate bools in Rust for clarity)
    pub flag: i32, // Bitfield for various flags (kept for compatibility)

    // Scoring matrix (5x5 for A,C,G,T,N)
    pub mat: [i8; 25], // Scoring matrix; mat[0] == 0 if unset

    // Output formatting options (Phase 6)
    pub read_group: Option<String>, // Read group header line (@RG\tID:foo\tSM:bar)
    pub header_lines: Vec<String>,  // Additional header lines to insert

    // Advanced options (Phase 7)
    pub insert_size_override: Option<InsertSizeOverride>, // Manual insert size specification
    pub verbosity: i32, // Verbosity level (1=error, 2=warning, 3=message, 4+=debug)

    // Advanced flags
    pub smart_pairing: bool,          // -p: Smart pairing (ignoring in2.fq)
    pub treat_alt_as_primary: bool,   // -j: Treat ALT contigs as part of primary assembly
    pub smallest_coord_primary: bool, // -5: For split alignment, take smallest coordinate as primary
    pub output_all_alignments: bool,  // -a: Output all alignments for SE or unpaired PE

    // Experimental: Deferred CIGAR architecture (Session 40)
    // When enabled, uses score-only SIMD extension followed by CIGAR regeneration
    // for survivors only (~10-20%), reducing CIGAR computation by 80-90%
    pub deferred_cigar: bool,
}

/// Manual insert size specification (overrides auto-inference)
#[derive(Debug, Clone)]
pub struct InsertSizeOverride {
    pub mean: f64,   // Mean insert size
    pub stddev: f64, // Standard deviation (default: 10% of mean)
    pub max: i32,    // Maximum insert size (default: mean + 4*stddev)
    pub min: i32,    // Minimum insert size (default: 0)
}

// ============================================================================
// STAGE-SPECIFIC PARAMETER BUNDLES
// ============================================================================

/// Parameters for the seeding stage (SMEM generation)
#[derive(Debug, Clone)]
pub struct SeedingParams {
    pub min_seed_len: i32,
    pub split_factor: f32,
    pub split_width: i32,
    pub max_occ: i32,
    pub max_mem_intv: u64,
}

/// Parameters for the chaining stage (seed grouping)
#[derive(Debug, Clone)]
pub struct ChainingParams {
    pub band_width: i32,
    pub max_chain_gap: i32,
    pub drop_ratio: f32,
    pub min_chain_weight: i32,
}

/// Parameters for the extension stage (Smith-Waterman alignment)
#[derive(Debug, Clone)]
pub struct ExtensionParams {
    pub band_width: i32,
    pub zdrop: i32,
    pub match_score: i32,
    pub mismatch_penalty: i32,
    pub gap_open_del: i32,
    pub gap_extend_del: i32,
    pub gap_open_ins: i32,
    pub gap_extend_ins: i32,
}

/// Parameters for the output/filtering stage
#[derive(Debug, Clone)]
pub struct OutputParams {
    pub score_threshold: i32,
    pub max_xa_hits: i32,
    pub max_xa_hits_alt: i32,
    pub xa_drop_ratio: f32,
    pub mask_level: f32,
    pub output_all_alignments: bool,
}

#[derive(Debug, Clone, Args)]
pub struct MemCliOptions {
    /// Index prefix (built with 'index' command)
    #[arg(value_name = "INDEX")]
    pub index: PathBuf,

    /// Input FASTQ file(s) - single file for single-end, two files for paired-end
    #[arg(value_name = "READS.FQ", required = true)]
    pub reads: Vec<PathBuf>,

    // ===== Algorithm Options =====
    /// Minimum seed length
    #[arg(short = 'k', long, value_name = "INT", default_value_t = 19)]
    pub min_seed_len: i32,

    /// Band width for banded alignment
    #[arg(short = 'w', long, value_name = "INT", default_value_t = 100)]
    pub band_width: i32,

    /// Off-diagonal X-dropoff
    #[arg(short = 'z', long, value_name = "INT", default_value_t = 100)]
    pub off_diagonal_dropoff: i32,

    /// Look for internal seeds inside a seed longer than {-k} * FLOAT
    #[arg(short = 'r', long, value_name = "FLOAT", default_value_t = 1.5)]
    pub reseed_factor: f32,

    /// Seed occurrence for the 3rd round seeding
    #[arg(short = 'y', long, value_name = "INT", default_value_t = 20)]
    pub seed_occurrence_3rd: u64,

    /// Skip seeds with more than INT occurrences
    #[arg(short = 'X', long, value_name = "INT", default_value_t = 500)]
    pub max_occurrences: i32,

    /// Drop chains shorter than FLOAT fraction of the longest overlapping chain
    #[arg(short = 'D', long, value_name = "FLOAT", default_value_t = 0.50)]
    pub drop_chain_fraction: f32,

    /// Discard a chain if seeded bases shorter than INT
    #[arg(short = 'm', long, value_name = "INT", default_value_t = 0)]
    pub min_chain_weight: i32,

    /// Perform at most INT rounds of mate rescues for each read
    #[arg(short = 'r', long, value_name = "INT", default_value_t = 50)]
    pub max_mate_rescues: i32,

    /// Skip mate rescue
    #[arg(long)]
    pub skip_mate_rescue: bool,

    /// Skip pairing; mate rescue performed unless -S also in use
    #[arg(long)]
    pub skip_pairing: bool,

    // ===== Scoring Options =====
    /// Score for a sequence match, which scales options -TdBOELU unless overridden
    #[arg(short = 'A', long, value_name = "INT", default_value_t = 1)]
    pub match_score: i32,

    /// Penalty for a mismatch
    #[arg(short = 'B', long, value_name = "INT", default_value_t = 4)]
    pub mismatch_penalty: i32,

    /// Gap open penalties for deletions and insertions [6,6]
    #[arg(short = 'O', long, value_name = "INT[,INT]", default_value = "6")]
    pub gap_open: String,

    /// Gap extension penalty; a gap of size k cost '{-O} + {-E}*k' [1,1]
    #[arg(short = 'E', long, value_name = "INT[,INT]", default_value = "1")]
    pub gap_extend: String,

    /// Penalty for 5'- and 3'-end clipping [5,5]
    #[arg(short = 'L', long, value_name = "INT[,INT]", default_value = "5")]
    pub clipping_penalty: String,

    /// Penalty for an unpaired read pair
    #[arg(short = 'U', long, value_name = "INT", default_value_t = 17)]
    pub unpaired_penalty: i32,

    // ===== I/O Options =====
    /// Output SAM file (default: stdout)
    #[arg(short = 'o', long, value_name = "FILE")]
    pub output: Option<PathBuf>,

    /// Read group header line such as '@RG\tID:foo\tSM:bar'
    #[arg(short = 'R', long, value_name = "STR")]
    pub read_group: Option<String>,

    /// Insert STR to header if it starts with @; or insert lines in FILE
    #[arg(short = 'H', long, value_name = "FILE|STR")]
    pub header: Option<String>,

    /// Treat ALT contigs as part of the primary assembly (i.e. ignore <idxbase>.alt file)
    #[arg(short = 'j', long)]
    pub treat_alt_as_primary: bool,

    /// For split alignment, take the alignment with the smallest coordinate as primary
    #[arg(short = '5', long)]
    pub smallest_coord_primary: bool,

    /// Don't modify mapQ of supplementary alignments
    #[arg(long)]
    pub no_modify_mapq: bool,

    /// Process INT input bases in each batch regardless of nThreads (for reproducibility)
    #[arg(short = 'K', long, value_name = "INT")]
    pub chunk_size: Option<i64>,

    /// Verbose level: 1=error, 2=warning, 3=message, 4=debug, 5+=trace
    #[arg(short = 'v', long, value_name = "INT", default_value_t = 3)]
    pub verbosity: u8,

    /// Minimum score to output
    #[arg(short = 'T', long, value_name = "INT", default_value_t = 30)]
    pub min_score: i32,

    /// If there are <INT hits with score >80% of the max score, output all in XA [5,200]
    /// Note: bwa-mem2 uses -h, but we use --max-xa-hits to avoid conflict with --help
    #[arg(short = 'x', long, value_name = "INT[,INT]", default_value = "5")]
    pub max_xa_hits: String,

    /// Output all alignments for SE or unpaired PE
    #[arg(short = 'a', long)]
    pub output_all: bool,

    /// Append FASTA/FASTQ comment to SAM output
    #[arg(short = 'C', long)]
    pub append_comment: bool,

    /// Output the reference FASTA header in the XR tag
    #[arg(long)]
    pub output_ref_header: bool,

    /// Use soft clipping for supplementary alignments
    #[arg(long)]
    pub soft_clip_supplementary: bool,

    /// Mark shorter split hits as secondary
    #[arg(long)]
    pub mark_secondary: bool,

    /// Smart pairing (ignoring in2.fq)
    #[arg(short = 'P', long)]
    pub smart_pairing: bool,

    /// Specify the mean, standard deviation (10% of the mean if absent), max
    /// (4 sigma from the mean if absent) and min of the insert size distribution.
    /// FR orientation only. [inferred]
    #[arg(short = 'I', long, value_name = "FLOAT[,FLOAT[,INT[,INT]]]")]
    pub insert_size: Option<String>,

    // ===== Processing Options =====
    /// Number of threads (default: all available cores)
    #[arg(short = 't', long, value_name = "INT")]
    pub threads: Option<usize>,

    // ===== Pipeline Options =====
    /// Use original CIGAR pipeline (legacy mode)
    /// By default, the deferred CIGAR pipeline is used (generates CIGARs only for
    /// high-scoring alignments, reducing computation by 80-90% and matching BWA-MEM2)
    #[arg(long)]
    pub standard_cigar: bool,
}

/// Parse XA hits string "INT" or "INT,INT"
pub fn parse_xa_hits(s: &str) -> Result<(i32, i32), String> {
    let parts: Vec<&str> = s.split(',').collect();
    match parts.len() {
        1 => {
            let val = parts[0]
                .parse::<i32>()
                .map_err(|_| format!("Invalid XA hits value: {}", s))?;
            Ok((val, 200)) // Default alt hits to 200
        }
        2 => {
            let primary = parts[0]
                .parse::<i32>()
                .map_err(|_| format!("Invalid primary XA hits: {}", parts[0]))?;
            let alt = parts[1]
                .parse::<i32>()
                .map_err(|_| format!("Invalid alt XA hits: {}", parts[1]))?;
            Ok((primary, alt))
        }
        _ => Err(format!("XA hits must be INT or INT,INT: {}", s)),
    }
}

impl Default for MemOpt {
    /// Create MemOpt with default values matching C++ mem_opt_init()
    /// See: src/bwamem.cpp mem_opt_init() function
    fn default() -> Self {
        let mut opt = MemOpt {
            // Scoring (matching C++ defaults)
            a: 1,
            b: 4,
            o_del: 6,
            e_del: 1,
            o_ins: 6,
            e_ins: 1,
            pen_unpaired: 17,
            pen_clip5: 5,
            pen_clip3: 5,

            // Alignment
            w: 100,
            zdrop: 100,

            // Seeding
            max_mem_intv: 20,
            min_seed_len: 19,
            split_factor: 1.5,
            split_width: 10,
            max_occ: 500,

            // Chaining
            min_chain_weight: 0,
            max_chain_extend: 50, // Limit chains to extend (was 1<<30 which caused memory explosion)
            max_chain_gap: 10000,

            // Filtering
            mask_level: 0.50,
            drop_ratio: 0.50,
            xa_drop_ratio: 0.80,
            mask_level_redun: 0.95,

            // Output
            t: 30,
            max_xa_hits: 5,
            max_xa_hits_alt: 200,

            // Paired-end
            max_ins: 10000,
            max_matesw: 50,

            // Processing
            n_threads: 1,
            chunk_size: 10_000_000,

            // Mapping quality
            mapq_coef_len: 50.0,
            mapq_coef_fac: 0, // Will be set to log(mapq_coef_len)

            // Flags
            flag: 0,

            // Scoring matrix (initialized to zeros, will be filled)
            mat: [0; 25],

            // Output formatting
            read_group: None,
            header_lines: Vec::new(),

            // Advanced options
            insert_size_override: None,
            verbosity: 3, // Default: message level

            // Advanced flags
            smart_pairing: false,
            treat_alt_as_primary: false,
            smallest_coord_primary: false,
            output_all_alignments: false, // Default: only output primary alignments (matching bwa-mem2)

            // Pipeline architecture (Session 46)
            deferred_cigar: true, // Default: enabled, use deferred CIGAR pipeline (matches BWA-MEM2)
        };

        // Calculate mapq_coef_fac as log of mapq_coef_len (matching C++)
        opt.mapq_coef_fac = opt.mapq_coef_len.ln() as i32;

        // Fill scoring matrix using match/mismatch scores
        opt.fill_scoring_matrix();

        opt
    }
}

impl MemOpt {
    // ========================================================================
    // STAGE-SPECIFIC PARAMETER ACCESSORS
    // ========================================================================

    /// Get seeding-stage parameters as a bundle
    pub fn seeding_params(&self) -> SeedingParams {
        SeedingParams {
            min_seed_len: self.min_seed_len,
            split_factor: self.split_factor,
            split_width: self.split_width,
            max_occ: self.max_occ,
            max_mem_intv: self.max_mem_intv,
        }
    }

    /// Get chaining-stage parameters as a bundle
    pub fn chaining_params(&self) -> ChainingParams {
        ChainingParams {
            band_width: self.w,
            max_chain_gap: self.max_chain_gap,
            drop_ratio: self.drop_ratio,
            min_chain_weight: self.min_chain_weight,
        }
    }

    /// Get extension-stage parameters as a bundle
    pub fn extension_params(&self) -> ExtensionParams {
        ExtensionParams {
            band_width: self.w,
            zdrop: self.zdrop,
            match_score: self.a,
            mismatch_penalty: self.b,
            gap_open_del: self.o_del,
            gap_extend_del: self.e_del,
            gap_open_ins: self.o_ins,
            gap_extend_ins: self.e_ins,
        }
    }

    /// Get output/filtering-stage parameters as a bundle
    pub fn output_params(&self) -> OutputParams {
        OutputParams {
            score_threshold: self.t,
            max_xa_hits: self.max_xa_hits,
            max_xa_hits_alt: self.max_xa_hits_alt,
            xa_drop_ratio: self.xa_drop_ratio,
            mask_level: self.mask_level,
            output_all_alignments: self.output_all_alignments,
        }
    }

    /// Validate parameters for consistency across stages
    /// Returns Ok(()) if valid, or Err with description of issues
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        // Seeding validation
        if self.min_seed_len < 1 {
            errors.push(format!(
                "min_seed_len must be >= 1, got {}",
                self.min_seed_len
            ));
        }
        if self.split_factor <= 1.0 {
            errors.push(format!(
                "split_factor must be > 1.0, got {}",
                self.split_factor
            ));
        }
        if self.max_occ < 1 {
            errors.push(format!("max_occ must be >= 1, got {}", self.max_occ));
        }

        // Chaining validation
        if self.w < 1 {
            errors.push(format!("band_width must be >= 1, got {}", self.w));
        }
        if self.max_chain_gap < 1 {
            errors.push(format!(
                "max_chain_gap must be >= 1, got {}",
                self.max_chain_gap
            ));
        }
        if !(0.0..=1.0).contains(&self.drop_ratio) {
            errors.push(format!(
                "drop_ratio must be in [0, 1], got {}",
                self.drop_ratio
            ));
        }

        // Scoring validation
        if self.a < 1 {
            errors.push(format!("match_score must be >= 1, got {}", self.a));
        }
        if self.b < 1 {
            errors.push(format!("mismatch_penalty must be >= 1, got {}", self.b));
        }

        // Output validation
        if self.t < 0 {
            errors.push(format!("score_threshold must be >= 0, got {}", self.t));
        }
        if !(0.0..=1.0).contains(&self.xa_drop_ratio) {
            errors.push(format!(
                "xa_drop_ratio must be in [0, 1], got {}",
                self.xa_drop_ratio
            ));
        }
        if !(0.0..=1.0).contains(&self.mask_level) {
            errors.push(format!(
                "mask_level must be in [0, 1], got {}",
                self.mask_level
            ));
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    // ========================================================================
    // SCORING MATRIX
    // ========================================================================

    /// Fill the scoring matrix based on match score (a) and mismatch penalty (b)
    /// Matches C++ bwa_fill_scmat() function
    pub fn fill_scoring_matrix(&mut self) {
        // Matrix is 5x5 for bases A=0, C=1, G=2, T=3, N=4
        // mat[i*5 + j] is the score for aligning base i with base j

        for i in 0..4 {
            for j in 0..4 {
                if i == j {
                    // Match
                    self.mat[i * 5 + j] = self.a as i8;
                } else {
                    // Mismatch (penalty, so negative)
                    self.mat[i * 5 + j] = -(self.b as i8);
                }
            }
            // N (ambiguous base) always gets a small penalty
            self.mat[i * 5 + 4] = -1;
            self.mat[4 * 5 + i] = -1;
        }
        // N vs N
        self.mat[4 * 5 + 4] = -1;
    }

    /// Update scoring matrix when match/mismatch scores change
    pub fn update_scoring(&mut self, match_score: i32, mismatch_penalty: i32) {
        self.a = match_score;
        self.b = mismatch_penalty;
        self.fill_scoring_matrix();

        // Note: In C++, -A flag scales other options unless overridden
        // We don't implement auto-scaling here; users must set options explicitly
    }

    /// Parse gap penalties from comma-separated string (e.g., "6,6" or "6")
    /// Returns (del_penalty, ins_penalty)
    pub fn parse_gap_penalties(s: &str) -> Result<(i32, i32), String> {
        let parts: Vec<&str> = s.split(',').collect();
        match parts.len() {
            1 => {
                let val = parts[0]
                    .parse::<i32>()
                    .map_err(|_| format!("Invalid gap penalty: {}", s))?;
                Ok((val, val))
            }
            2 => {
                let del = parts[0]
                    .parse::<i32>()
                    .map_err(|_| format!("Invalid deletion penalty: {}", parts[0]))?;
                let ins = parts[1]
                    .parse::<i32>()
                    .map_err(|_| format!("Invalid insertion penalty: {}", parts[1]))?;
                Ok((del, ins))
            }
            _ => Err(format!("Gap penalty must be INT or INT,INT: {}", s)),
        }
    }

    /// Parse clipping penalties from comma-separated string (e.g., "5,5" or "5")
    /// Returns (clip5, clip3)
    pub fn parse_clip_penalties(s: &str) -> Result<(i32, i32), String> {
        let parts: Vec<&str> = s.split(',').collect();
        match parts.len() {
            1 => {
                let val = parts[0]
                    .parse::<i32>()
                    .map_err(|_| format!("Invalid clipping penalty: {}", s))?;
                Ok((val, val))
            }
            2 => {
                let clip5 = parts[0]
                    .parse::<i32>()
                    .map_err(|_| format!("Invalid 5' clipping penalty: {}", parts[0]))?;
                let clip3 = parts[1]
                    .parse::<i32>()
                    .map_err(|_| format!("Invalid 3' clipping penalty: {}", parts[1]))?;
                Ok((clip5, clip3))
            }
            _ => Err(format!("Clipping penalty must be INT or INT,INT: {}", s)),
        }
    }

    /// Extract read group ID from read group header line
    /// Example input: "@RG\tID:foo\tSM:bar" -> "foo"
    pub fn extract_rg_id(rg_line: &str) -> Option<String> {
        // Read group line should start with @RG or just have tab-separated fields
        let stripped = rg_line.strip_prefix("@RG\t").unwrap_or(rg_line);

        // Find ID: field
        for field in stripped.split('\t') {
            if let Some(id) = field.strip_prefix("ID:") {
                return Some(id.to_string());
            }
        }

        None
    }

    /// Parse header lines from string or file
    /// If input starts with '@', treat as header line(s)
    /// Otherwise, treat as file path and read lines from file
    pub fn parse_header_input(input: &str) -> Result<Vec<String>, String> {
        use std::fs::File;
        use std::io::{BufRead, BufReader};

        if input.starts_with('@') {
            // Direct header line(s) - could be multiple separated by \n
            Ok(input.split('\n').map(|s| s.to_string()).collect())
        } else {
            // File path - read lines from file
            let file = File::open(input)
                .map_err(|e| format!("Failed to open header file {}: {}", input, e))?;
            let reader = BufReader::new(file);

            let mut lines = Vec::new();
            for line in reader.lines() {
                let line = line.map_err(|e| format!("Failed to read header file: {}", e))?;
                if !line.is_empty() {
                    lines.push(line);
                }
            }
            Ok(lines)
        }
    }

    /// Parse insert size specification from string
    /// Format: "mean[,stddev[,max[,min]]]"
    /// Example: "500" or "500,50" or "500,50,800,200"
    pub fn parse_insert_size(s: &str) -> Result<InsertSizeOverride, String> {
        let parts: Vec<&str> = s.split(',').collect();

        if parts.is_empty() || parts.len() > 4 {
            return Err(format!(
                "Insert size must be FLOAT[,FLOAT[,INT[,INT]]]: {}",
                s
            ));
        }

        // Parse mean (required)
        let mean = parts[0]
            .parse::<f64>()
            .map_err(|_| format!("Invalid insert size mean: {}", parts[0]))?;

        if mean <= 0.0 {
            return Err(format!("Insert size mean must be positive: {}", mean));
        }

        // Parse stddev (default: 10% of mean)
        let stddev = if parts.len() > 1 {
            parts[1]
                .parse::<f64>()
                .map_err(|_| format!("Invalid insert size stddev: {}", parts[1]))?
        } else {
            mean * 0.1
        };

        // Parse max (default: mean + 4*stddev)
        let max = if parts.len() > 2 {
            parts[2]
                .parse::<i32>()
                .map_err(|_| format!("Invalid insert size max: {}", parts[2]))?
        } else {
            (mean + 4.0 * stddev) as i32
        };

        // Parse min (default: 0)
        let min = if parts.len() > 3 {
            parts[3]
                .parse::<i32>()
                .map_err(|_| format!("Invalid insert size min: {}", parts[3]))?
        } else {
            0
        };

        Ok(InsertSizeOverride {
            mean,
            stddev,
            max,
            min,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_values_match_cpp() {
        let opt = MemOpt::default();

        // Verify key defaults match C++ mem_opt_init()
        assert_eq!(opt.a, 1, "Match score should be 1");
        assert_eq!(opt.b, 4, "Mismatch penalty should be 4");
        assert_eq!(opt.o_del, 6, "Gap open (del) should be 6");
        assert_eq!(opt.o_ins, 6, "Gap open (ins) should be 6");
        assert_eq!(opt.e_del, 1, "Gap extend (del) should be 1");
        assert_eq!(opt.e_ins, 1, "Gap extend (ins) should be 1");
        assert_eq!(opt.w, 100, "Band width should be 100");
        assert_eq!(opt.zdrop, 100, "Z-dropoff should be 100");
        assert_eq!(opt.min_seed_len, 19, "Min seed length should be 19");
        assert_eq!(opt.max_occ, 500, "Max occurrences should be 500");
        assert_eq!(opt.split_factor, 1.5, "Split factor should be 1.5");
        assert_eq!(opt.t, 30, "Score threshold should be 30");
        assert_eq!(opt.max_matesw, 50, "Max mate-SW rounds should be 50");
        assert_eq!(opt.pen_unpaired, 17, "Unpaired penalty should be 17");
        assert_eq!(opt.pen_clip5, 5, "5' clip penalty should be 5");
        assert_eq!(opt.pen_clip3, 5, "3' clip penalty should be 5");
    }

    #[test]
    fn test_scoring_matrix() {
        let opt = MemOpt::default();

        // Check diagonal (matches) - should be +1
        assert_eq!(opt.mat[0 * 5 + 0], 1, "A-A match");
        assert_eq!(opt.mat[1 * 5 + 1], 1, "C-C match");
        assert_eq!(opt.mat[2 * 5 + 2], 1, "G-G match");
        assert_eq!(opt.mat[3 * 5 + 3], 1, "T-T match");

        // Check mismatches - should be -4
        assert_eq!(opt.mat[0 * 5 + 1], -4, "A-C mismatch");
        assert_eq!(opt.mat[0 * 5 + 2], -4, "A-G mismatch");
        assert_eq!(opt.mat[1 * 5 + 3], -4, "C-T mismatch");

        // Check N (ambiguous) - should be -1
        assert_eq!(opt.mat[0 * 5 + 4], -1, "A-N");
        assert_eq!(opt.mat[4 * 5 + 0], -1, "N-A");
        assert_eq!(opt.mat[4 * 5 + 4], -1, "N-N");
    }

    #[test]
    fn test_update_scoring() {
        let mut opt = MemOpt::default();
        opt.update_scoring(2, 6);

        assert_eq!(opt.a, 2);
        assert_eq!(opt.b, 6);
        assert_eq!(opt.mat[0 * 5 + 0], 2, "Match should be 2");
        assert_eq!(opt.mat[0 * 5 + 1], -6, "Mismatch should be -6");
    }

    #[test]
    fn test_parse_gap_penalties() {
        // Single value
        let (del, ins) = MemOpt::parse_gap_penalties("6").unwrap();
        assert_eq!(del, 6);
        assert_eq!(ins, 6);

        // Two values
        let (del, ins) = MemOpt::parse_gap_penalties("6,5").unwrap();
        assert_eq!(del, 6);
        assert_eq!(ins, 5);

        // Error cases
        assert!(MemOpt::parse_gap_penalties("not_a_number").is_err());
        assert!(MemOpt::parse_gap_penalties("1,2,3").is_err());
    }

    #[test]
    fn test_parse_clip_penalties() {
        let (clip5, clip3) = MemOpt::parse_clip_penalties("5").unwrap();
        assert_eq!(clip5, 5);
        assert_eq!(clip3, 5);

        let (clip5, clip3) = MemOpt::parse_clip_penalties("3,7").unwrap();
        assert_eq!(clip5, 3);
        assert_eq!(clip3, 7);
    }

    #[test]
    fn test_mapq_coef_fac() {
        let opt = MemOpt::default();
        // ln(50) ≈ 3.912, should be truncated to 3
        assert_eq!(
            opt.mapq_coef_fac, 3,
            "mapq_coef_fac should be ln(50) truncated"
        );
    }

    // ========================================================================
    // STAGE-SPECIFIC PARAMETER BUNDLE TESTS
    // ========================================================================

    #[test]
    fn test_seeding_params() {
        let opt = MemOpt::default();
        let params = opt.seeding_params();

        assert_eq!(params.min_seed_len, 19);
        assert_eq!(params.split_factor, 1.5);
        assert_eq!(params.max_occ, 500);
    }

    #[test]
    fn test_chaining_params() {
        let opt = MemOpt::default();
        let params = opt.chaining_params();

        assert_eq!(params.band_width, 100);
        assert_eq!(params.max_chain_gap, 10000);
        assert_eq!(params.drop_ratio, 0.50);
    }

    #[test]
    fn test_extension_params() {
        let opt = MemOpt::default();
        let params = opt.extension_params();

        assert_eq!(params.match_score, 1);
        assert_eq!(params.mismatch_penalty, 4);
        assert_eq!(params.gap_open_del, 6);
        assert_eq!(params.gap_extend_del, 1);
    }

    #[test]
    fn test_output_params() {
        let opt = MemOpt::default();
        let params = opt.output_params();

        assert_eq!(params.score_threshold, 30);
        assert_eq!(params.max_xa_hits, 5);
        assert_eq!(params.xa_drop_ratio, 0.80);
        assert!(!params.output_all_alignments);
    }

    #[test]
    fn test_validate_default_passes() {
        let opt = MemOpt::default();
        assert!(opt.validate().is_ok());
    }

    #[test]
    fn test_validate_catches_invalid_params() {
        let mut opt = MemOpt::default();

        // Invalid min_seed_len
        opt.min_seed_len = 0;
        let result = opt.validate();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .iter()
                .any(|e| e.contains("min_seed_len"))
        );

        // Reset and try invalid drop_ratio
        opt = MemOpt::default();
        opt.drop_ratio = 1.5;
        let result = opt.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().iter().any(|e| e.contains("drop_ratio")));
    }
}
