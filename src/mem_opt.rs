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
}

/// Manual insert size specification (overrides auto-inference)
#[derive(Debug, Clone)]
pub struct InsertSizeOverride {
    pub mean: f64,   // Mean insert size
    pub stddev: f64, // Standard deviation (default: 10% of mean)
    pub max: i32,    // Maximum insert size (default: mean + 4*stddev)
    pub min: i32,    // Minimum insert size (default: 0)
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
            max_chain_extend: 1 << 30,
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
        };

        // Calculate mapq_coef_fac as log of mapq_coef_len (matching C++)
        opt.mapq_coef_fac = opt.mapq_coef_len.ln() as i32;

        // Fill scoring matrix using match/mismatch scores
        opt.fill_scoring_matrix();

        opt
    }
}

impl MemOpt {
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
}
