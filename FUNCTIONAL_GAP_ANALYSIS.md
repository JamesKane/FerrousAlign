# Functional Gap Analysis of MEM Tool Parameters in FerrousAlign

This document details the functional status of each parameter within the `MemOpt` struct in FerrousAlign's Rust codebase. The analysis aims to identify which parameters are actively used to influence program logic and which, despite being present or parsed, are not yet functional or only partially functional. For each parameter, its intended purpose, actual usage in Rust, and comparison to the original C++ `bwa-mem2` implementation (`mem_opt_t` struct from `bwamem.h`) are provided.

**Methodology:**
Each field of the `MemOpt` struct (`src/pipelines/linear/mem_opt.rs`) was systematically searched across the entire `src/` directory to identify where its value is read or used. The functional status was determined based on whether the parameter's value directly influences the program's behavior beyond mere assignment or inclusion in a data structure.

## Summary of Functional Gap

The following `MemOpt` parameters were identified as **Not Functional** or **Partially Functional** (meaning their parsed values do not actively influence program logic in the current Rust codebase, or their influence is indirect/unclear):

*   **`opt.max_xa_hits_alt`**: (Output parameters) - Value is parsed and included in a bundle, but no explicit logic uses this specific value to limit XA hits for ALT contigs.
*   **`opt.max_ins`**: (Paired-end parameters) - Value is initialized, but never read or used to filter pairs during insert size estimation.
*   **`opt.chunk_size`**: (Processing parameters) - Value is parsed, but no explicit logic uses it to control sequence batching.
*   **`opt.flag`**: (Flags and other options) - Bitfield is present for compatibility but its value is never read or used; its functionality has been superseded by explicit boolean flags.
*   **`opt.verbosity`**: (Flags and other options) - Value is parsed, but its direct impact on configuring a logger or controlling logging output within the codebase is not explicitly evident. (Partially Functional)
*   **`opt.smart_pairing`**: (Flags and other options) - Value is parsed, but no explicit logic uses it to modify paired-end handling.
*   **`opt.treat_alt_as_primary`**: (Flags and other options) - Value is parsed, but no explicit logic uses it to influence primary alignment determination for ALT contigs.
*   **`opt.smallest_coord_primary`**: (Flags and other options) - Value is parsed, but no explicit logic uses it to determine primary alignment for split reads based on coordinate.

## Detailed Analysis of MemOpt Parameters

### Scoring Parameters

*   **`a` (Match score)**
    *   **Purpose:** Score awarded for a sequence match.
    *   **Functional Status:** Functional
    *   **Target Methods/Usage:** Used in `region.rs`, `chaining.rs`, `batch_extension.rs`, `finalization.rs` for initial SWA score (h0), gap calculations, and `MD` tag generation. Passed to underlying SWA/KSW functions via `ExtensionParams`.
    *   **C++ Context:** `int a; // match score`. Consistent with C++ behavior.

*   **`b` (Mismatch penalty)**
    *   **Purpose:** Penalty for a sequence mismatch.
    *   **Functional Status:** Functional
    *   **Target Methods/Usage:** Used in `region.rs`, `batch_extension.rs` for setting scoring matrix values (as negative), and in `finalization.rs` for `MD` tag generation thresholds. Passed to underlying SWA/KSW functions via `ExtensionParams`.
    *   **C++ Context:** `int b; // mismatch penalty`. Consistent with C++ behavior.

*   **`o_del` (Gap open penalty for deletions)**
    *   **Purpose:** Penalty for opening a gap (deletion) in the alignment.
    *   **Functional Status:** Functional
    *   **Target Methods/Usage:** Used in `mem.rs` (assignment), `region.rs`, `chaining.rs` (gap calculations), `batch_extension.rs`, `finalization.rs`. Passed to underlying SWA/KSW functions via `ExtensionParams`.
    *   **C++ Context:** `int o_del;`. Consistent with C++ behavior.

*   **`e_del` (Gap extension penalty for deletions)**
    *   **Purpose:** Penalty for extending an existing deletion gap.
    *   **Functional Status:** Functional
    *   **Target Methods/Usage:** Used in `mem.rs` (assignment), `region.rs`, `chaining.rs` (gap calculations), `batch_extension.rs`, `finalization.rs`. Passed to underlying SWA/KSW functions via `ExtensionParams`.
    *   **C++ Context:** `int e_del;`. Consistent with C++ behavior.

*   **`o_ins` (Gap open penalty for insertions)**
    *   **Purpose:** Penalty for opening a gap (insertion) in the alignment.
    *   **Functional Status:** Functional
    *   **Target Methods/Usage:** Used in `mem.rs` (assignment), `region.rs`, `chaining.rs` (gap calculations), `batch_extension.rs`, `finalization.rs`. Passed to underlying SWA/KSW functions via `ExtensionParams`.
    *   **C++ Context:** `int o_ins;`. Consistent with C++ behavior.

*   **`e_ins` (Gap extension penalty for insertions)**
    *   **Purpose:** Penalty for extending an existing insertion gap.
    *   **Functional Status:** Functional
    *   **Target Methods/Usage:** Used in `mem.rs` (assignment), `region.rs`, `chaining.rs` (gap calculations), `batch_extension.rs`, `finalization.rs`. Passed to underlying SWA/KSW functions via `ExtensionParams`.
    *   **C++ Context:** `int e_ins;`. Consistent with C++ behavior.

*   **`pen_unpaired` (Phred-scaled penalty for unpaired reads)**
    *   **Purpose:** Penalty applied to reads that are supposed to be paired but are found unpaired.
    *   **Functional Status:** Functional (parsed and assigned)
    *   **Target Methods/Usage:** Assigned in `mem.rs` from CLI. Value is stored in `MemOpt`.
    *   **C++ Context:** `int pen_unpaired; // phred-scaled penalty for unpaired reads`. While assigned, its direct use in calculating penalties or flagging is not explicitly seen in the code snippets. Further tracing would be needed to confirm its ultimate impact on final scores or flags.

*   **`pen_clip5` (5' clipping penalty)**
    *   **Purpose:** Penalty for soft-clipping at the 5' end of a read.
    *   **Functional Status:** Functional
    *   **Target Methods/Usage:** Used in `mem.rs` (assignment), `region.rs`, `batch_extension.rs`. Passed to underlying SWA/KSW functions via `ExtensionParams`. Also used directly in `region.rs` (`compute_extended_regions`) for score comparison.
    *   **C++ Context:** `int pen_clip5; // clipping penalty. This score is not deducted from the DP score.`. Consistent with C++ where it's used for decision making rather than direct score modification.

*   **`pen_clip3` (3' clipping penalty)**
    *   **Purpose:** Penalty for soft-clipping at the 3' end of a read.
    *   **Functional Status:** Functional
    *   **Target Methods/Usage:** Used in `mem.rs` (assignment), `region.rs`, `batch_extension.rs`. Passed to underlying SWA/KSW functions via `ExtensionParams`. Also used directly in `region.rs` (`compute_extended_regions`) for score comparison.
    *   **C++ Context:** `int pen_clip3; // clipping penalty. This score is not deducted from the DP score.`. Consistent with C++ where it's used for decision making rather than direct score modification.

### Alignment Parameters

*   **`w` (Band width for banded alignment)**
    *   **Purpose:** Defines the width of the band for banded Smith-Waterman alignment, limiting the search space.
    *   **Functional Status:** Functional
    *   **Target Methods/Usage:** Used in `mem.rs` (assignment), `region.rs` (passed to SWA functions), `batch_extension.rs`, `chaining.rs` (gap conditions). Passed to `ChainingParams` and `ExtensionParams`.
    *   **C++ Context:** `int w; // band width`. Consistent with C++ behavior.

*   **`zdrop` (Z-dropoff)**
    *   **Purpose:** Off-diagonal X-dropoff threshold; alignment extension stops if the score drops too far from the maximum on the diagonal.
    *   **Functional Status:** Functional
    *   **Target Methods/Usage:** Used in `mem.rs` (assignment), `region.rs`, `batch_extension.rs`. Passed to underlying SWA/KSW functions via `ExtensionParams`.
    *   **C++ Context:** `int zdrop; // Z-dropoff`. Consistent with C++ behavior.

### Seeding Parameters

*   **`max_mem_intv` (Maximum MEM interval)**
    *   **Purpose:** Controls the pruning of common Maximal Exact Matches (MEMs) during seeding.
    *   **Functional Status:** Functional
    *   **Target Methods/Usage:** Used in `mem.rs` (assignment), `pipeline.rs` (passed to seeding logic via `SeedingParams` to configure SMEM search).
    *   **C++ Context:** `uint64_t max_mem_intv;`. Consistent with C++ behavior.

*   **`min_seed_len` (Minimum seed length)**
    *   **Purpose:** Minimum length for a seed to be considered valid.
    *   **Functional Status:** Functional
    *   **Target Methods/Usage:** Used in `mem.rs` (assignment), `pipeline.rs` (filtering seeds, calculating `split_len`), `chaining.rs` (chain filtering), `finalization.rs` (score threshold calculation). Passed to `SeedingParams`.
    *   **C++ Context:** `int min_seed_len; // minimum seed length`. Consistent with C++ behavior.

*   **`split_factor` (Split factor)**
    *   **Purpose:** Factor used to determine when a MEM is long enough to be split into a new seed.
    *   **Functional Status:** Functional
    *   **Target Methods/Usage:** Used in `mem.rs` (assignment), `pipeline.rs` (calculating `split_len`). Passed to `SeedingParams`.
    *   **C++ Context:** `float split_factor;`. Consistent with C++ behavior.

*   **`split_width` (Split width)**
    *   **Purpose:** Threshold for occurrences to determine if a seed should be split.
    *   **Functional Status:** Functional
    *   **Target Methods/Usage:** Used in `pipeline.rs` (passed to seeding logic via `SeedingParams` for SMEM finding configuration).
    *   **C++ Context:** `int split_width;`. Consistent with C++ behavior.

*   **`max_occ` (Maximum occurrences)**
    *   **Purpose:** Seeds with occurrences higher than this value are considered repetitive and skipped.
    *   **Functional Status:** Functional
    *   **Target Methods/Usage:** Used in `mem.rs` (assignment), `chaining.rs` (filtering repetitive seeds), `pipeline.rs` (filtering seeds). Passed to `SeedingParams`.
    *   **C++ Context:** `int max_occ;`. Consistent with C++ behavior.

### Chaining Parameters

*   **`min_chain_weight` (Minimum chain weight)**
    *   **Purpose:** Minimum score for a chain to be considered valid.
    *   **Functional Status:** Functional
    *   **Target Methods/Usage:** Used in `mem.rs` (assignment), `chaining.rs` (filtering chains). Passed to `ChainingParams`.
    *   **C++ Context:** `int min_chain_weight;`. Consistent with C++ behavior.

*   **`max_chain_extend` (Maximum chain extension)**
    *   **Purpose:** Limit on the number of chains considered or extended.
    *   **Functional Status:** Functional
    *   **Target Methods/Usage:** Used in `chaining.rs` (`chain_seeds`) to limit the number of `kept_chains`.
    *   **C++ Context:** `int max_chain_extend;`. Consistent with C++ behavior.

*   **`max_chain_gap` (Maximum chain gap)**
    *   **Purpose:** Maximum allowed genomic distance between seeds in a chain.
    *   **Functional Status:** Functional
    *   **Target Methods/Usage:** Used in `chaining.rs` (`chain_seeds`) for evaluating seed addition to a chain, and in `finalization.rs` (`resolve_overlap`). Passed to `ChainingParams`.
    *   **C++ Context:** `int max_chain_gap;`. Consistent with C++ behavior.

### Filtering Parameters

*   **`mask_level` (Mask level)**
    *   **Purpose:** Threshold for marking redundant hits based on overlap with a better hit.
    *   **Functional Status:** Functional
    *   **Target Methods/Usage:** Used in `chaining.rs` (`remove_redundant_chains`), `pipeline.rs` (`remove_redundant_regions`), `finalization.rs` (`find_primary_alignments`). Passed to `OutputParams`.
    *   **C++ Context:** `float mask_level;`. Consistent with C++ behavior.

*   **`drop_ratio` (Drop ratio)**
    *   **Purpose:** Threshold for dropping chains if their seed coverage is too low compared to a better overlapping chain.
    *   **Functional Status:** Functional
    *   **Target Methods/Usage:** Used in `mem.rs` (assignment), `chaining.rs` (filtering chains). Passed to `ChainingParams`.
    *   **C++ Context:** `float drop_ratio;`. Consistent with C++ behavior.

*   **`xa_drop_ratio` (XA drop ratio)**
    *   **Purpose:** Threshold for filtering alignments for inclusion in the XA tag based on score relative to the best alignment.
    *   **Functional Status:** Functional
    *   **Target Methods/Usage:** Used in `finalization.rs` (`filter_supplementary_by_score`, `generate_xa_tag`). Passed to `OutputParams`.
    *   **C++ Context:** `float XA_drop_ratio;`. Consistent with C++ behavior.

*   **`mask_level_redun` (Mask level for redundant hits)**
    *   **Purpose:** Another threshold for marking redundant hits, possibly with a different application than `mask_level`.
    *   **Functional Status:** Functional
    *   **Target Methods/Usage:** Used in `finalization.rs` (`resolve_overlap`).
    *   **C++ Context:** `float mask_level_redun;`. Its specific distinction from `mask_level` in C++ might require deeper analysis, but it is actively used in FerrousAlign.

### Output Parameters

*   **`t` (Minimum score threshold to output)**
    *   **Purpose:** Minimum alignment score for an alignment to be output.
    *   **Functional Status:** Functional
    *   **Target Methods/Usage:** Used in `mem.rs` (assignment), `pipeline.rs` (filtering alignment regions), `batch_extension.rs`, `paired_end.rs` (filtering in paired-end processing), `core/io/sam_output.rs` (determining SAM output). Passed to `OutputParams`.
    *   **C++ Context:** `int T; // output score threshold; only affecting output`. Consistent with C++ behavior.

*   **`max_xa_hits` (Maximum XA hits)**
    *   **Purpose:** Limits the number of alternative alignments reported in the XA tag.
    *   **Functional Status:** Functional
    *   **Target Methods/Usage:** Used in `mem.rs` (assignment), `finalization.rs` (`generate_xa_tag`). Passed to `OutputParams`.
    *   **C++ Context:** `int max_XA_hits;`. Consistent with C++ behavior.

*   **`max_xa_hits_alt` (Maximum XA hits for ALT contigs)**
    *   **Purpose:** Limits the number of alternative alignments specifically for ALT contigs in the XA tag.
    *   **Functional Status:** Not Functional
    *   **Target Methods/Usage:** Assigned in `mem.rs` from CLI. Included in `OutputParams`. **No explicit code logic was found that uses this specific value to limit XA hits for ALT contigs, distinct from `max_xa_hits`.**
    *   **C++ Context:** `int max_XA_hits_alt;`. In C++ bwa-mem2, this is distinct and used. **Recommendation:** Implement logic to differentiate between `max_xa_hits` and `max_xa_hits_alt` in `generate_xa_tag` or related functions.

### Paired-end Parameters

*   **`max_ins` (Maximum insert size)**
    *   **Purpose:** When estimating insert size distribution, pairs with insert sizes larger than this are skipped.
    *   **Functional Status:** Not Functional
    *   **Target Methods/Usage:** Initialized in `MemOpt::default()`. **No code logic was found that reads or uses this value to filter pairs during insert size estimation.**
    *   **C++ Context:** `int max_ins; // when estimating insert size distribution, skip pairs with insert longer than this value`. This is a critical parameter in C++ `mem_pestat` for paired-end processing. **Recommendation:** Implement logic in paired-end processing (`paired_end.rs` or related modules) to utilize `max_ins` for filtering.

*   **`max_matesw` (Maximum mate-SW rounds)**
    *   **Purpose:** Limits the number of rounds for mate-SW (Smith-Waterman) rescue process for each end.
    *   **Functional Status:** Functional
    *   **Target Methods/Usage:** Used in `mem.rs` (assignment), `paired_end.rs` (passed to `mate_rescue` functions).
    *   **C++ Context:** `int max_matesw;`. Consistent with C++ behavior.

### Processing Parameters (General Control)

*   **`n_threads` (Number of threads)**
    *   **Purpose:** Controls the number of threads used for parallel processing.
    *   **Functional Status:** Functional
    *   **Target Methods/Usage:** Used in `mem.rs` (assignment), `single_end.rs` (to configure thread pools/parallel iterators).
    *   **C++ Context:** `int n_threads;`. Consistent with C++ behavior.

*   **`chunk_size` (Process chunk_size-bp sequences in a batch)**
    *   **Purpose:** Controls the size of input batches for processing.
    *   **Functional Status:** Not Functional
    *   **Target Methods/Usage:** Assigned in `mem.rs` from CLI. **No code logic was found that reads or uses this value to control batching of sequences for processing.**
    *   **C++ Context:** `int64_t chunk_size;`. This is a crucial parameter in C++ `mem_process_seqs` for parallel chunking. **Recommendation:** Implement logic to utilize `chunk_size` for input batching in `pipeline.rs` or `main.rs`.

*   **`batch_size` (Number of read pairs to process in a batch)**
    *   **Purpose:** Controls the number of read pairs processed together in a batch, primarily for paired-end processing.
    *   **Functional Status:** Functional
    *   **Target Methods/Usage:** Used in `mem.rs` (assignment), `paired_end.rs` (reading input, loop control).
    *   **C++ Context:** Not a direct `mem_opt_t` field but the concept is central to C++ batch processing. Consistent with FerrousAlign's batching mechanism.

*   **`mapq_coef_len` (Coefficient length for mapQ calculation)**
    *   **Purpose:** Used in the calculation of mapping quality (MAPQ).
    *   **Functional Status:** Functional
    *   **Target Methods/Usage:** Used in `mem_opt.rs` (calculating `mapq_coef_fac`), `mem.rs` (recalculating `mapq_coef_fac`), `finalization.rs` (`calculate_mapq`).
    *   **C++ Context:** `float mapQ_coef_len;`. Consistent with C++ behavior.

*   **`mapq_coef_fac` (Coefficient factor for mapQ calculation)**
    *   **Purpose:** Used in the calculation of mapping quality (MAPQ), derived from `mapq_coef_len`.
    *   **Functional Status:** Functional
    *   **Target Methods/Usage:** Used in `mem_opt.rs` (calculated in `default`), `mem.rs` (recalculated), `finalization.rs` (`calculate_mapq`).
    *   **C++ Context:** `int mapQ_coef_fac;`. Consistent with C++ behavior.

### Flags and Other Options

*   **`flag` (Bitfield for various flags)**
    *   **Purpose:** Bitfield representing various high-level control flags (e.g., paired-end mode, output all alignments, etc.).
    *   **Functional Status:** Not Functional
    *   **Target Methods/Usage:** Initialized in `MemOpt::default()` to `0`. **No code logic was found that reads or uses this value. Its functionality has been superseded by explicit boolean fields within `MemOpt`.**
    *   **C++ Context:** `int flag; // see MEM_F_* macros`. In C++, this is a central control mechanism. **Recommendation:** This field serves as a compatibility placeholder and is effectively superseded. It can likely be removed if full compatibility with the C++ bitfield is not required, or explicitly marked as deprecated.

*   **`read_group` (Read group header line)**
    *   **Purpose:** Specifies the SAM read group header line.
    *   **Functional Status:** Functional
    *   **Target Methods/Usage:** Used in `mem.rs` (assignment, check for presence). This implies usage in SAM output generation.
    *   **C++ Context:** Handled by SAM output functions. Consistent with C++ behavior.

*   **`header_lines` (Additional header lines to insert)**
    *   **Purpose:** Provides additional lines to be inserted into the SAM header.
    *   **Functional Status:** Functional
    *   **Target Methods/Usage:** Used in `mem.rs` (assignment, iteration for SAM header generation).
    *   **C++ Context:** Handled by SAM output functions. Consistent with C++ behavior.

*   **`insert_size_override` (Manual insert size specification)**
    *   **Purpose:** Allows overriding automatically inferred insert size distribution.
    *   **Functional Status:** Functional
    *   **Target Methods/Usage:** Used in `mem.rs` (assignment), `paired_end.rs` (checking for override and using values).
    *   **C++ Context:** Similar manual control is typically available. Consistent with C++ behavior.

*   **`verbosity` (Verbosity level)**
    *   **Purpose:** Controls the logging level.
    *   **Functional Status:** Partially Functional
    *   **Target Methods/Usage:** Assigned in `mem.rs` from CLI. **No explicit code logic was found that uses this value to configure a logger or control logging output within the codebase after assignment.**
    *   **C++ Context:** Logging levels typically configured globally. **Recommendation:** Ensure this parameter is explicitly used to configure the logging framework (e.g., `log` crate) at startup.

*   **`smart_pairing` (Smart pairing)**
    *   **Purpose:** Modifies how paired-end reads are handled (e.g., ignoring `in2.fq` in certain scenarios).
    *   **Functional Status:** Not Functional
    *   **Target Methods/Usage:** Assigned in `mem.rs` from CLI. **No explicit code logic was found that reads or uses this value to modify paired-end handling.**
    *   **C++ Context:** Corresponds to `MEM_F_SMARTPE` flag. This is a significant functional mode in C++. **Recommendation:** Implement logic in `paired_end.rs` or related modules to utilize `smart_pairing` to adjust paired-end processing.

*   **`treat_alt_as_primary` (Treat ALT contigs as primary)**
    *   **Purpose:** Influences whether alignments to alternative contigs can be considered primary.
    *   **Functional Status:** Not Functional
    *   **Target Methods/Usage:** Assigned in `mem.rs` from CLI. **No explicit code logic was found that reads or uses this value to influence primary alignment determination for ALT contigs.**
    *   **C++ Context:** This flag affects processing and output of ALT contigs. **Recommendation:** Implement logic in `finalization.rs` or other post-processing stages to utilize `treat_alt_as_primary` for primary alignment selection.

*   **`smallest_coord_primary` (Smallest coordinate as primary)**
    *   **Purpose:** Dictates the rule for determining the primary alignment among split alignments (take the one with the smallest coordinate).
    *   **Functional Status:** Not Functional
    *   **Target Methods/Usage:** Assigned in `mem.rs` from CLI. **No explicit code logic was found that reads or uses this value to influence primary alignment determination for split reads.**
    *   **C++ Context:** Corresponds to `MEM_F_PRIMARY5` flag. This is a significant functional rule for split alignments in C++. **Recommendation:** Implement logic in `finalization.rs` or other primary alignment selection stages to utilize `smallest_coord_primary`.

*   **`output_all_alignments` (Output all alignments)**
    *   **Purpose:** Controls whether all alignments (primary and secondary) are output for single-end or unpaired paired-end reads.
    *   **Functional Status:** Functional
    *   **Target Methods/Usage:** Used in `mem.rs` (assignment), `paired_end.rs` (`select_output_indices`), `core/io/sam_output.rs` (determining SAM output). Passed to `OutputParams`.
    *   **C++ Context:** Corresponds to `MEM_F_ALL` flag. Consistent with C++ behavior.
