# Gap Analysis of MEM Tool Parameters in FerrousAlign

This document identifies parameters defined in the `MemOpt` struct that are not directly passed into the stage-specific parameter bundles (`SeedingParams`, `ChainingParams`, `ExtensionParams`, `OutputParams`). These bundles are intended to encapsulate the parameters required by the core alignment algorithms (seeding, chaining, extension, and final output filtering). Parameters not included in these bundles are either handled at a higher level of the MEM tool's logic, are related to I/O or general control, or represent potential areas where parameter passing could be refined.

The analysis references the original C++ `mem_opt_t` struct from `bwa-mem2/src/bwamem.h` for context.

## Parameters Not Included in Stage-Specific Bundles

### 1. Scoring Parameters

*   **`pen_unpaired`**
    *   **FerrousAlign `MemOpt` description:** Phred-scaled penalty for unpaired reads.
    *   **C++ `mem_opt_t` context:** `int pen_unpaired; // phred-scaled penalty for unpaired reads`
    *   **Gap Explanation:** This parameter is relevant for paired-end alignment post-processing (e.g., when determining mapping quality for unpaired reads in a pair). It doesn't directly influence the core dynamic programming alignment (extension) or seed chaining. It's likely used in the final scoring or flagging of alignments.
    *   **Changes Needed:** No immediate changes needed for core algorithms. This is correctly handled outside the `ExtensionParams` as it's a post-alignment penalty. If paired-end specific functions were to have their own parameter bundle, this would belong there.

*   **`pen_clip5`, `pen_clip3`**
    *   **FerrousAlign `MemOpt` description:** 5' clipping penalty, 3' clipping penalty.
    *   **C++ `mem_opt_t` context:** `int pen_clip5,pen_clip3;// clipping penalty. This score is not deducted from the DP score.`
    *   **Gap Explanation:** As noted in the C++ comment, these clipping penalties are *not deducted from the DP score*. This implies they are applied during a post-processing step related to CIGAR string generation or final alignment score adjustment, rather than directly influencing the banded dynamic programming during the extension phase.
    *   **Changes Needed:** No immediate changes needed for core algorithms. These are correctly handled outside `ExtensionParams`.

*   **`mat`**
    *   **FerrousAlign `MemOpt` description:** Scoring matrix (5x5 for A,C,G,T,N).
    *   **C++ `mem_opt_t` context:** `int8_t mat[25]; // scoring matrix; mat[0] == 0 if unset`
    *   **Gap Explanation:** While the individual match (`a`) and mismatch (`b`) scores are passed to `ExtensionParams`, the full pre-computed scoring matrix (`mat`) is not. The `ExtensionParams` bundle contains `match_score` (`a`) and `mismatch_penalty` (`b`), from which the scoring matrix could be (and *is*, in `MemOpt::fill_scoring_matrix()`) derived. The core extension algorithms likely take individual match/mismatch scores or construct their own internal scoring representations.
    *   **Changes Needed:** The current approach is valid. If the extension algorithms were refactored to directly accept a pre-computed scoring matrix, `mat` could be added to `ExtensionParams`.

### 2. Chaining Parameters

*   **`max_chain_extend`**
    *   **FerrousAlign `MemOpt` description:** Maximum chain extension.
    *   **C++ `mem_opt_t` context:** `int max_chain_extend;`
    *   **Gap Explanation:** This parameter is present in `MemOpt` but not explicitly in `ChainingParams`. Its exact role in the chaining process is not immediately clear from the `FerrousAlign` code, as it's not a parameter passed to the `ChainingParams` bundle. In `MemOpt::default()`, it's set to `100` with the comment `// Limit chains to extend (was 1<<30 which caused memory explosion)`. This suggests it's an internal limit on chain growth.
    *   **Changes Needed:** If this parameter directly influences the chaining algorithm's logic for extending chains, it should be added to `ChainingParams`. Further investigation into its usage within the C++ `bwa-mem2` chaining functions would clarify its role. It might be used implicitly in some calculation or as a hard limit in the chaining loop.

### 3. Filtering Parameters

*   **`mask_level_redun`**
    *   **FerrousAlign `MemOpt` description:** Mask level for redundant hits.
    *   **C++ `mem_opt_t` context:** `float mask_level_redun;`
    *   **Gap Explanation:** This parameter is distinct from `mask_level` (which *is* in `OutputParams`) and is not included in `OutputParams`. This suggests it's used in a different filtering context, potentially for highly redundant or secondary alignments, or in a specific post-processing phase not covered by the `OutputParams` bundle.
    *   **Changes Needed:** Clarify its exact usage in the C++ codebase to determine if it should be added to `OutputParams` (if it's an output filtering parameter) or if a separate parameter bundle is needed for a specific "redundancy masking" stage.

### 4. Paired-End Parameters

*   **`max_ins`**
    *   **FerrousAlign `MemOpt` description:** Maximum insert size (when estimating distribution, skip pairs with insert > this).
    *   **C++ `mem_opt_t` context:** `int max_ins; // when estimating insert size distribution, skip pairs with insert longer than this value`
    *   **Gap Explanation:** This parameter is crucial for paired-end mapping, specifically for filtering reads when estimating the insert size distribution. It's not directly part of the core seeding, chaining, or single-read extension algorithms, but rather a control parameter for paired-end specific logic (e.g., `mem_pestat` in C++).
    *   **Changes Needed:** No changes needed for the core alignment algorithms. This parameter would belong in a dedicated `PairedEndParams` bundle if such a structure were created for paired-end specific processing logic.

*   **`max_matesw`**
    *   **FerrousAlign `MemOpt` description:** Perform maximally `max_matesw` rounds of mate-SW for each end.
    *   **C++ `mem_opt_t` context:** `int max_matesw; // perform maximally max_matesw rounds of mate-SW for each end`
    *   **Gap Explanation:** This parameter directly controls the behavior of the mate-SW (Smith-Waterman) rescue process, a specialized alignment routine for paired-end reads. It is not part of the general `ExtensionParams` because it's for a specific rescue mechanism.
    *   **Changes Needed:** Similar to `max_ins`, this would belong in a `PairedEndParams` bundle, indicating it's used by paired-end specific algorithms.

### 5. Processing Parameters (General Control)

*   **`n_threads`**
    *   **FerrousAlign `MemOpt` description:** Number of threads.
    *   **C++ `mem_opt_t` context:** `int n_threads; // number of threads`
    *   **Gap Explanation:** This is a high-level concurrency parameter that controls the overall parallel execution of the MEM tool. It's not a parameter for an individual alignment algorithm but for the entire pipeline's parallelization. It's passed to functions like `mem_process_seqs`.
    *   **Changes Needed:** Correctly handled at a higher level; no changes needed for algorithm-specific bundles.

*   **`chunk_size`**
    *   **FerrousAlign `MemOpt` description:** Process `chunk_size`-bp sequences in a batch.
    *   **C++ `mem_opt_t` context:** `int64_t chunk_size; // process chunk_size-bp sequences in a batch`
    *   **Gap Explanation:** This parameter dictates the size of input data processed in each batch, affecting memory usage and throughput. It's an operational parameter for the overall workflow, not directly for the alignment algorithms themselves.
    *   **Changes Needed:** Correctly handled at a higher level; no changes needed for algorithm-specific bundles.

*   **`batch_size`**
    *   **FerrousAlign `MemOpt` description:** Number of read pairs to process in a batch.
    *   **C++ `mem_opt_t` context:** (Not explicitly present with this name in C++ `mem_opt_t`, but similar concepts might exist in other structs or implicitly.) In FerrousAlign, this seems to be a Rust-specific addition or refinement for batching reads.
    *   **Gap Explanation:** Similar to `chunk_size`, this is an operational parameter for managing the number of reads processed concurrently or in a single pass.
    *   **Changes Needed:** Correctly handled at a higher level; no changes needed for algorithm-specific bundles.

*   **`mapq_coef_len`, `mapq_coef_fac`**
    *   **FerrousAlign `MemOpt` description:** Coefficient length/factor for mapQ calculation.
    *   **C++ `mem_opt_t` context:** `float mapQ_coef_len;`, `int mapQ_coef_fac;` (Also `#define MEM_MAPQ_COEF 30.0` and `#define MEM_MAPQ_MAX 60` are related).
    *   **Gap Explanation:** These parameters are specifically for calculating mapping quality (MAPQ). MAPQ calculation happens after alignments are generated and scored. It's a post-alignment filtering/scoring step.
    *   **Changes Needed:** Correctly handled outside core algorithm bundles. If a `MapQParams` bundle were created, they would belong there.

### 6. Flags and Other Options

*   **`flag`**
    *   **FerrousAlign `MemOpt` description:** Bitfield for various flags (kept for compatibility).
    *   **C++ `mem_opt_t` context:** `int flag; // see MEM_F_* macros` (e.g., `MEM_F_PE`, `MEM_F_NOPAIRING`, `MEM_F_ALL`, etc.)
    *   **Gap Explanation:** In FerrousAlign, the individual flags from the C++ bitfield have been largely separated into distinct boolean fields (e.g., `smart_pairing`, `output_all_alignments`). The `flag` field itself remains for compatibility but is likely not directly used by the Rust logic's control flow in the same way as the original C++ bitfield.
    *   **Changes Needed:** The individual boolean flags derived from this bitfield (`smart_pairing`, `treat_alt_as_primary`, `smallest_coord_primary`, `output_all_alignments`) are already used. The `flag` field itself serves as a compatibility placeholder. No changes are needed for algorithm-specific bundles; its function has been superseded by more granular boolean fields.

*   **`read_group`**
    *   **FerrousAlign `MemOpt` description:** Read group header line.
    *   **C++ `mem_opt_t` context:** Not directly in `mem_opt_t` as a field, but `mem_reg2sam` and `mem_aln2sam` functions would handle SAM header information derived from command-line options.
    *   **Gap Explanation:** This is an I/O and SAM output formatting parameter. It concerns how the output SAM file is generated, not the alignment process itself.
    *   **Changes Needed:** Correctly handled outside algorithm-specific bundles.

*   **`header_lines`**
    *   **FerrousAlign `MemOpt` description:** Additional header lines to insert.
    *   **C++ `mem_opt_t` context:** Not directly in `mem_opt_t` as a field, but typically handled by `bwa.c` or similar for SAM output.
    *   **Gap Explanation:** Similar to `read_group`, this is an I/O and SAM output formatting parameter.
    *   **Changes Needed:** Correctly handled outside algorithm-specific bundles.

*   **`insert_size_override`**
    *   **FerrousAlign `MemOpt` description:** Manual insert size specification.
    *   **C++ `mem_opt_t` context:** No direct equivalent field in `mem_opt_t`. The C++ version might handle this through command-line parameters that influence the `mem_pestat_t` structure.
    *   **Gap Explanation:** This parameter provides manual control over the insert size distribution, overriding auto-inference. It's used by paired-end processing logic, similar to `max_ins`, but not by the core alignment algorithms.
    *   **Changes Needed:** Would belong in a `PairedEndParams` bundle if one is created.

*   **`verbosity`**
    *   **FerrousAlign `MemOpt` description:** Verbosity level.
    *   **C++ `mem_opt_t` context:** No explicit `verbosity` field in `mem_opt_t`. Logging levels are typically handled globally or via preprocessor macros in C++.
    *   **Gap Explanation:** This is a debugging and logging control parameter, not directly related to the computational steps of alignment.
    *   **Changes Needed:** Correctly handled at a higher level (e.g., configuring a logger).

*   **`smart_pairing`**
    *   **FerrousAlign `MemOpt` description:** Smart pairing.
    *   **C++ `mem_opt_t` context:** This corresponds to the `MEM_F_SMARTPE` flag in `mem_opt_t::flag`.
    *   **Gap Explanation:** This is a high-level flag that modifies how paired-end reads are handled. It affects the overall strategy for read pairing, not the internal mechanics of seeding, chaining, or extension.
    *   **Changes Needed:** Correctly handled at a higher level; no changes needed for algorithm-specific bundles.

*   **`treat_alt_as_primary`**
    *   **FerrousAlign `MemOpt` description:** Treat ALT contigs as part of primary assembly.
    *   **C++ `mem_opt_t` context:** This corresponds to the `-j` command-line option in bwa-mem2. It influences outputting to XA tag, etc.
    *   **Gap Explanation:** This flag influences how alignments to alternative contigs are handled, affecting classification and reporting. It's a post-alignment filtering/flagging decision.
    *   **Changes Needed:** Correctly handled at a higher level; no changes needed for algorithm-specific bundles.

*   **`smallest_coord_primary`**
    *   **FerrousAlign `MemOpt` description:** For split alignment, take smallest coordinate as primary.
    *   **C++ `mem_opt_t` context:** This corresponds to the `MEM_F_PRIMARY5` flag in `mem_opt_t::flag`.
    *   **Gap Explanation:** This flag dictates the rule for determining the primary alignment among split alignments. It's a post-alignment decision for reporting.
    *   **Changes Needed:** Correctly handled at a higher level; no changes needed for algorithm-specific bundles.
