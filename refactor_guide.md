# Refactoring Guide: `align.rs` to `alignment/`

This document outlines the proposed refactoring of the monolithic `align.rs` module into a more organized structure under the `src/alignment/` directory.

---

## 1. `src/alignment/seeding.rs`

This module will be responsible for generating seeds (SMEMs) from the query sequence using the FM-Index.

**Structs to move:**
- `Seed`
- `SMEM`

**Functions to move:**
- `generate_smems_for_strand`
- `get_bwt_base_from_cp_occ` (as a helper for seeding)
- `get_bwt` (as a helper for seeding)
- `get_sa_entry` (as a helper for seeding)

---

## 2. `src/alignment/chaining.rs`

This module will handle the chaining of seeds into potential alignment candidates.

**Structs to move:**
- `Chain`

**Functions to move:**
- `chain_seeds`
- `filter_chains`
- `calculate_chain_weight`
- `cal_max_gap` (helper function used in chaining)

---

## 3. `src/alignment/extension.rs`

This module will be responsible for extending seeds or chains using Smith-Waterman alignment.

**Structs to move:**
- `AlignmentJob`

**Functions to move:**
- `execute_adaptive_alignments`
- `execute_batched_alignments_with_size`
- `execute_scalar_alignments`
- `determine_optimal_batch_size`
- `partition_jobs_by_divergence`
- `estimate_divergence_score`
- `execute_batched_alignments` (deprecated, but can be moved)

---

## 4. `src/alignment/finalization.rs`

This module will contain the final `Alignment` data structure and all logic related to scoring, formatting, and filtering final alignments.

**Modules to move:**
- `sam_flags`

**Structs to move:**
- `Alignment`

**Functions and `impl` blocks to move:**
- `impl Alignment` (containing `cigar_string`, `to_sam_string`, `calculate_tlen`, etc.)
- `mark_secondary_alignments`
- `calculate_mapq`
- `alignments_overlap`
- `generate_xa_tags`
- `generate_sa_tags`

---

## 5. `src/alignment/pipeline.rs`

This module will orchestrate the entire alignment process for a single read, from seed generation to final alignments.

**Functions to move:**
- `generate_seeds` (This is the main top-level function in `align.rs` that drives the process)
- `generate_seeds_with_mode` (The internal implementation)

---

## 6. Utility and Common Code

The following items are general utilities. They could be placed in a new `src/alignment/utils.rs` file or be made private within the modules that use them.

- `DEFAULT_SCORING_MATRIX`
- `base_to_code`
- `reverse_complement_code`
- `encode_sequence`
- `reverse_complement_sequence`
- `hash_64` (from `crate::utils`) will be a dependency for `finalization.rs`.

---

## 7. `src/alignment/mod.rs`

This file will declare the new modules.

```rust
// src/alignment/mod.rs

pub mod chaining;
pub mod extension;
pub mod finalization;
pub mod pipeline;
pub mod seeding;

// Optional:
// pub mod utils;
```
