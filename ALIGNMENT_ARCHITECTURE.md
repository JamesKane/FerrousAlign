# FerrousAlign Alignment Module Architecture - Comprehensive Summary

## Executive Summary

The alignment module is a 4-stage pipeline processing reads from seeding through final SAM output. The current architecture has **significant front-loading opportunities** where work done late in the pipeline could be moved earlier for better cache locality, reduced redundant computation, and faster filtering.

### Key Statistics
- **Total Source**: 17,295 lines across 18 files
- **Largest files**: banded_swa.rs (3,713 lines), pipeline.rs (2,098 lines), finalization.rs (1,857 lines)
- **4-stage pipeline**: Seeding → Chaining → Extension → Finalization
- **Decision points**: 15+ major locations where alignment filtering/ranking occurs

---

## Stage 1: Seeding (FM-Index Search)

**File**: `/home/jkane/RustroverProjects/FerrousAlign/src/alignment/seeding.rs` (1,160 lines)

**Key Functions**:
- `generate_smems_for_strand()` - Lines 39-395
  - 3-phase algorithm: Forward extension, Backward search, Unique interval tracking
  - Outputs SMEM structs with BWT interval bounds
- `generate_smems_from_position()` - Re-seeding for chimeric detection
- `forward_only_seed_strategy()` - 3rd-round seeding for repetitive regions
- `get_sa_entries()` - Convert BWT intervals to reference positions

**Data Structures**:
```rust
pub struct SMEM {
    query_start: i32,          // Position in read
    query_end: i32,
    bwt_interval_start: u64,   // Range in suffix array
    bwt_interval_end: u64,
    interval_size: u64,        // Occurrence count
    is_reverse_complement: bool // Determined later from position
}

pub struct Seed {
    query_pos: i32,
    ref_pos: u64,              // FM-index position
    len: i32,
    is_rev: bool,              // Determined from ref_pos >= l_pac
    interval_size: u64,        // BWT interval size (repetitiveness)
    rid: i32,                  // Chromosome ID (-1 if spans boundary)
}
```

**Filtering Decision Points**:
- **Line 806**: Basic filter: `seed_len >= min_seed_len && occurrences <= max_occ`
- **Line 776-796**: Relaxed max_occ for 3rd round seeding (adaptive threshold)
- **Line 912-918**: Chromosome boundary check - seeds spanning boundaries rejected
- **Line 854-872**: SEEDS_PER_READ limit (500) prevents memory explosion

**Key Properties**:
- SMEMs come from bidirectional FM-index (both strands searched automatically)
- Strand determined by FM-index position, NOT by which strand was searched
- Multiple seeds can come from same SMEM if it appears many times
- Filtering is **conservative** - few decisions made here

---

## Stage 2: Chaining (O(n log n) Seed Grouping)

**File**: `/home/jkane/RustroverProjects/FerrousAlign/src/alignment/chaining.rs` (1,086 lines)

**Key Functions**:
- `chain_seeds()` - Lines 121-264 - B-tree based chaining
- `test_and_merge()` - Lines 47-117 - Attempts to merge seed into existing chain
- `filter_chains()` - Lines 275-498 - Chain filtering with weight-based selection
- `calculate_chain_weight()` - Computes weight from seed coverage

**Data Structures**:
```rust
pub struct Chain {
    score: i32,                // Sum of seed lengths (alignment score)
    seeds: Vec<usize>,         // Indices into seeds array
    query_start: i32,          // Bounds on query/reference
    query_end: i32,
    ref_start: u64,
    ref_end: u64,
    is_rev: bool,
    weight: i32,               // Seed coverage (sum of seed lengths)
    kept: i32,                 // Status: 0=discarded, 1=shadowed, 2=partial, 3=primary
    frac_rep: f32,             // Fraction of repetitive seeds
    rid: i32,                  // Chromosome ID
    last_qbeg: i32,            // Last seed info for test_and_merge
    last_rbeg: u64,
    last_len: i32,
}
```

**Critical Decision Points**:

1. **Mergeability (test_and_merge, lines 47-117)**:
   - Line 55: Different chromosome → reject
   - Line 65-72: Seed fully contained in chain → accept (no-op)
   - Line 74-81: Different strands → reject
   - Line 92-97: Check diagonal band constraint + gap constraints
   
   **OPTIMIZATION OPPORTUNITY**: This logic runs O(n²) for unoptimized chaining. Current B-tree approach is good, but constraint verification could be cached.

2. **Weight Calculation (calculate_chain_weight, lines 286-304)**:
   - Sum of seed lengths weighted by repetitiveness
   - `frac_rep = l_rep / query_length` - fraction of seeds in repetitive regions
   - **FRONT-LOADING IDEA**: Compute repetitiveness in Stage 1 when seeds are created

3. **Filtering by Drop Ratio (lines 324-420)**:
   - Line 383: `overlap >= (min_len * mask_level)` - significant overlap threshold
   - Line 389-399: `drop_ratio` only applies to overlapping chains
   - Non-overlapping chains kept regardless of score
   
   **Key insight**: Filtering is **score-independent** here - done based on weight/overlap

**Key Properties**:
- Only chains with `weight >= min_chain_weight` survive
- Overlapping chains filtered by drop_ratio (multiplicative threshold)
- Each chain gets `frac_rep` metric for later MAPQ calculation
- Strand information preserved (forward vs reverse)

---

## Stage 3: Extension (Smith-Waterman Alignment)

**Files**: 
- `/home/jkane/RustroverProjects/FerrousAlign/src/alignment/extension.rs` (940 lines)
- `/home/jkane/RustroverProjects/FerrousAlign/src/alignment/pipeline.rs` (lines 1113-1379)

**Key Functions**:
- `extend_chains_to_alignments()` - Lines 1113-1379 in pipeline.rs
  - Creates LEFT and RIGHT alignment jobs (separate extensions on each side of seed)
  - Routes to execute_adaptive_alignments() or execute_scalar_alignments()
  - **NOW INCLUDES CIGAR MERGING** (moved from finalization in Phase 2)
- `execute_adaptive_alignments()` - extension.rs lines 85-186
- `merge_cigars_for_chains()` - pipeline.rs lines 119-294 ✨ **FRONT-LOADED**

**Data Structures**:
```rust
pub struct AlignmentJob {
    seed_idx: usize,
    query: Vec<u8>,
    target: Vec<u8>,
    band_width: i32,
    query_offset: i32,
    direction: Option<ExtensionDirection>,  // LEFT or RIGHT
    seed_len: i32,                          // h0 = initial alignment score
}

struct MergedChainResult {  // ✨ Phase 2 refactoring
    chain_idx: usize,
    cigar: Vec<(u8, i32)>,  // Merged CIGAR: left + seed + right
    score: i32,
    // Reference location (Phase 3 - coordinate conversion in this stage)
    ref_name: String,
    ref_id: usize,
    chr_pos: u64,           // 0-based chromosome position (NOT FM-index)
    fm_index_pos: u64,
    fm_is_rev: bool,
    query_start: i32,       // Query bounds from CIGAR analysis
    query_end: i32,
}
```

**Critical Decision Points**:

1. **Job Creation (lines 1137-1267)**:
   - Line 1160-1164: Calculate left/right extension margins based on seed position
   - Line 1177-1185: Get reference sequence segment (rseq)
   - Line 1193-1223: Create LEFT job if seed.query_pos > 0
   - Line 1225-1257: Create RIGHT job if seed.query_end < query_len

2. **Score Computation (lines 1173-1181)**:
   - Merges left extension score + seed score + right extension score
   - Must avoid double-counting seed (h0 parameter)
   - Line 177-181: Score logic depends on which extensions exist

3. **Coordinate Conversion ✨ (lines 245-276)**:
   - **FRONT-LOADED** from finalization in Phase 3
   - FM-index position → chromosome position
   - Line 258-266: Calls `fm_to_chromosome_coords()`
   - Result: `ref_name`, `ref_id`, `chr_pos` already computed

4. **CIGAR Analysis (lines 223-243)**:
   - Calculate `query_start` and `query_end` from CIGAR clips
   - Sum leading/trailing soft/hard clips

**Key Properties**:
- LEFT extension reverses both query and target (C++ bwamem.cpp:2278)
- RIGHT extension uses forward direction
- Each chain can produce multiple seeds, each with own extensions
- **MAJOR OPTIMIZATION**: CIGAR merging moved here from finalization
  - Reduces data flowing to finalization
  - Enables early filtering before NM/MD computation

---

## Stage 4: Finalization (Scoring, Dedup, SAM Output)

**File**: `/home/jkane/RustroverProjects/FerrousAlign/src/alignment/finalization.rs` (1,857 lines)

**Key Functions**:
- `finalize_alignments()` - pipeline.rs lines 1391-1421
- `build_candidate_alignments()` - pipeline.rs lines 1431-1567 ✨ **Simpler after Phase 2**
- `finalize_candidates()` - pipeline.rs lines 1580-1679
- `mark_secondary_alignments()` - finalization.rs (complex logic)
- `remove_redundant_alignments()` - finalization.rs

**Data Structures**:
```rust
struct CandidateAlignment {  // ✨ Phase 1: Intermediate type
    ref_id: usize,
    ref_name: String,
    pos: u64,
    strand_rev: bool,
    cigar: Vec<(u8, i32)>,
    score: i32,
    query_start: i32,
    query_end: i32,
    seed_coverage: i32,
    frac_rep: f32,
    hash: u64,
    md_tag: String,           // Computed here
    nm: i32,                  // Computed here
    chain_id: usize,
}

pub struct Alignment {
    query_name: String,
    flag: u16,                // SAM flags
    ref_name: String,
    ref_id: usize,
    pos: u64,
    mapq: u8,
    score: i32,
    cigar: Vec<(u8, i32)>,
    rnext: String,
    pnext: u64,
    tlen: i32,
    seq: String,
    qual: String,
    tags: Vec<(String, String)>,
    // Internal fields for ranking
    query_start: i32,
    query_end: i32,
    seed_coverage: i32,
    hash: u64,
    frac_rep: f32,
}
```

**Critical Decision Points**:

1. **NM/MD Computation (pipeline.rs lines 1459-1521)**:
   - Line 1459-1478: Fetch FORWARD reference at chromosome coordinates
   - Line 1491-1507: Extract query portion (handles reverse strand inversion)
   - Line 1510: Call `edit_distance::compute_nm_and_md()`
   
   **FRONT-LOADING OPPORTUNITY**: This could happen in Stage 3 immediately after CIGAR merge
   
   **Reverse Strand Complexity**:
   ```rust
   if merged.fm_is_rev {
       let read_len = encoded_query.len();
       let orig_start = read_len - merged.query_end as usize;
       let orig_end = read_len - merged.query_start as usize;
       // Take portion from original coords and revcomp
       encoded_query[orig_start..orig_end]
           .iter()
           .rev()
           .map(|&b| 3 - b)
           .collect()
   }
   ```

2. **Sorting and Hash Assignment (lines 1550-1564)**:
   - Line 1554-1559: Sort by `(score DESC, ref_pos ASC, query_start ASC)`
   - Line 1562-1564: Assign hash based on sorted order
   - **Purpose**: Deterministic tie-breaking for equal scores
   
   **OPTIMIZATION**: Hash could be computed in Stage 3 after scoring

3. **Score Filtering (pipeline.rs lines 1607-1615)**:
   - Line 1608: `alignments.retain(|a| a.score >= opt.t)`
   - **Threshold**: opt.t (default varies, matches C++ bwa-mem2)
   - **Note**: Skipped for paired-end (deferred to after mate rescue)
   
   **FRONT-LOADING**: Could filter before NM/MD computation!

4. **Redundancy Removal (finalization.rs lines 1617-1626)**:
   - Line 1620: Call `remove_redundant_alignments()`
   - Removes alignments with >95% overlap on both ref and query
   
   **Logic** (lines 292-351 in finalization.rs):
   ```rust
   // Compare each alignment against all kept alignments
   // Mark as redundant if:
   // - Overlaps >95% on query
   // - Overlaps >95% on reference
   // - Keep higher-scoring alignment
   ```

5. **Secondary/Supplementary Marking (finalization.rs lines 1631-1635)**:
   - Call `mark_secondary_alignments()`
   - Complex logic for distinguishing overlapping (SECONDARY) vs non-overlapping (SUPPLEMENTARY)
   - Also generates XA/SA tags
   
   **Logic** (finalization.rs lines 576-836):
   - Overlapping alignments: mark primary as 0, rest as 0x100 (SECONDARY)
   - Non-overlapping: mark as 0x800 (SUPPLEMENTARY)
   - MAPQ calculation: based on score gap and alignment quality

6. **Unmapped Record Creation (lines 1646-1676)**:
   - If all alignments filtered: create placeholder unmapped record
   - Ensures SAM spec compliance (all reads must appear in output)

**Key Properties**:
- NM/MD computation requires reference sequence - **cannot move much earlier**
- But could move immediately after CIGAR merging instead of bundling with scoring
- Filtering is multi-stage: score → redundancy → secondary marking
- Secondary marking is **deferred for paired-end** to allow mate-based ranking

---

## File-by-File Breakdown

| File | Lines | Purpose | Key Decision Points |
|------|-------|---------|-------------------|
| **pipeline.rs** | 2,098 | Main 4-stage pipeline | align_read() entry, merge_cigars_for_chains(), finalize_alignments() |
| **finalization.rs** | 1,857 | Score/secondary marking | mark_secondary_alignments(), remove_redundant_alignments() |
| **banded_swa.rs** | 3,713 | Smith-Waterman (scalar) | CIGAR generation, diagonal band tracking |
| **seeding.rs** | 1,160 | SMEM generation | 3-phase FM-index search, filtering |
| **chaining.rs** | 1,086 | Seed grouping | test_and_merge(), filter_chains() |
| **region.rs** | 1,292 | Deferred CIGAR (experimental) | extend_chains_to_regions() - Phase 5 CIGAR generation |
| **ksw_affine_gap.rs** | 1,595 | Scalar fallback | Affine-gap SW (non-SIMD) |
| **banded_swa_avx2.rs** | 585 | SIMD 256-bit | 32-way parallel alignment |
| **banded_swa_avx512.rs** | 585 | SIMD 512-bit | 64-way parallel alignment (experimental) |
| **extension.rs** | 940 | Job dispatch | execute_adaptive_alignments(), batching strategy |
| **edit_distance.rs** | 405 | NM/MD computation | compute_nm_and_md() - **authoritative** |
| **coordinates.rs** | 175 | Coordinate conversion | fm_to_chromosome_coords() - canonical implementation |
| **cigar.rs** | 305 | CIGAR manipulation | normalize() - merge adjacent ops |
| **mem_opt.rs** | 890 | Parameter handling | mem_opt defaults, validation |
| **mem.rs** | 269 | CLI integration | Argument parsing, index loading |
| **single_end.rs** | 230 | Read batching | Rayon parallel processing |
| **utils.rs** | 86 | Helper functions | base_to_code(), reverse_complement_code() |
| **mod.rs** | 24 | Module exports | Public API |

---

## Current Pipeline Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: SEEDING (find_seeds, 420-1019)                         │
│ - 3 SMEM passes: initial + re-seeding + 3rd-round              │
│ - Filtering: min_seed_len, max_occ, effective_max_occ         │
│ - Output: ~500 Seed structs (with ref_pos, query_pos, len)    │
│ - NO DECISIONS: all filtering is conservative                  │
└────────────────┬────────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────────┐
│ Stage 2: CHAINING (build_and_filter_chains, 1022-1087)         │
│ - B-tree O(n log n) merging: test_and_merge()                 │
│ - Weight calculation: sum of seed lengths                      │
│ - Filtering: min_chain_weight, drop_ratio, overlap             │
│ - Output: ~20-50 Chain structs with weight/frac_rep           │
│ - DECISIONS: min_weight, drop_ratio thresholds                │
└────────────────┬────────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────────┐
│ Stage 3: EXTENSION (extend_chains_to_alignments, 1113-1379)    │
│ - Create alignment jobs: LEFT and RIGHT extensions             │
│ - Execute SIMD/scalar banded SW for each job                   │
│ ✨ NEW: CIGAR MERGE + COORDINATE CONVERSION (Phase 2)         │
│ - Output: ~10-20 MergedChainResult structs                     │
│   * WITH: merged CIGAR, chromosome coordinates                 │
│   * WITH: query_start/query_end (from CIGAR clips)            │
│ - NO NM/MD: deferred to next stage                             │
└────────────────┬────────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────────┐
│ Stage 4: FINALIZATION (finalize_alignments, 1391-1679)         │
│ - Build candidates: compute NM/MD (AFTER coordinate conv)      │
│ - Sort by (score DESC, pos ASC, qpos ASC) + assign hash       │
│ - Filter by score: opt.t threshold                             │
│ - Remove redundant: >95% overlap on query+ref                  │
│ - Mark secondary/supplementary: overlapping logic              │
│ - Calculate MAPQ: based on score gap                           │
│ - Generate XA/SA tags: alternative alignment strings           │
│ - Output: final Alignment structs ready for SAM                │
└──────────────────────────────────────────────────────────────────┘
```

---

## Front-Loading Opportunities (Ranked by Impact)

### **PRIORITY 1: Move Score Filtering Earlier** (15-20% improvement)
**Before**: Finalization stage (line 1608) - after NM/MD computation
**After**: After Stage 3 merging, before NM/MD
**Impact**: Skip NM/MD computation (10-20% of work) for ~30-40% of chains

**Implementation**:
```rust
// In extend_chains_to_alignments() after merge_cigars_for_chains()
let mut merged = merge_cigars_for_chains(...);
merged.retain(|m| m.score >= opt.t);  // Early filter!
```

### **PRIORITY 2: Move Redundancy Detection Earlier** (10-15% improvement)
**Before**: Finalization stage (line 1620)
**After**: After Stage 3, using only CIGAR + coordinates
**Impact**: Remove duplicate work before NM/MD

**Implementation**:
```rust
// Redundancy check only needs: ref bounds, query bounds, CIGAR
// Can run on MergedChainResult before NM/MD
let filtered = remove_redundant_by_coords(&merged_results);
```

### **PRIORITY 3: Early Hash Assignment** (5% improvement)
**Before**: After sorting in finalization (line 1562)
**After**: After Stage 3 scoring, before sorting
**Impact**: Reduce late-stage data movement

**Implementation**:
```rust
// In extend_chains_to_alignments():
// Pre-assign hash based on chain index and score
for (i, merged) in merged_results.iter_mut().enumerate() {
    merged.hash = hash_64(read_id + i as u64);
}
```

### **PRIORITY 4: Separate SMEM Filtering from Seed Conversion** (3-5% improvement)
**Before**: Mixed logic in find_seeds() (lines 798-819)
**After**: Two passes - filter SMEMs, then convert to Seeds
**Impact**: Cleaner code, easier to optimize

**Implementation**:
```rust
// Pass 1: Filter SMEMs
let filtered_smems = filter_smems(all_smems, opt);
// Pass 2: Convert to Seeds
let seeds = smems_to_seeds(filtered_smems);
```

### **PRIORITY 5: Cache Repetitiveness in Stage 1** (2-3% improvement)
**Before**: Calculated in Stage 2 (line 288)
**After**: Computed when Seed created
**Impact**: Avoid recalculation, better cache locality

**Implementation**:
```rust
// In Seed struct:
pub struct Seed {
    ...
    interval_size: u64,     // Already here - just use for frac_rep
    is_repetitive: bool,    // Add this for quick filtering
}
```

---

## Critical Code Patterns

### **Pattern 1: Coordinate Conversion Centralization**
**File**: `coordinates.rs` (175 lines)
**Function**: `fm_to_chromosome_coords()` - lines 77-121

**Key Insight**: Canonical implementation prevents bugs from duplicated logic
- FM-index position (0 to 2*l_pac) → chromosome position
- Handles reverse strand via `re - 1` trick
- Used in: pipeline.rs:258-266 ✨ (NOW in Stage 3!)

### **Pattern 2: Edit Distance Consolidation**
**File**: `edit_distance.rs` (405 lines)
**Function**: `compute_nm_and_md()` - lines 57-150

**Key Insight**: Single pass computes both NM and MD
- Replaces deprecated methods in Alignment struct
- Requires ref_seq + query_seq + CIGAR
- Currently called in finalization (line 1510)
- **Could be moved to Stage 3** if reference data available

### **Pattern 3: CIGAR Merging Pipeline**
**File**: `pipeline.rs` (lines 119-294)
**Function**: `merge_cigars_for_chains()`

**Key Insight**: ✨ **ALREADY FRONT-LOADED** in Phase 2
- Was in finalization, moved to extension stage
- Combines left + seed + right CIGARS
- Normalizes with `cigar::normalize()`
- Includes score calculation with h0 (seed_len) handling

### **Pattern 4: Score-Only vs Full Alignment**
**File**: `pipeline.rs` (lines 1697-1977)
**Function**: `align_read_deferred()` - experimental deferred CIGAR pipeline

**Key Insight**: Scores can be computed before CIGAR!
- Phase 3: Score-only extension → AlignmentRegion
- Phase 4: Filter by scores (eliminates 80-90% of work)
- Phase 5: CIGAR generation for survivors only
- Expected 3-5x speedup but requires `extend_chains_to_regions()`

---

## Data Flow and Memory Patterns

### **Memory Allocation Points**:
1. Stage 1: `Vec<Seed>` grows from 0 → ~500
2. Stage 2: `Vec<Chain>` shrinks 500 → ~20-50 (heavy filtering)
3. Stage 3: `Vec<MergedChainResult>` stays ~20-50
4. Stage 4: `Vec<Alignment>` stays ~20-50 (add NM/MD strings)

**Optimization Opportunities**:
- Pre-allocate seeds vector with capacity (~500)
- **Don't pre-allocate** candidates - filter heavily first
- NM/MD strings: average 20-100 chars, total ~1-2KB per read

### **Reference Data Access Patterns**:
1. **FM-Index**: Accessed in Stage 1 (dense random access, high cache miss rate)
2. **Reference Sequence**: Accessed in Stage 3 (sequential per chain, good locality)
3. **Reference for MD/NM**: Accessed in Stage 4 (sequential, same as CIGAR regions)

**Optimization Insight**: Stage 4 could pipeline with Stage 3 by buffering reference segments

---

## Deferred CIGAR Architecture (Experimental)

**Status**: Implemented but marked `#[allow(dead_code)]`

**File**: `pipeline.rs` (lines 1697-1977)

**Key Phases**:
1. **Seeding**: Same as standard pipeline
2. **Chaining**: Same as standard pipeline
3. **Score-only Extension**: Returns AlignmentRegion with score but NO CIGAR
4. **Region Filtering**: Filter by score (opt.t threshold)
5. **Redundancy Removal**: Remove overlapping regions
6. **CIGAR Generation**: Call `generate_cigar_from_region()` for survivors only

**Expected Benefits**:
- 3-5x faster: Skip CIGAR computation for 80-90% of alignments
- Requires: `region::extend_chains_to_regions()` + `region::generate_cigar_from_region()`

**Current Blocker**: Region module not fully integrated with pipeline

---

## Known Issues and TODOs

### **Correctness Issues**:
1. **Proper pairing rate gap**: 90.28% vs 97.11% (6.83% gap) on HG002 10K test
   - Root cause: Insert size estimation or pair scoring differences
   - **Not related to this module** (paired-end logic is separate)

### **Performance TODOs**:
1. Faster suffix array reconstruction (cache recent lookups)
2. Streaming mode for paired-end (avoid buffering all alignments)
3. Vectorize backward_ext() for FM-Index search
4. Stabilize AVX-512 (waiting on Rust compiler stabilization)

### **Code Quality TODOs**:
1. Remove deprecated methods from Alignment struct
2. Consolidate all MAPQ calculation to single location
3. Add integration tests for coordinate conversion edge cases

---

## Integration Points for Heterogeneous Compute

**GPU Dispatch Point**: `extend_chains_to_alignments()`, lines 1303-1328
```rust
let effective_backend = compute_backend.effective_backend();
let extended_cigars: Vec<RawAlignment> = match effective_backend {
    ComputeBackend::CpuSimd(_) => {
        if alignment_jobs.len() >= 8 {
            execute_adaptive_alignments(&sw_params, &alignment_jobs)
        } else {
            execute_scalar_alignments(&sw_params, &alignment_jobs)
        }
    }
    ComputeBackend::Gpu => {
        // TODO: execute_gpu_alignments()
        execute_adaptive_alignments(&sw_params, &alignment_jobs)
    }
    ComputeBackend::Npu => {
        // TODO: NPU seed pre-filtering
        execute_adaptive_alignments(&sw_params, &alignment_jobs)
    }
}
```

**NPU Integration**: Seed pre-filtering before job creation (lines 1140-1267)
- Use `compute::encoding::EncodingStrategy::OneHot` for NPU-friendly format
- Would filter seeds BEFORE alignment job creation

---

## Summary Table: Stage Comparison

| Aspect | Stage 1 (Seeding) | Stage 2 (Chaining) | Stage 3 (Extension) | Stage 4 (Finalization) |
|--------|-------------------|-------------------|-------------------|----------------------|
| **Input** | Query sequence | 500 Seeds | 50 Chains | 50 MergedChainResults |
| **Output** | 500 Seeds | 50 Chains | 50 MergedChainResults | 3-5 Alignments |
| **Compute** | FM-Index search | Weight calc + overlap | Smith-Waterman + CIGAR merge | NM/MD + secondary marking |
| **Memory** | ~100KB (index) | ~50KB (chains) | ~500KB (sequences) | ~50KB (tags) |
| **Time** | 30-40% | 5-10% | 40-50% | 10-20% |
| **Filtering** | Conservative | Weight-based | **(OPPORTUNITY)** | Score-based |
| **Decisions** | Few | Weight/overlap | **(MOVED HERE)** | Ranking/MAPQ |

---

## Conclusions

1. **Architecture is well-structured**: Clear 4-stage pipeline with good separation of concerns
2. **Phase 2 refactoring partially done**: CIGAR merging + coordinate conversion moved to Stage 3 ✅
3. **Major opportunity in Stage 4**: Score filtering, redundancy detection, NM/MD computation could be spread across earlier stages
4. **Deferred CIGAR pipeline available**: If fully integrated, could give 3-5x speedup for score-only path
5. **Heterogeneous compute ready**: GPU/NPU integration points already in place, just need implementations

