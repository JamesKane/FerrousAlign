# Alignment Module - Quick Reference Guide

## Key Decision Points (Line References)

### Stage 1: Seeding (`seeding.rs`)
| Decision | Line | Threshold | Impact |
|----------|------|-----------|--------|
| Min seed length | 806 | opt.min_seed_len | Rejects short matches |
| Max seed occurrence | 806 | opt.max_occ | Rejects repetitive regions |
| Seed/chromosome boundary | 912-918 | rid >= 0 | Rejects cross-chromosome seeds |
| Per-read limit | 854 | 500 seeds | Prevents memory explosion |

### Stage 2: Chaining (`chaining.rs`)
| Decision | Line | Threshold | Impact |
|----------|------|-----------|--------|
| Merge constraint | 92-97 | diagonal band + gaps | Determines which seeds chain |
| Min chain weight | 331 | opt.min_chain_weight | Rejects low-coverage chains |
| Overlap detection | 363 | qe_min > qb_max | Identifies competing chains |
| Drop ratio | 389-399 | opt.drop_ratio | Rejects weak overlapping chains |

### Stage 3: Extension (`pipeline.rs:1113-1379`)
| Decision | Line | Purpose | Impact |
|----------|------|---------|--------|
| CIGAR merge | 119-294 | Combine left+seed+right | Produces merged CIGAR ✨ |
| Coordinate conversion | 258-266 | FM-index → chromosome | Produces chr_pos ✨ |
| Score calculation | 173-181 | h0 handling for extensions | Produces final score |
| **(OPPORTUNITY)** | **1608** | **Score filtering** | **Skip NM/MD for ~30-40%** |
| **(OPPORTUNITY)** | **1620** | **Redundancy removal** | **Detect overlaps early** |

### Stage 4: Finalization (`pipeline.rs:1391-1679`)
| Decision | Line | Threshold | Impact |
|----------|------|-----------|--------|
| NM/MD computation | 1510 | (no threshold) | ~10-20% of work |
| Sorting | 1554-1559 | (score, pos, qpos) | Deterministic ranking |
| Hash assignment | 1562 | read_id + index | Tie-breaking |
| Score filter | 1608 | opt.t | Rejects low-score aligns |
| Redundancy removal | 1620 | 95% overlap | Removes duplicates |
| Secondary marking | 1635 | overlapping logic | Sets SAM flags |

---

## Data Structures and Their Flow

```
STAGE 1                    STAGE 2                STAGE 3              STAGE 4
───────────────────────────────────────────────────────────────────────────────

Seed struct                Chain struct          MergedChainResult    Alignment struct
├─ query_pos              ├─ score              ├─ cigar ✨           ├─ all from MergedChainResult
├─ ref_pos                ├─ seeds (indices)    ├─ score              ├─ md_tag (computed)
├─ len                    ├─ weight             ├─ ref_name ✨        ├─ nm (computed)
├─ is_rev                 ├─ frac_rep           ├─ ref_id ✨          ├─ mapq (calculated)
├─ interval_size          ├─ rid                ├─ chr_pos ✨         ├─ flag (calculated)
└─ rid                    ├─ query/ref bounds   ├─ fm_is_rev          └─ tags (XA/SA)
                          └─ is_rev             ├─ query_start ✨
                                                └─ query_end ✨
```

---

## Key Files and Their Responsibilities

### Critical Path Files (in order of execution)
1. **seeding.rs** - Find seed matches (30-40% of time)
2. **chaining.rs** - Group seeds into chains (5-10% of time)
3. **pipeline.rs (Stage 3)** - Smith-Waterman alignment (40-50% of time)
   - Calls extension.rs for job dispatch
   - Calls edit_distance.rs for NM/MD
4. **finalization.rs** - Secondary marking + SAM flags (10-20% of time)

### Key Support Files
- **coordinates.rs** - Canonical coordinate conversion (FM-index → chromosome)
- **edit_distance.rs** - Authoritative NM/MD computation
- **cigar.rs** - CIGAR manipulation and normalization
- **banded_swa.rs** - Scalar Smith-Waterman (contains SIMD variants)
- **extension.rs** - Job dispatch and adaptive batching

---

## Important Constants and Thresholds

### Seeding
```rust
const SEEDS_PER_READ: usize = 500;           // Hard limit on seeds
opt.min_seed_len                              // Default: varies
opt.max_occ                                   // Default: varies
opt.max_mem_intv                              // 3rd-round seeding threshold
```

### Chaining
```rust
opt.min_chain_weight                          // Minimum seed coverage
opt.drop_ratio                                // Weak chain threshold
opt.mask_level                                // Overlap percentage for redundancy
const MAX_SEEDS_PER_READ: usize = 100_000;   // Runaway guard
const MAX_CHAINS_PER_READ: usize = 10_000;   // Runaway guard
```

### Extension
```rust
opt.w                                         // Band width for SW
opt.o_del, opt.e_del                         // Gap open/extend penalties
opt.o_ins, opt.e_ins                         // Gap open/extend penalties
opt.a, opt.b                                  // Match/mismatch scores
```

### Finalization
```rust
opt.t                                         // Score threshold (filters ~30-40% of aligns)
mask_level (0.95)                             // Overlap threshold for redundancy
```

---

## Optimization Targets (Priority Order)

### PRIORITY 1: Early Score Filtering (15-20% improvement)
**Location**: pipeline.rs:1608
**Current**: Filters AFTER NM/MD computation
**Proposed**: Filter immediately after merge_cigars_for_chains()
**Expected Gain**: Skip NM/MD for ~30-40% of chains

```rust
// Add after line 1369:
let mut merged = merge_cigars_for_chains(...);
merged.retain(|m| m.score >= opt.t);  // EARLY FILTER!
```

### PRIORITY 2: Early Redundancy Detection (10-15% improvement)
**Location**: pipeline.rs:1620
**Current**: Filters in finalization with full Alignment structs
**Proposed**: Filter in Stage 3 using only coordinates + CIGAR bounds
**Expected Gain**: Avoid building CandidateAlignment for ~10-20%

```rust
// Add after early score filtering:
let filtered = remove_redundant_by_coords(&merged);
```

### PRIORITY 3: Precompute Hash (5% improvement)
**Location**: pipeline.rs:1562
**Current**: Assigned after sorting in finalization
**Proposed**: Assign in Stage 3 based on chain order
**Expected Gain**: Reduce late-stage data movement

### PRIORITY 4: Deferred CIGAR (3-5x improvement - experimental)
**Location**: pipeline.rs:1697-1977
**Status**: Implemented but not integrated
**Requires**: Completing region.rs integration
**Expected Gain**: Skip CIGAR for 80-90% of alignments

---

## Common Pitfalls

### Pitfall 1: Reverse Strand Coordinate Confusion
**File**: coordinates.rs:77-121
**Issue**: For reverse strand, must use `re - 1` not `rb` for correct SAM position
**Pattern**: 
```rust
let depos_input = if is_rev { re.saturating_sub(1) } else { rb };
```

### Pitfall 2: Seed Length vs Score Confusion
**File**: pipeline.rs:173-181
**Issue**: Must not double-count h0 (seed length) when merging extension scores
**Pattern**:
```rust
if !has_left && !has_right {
    combined_score += seed.len;  // Only if no extensions
} else if has_left && has_right {
    combined_score -= seed.len;  // Avoid double-count
}
```

### Pitfall 3: Query Coordinate Extraction for Reverse Strand
**File**: pipeline.rs:1491-1507
**Issue**: For reverse strand alignments, must invert query coordinates
**Pattern**:
```rust
if merged.fm_is_rev {
    let orig_start = read_len - merged.query_end as usize;
    let orig_end = read_len - merged.query_start as usize;
    query_for_md = reverse_and_complement(original_query[orig_start..orig_end]);
}
```

### Pitfall 4: LEFT Extension Direction
**File**: extension.rs:217-223, pipeline.rs:1215-1223
**Issue**: LEFT extension requires reversing both query AND target
**Pattern**:
```rust
let (query, target) = if direction == Some(ExtensionDirection::Left) {
    (query.iter().copied().rev().collect(),
     target.iter().copied().rev().collect())
} else {
    (query.clone(), target.clone())
};
```

---

## Testing Guidance

### Unit Tests by Stage
- **Seeding**: `seeding.rs` - SMEM generation, interval logic
- **Chaining**: `chaining.rs` - test_and_merge, weight calculation
- **Extension**: `banded_swa.rs` - CIGAR generation, score calculation
- **Finalization**: `finalization.rs` - secondary marking, XA/SA tags

### Integration Tests
- `tests/integration_test.rs` - End-to-end single-end
- `tests/paired_end_integration_test.rs` - Insert size + mate rescue
- `tests/complex_integration_test.rs` - Edge cases

### Golden Reads Tests
- Location: `tests/golden_reads/`
- Purpose: Regression testing for pipeline refactoring
- Usage: Compare against frozen baseline for correctness

---

## Coordinate System Reference

### FM-Index Positions (0 to 2*l_pac)
```
[0 ................. l_pac ................. 2*l_pac)
 ↑                    ↑                       ↑
 Forward strand    Boundary         Reverse complement strand
 alignments start   here            alignments are in second half
```

### Chromosome Coordinates (0 to chromosome_len)
```
[0 ................. chromosome_len)
 All positions are on forward strand,
 reverse alignments marked by REVERSE flag
```

### SAM Position (1-based)
```
SAM uses 1-based coordinates:
POS = 0-based position + 1
For unmapped: POS = 0
```

---

## Performance Profiling

### Expected Time Distribution (Single-End)
- Stage 1 (Seeding): 30-40%
- Stage 2 (Chaining): 5-10%
- Stage 3 (Extension): 40-50%
  - Job creation: 5-10%
  - Smith-Waterman: 30-35%
  - CIGAR merge: 5%
- Stage 4 (Finalization): 10-20%
  - NM/MD: 5-10% (largest single component)
  - Secondary marking: 3-5%
  - Other: 2-5%

### Bottleneck Locations
1. **FM-Index search** (Stage 1) - Random memory access
2. **Smith-Waterman** (Stage 3) - Quadratic alignment
3. **NM/MD computation** (Stage 4) - Sequence comparison (10-20% of total)

---

## References

- **Main pipeline**: `pipeline.rs:406-474` (align_read function)
- **Deferred pipeline**: `pipeline.rs:1709-1907` (align_read_deferred - experimental)
- **BWA-MEM2 C++ equivalent**: bwamem.cpp:1398-1544 (mem function)
- **Coordinate conversion**: `coordinates.rs:77-121`
- **Edit distance**: `edit_distance.rs:57-150`

