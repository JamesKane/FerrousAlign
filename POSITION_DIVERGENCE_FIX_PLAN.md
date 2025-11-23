# Position Divergence Fix Plan

## Problem Statement

FerrousAlign produces biologically correct alignments but selects different primary alignments than BWA-MEM2 in ~96% of cases where both tools find alignments at the same genomic location on opposite strands. This erodes user confidence even though the alignments are scientifically equivalent.

### Root Cause Analysis

When a read aligns equally well to both strands at the same location (common for palindromic or low-complexity regions), both tools:
1. Find both alignments with identical scores
2. Use hash-based tie-breaking to select the primary
3. **But select opposite strands** due to different alignment generation order

The hash is computed as `hash_64(read_id + alignment_index)` where `alignment_index` is the position in the alignment array at generation time. Since alignments are generated in different orders, the same genomic alignment gets different hash values.

### Evidence

```
Read: HISEQ1:18:H8VC6ADXX:1:1101:2101:2190
Ferrous: chr7:67600541 flag=99 (forward) AS:143
BWA-MEM2: chr7:67600394 flag=83 (reverse) AS:143
Position difference: 147bp (read_len - 1) = same location, opposite strand
```

## BWA-MEM2 Alignment Generation Order

### 1. SMEM Generation (`FMI_search.cpp`)

BWA-MEM2 generates SMEMs in a specific order:

```cpp
// FMI_search.cpp: getSMEMs()
// For each position in the query:
//   1. Extend backward (left) as far as possible
//   2. Record SMEM interval
//   3. Move to next position

// SMEMs are stored in order of query position (left to right)
```

**Key insight**: Forward strand SMEMs are generated first (query positions 0→len), then the query is reverse-complemented and searched again.

### 2. Seed Sorting (`bwamem.cpp`)

After SMEM generation, seeds are sorted:

```cpp
// bwamem.cpp:mem_chain() calls:
ks_introsort(mem_seed, n, seeds);

// mem_seed comparison (from bwamem.cpp):
#define mem_seed_cmp(a, b) ((a).rbeg < (b).rbeg)
```

Seeds are sorted by **reference position** (rbeg), not query position.

### 3. Chain Building (`bwamem.cpp`)

Chains are built from sorted seeds:

```cpp
// For each seed (in rbeg order):
//   Try to add to existing chain
//   If no compatible chain, start new chain
```

### 4. Alignment Generation (`bwamem.cpp`)

Alignments are generated per-chain:

```cpp
// bwamem.cpp:mem_chain2aln()
// For each chain (in chain array order):
//   Generate alignment via Smith-Waterman
//   Store in regs array
```

### 5. Dedup and Sort (`bwamem.cpp:292-353`)

```cpp
// mem_sort_dedup_patch():
ks_introsort(mem_ars2, n, a);  // Sort by reference END position (re)
// ... dedup logic ...
ks_introsort(mem_ars, n, a);   // Sort by score desc, then rb, then qb
```

### 6. Primary Marking (`bwamem.cpp:1420-1460`)

```cpp
// mem_mark_primary_se():
for (i = 0; i < n; ++i) {
    a[i].hash = hash_64(id + i);  // Hash based on current position
}
ks_introsort(mem_ars_hash, n, a);  // Sort by score, is_alt, hash
// ... mark secondaries ...
```

**Critical**: Hash is assigned BEFORE the final sort, based on position in the array AFTER `mem_sort_dedup_patch()`.

## FerrousAlign Current Order

### Current Flow (`alignment/pipeline.rs`)

```rust
// 1. find_seeds(): Generate SMEMs
//    - Forward strand searched
//    - Reverse complement searched
//    - Seeds converted to reference positions

// 2. build_and_filter_chains(): Chain seeds
//    - Seeds sorted by (ref_pos, query_pos)
//    - Chains built via DP

// 3. extend_chains_to_alignments(): SW extension
//    - For each chain, generate alignment

// 4. build_candidate_alignments(): Build candidates
//    - Hash assigned here: hash_64(read_id + candidates.len())
//    - candidates.len() = current position in output array
```

### Key Differences

| Stage | BWA-MEM2 | FerrousAlign |
|-------|----------|--------------|
| SMEM order | Forward first, then RC | Forward first, then RC (**same**) |
| Seed sort | By `rbeg` only | By `(ref_pos, query_pos)` (**different**) |
| Chain order | Based on seed order | Based on seed order |
| Dedup sort | By `re`, then by score/rb/qb | By score, then hash (**different**) |
| Hash assignment | After dedup, before final sort | During candidate building |

## Fix Plan

### Phase 1: Seed Sorting Alignment (Estimated: 2-4 hours)

**Goal**: Match BWA-MEM2's seed sorting order.

**File**: `src/alignment/chaining.rs`

**Changes**:
1. Sort seeds by `ref_pos` only (not `(ref_pos, query_pos)`)
2. Verify chain building produces same chains

**Verification**:
```bash
# Add debug logging to compare seed order
RUST_LOG=debug cargo run -- mem ref.fa test.fq 2>&1 | grep "SEED:"
```

### Phase 2: Dedup Sort Order (Estimated: 4-6 hours)

**Goal**: Match BWA-MEM2's `mem_sort_dedup_patch()` behavior.

**File**: `src/alignment/finalization.rs`

**Changes**:
1. Add intermediate sort by reference END position (`re`) before dedup
2. After dedup, sort by score desc, then `rb`, then `qb`
3. Move hash assignment to AFTER dedup sort

**Current** (`remove_redundant_alignments`):
```rust
// No intermediate sort by re
// Dedup based on overlap
// Sort by score, then hash
```

**Target**:
```rust
// Step 1: Sort by re (reference end)
alignments.sort_by_key(|a| a.pos + cigar_ref_len(&a.cigar));

// Step 2: Dedup with overlap logic

// Step 3: Sort by score desc, rb, qb
alignments.sort_by(|a, b| {
    b.score.cmp(&a.score)
        .then_with(|| a.pos.cmp(&b.pos))
        .then_with(|| a.query_start.cmp(&b.query_start))
});

// Step 4: Assign hash based on position in sorted array
for (i, aln) in alignments.iter_mut().enumerate() {
    aln.hash = hash_64(read_id + i as u64);
}
```

### Phase 3: Hash Assignment Timing (Estimated: 2-3 hours)

**Goal**: Assign hash at the correct point in the pipeline.

**File**: `src/alignment/pipeline.rs`

**Changes**:
1. Remove hash assignment from `build_candidate_alignments()`
2. Add hash assignment in `finalize_candidates()` after dedup sort
3. Pass `read_id` to `finalize_candidates()`

### Phase 4: Verification and Testing (Estimated: 4-6 hours)

**Goal**: Verify position parity with BWA-MEM2.

**Tests**:
1. Run golden reads test (10K pairs):
   ```bash
   ./target/release/ferrous-align mem -t 4 $REF golden_R1.fq golden_R2.fq > ferrous.sam
   bwa-mem2 mem -t 4 $REF golden_R1.fq golden_R2.fq > bwamem2.sam

   # Compare positions
   grep -v '^@' ferrous.sam | cut -f1,3,4 | sort > ferrous_pos.txt
   grep -v '^@' bwamem2.sam | cut -f1,3,4 | sort > bwamem2_pos.txt
   comm -12 ferrous_pos.txt bwamem2_pos.txt | wc -l  # Should be ~100%
   ```

2. Add regression test:
   ```rust
   #[test]
   fn test_alignment_order_matches_bwamem2() {
       // Test specific reads known to have tie-break scenarios
   }
   ```

### Phase 5: Edge Cases (Estimated: 2-4 hours)

**Goal**: Handle edge cases that may differ.

**Cases to verify**:
1. Reads with >2 equal-score alignments
2. Supplementary alignment ordering
3. ALT contig handling (is_alt flag)
4. Reads spanning chromosome boundaries

## Implementation Order

```
Phase 2 (Dedup Sort) → Phase 3 (Hash Timing) → Phase 1 (Seed Sort) → Phase 4 (Verify) → Phase 5 (Edge Cases)
```

**Rationale**: Phase 2 and 3 have the highest impact on position selection. Phase 1 may not be necessary if 2+3 achieve parity.

## Estimated Total Effort

| Phase | Hours | Priority |
|-------|-------|----------|
| Phase 2: Dedup Sort | 4-6 | High |
| Phase 3: Hash Timing | 2-3 | High |
| Phase 1: Seed Sort | 2-4 | Medium |
| Phase 4: Verification | 4-6 | High |
| Phase 5: Edge Cases | 2-4 | Low |
| **Total** | **14-23** | |

## Success Criteria

1. **Position match rate**: >99% of alignments at same position as BWA-MEM2
2. **Score parity**: 100% of alignments have same AS score
3. **Flag parity**: >99% of alignments have same SAM flags
4. **No regression**: All existing tests pass
5. **Performance**: No significant slowdown (<5%)

## Files to Modify

| File | Changes |
|------|---------|
| `src/alignment/finalization.rs` | Dedup sort order, hash assignment |
| `src/alignment/pipeline.rs` | Remove early hash, pass read_id |
| `src/alignment/chaining.rs` | Seed sort order (if needed) |
| `tests/golden_reads/` | Add position parity test |

## References

- BWA-MEM2 source: `bwa-mem2 (Copy)/src/bwamem.cpp:292-353` (dedup)
- BWA-MEM2 source: `bwa-mem2 (Copy)/src/bwamem.cpp:1420-1460` (primary marking)
- Current hash fix: Session 31 (this session)
- Previous research: `dev_notes/ALIGNMENT_DISCREPANCY_RESEARCH.md`

## Notes

- The biological correctness of alignments is NOT affected by this fix
- This is purely for user confidence and reproducibility with BWA-MEM2
- Some edge cases may never achieve 100% parity due to floating-point or algorithmic differences
