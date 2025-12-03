# Tie-Breaking Logic Comparison: BWA-MEM2 vs FerrousAlign

**Date**: December 3, 2025
**Finding**: FerrousAlign's tie-breaking is **almost correct** but missing one key component

---

## BWA-MEM2 Sorting Logic

### Primary Sort (bwamem.cpp:1485)

```c
#define alnreg_hlt(a, b)  ((a).score > (b).score || \
                           ((a).score == (b).score && \
                            ((a).is_alt < (b).is_alt || \
                             ((a).is_alt == (b).is_alt && (a).hash < (b).hash))))
```

**Breakdown**:
1. **Primary**: Score (descending - higher is better)
2. **Tie-break 1**: `is_alt` (ascending - non-alt alignments first)
3. **Tie-break 2**: Hash (ascending - deterministic ordering)

**Key insight**: The `is_alt` field prioritizes alignments to the primary assembly over alternate haplotypes.

---

## FerrousAlign Sorting Logic

### Current Sort (redundancy.rs:95)

```rust
alignments.sort_by(|a, b| b.score.cmp(&a.score)
                          .then_with(|| a.hash.cmp(&b.hash)));
```

**Breakdown**:
1. **Primary**: Score (descending)
2. **Tie-break**: Hash (ascending)

**Missing**: No `is_alt` check!

---

## The Problem

When reads map to both primary and alternate contigs with equal scores:
- **BWA-MEM2**: Prefers primary assembly (is_alt=false) over alternates (is_alt=true)
- **FerrousAlign**: Uses hash only, may select alternate when primary exists

### Impact on Pairing

If Read1 selects alternate contig and Read2 selects primary contig (or vice versa):
- Mates map to "different chromosomes" (technically different contigs)
- Pairing fails even though both are valid alignments
- Shows up as "mate mapped to different chr" in flagstat

---

## Additional Sorting Differences

### BWA-MEM2 Position-Based Sort (alnreg_slt)

Used for some operations (bwamem.cpp:152):
```c
#define alnreg_slt(a, b) ((a).score > (b).score || \
                          ((a).score == (b).score && \
                           ((a).rb < (b).rb || \
                            ((a).rb == (b).rb && (a).qb < (b).qb))))
```

**Breakdown**:
1. **Primary**: Score (descending)
2. **Tie-break 1**: Reference begin position `rb` (ascending)
3. **Tie-break 2**: Query begin position `qb` (ascending)

**Use case**: When hash isn't initialized, sort by genomic position instead.

### FerrousAlign's Position Sort (Not Currently Used)

FerrousAlign does have position-based sorting in redundancy.rs:19-23:
```rust
alignments.sort_by(|a, b| {
    let a_ref_end = a.pos + alignment_ref_length(a);
    let b_ref_end = b.pos + alignment_ref_length(b);
    a.ref_id.cmp(&b.ref_id).then_with(|| a_ref_end.cmp(&b_ref_end))
});
```

But this sorts by **END** position, not BEGIN, and is only used for redundancy filtering.

---

## What is `is_alt`?

From BWA-MEM2's perspective:
- `is_alt` = true: Alignment to an alternate contig/haplotype
- `is_alt` = false: Alignment to primary assembly

**In human reference genomes** (like CHM13):
- Primary contigs: chr1, chr2, ..., chrX, chrY, chrM
- Alternate contigs: chr1_KI270706v1_random, HLA-*, etc.

**Why it matters for pairing**:
- If R1 maps to chr1 (primary) and R2 maps to chr1_alt (alternate), they appear as "different chromosomes"
- BWA-MEM2 would prefer both on chr1 (primary) for proper pairing
- FerrousAlign's hash-only tie-breaking might split the pair

---

## The Fix

### Option 1: Add is_alt Field (Full BWA-MEM2 Compatibility)

**Changes needed**:
1. Add `is_alt: bool` to `Alignment` struct
2. Set `is_alt` during finalization (check if ref_name contains "alt", "random", "fix", "HLA")
3. Update sort in redundancy.rs:

```rust
alignments.sort_by(|a, b| {
    b.score.cmp(&a.score)
        .then_with(|| a.is_alt.cmp(&b.is_alt))  // NEW: prefer primary assembly
        .then_with(|| a.hash.cmp(&b.hash))
});
```

**Effort**: 1-2 hours
**Risk**: Low (straightforward addition)

### Option 2: Use Position-Based Tie-Breaking (Simpler)

If `is_alt` detection is complex, use position-based tie-breaking:

```rust
alignments.sort_by(|a, b| {
    b.score.cmp(&a.score)
        .then_with(|| a.ref_id.cmp(&b.ref_id))      // chromosome order
        .then_with(|| a.pos.cmp(&b.pos))             // position within chr
        .then_with(|| a.query_start.cmp(&b.query_start))
});
```

**Rationale**:
- Deterministic ordering based on genomic position
- Primary chromosomes (chr1-22, X, Y) typically have lower IDs than alternates
- Matches BWA-MEM2's `alnreg_slt` comparator

**Effort**: 5 minutes
**Risk**: Very low (no new fields needed)

---

## Which Option?

### Recommendation: **Option 2 (Position-Based)** for immediate fix

**Reasons**:
1. **Faster to implement** - no struct changes needed
2. **Deterministic** - same alignment order every time
3. **Genomically meaningful** - alignments sorted left-to-right on genome
4. **Covers most cases** - primary chrs typically sorted before alternates in ref

**Expected impact on pairing**: Should fix **most** of the 73% tie-breaking cases.

### Follow-up: **Option 1** if Option 2 insufficient

If position-based sorting doesn't fully close the pairing gap:
- Implement full `is_alt` detection
- Requires parsing reference names to identify alternate contigs
- More complete BWA-MEM2 compatibility

---

## Implementation Plan

### Immediate (Today)

1. **Test current sorting**:
   - Add logging to show how alignments are sorted
   - Check if hash values are unique/meaningful

2. **Implement Option 2**:
   - Update sort in `redundancy.rs:95`
   - Add position-based tie-breaking

3. **Benchmark**:
   - Re-run pairing comparison
   - Measure improvement in properly paired rate

### If Needed (Tomorrow)

4. **Implement Option 1**:
   - Add `is_alt` field to `Alignment`
   - Implement alternate contig detection
   - Update sorting logic

---

## Expected Results

### With Option 2 (Position-Based)

**Optimistic**: 93.80% → 96-97% properly paired
- Fixes cases where hash tie-breaking was non-deterministic
- Fixes cases where primary/alternate split occurred

**Pessimistic**: 93.80% → 95% properly paired
- Helps but doesn't fully match BWA-MEM2
- May need Option 1 for remaining cases

### With Option 1 (Full is_alt)

**Expected**: 93.80% → 97-98% properly paired
- Matches BWA-MEM2's tie-breaking exactly
- Should fix all 73% of tie-breaking cases

---

## Code Locations

### BWA-MEM2
- `src/bwamem.cpp:152-159` - Sort comparator definitions
- `src/bwamem.cpp:1474-1518` - `mem_mark_primary_se()` function
- `src/bwamem.cpp:1485` - Key sort: `ks_introsort(mem_ars_hash, n, a)`

### FerrousAlign
- `src/pipelines/linear/finalization/redundancy.rs:95` - Current sort (NEEDS UPDATE)
- `src/pipelines/linear/finalization/alignment.rs` - `Alignment` struct definition
- `src/pipelines/linear/finalization/secondary.rs` - Secondary alignment marking

---

**Document Version**: 1.0
**Author**: Claude Code + Source Code Analysis
**Status**: Fix Ready for Implementation
