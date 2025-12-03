# is_alt Field Implementation Status

**Date**: December 3, 2025
**Status**: ✅ Complete - Ready for testing with GRCh38

---

## Summary

Implemented BWA-MEM2-compatible `is_alt` field for tie-breaking logic in alignment selection. This adds a critical missing component to match BWA-MEM2's sorting algorithm.

### BWA-MEM2 Tie-Breaking Logic (bwamem.cpp:155)
```c
#define alnreg_hlt(a, b)  ((a).score > (b).score || \
                           ((a).score == (b).score && \
                            ((a).is_alt < (b).is_alt || \
                             ((a).is_alt == (b).is_alt && (a).hash < (b).hash))))
```

**Breakdown**:
1. **Primary**: Score (descending - higher is better)
2. **Tie-break 1**: `is_alt` (ascending - prefer primary assembly over alternates)
3. **Tie-break 2**: Hash (ascending - deterministic ordering)

---

## Implementation Details

### 1. Added `is_alt` Field to Alignment Struct

**File**: `src/pipelines/linear/finalization/alignment.rs:26`

```rust
pub struct Alignment {
    // ... existing fields ...
    pub(crate) is_alt: bool,  // True if alignment to alternate contig/haplotype
}
```

### 2. Implemented Alternate Contig Detection

**File**: `src/pipelines/linear/finalization/alignment.rs:30-45`

```rust
/// Detect if reference name refers to an alternate contig/haplotype
/// Matches BWA-MEM2's logic for preferring primary assembly in tie-breaking
#[inline]
pub fn is_alternate_contig(ref_name: &str) -> bool {
    // Common patterns in alternate contig names:
    // - Contains "alt" (e.g., chr1_KI270706v1_alt)
    // - Contains "random" (e.g., chr1_KI270706v1_random)
    // - Contains "fix" (e.g., chr1_KI270706v1_fix)
    // - Contains "HLA" (e.g., HLA-A*01:01:01:01)
    // - Contains underscore after chr prefix (common in alternate names)
    ref_name.contains("alt")
        || ref_name.contains("random")
        || ref_name.contains("fix")
        || ref_name.contains("HLA")
        || (ref_name.starts_with("chr") && ref_name.contains('_'))
}
```

### 3. Updated Sorting Logic

**File**: `src/pipelines/linear/finalization/redundancy.rs:95-103`

```rust
// Match BWA-MEM2's tie-breaking logic:
// 1. Primary: Score (descending - higher is better)
// 2. Tie-break 1: is_alt (ascending - prefer primary assembly over alternates)
// 3. Tie-break 2: Hash (ascending - deterministic ordering)
alignments.sort_by(|a, b| {
    b.score.cmp(&a.score)
        .then_with(|| a.is_alt.cmp(&b.is_alt))  // NEW: prefer primary assembly (false < true)
        .then_with(|| a.hash.cmp(&b.hash))
});
```

### 4. Updated All Alignment Creation Points

Propagated `is_alt` field through:
- ✅ SoA finalization (`batch_extension/finalize_soa.rs`)
- ✅ SoA structure (`batch_extension/types.rs`)
- ✅ AoS pipeline (`pipeline.rs`)
- ✅ Mate rescue (all variants)
- ✅ Unmapped alignment creation
- ✅ Test code (all test helpers)

---

## Testing Results

### CHM13v2.0 Reference (No Alternate Contigs)

**Baseline (before is_alt)**:
- Properly paired: 177,348 / 189,068 (93.80%)

**After is_alt implementation**:
- Properly paired: 177,348 / 189,068 (93.80%)
- **No change** - Expected, as CHM13v2.0 has no alternate contigs

**Why no improvement?**

CHM13v2.0 is a complete telomere-to-telomere assembly with only primary chromosomes:
- chr1-22, X, Y, M (25 contigs total)
- **No alternate haplotypes or random contigs**
- All alignments have `is_alt = false`
- Tie-breaking defaults to hash-only

### Expected Impact with GRCh38

GRCh38 includes alternate contigs:
- Primary: chr1-22, X, Y, M
- Alternates: chr1_KI270706v1_alt, chr1_KI270762v1_alt, etc.
- HLA haplotypes: HLA-A*01:01:01:01, etc.
- Random contigs: chrUn_*, chr*_random

**Expected improvement**: When reads map equally well to both primary and alternate contigs, `is_alt` logic will prefer primary assembly, improving pairing consistency.

---

## Remaining Pairing Accuracy Gap

**Current**: 93.80% properly paired (FerrousAlign on CHM13v2.0)
**Target**: 98.11% properly paired (BWA-MEM2 on CHM13v2.0)
**Gap**: 4.31 percentage points (~8,200 reads)

### Root Cause Investigation

The `is_alt` field alone did not close the gap on CHM13v2.0. Further investigation revealed:

**Hash discrepancies**: Many alignment creation points set `hash: 0`:
- Mate rescue alignments (all variants)
- Unmapped alignments
- Test alignment helpers

**Implications**:
- When hash = 0, tie-breaking becomes non-deterministic
- BWA-MEM2 computes hash for all alignments: `hash_64(read_id + alignment_idx)`
- FerrousAlign correctly computes hash in finalization but not in mate rescue

**Next Steps**:
1. Add proper hash computation to mate rescue paths
2. Verify hash consistency across all alignment creation points
3. Test with GRCh38 to validate `is_alt` logic works as expected

---

## Files Modified

### Core Implementation
- `src/pipelines/linear/finalization/alignment.rs` - Added field + detection logic
- `src/pipelines/linear/finalization/redundancy.rs` - Updated sorting
- `src/pipelines/linear/batch_extension/finalize_soa.rs` - SoA finalization
- `src/pipelines/linear/batch_extension/types.rs` - SoA structure
- `src/pipelines/linear/pipeline.rs` - AoS pipeline
- `src/pipelines/linear/paired/mate_rescue.rs` - Mate rescue (legacy)
- `src/pipelines/linear/paired/mate_rescue_aos.rs` - Mate rescue (AoS)
- `src/core/io/sam_output.rs` - Unmapped alignment helpers

### Test Code
- `src/pipelines/linear/finalization/alignment.rs` - Test helpers
- `src/pipelines/linear/finalization/redundancy.rs` - Test helpers
- `src/pipelines/linear/finalization/secondary.rs` - Test helpers
- `src/pipelines/linear/stages/finalization/mod.rs` - Test helpers
- `src/pipelines/linear/orchestrator/paired_end/helpers.rs` - Test helpers
- `src/core/io/sam_output.rs` - Test helpers

---

## Validation

### Compilation
✅ Release build: Success
✅ Test compilation: Success
✅ All tests pass: 234 passed; 0 failed

### Runtime
✅ Runs without errors on CHM13v2.0 dataset
✅ Produces valid SAM output
✅ No regressions in pairing accuracy

---

## Recommendations

### Immediate (v0.8.0)
1. **Test with GRCh38**: Validate `is_alt` logic provides expected improvement
2. **Fix hash computation**: Add proper hash values to mate rescue alignments
3. **Measure improvement**: Re-run pairing comparison after hash fix

### Future (v0.9.0+)
1. **Optimize alternate contig detection**: Could be done once during index loading
2. **Add is_alt to SoA metadata**: Store per-reference instead of per-alignment
3. **Performance profiling**: Measure impact of string-based detection

---

**Document Version**: 1.0
**Author**: Claude Code
**Status**: Implementation Complete - Ready for GRCh38 Testing
