# Hash Computation Fix Results

**Date**: December 3, 2025
**Status**: ✅ Implemented - No pairing improvement on CHM13v2.0

---

## Summary

Implemented post-mate-rescue hash computation to ensure all alignments have proper hash values for tie-breaking. This matches BWA-MEM2's approach where `mem_mark_primary_se()` computes hashes after mate rescue.

### Implementation

**File**: `src/pipelines/linear/orchestrator/paired_end/helpers.rs`

Added `fix_missing_hashes()` function that runs after mate rescue:
```rust
fn fix_missing_hashes(all_alignments: &mut [Vec<Alignment>]) {
    for (read_idx, alignments) in all_alignments.iter_mut().enumerate() {
        for (aln_idx, aln) in alignments.iter_mut().enumerate() {
            if aln.hash == 0 {
                // Compute hash using same formula as finalization
                aln.hash = crate::utils::hash_64(read_idx as u64 + aln_idx as u64);
            }
        }
    }
}
```

**Called after mate rescue** (line 120-121):
```rust
// Convert back to AoS
*alignments1 = soa1.to_aos();
*alignments2 = soa2.to_aos();

// Fix hash values for any alignments that don't have one
Self::fix_missing_hashes(alignments1);
Self::fix_missing_hashes(alignments2);
```

---

## Testing Results

### CHM13v2.0 Reference

**Before hash fix** (with is_alt only):
- Properly paired: 177,348 / 189,068 (93.80%)

**After hash fix**:
- Properly paired: 177,348 / 189,068 (93.80%)
- **No improvement**

### Analysis

The hash fix did not improve pairing accuracy for several possible reasons:

1. **Hash collision isn't the issue**: Within a single read's alignments, relative hash ordering matters more than absolute values. Even with hash=0, the sorting would be consistent within that read.

2. **Main finalization already has correct hashes**: The primary finalization paths (`finalize_soa.rs`, `pipeline.rs`) already compute hashes correctly using `hash_64(read_id + idx)`. Only mate-rescued and unmapped alignments had hash=0.

3. **Mate rescued alignments are typically lower scoring**: Rescued alignments usually have lower scores than primary alignments, so they rarely participate in tie-breaking for primary selection anyway.

4. **The real issue may be elsewhere**: The 4.31pp gap vs BWA-MEM2 likely stems from:
   - Different mate rescue strategies
   - Different insert size inference
   - Different proper pair detection logic
   - Differences in how alignments are selected before pairing

---

## Observations

### Hash Values in Different Contexts

**Regular finalization** (finalize_soa.rs:368):
```rust
let hash = crate::utils::hash_64(read_id + idx as u64);
```
- Uses global `read_id` (batch_start_id + read_idx)
- Unique across entire run
- ✅ Correct

**Mate rescue** (after our fix):
```rust
aln.hash = crate::utils::hash_64(read_idx as u64 + aln_idx as u64);
```
- Uses batch-local `read_idx`
- Could collide across batches
- ⚠️ Not globally unique, but doesn't matter for intra-read tie-breaking

**Unmapped alignments** (various locations):
```rust
hash: 0
```
- Never participate in tie-breaking (only one unmapped alignment per read)
- ✅ Doesn't matter

---

## BWA-MEM2 Comparison

### BWA-MEM2 Approach

BWA-MEM2 computes hashes in `mem_mark_primary_se()` (bwamem.cpp:1482):
```c
for (i = 0; i < n; ++i)
    a[i].hash = hash_64(id+i);
```

This runs **AFTER**:
1. Alignment generation
2. Mate rescue
3. Before final sorting and primary selection

### FerrousAlign Approach

**Now matches BWA-MEM2's pattern**:
1. ✅ Finalization computes hashes for regular alignments
2. ✅ Mate rescue adds alignments (initially hash=0)
3. ✅ `fix_missing_hashes()` computes hashes for rescued alignments
4. ✅ Sorting uses hash for tie-breaking

---

## Remaining Pairing Gap Analysis

**Current**: 93.80% properly paired (FerrousAlign on CHM13v2.0)
**Target**: 98.11% properly paired (BWA-MEM2 on CHM13v2.0)
**Gap**: 4.31 percentage points (~8,200 reads)

### Likely Causes (Not Hash-Related)

1. **Insert Size Inference Differences**
   - Different percentile calculations
   - Different outlier filtering
   - Different orientation weighting

2. **Mate Rescue Strategy Differences**
   - Different score thresholds
   - Different search regions
   - Different SW parameters

3. **Proper Pair Detection Logic**
   - Different insert size bounds checking
   - Different orientation requirements
   - Different MAPQ thresholds

4. **Alignment Selection Before Pairing**
   - Differences in which alignments make it to pairing stage
   - Different redundancy filtering
   - Different secondary/supplementary marking

---

## Recommendations

### Next Investigation Steps

1. **Compare alignment counts before pairing**:
   ```bash
   # Check if BWA-MEM2 and FerrousAlign produce same number of alignments per read
   samtools view bwa.sam | cut -f1 | sort | uniq -c | sort -rn | head -20
   samtools view ferrous.sam | cut -f1 | sort | uniq -c | sort -rn | head -20
   ```

2. **Compare insert size distributions**:
   - Extract insert size stats from BWA-MEM2 vs FerrousAlign
   - Check if bounds are calculated identically

3. **Analyze specific mismatched pairs**:
   - Take 100 read pairs where BWA-MEM2 has proper pair but FerrousAlign doesn't
   - Check insert sizes, orientations, scores
   - Identify systematic differences

4. **Test with different references**:
   - Try GRCh38 to validate is_alt logic works
   - Try smaller references to isolate variables

### Code Quality Improvements

Even though hash fix didn't improve pairing:
- ✅ Better matches BWA-MEM2's architecture
- ✅ More robust (no hash=0 alignments in tie-breaking)
- ✅ Easier to debug (all alignments have proper hashes)
- ✅ Good foundation for future improvements

---

## Conclusion

The hash computation fix brings FerrousAlign closer to BWA-MEM2's implementation architecture, but does not improve pairing accuracy on CHM13v2.0. The 4.31pp pairing gap likely stems from differences in:
- Insert size inference
- Mate rescue strategy
- Proper pair detection logic

Further investigation needed to identify the root cause.

---

**Document Version**: 1.0
**Author**: Claude Code
**Status**: Hash fix implemented - Pairing gap investigation ongoing
