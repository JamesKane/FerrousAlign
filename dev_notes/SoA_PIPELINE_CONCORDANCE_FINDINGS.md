# SoA Pipeline Concordance Analysis

**Date**: 2025-12-01 (Updated: 2025-12-02)
**Branch**: feature/core-rearch
**Status**: Research needed - 81% concordance vs BWA-MEM2 baseline

**Latest**: AVX-512 crash FIXED (2025-12-02) - misaligned buffer allocation issue resolved

## Summary

The pure Structure-of-Arrays (SoA) paired-end pipeline has been successfully implemented with all AoS code paths removed (~1,706 lines deleted). However, alignment concordance testing reveals significant regressions compared to both the v0.6.0 baseline and BWA-MEM2.

## Test Results (10K Golden Reads)

### Concordance Metrics

| Comparison | Concordance | Status | Notes |
|------------|-------------|--------|-------|
| **SoA vs BWA-MEM2** | **81.48%** | ❌ FAIL | Target: >98% for variant callers |
| **SoA vs v0.6.0** | **82.23%** | ❌ FAIL | 11% regression from baseline |
| **v0.6.0 vs BWA-MEM2** | **92.21%** | ⚠️ BASELINE | Pre-existing 8% gap |

### Pairing Metrics

| Metric | SoA | v0.6.0 | BWA-MEM2 | Notes |
|--------|-----|--------|----------|-------|
| Properly paired | 94.44% | 97.65% | ~98% | -3.21% vs v0.6.0 |
| Mapped | 98.65% | 99.44% | ~99% | -0.79% vs v0.6.0 |
| Supplementary | 0 | 104 | ~100 | Missing feature |

## Key Findings

### 1. Proper Pairing Success ✅

Fixed critical bug in `is_proper_pair_soa()`:
- **Before fix**: 48.16% properly paired (completely broken)
- **After fix**: 94.44% properly paired (production quality)
- **Root cause**: Reimplemented orientation calculation instead of using existing `infer_orientation()`
- **Fix**: Replaced custom logic with `super::insert_size::infer_orientation()`

### 2. Alignment Position Discordance ❌

**18% of alignments differ in position/strand from both baselines**

Example cases:
```
Read: HISEQ1:18:H8VC6ADXX:1:1101:10009:11965/1
BWA-MEM2:  chr5:50141532:+ (148M)
v0.6.0:    chr5:50014683:- (148M1D) 
SoA:       chr5:49956952:- (1S129M18S) ← Different position AND strand

Read: HISEQ1:18:H8VC6ADXX:1:1101:10014:8135/1
BWA-MEM2:  chr6:12235803:+ (148M, score 143)
SoA:       chr6:12235848:+ (45S102M1S, score 102) ← Different alignment, lower score
```

**Observations**:
- SoA alignments are valid (not random/broken)
- SoA often has more soft-clipping (S operations)
- SoA scores are sometimes lower
- Position differences range from 12bp to >1Mbp
- Some reads map to entirely different chromosomes

### 3. Missing Supplementary Alignments ⚠️

- **v0.6.0**: 104 supplementary alignments (chimeric/split reads)
- **SoA**: 0 supplementary alignments
- Impact: Affects ~0.5% of reads, contributes to concordance gap

## Root Cause Analysis

The 18% discordance indicates **upstream pipeline differences** in:

### Likely Culprits (in order of probability):

1. **Seeding Stage** (`find_seeds_batch`)
   - Different SMEM (Supermaximal Exact Match) generation
   - Possible bug in SoA batch seeding logic
   - May be finding different seed sets

2. **Chaining Stage** (`chain_seeds_batch`)
   - Different chain selection/scoring
   - Tie-breaking differences in SoA vs AoS iteration order
   - Possible bug in SoA batch chaining

3. **Extension Stage** (`collect_extension_jobs_batch_soa`)
   - Different left/right extension boundaries
   - CIGAR generation differences (more soft-clipping in SoA)
   - Extension job collection bug

4. **Primary Alignment Selection** (`pair_alignments_soa`)
   - Different scoring/filtering of alignments
   - Tie-breaking differences when multiple alignments have same score

### Less Likely:

- Pairing logic (94% proper pairing shows this works correctly)
- SIMD kernels (these are shared with v0.6.0 baseline)
- Reference fetching (no evidence of coordinate bugs)

## Detailed Discordance Patterns

### Pattern 1: Small Position Shifts (12-100bp)
```
Examples: pos_diff:12, pos_diff:45, pos_diff:58
Likely cause: Different extension boundaries or seed selection
```

### Pattern 2: Large Position Jumps (>1Kbp)
```
Examples: pos_diff:57731, pos_diff:647662, pos_diff:1515921
Likely cause: Selecting different seed regions entirely
```

### Pattern 3: Strand Flips
```
Examples: different_strand (chr5:50141532:+ → chr5:49956952:-)
Likely cause: Reverse complement seed preference differences
```

### Pattern 4: Chromosome Changes
```
Examples: different_chr:chr18vschr12
Likely cause: Multi-mapping reads selecting different primary
```

## Investigation Plan

### Phase 1: Isolate Divergence Point

Run SoA pipeline with detailed logging on discordant reads:

1. **Seeding**: Compare seed sets (SMEMs) between v0.6.0 and SoA
   - Log all seeds for discordant reads
   - Check if seed positions match
   - Verify seed counts match

2. **Chaining**: Compare chain sets
   - Log all chains for discordant reads  
   - Check chain scores match
   - Verify chain selection logic

3. **Extension**: Compare extension results
   - Log extension jobs created
   - Check left/right extension boundaries
   - Verify CIGAR generation

### Phase 2: Root Cause Fix

Once divergence point is identified:
- Add unit tests for SoA batch functions
- Compare SoA batch logic to v0.6.0 AoS logic line-by-line
- Fix bugs in SoA implementation
- Re-test concordance

### Phase 3: Validation

- Target: >98% concordance with BWA-MEM2
- Run on full 4M read dataset
- GATK ValidateSamFile
- Variant calling comparison

## Commits Related to This Issue

```
699802c fix(pairing): Fix PROPER_PAIR flag logic - use infer_orientation()
1fc9ee9 fix(tests): Merge duplicate test modules in mate_rescue.rs  
96e8281 refactor(batch_extension): Remove unused AoS batch processing modules
85563c4 feat(soa): Phase 5 - Remove AoS paths, pure SoA pipeline
57dd857 feat(soa): Phase 4 - Implement SoA Pair Finalization
```

## References

- Comparison script: `scripts/compare_sam_outputs.py`
- Golden reads: `tests/golden_reads/` (10K pairs from HG002)
- v0.6.0 baseline: Achieved 97.65% proper pairing, 92.21% BWA-MEM2 concordance
- BWA-MEM2 baseline: Reference implementation

## Next Steps

1. ❌ **DO NOT MERGE** to main until concordance >98%
2. 🔍 Investigate seeding stage differences (highest priority)
3. 🔍 Compare SoA vs AoS batch processing logic
4. 🐛 Fix identified bugs
5. ✅ Re-test and validate

---

**Conclusion**: The SoA pipeline is functionally complete but has significant alignment selection regressions. The 94% proper pairing shows pairing logic works, but upstream stages (seeding/chaining/extension) are producing different alignments. This is a **blocking issue** for production use as variant callers require >98% positional concordance.
