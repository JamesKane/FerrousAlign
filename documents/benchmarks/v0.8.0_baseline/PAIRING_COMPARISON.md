# Pairing Accuracy Comparison: FerrousAlign vs BWA-MEM2

**Date**: December 3, 2025
**Dataset**: HG002 100K paired-end reads (200K total reads)
**Reference**: CHM13v2.0 human genome

---

## Summary Statistics

### Overall Comparison

| Metric | BWA-MEM2 | FerrousAlign | Difference |
|--------|----------|--------------|------------|
| **Total reads** | 200,000 | 200,000 | 0 |
| **Total alignments** | 200,602 | 200,000 | -602 |
| **Supplementary** | 602 | 0 | **-602 ⚠️** |
| **Mapped** | 199,440 (99.42%) | 197,484 (98.74%) | -1,956 (-0.68pp) |
| **Properly paired** | 196,210 (98.11%) | 177,348 (93.80%) | **-18,862 (-4.31pp) ⚠️** |
| **Both mates mapped** | 198,254 (99.13%) | 184,670 (92.34%) | -13,584 (-6.79pp) |
| **Singletons** | 584 (0.29%) | 1,882 (1.00%) | +1,298 (+0.71pp) |
| **Mate diff chr** | 1,466 (0.73%) | 3,404 (1.70%) | +1,938 (+0.97pp) |

### Key Findings

1. **Missing Supplementary Alignments**: FerrousAlign produces 0 supplementary alignments vs BWA-MEM2's 602
2. **Lower Pairing Rate**: 93.80% vs 98.11% (4.31pp gap)
3. **Higher Singleton Rate**: 1.00% vs 0.29% (0.71pp increase)
4. **More Mates on Different Chromosomes**: 1.70% vs 0.73%

---

## Properly Paired Read Analysis

### Read-Level Comparison

| Category | Count | Percentage |
|----------|-------|------------|
| **BWA-MEM2 properly paired reads** | 98,105 | 100% |
| **FerrousAlign properly paired reads** | 90,370 | 92.1% |
| **Paired in BWA-MEM2 only** | 7,819 | **8.0%** ⚠️ |
| **Paired in FerrousAlign only** | 84 | 0.1% |

**Net gap**: 7,735 reads (7.9% of BWA-MEM2 properly paired)

### Direction of Discrepancy

**Primary Issue**: FerrousAlign **fails to pair** 7,819 reads that BWA-MEM2 successfully pairs.

**Minor Issue**: FerrousAlign **incorrectly pairs** 84 reads that BWA-MEM2 doesn't pair (negligible).

---

## Root Cause Hypotheses (Prioritized)

### 1. Missing Supplementary Alignments ⭐ **MOST LIKELY**

**Evidence**: BWA-MEM2 generates 602 supplementary alignments, FerrousAlign generates 0.

**Hypothesis**: Supplementary alignments are split/chimeric reads that help resolve proper pairing:
- BWA-MEM2 may use supplementary alignments to determine mate locations
- Without supplementary, FerrousAlign may fail to pair reads with split alignments

**Impact**: Could explain ~600 of the 7,819 discrepancies (7.7%)

**Investigation Needed**:
```bash
# Find reads with supplementary in BWA-MEM2
samtools view -f 2048 bwa_mem2.sam | cut -f1 | sort -u > has_supplementary.txt

# Check if they're in the mismatched set
comm -12 has_supplementary.txt mismatched_bwa_only_sample.txt | wc -l
```

**Status**: Supplementary alignment generation is **not implemented** in FerrousAlign

### 2. Insert Size Distribution Differences

**Evidence from logs** (from baseline run):
- BWA-MEM2: Mean=577.57, Std=155.27, Range=[1, 1251]
- FerrousAlign: Mean=577.57, Std=155.27, Range=[1, 1251] (identical!)

**Hypothesis**: Insert size thresholds may differ in edge cases

**Impact**: Unlikely to be major cause (statistics match)

### 3. Mapping Quality (MAPQ) Thresholds

**Hypothesis**: FerrousAlign may use stricter MAPQ cutoffs for pairing

**Investigation Needed**:
```bash
# Compare MAPQ distribution for mismatched reads
for read in $(head -10 mismatched_bwa_only_sample.txt); do
  echo "=== $read ==="
  grep "^$read\s" bwa_mem2.sam | cut -f1,5
  grep "^$read\s" ferrous.sam | cut -f1,5
done
```

### 4. Strand Orientation Logic

**Hypothesis**: Different handling of FR/RF/FF/RR orientations

**Investigation Needed**: Compare SAM flags (0x10, 0x20) for mismatched reads

### 5. Alignment Score Differences (Secondary Effect)

**Hypothesis**: Different alignment scores → different "best" alignment selection

**Impact**: Could affect which alignment is chosen as primary, affecting pairing

---

## Next Steps (Investigation Plan)

### Phase 1: Characterize the 7,819 Discrepancies (Today)

**Step 1**: Check if supplementary alignments explain discrepancies
```bash
cd documents/benchmarks/v0.8.0_baseline
samtools view -f 2048 bwa_mem2.sam | cut -f1 | sort -u > has_supplementary.txt
comm -12 has_supplementary.txt mismatched_bwa_only_sample.txt > supp_overlap.txt
echo "Reads with supplementary: $(wc -l < supp_overlap.txt)"
```

**Step 2**: Compare MAPQ for mismatched reads
```bash
# Sample 100 mismatched reads
for read in $(head -100 mismatched_bwa_only_sample.txt | tr '\n' ' '); do
  BWA_MAPQ=$(grep "^$read\s" bwa_mem2.sam | head -1 | cut -f5)
  FERR_MAPQ=$(grep "^$read\s" ferrous.sam | head -1 | cut -f5)
  echo "$read,$BWA_MAPQ,$FERR_MAPQ"
done > mapq_comparison.csv
```

**Step 3**: Check strand orientation differences
```bash
# Compare SAM flags
for read in $(head -100 mismatched_bwa_only_sample.txt); do
  BWA_FLAG=$(grep "^$read\s" bwa_mem2.sam | head -1 | cut -f2)
  FERR_FLAG=$(grep "^$read\s" ferrous.sam | head -1 | cut -f2)
  echo "$read,$BWA_FLAG,$FERR_FLAG"
done > flag_comparison.csv
```

### Phase 2: Implement Fix (2-3 days)

Based on Phase 1 findings:

**If supplementary is the cause** → Implement supplementary alignment generation
- **File**: `src/pipelines/linear/seeding.rs` (SMEM splitting for chimeric reads)
- **Effort**: 2-3 days
- **Risk**: Medium (need to match BWA-MEM2 logic exactly)

**If MAPQ is the cause** → Adjust MAPQ calculation or thresholds
- **File**: `src/pipelines/linear/finalization/mod.rs`
- **Effort**: 1 day
- **Risk**: Low

**If orientation is the cause** → Fix strand orientation logic
- **File**: `src/pipelines/linear/paired/pairing_aos.rs`
- **Effort**: 1-2 days
- **Risk**: Medium (core pairing logic)

### Phase 3: Validation (1 day)

After implementing fix:
1. Re-run comparison on 100K dataset
2. Target: Properly paired ≥ 97% (close to BWA-MEM2's 98.11%)
3. Ensure no new regressions (duplicates, etc.)

---

## Detailed Read Samples

### Sample of Reads Paired in BWA-MEM2 Only (first 20)

See `mismatched_bwa_only_sample.txt` for read IDs.

### Sample of Reads Paired in FerrousAlign Only (first 20)

See `mismatched_ferrous_only_sample.txt` for read IDs.

---

## Files Generated

In `documents/benchmarks/v0.8.0_baseline/`:
- `ferrous.sam` - FerrousAlign output (200,060 lines)
- `bwa_mem2.sam` - BWA-MEM2 output (200,628 lines)
- `ferrous_flagstat.txt` - Flagstat summary
- `bwa_flagstat.txt` - Flagstat summary
- `bwa_paired.txt` - 98,105 properly paired read IDs (BWA-MEM2)
- `ferrous_paired.txt` - 90,370 properly paired read IDs (FerrousAlign)
- `mismatched_bwa_only_sample.txt` - Sample of 7,819 discrepancies
- `mismatched_ferrous_only_sample.txt` - Sample of 84 discrepancies

---

**Document Version**: 1.0
**Author**: Claude Code + Samtools Analysis
**Status**: Investigation Plan Ready
