# Pairing Accuracy Root Cause Analysis

**Date**: December 3, 2025
**Finding**: 7,819 reads properly paired in BWA-MEM2 but not in FerrousAlign

---

## Root Cause Identified ⚠️

**Primary Issue**: FerrousAlign is **selecting different alignment positions** for reads, causing mate coordinates to be wrong or unmapped.

### Key Observations from Sample Analysis

#### Pattern 1: Both Mates Mapped but Different Positions (Most Common)

**Example 1**: `HISEQ1:18:H8VC6ADXX:1:1101:10000:26291`
- **BWA-MEM2**: Both map to `chrY:38889335` and `chrY:38889889` (554bp apart, properly paired)
- **FerrousAlign**: Both map to `chrY:53810321` and `chrY:53909372` (**99,051bp apart!**, not properly paired)
- **Issue**: FerrousAlign chose completely different genomic positions

**Example 2**: `HISEQ1:18:H8VC6ADXX:1:1101:10007:79276`
- **BWA-MEM2**: Both on `chr3` ~500bp apart (properly paired)
- **FerrousAlign**: R1 on `chr7`, R2 on `chr3` (different chromosomes, not properly paired)

**Example 3**: `HISEQ1:18:H8VC6ADXX:1:1101:10011:55043`
- **BWA-MEM2**: Both on `chr22` ~800bp apart (properly paired)
- **FerrousAlign**: Both on `chr22` but **1,215,653bp apart!** (not properly paired)

#### Pattern 2: Mate Completely Unmapped

**Example**: `HISEQ1:18:H8VC6ADXX:1:1101:10009:11965`
- **BWA-MEM2**: Both mapped to `chr19` ~400bp apart (properly paired)
- **FerrousAlign**: Both mapped to `chr19` but **mate field shows `*`** (unmapped/not paired)
  - Flag shows `0x10` (reverse strand) and `0x0` (forward strand)
  - Mate positions at `0` (indicating not paired)

#### Pattern 3: Insert Size Calculation Issues

Notice in FerrousAlign many have `TLEN=0` (column 9), while BWA-MEM2 has proper insert sizes (e.g., `702`, `-651`, `581`).

This suggests:
1. **Pairing logic isn't running correctly** OR
2. **Insert size is being zeroed out** OR
3. **Mates aren't being properly associated**

---

## Technical Analysis

### SAM Flag Interpretation

| Read | BWA-MEM2 Flags | FerrousAlign Flags | Interpretation |
|------|----------------|-------------------|----------------|
| Read 1 | `83` (0x53) | `81` (0x51) | Both paired, read1, reverse strand |
| Read 1 | | But mate coord wrong | Pairing incomplete |
| Read 2 | `163` (0xa3) | `161` (0xa1) | Both paired, read2, forward strand |
| Read 2 | | But mate coord wrong | Pairing incomplete |

**Key Flag Bits**:
- `0x1` (1): Paired in sequencing
- `0x2` (2): **Properly paired** ← This is MISSING in FerrousAlign!
- `0x4` (4): Read unmapped
- `0x8` (8): Mate unmapped
- `0x10` (16): Reverse strand
- `0x20` (32): Mate reverse strand
- `0x40` (64): First in pair
- `0x80` (128): Second in pair

**FerrousAlign FLAGS**:
- `81` = `0x51` = 1 + 16 + 64 = paired + reverse + read1, **BUT NO 0x2 (properly paired)**
- `161` = `0xa1` = 1 + 32 + 128 = paired + mate reverse + read2, **BUT NO 0x2 (properly paired)**

**BWA-MEM2 FLAGS**:
- `83` = `0x53` = **2** + 1 + 16 + 64 = **properly paired** + paired + reverse + read1 ✅
- `163` = `0xa3` = **2** + 1 + 32 + 128 = **properly paired** + paired + mate reverse + read2 ✅

---

## Root Causes (Ranked by Likelihood)

### 1. ⭐ **Alignment Position Selection Differs** (PRIMARY CAUSE)

**Evidence**: FerrousAlign consistently chooses different genomic positions than BWA-MEM2.

**Possible Reasons**:
a. **Seeding differences**: Different SMEM generation → different candidate positions
b. **Chaining differences**: Different chain scoring → different "best" position selection
c. **Extension differences**: Different alignment scores → different position ranking
d. **Tie-breaking differences**: When multiple positions have same score, different tie-breaking logic

**Impact**: If R1 maps to a different location than BWA-MEM2, its mate R2 will have wrong mate coordinates, causing pairing to fail.

### 2. 🎯 **Proper Pairing Flag Not Set** (SYMPTOM, NOT ROOT CAUSE)

**Evidence**: FerrousAlign never sets the `0x2` (properly paired) flag, even when mates ARE mapped nearby.

**Hypothesis**:
- Pairing logic runs but **doesn't set the flag** due to overly strict insert size checks
- OR pairing logic **doesn't run at all** for some reads

**Code Location**: `src/pipelines/linear/paired/pairing_aos.rs`

**Investigation Needed**: Check insert size bounds and flag-setting logic.

### 3. 📊 **Insert Size Threshold Too Strict**

**Evidence**: Many FerrousAlign alignments have `TLEN=0`, suggesting insert size calculation failed or pairing was rejected.

**Hypothesis**: Insert size bounds are too strict, rejecting valid pairs.

**Investigation Needed**: Compare insert size distributions and thresholds.

---

## Investigation Plan

### Phase 1: Understand Alignment Position Differences (Priority #1)

**Goal**: Why does FerrousAlign choose different positions than BWA-MEM2?

**Step 1**: Compare seeds for a mismatched read
```bash
# Add debug logging to seeding stage
# Output: "Read X: Generated N seeds at positions [...]"
# Compare with BWA-MEM2 (if possible to extract)
```

**Step 2**: Compare chains for a mismatched read
```bash
# Add debug logging to chaining stage
# Output: "Read X: Chain 1 score=Y, Chain 2 score=Z, selected=1"
```

**Step 3**: Compare final alignment scores
```bash
# Extract AS (alignment score) tags from both outputs
grep "HISEQ1:18:H8VC6ADXX:1:1101:10000:26291" bwa_mem2.sam | grep -o "AS:i:[0-9]*"
grep "HISEQ1:18:H8VC6ADXX:1:1101:10000:26291" ferrous.sam | grep -o "AS:i:[0-9]*"
```

### Phase 2: Fix Pairing Logic (If Phase 1 doesn't reveal position selection issues)

**Goal**: Ensure proper pairing flag is set correctly

**Investigation**:
1. Add logging to `pair_alignments_aos()` function
2. Check insert size bounds being used
3. Verify flag-setting logic

**Code to Review**:
```rust
// src/pipelines/linear/paired/pairing_aos.rs
pub fn pair_alignments_aos(...) {
    // Check:
    // 1. Is this function being called?
    // 2. Are insert size bounds reasonable?
    // 3. Is 0x2 flag being set?
}
```

### Phase 3: Compare with BWA-MEM2 Source Code

**Goal**: Understand BWA-MEM2's alignment position selection logic

**Files to Review** (in bwa-mem2 source):
- `src/FMI_search.cpp` - Seeding
- `src/kswv.cpp` - Alignment scoring
- `src/fastmap.cpp` - Main pipeline + position selection

**Key Question**: What tie-breaking logic does BWA-MEM2 use when multiple positions have equal scores?

---

## Immediate Next Steps

### Today (December 3, 2025)

1. **Add instrumentation** to FerrousAlign to log:
   - Number of seeds generated per read
   - Number of chains generated
   - Final alignment score (AS tag)
   - Insert size calculation
   - Whether pairing flag was set

2. **Compare one mismatched read end-to-end**:
   - Pick `HISEQ1:18:H8VC6ADXX:1:1101:10000:26291`
   - Trace through seeding → chaining → extension → pairing
   - Identify where decision diverges from BWA-MEM2

3. **Quick check**: Extract AS scores for all mismatched reads
   ```bash
   # Are FerrousAlign's alignment scores consistently lower?
   for read in $(head -100 mismatched_bwa_only_full.txt); do
       BWA_AS=$(grep "^$read\s" bwa_mem2.sam | grep -o "AS:i:[0-9]*" | head -1 | cut -d: -f3)
       FERR_AS=$(grep "^$read\s" ferrous.sam | grep -o "AS:i:[0-9]*" | head -1 | cut -d: -f3)
       echo "$read,$BWA_AS,$FERR_AS"
   done > alignment_score_comparison.csv
   ```

---

## Expected Outcome

If alignment position selection is the root cause:
- **Fix**: Adjust seeding/chaining/extension scoring to match BWA-MEM2 logic
- **Effort**: 3-5 days (requires careful algorithm comparison)
- **Risk**: Medium-high (core algorithm changes)

If pairing flag logic is the root cause:
- **Fix**: Adjust insert size bounds or flag-setting logic
- **Effort**: 1 day
- **Risk**: Low (localized change)

**Most Likely**: Combination of both issues.

---

**Document Version**: 1.0
**Author**: Claude Code + SAM Analysis
**Status**: Root Cause Identified - Investigation Plan Ready
