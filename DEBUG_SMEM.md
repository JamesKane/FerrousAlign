# SMEM Generation Debug Report

## Problem Statement

After fixing the suffix array position bug (commit ceee0c2), alignment of real WGS data (GIAB HG002) produces **zero alignments**:

```bash
# Test with 1000 reads
./ferrous-align mem ref.idx test_1k.fq > output.sam
# Result: 0 alignments (only 197 SAM headers)
```

## Investigation Timeline

### Step 1: Initial Benchmarking Revealed Invalid Results

**Observation**: Original benchmarks appeared to show only 2.2% improvement (15.026s → 14.692s) for AVX2 vs baseline.

**Discovery**: Both runs crashed early and never completed alignment:
- Only 23 batches processed (~11K reads)
- Crashed with `index out of bounds` error
- Zero alignments actually produced
- Benchmark was measuring "time to crash", not "time to align"

### Step 2: Suffix Array Position Bug (FIXED)

**Root Cause**: `get_sa_entry()` returns positions in range [0, 2×l_pac) but code wasn't converting reverse complement positions.

**Evidence**:
```
Error: ref_pos: 6,148,247,576 (6.1 billion)
Genome size: 3,099,922,541 (3.1 billion)
```

Position is ~2x genome size, indicating it's in reverse complement region but not converted.

**Fix Applied** (src/align.rs:735-757):
```rust
let mut ref_pos = get_sa_entry(bwa_idx, smem.k);
let mut is_rev = smem.is_rev_comp;

// Convert positions in reverse complement region to forward strand
// BWT contains both forward [0, l_pac) and reverse [l_pac, 2*l_pac)
if ref_pos >= bwa_idx.bns.l_pac {
    ref_pos = (bwa_idx.bns.l_pac << 1) - 1 - ref_pos;
    is_rev = !is_rev; // Flip strand orientation
}
```

**Result**: Program no longer crashes, but still produces zero alignments.

### Step 3: Added Diagnostic Logging

Added comprehensive logging to track pipeline stages:

```rust
log::debug!("{}: Generated {} SMEMs, filtered to {} unique", ...);
log::debug!("{}: Using {} of {} filtered SMEMs for alignment", ...);
log::debug!("{}: Found {} seeds, {} alignment jobs", ...);
log::debug!("{}: Extended {} seeds, {} CIGARs produced", ...);
log::debug!("{}: Chaining produced {} chains", ...);
```

**Output**:
```
[DEBUG] HISEQ1:18:H8VC6ADXX:1:1101:2101:2190: Generated 4740 SMEMs, filtered to 0 unique
[DEBUG] HISEQ1:18:H8VC6ADXX:1:1101:2101:2190: Using 0 of 0 filtered SMEMs for alignment
[DEBUG] HISEQ1:18:H8VC6ADXX:1:1101:2101:2190: Found 0 seeds, 0 alignment jobs
[DEBUG] HISEQ1:18:H8VC6ADXX:1:1101:2101:2190: Extended 0 seeds, 0 CIGARs produced
[DEBUG] HISEQ1:18:H8VC6ADXX:1:1101:2101:2190: Chaining produced 0 chains
```

**Discovery**: Thousands of SMEMs generated, but **all filtered out** (filtered to 0 unique).

### Step 4: Seed Limitation Issue (FIXED)

**Finding**: Code was artificially limited to using only 1 SMEM per read:

```rust
// Original (line 723)
let useful_smems: Vec<_> = sorted_smems.into_iter().take(1).collect();
```

**Fix Applied**:
```rust
// Increased from 1 to 10
let max_seeds = std::cmp::min(smem_count, 10);
let useful_smems: Vec<_> = sorted_smems.into_iter().take(max_seeds).collect();
```

**Result**: Still zero alignments (because all SMEMs are being filtered out before this limit is applied).

### Step 5: Filter Diagnostics

Added tracking for why SMEMs are filtered:

```rust
let mut filtered_too_short = 0;      // len < min_seed_len (19)
let mut filtered_too_many_occ = 0;   // occurrences > max_occ (500)
let mut duplicates = 0;              // Duplicate SMEMs
```

**Output**:
```
[DEBUG] HISEQ1:18:H8VC6ADXX:1:1101:2101:2190: All SMEMs filtered out
  too_short=4659, too_many_occ=4734, duplicates=0
  min_len=19, max_occ=500
```

**Analysis**:
- 4,740 total SMEMs
- 4,659 too short (length < 19)
- 4,734 too many occurrences (> 500)
- Most SMEMs fail BOTH filters
- No duplicates found

### Step 6: Testing Higher max_occ Threshold

**Test**: Increased max_occ from 500 to 10,000:

```bash
./ferrous-align mem -c 10000 ref.idx test_1read.fq
```

**Output**:
```
[DEBUG] All SMEMs filtered out - too_short=4659, too_many_occ=4734, duplicates=0
  min_len=19, max_occ=10000
```

**Discovery**: Even with max_occ=10,000, ALL 4,734 SMEMs (that aren't too short) STILL have too many occurrences!

This is suspicious - suggests occurrence counts are grossly inflated.

### Step 7: CRITICAL BUG FOUND - Invalid BWT Intervals

**Investigation**: Added logging to inspect actual SMEM values:

```rust
for (i, smem) in all_smems.iter().take(5).enumerate() {
    let len = smem.n - smem.m + 1;
    let occ = smem.l - smem.k;  // BWT interval size
    log::debug!("Sample SMEM {}: len={}, occ={}, m={}, n={}, k={}, l={}",
                i, len, occ, smem.m, smem.n, smem.k, smem.l);
}
```

**Output**:
```
[DEBUG] Sample SMEM 0: len=1, occ=1817861263, m=0, n=0, k=0, l=1817861263
[DEBUG] Sample SMEM 1: len=1, occ=1817861263, m=0, n=0, k=4381983819, l=6199845082
[DEBUG] Sample SMEM 2: len=2, occ=6199845083, m=0, n=1, k=1344759809, l=7544604892
[DEBUG] Sample SMEM 3: len=2, occ=18446744072024130832, m=0, n=1, k=4785343326, l=3099922542
[DEBUG] Sample SMEM 4: len=3, occ=1916675856, m=0, n=2, k=1470369860, l=3387045716
```

## Critical Bug Analysis

### SMEM 3 - Invalid BWT Interval

```
k = 4,785,343,326
l = 3,099,922,542
occ = l - k = 18,446,744,072,024,130,832
```

**Problem**: `k > l` violates the fundamental BWT interval invariant!

- BWT interval is [k, l) where k is start, l is end (exclusive)
- MUST have k < l for valid interval
- When k > l, subtraction underflows:
  ```
  3,099,922,542 - 4,785,343,326 = -1,685,420,784 (as signed)
  = 18,446,744,072,024,130,832 (as unsigned u64)
  ```

**Impact**: ALL SMEMs have invalid (massive) occurrence counts, so ALL get filtered out.

### Other Suspicious Values

**SMEM 0**:
- `occ = 1,817,861,263` (1.8 billion occurrences for a single base!)
- This is ~60% of the genome size - unrealistic for a unique match

**SMEM 1**:
- `k = 4,381,983,819` and `l = 6,199,845,082`
- Both values > genome size (3.1 billion)
- l = 6,199,845,082 ≈ 2 × genome size (exactly 2 × l_pac)

**SMEM 2**:
- `l = 7,544,604,892` > 2 × genome size
- k and l are both way out of bounds

### BWT Structure Context

**Expected BWT Range**:
```
Genome size (l_pac):     3,099,922,541
Forward strand:          [0, 3,099,922,541)
Reverse complement:      [3,099,922,541, 6,199,845,082)
Total BWT length:        6,199,845,082 (= 2 × l_pac)
```

**Observed Values**:
- Many k and l values are > 2 × l_pac (outside valid BWT range)
- k > l violations
- Values suggest boundary crossing or overflow issues

## Root Cause Hypothesis

The `backward_ext()` function in `src/align.rs` is producing invalid BWT intervals. Possible causes:

### 1. Boundary Crossing Issue

BWT contains both forward and reverse complement strands. The backward search might be incorrectly handling intervals that cross the l_pac boundary.

### 2. Integer Overflow in Backward Search

The BWT iteration might be incrementing k or l beyond valid bounds without checking:

```rust
// Hypothetical bug pattern
let mut k = start_k;
let mut l = start_l;
for base in query.iter().rev() {
    // Get new interval for this base
    let (new_k, new_l) = bwt_extend(k, l, base);
    k = new_k;  // Might overflow beyond seq_len
    l = new_l;  // Might wrap around
}
```

### 3. cp_occ Checkpoint Calculation

The occurrence counting uses checkpoints (cp_occ) every 64 bases. If checkpoint indexing is wrong, calculated intervals could be invalid.

### 4. SA Sampling Confusion

Suffix array is sampled (not full array). The interaction between BWT positions and SA sampling might be causing position corruption.

## Evidence Summary

| Metric | Expected | Observed | Status |
|--------|----------|----------|--------|
| SMEMs generated | Hundreds | 4,740 | ✅ OK |
| SMEMs after filtering | Tens | 0 | ❌ ALL FILTERED |
| SMEM length | 19-100bp | 1-3bp mostly | ⚠️ TOO SHORT |
| SMEM occurrences | <500 | >1 billion | ❌ INVALID |
| BWT interval [k,l) | k < l | k > l | ❌ VIOLATION |
| k values | [0, 2×l_pac) | Up to 7.5B | ❌ OUT OF BOUNDS |
| l values | [0, 2×l_pac) | Up to 7.5B | ❌ OUT OF BOUNDS |

## Next Steps

### Immediate Actions

1. **Investigate `backward_ext()` implementation**
   - Location: `src/align.rs`
   - Check BWT interval calculation logic
   - Verify boundary conditions

2. **Check `get_bwt()` and checkpoint logic**
   - Verify cp_occ array indexing
   - Check popcount64 calculations
   - Verify one-hot encoding

3. **Add interval validation**
   ```rust
   // After backward_ext
   assert!(smem.k < smem.l, "Invalid BWT interval: k={} >= l={}", smem.k, smem.l);
   assert!(smem.l <= bwa_idx.bwt.seq_len, "BWT interval l={} exceeds seq_len={}",
           smem.l, bwa_idx.bwt.seq_len);
   ```

4. **Compare with C++ bwa-mem2**
   - Check backward_search implementation in fastmap.cpp
   - Verify SMEM generation logic in bwtbwt.cpp
   - Compare checkpoint calculation

### Testing Strategy

**Unit Tests**:
```rust
#[test]
fn test_backward_ext_simple() {
    // Test with known query sequence
    // Verify k < l invariant
    // Verify intervals within [0, 2*l_pac)
}

#[test]
fn test_backward_ext_boundary() {
    // Test sequences that cross forward/reverse boundary
    // Verify correct interval splitting
}
```

**Integration Tests**:
```bash
# Test with simple synthetic data (known to work)
./ferrous-align mem test.idx simple.fq

# Test with single known-good read
./ferrous-align mem ref.idx single_read.fq -v 4

# Validate SMEM intervals
# - All should have k < l
# - All should have k,l < 2*l_pac
# - Occurrence counts should be reasonable (<10000 for most)
```

## Related Code Locations

**SMEM Generation**:
- `src/align.rs:590-675` - Forward/reverse strand backward search
- `src/align.rs:152-244` - `backward_ext()` function

**BWT Operations**:
- `src/align.rs:47-70` - `get_bwt()` function
- `src/align.rs:73-149` - `get_sa_entry()` function
- `src/bwt.rs` - BWT data structure and checkpoints

**Filtering**:
- `src/align.rs:685-739` - SMEM filtering and diagnostics

## Commit History

- **ceee0c2**: Fix critical suffix array position bug and add SIMD instrumentation
  - Fixed SA position conversion for reverse complement
  - Added comprehensive logging
  - Added early batch completion optimization
  - Created INSTRUMENTATION_GUIDE.md and WGS_BENCHMARKING_GUIDE.md

## Current Status

**BLOCKED**: Cannot proceed with SIMD optimization validation until SMEM generation is fixed.

**Severity**: CRITICAL - Zero alignments produced on real data

**Workaround**: None - this is a fundamental bug in the core alignment algorithm

**Timeline**: Must fix before any benchmarking or performance testing can be valid
