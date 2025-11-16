# FerrousAlign Development - Active Investigation

## Current Issue

**Problem**: SMEM (Super Maximal Exact Match) generation differs from C++ bwa-mem2 reference implementation
- C++ bwa-mem2: Generates 4 long SMEMs (max 96bp) for test read `HISEQ1:18:H8VC6ADXX:1:1101:2101:2190`
- Rust implementation: Alignment behavior under investigation with real-world WGS data

**Impact**: Testing alignment correctness against human genome reference GRCh38

## What We're Solving

Ensuring FerrousAlign produces identical alignment results to C++ bwa-mem2 for production whole-genome sequencing data.

**Target Read**: `HISEQ1:18:H8VC6ADXX:1:1101:2101:2190` (148bp)
- Expected alignment: chr7:67600394
- Testing against: Full GRCh38 human reference genome (~3.1 Gbp)

## Test Data Access

### Reference Genome
```bash
/home/jkane/Genomics/Reference/b38/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna
```

### Test Read
```bash
/home/jkane/Genomics/HG002/test_1read.fq
```
Contains single 148bp read from HG002 whole-genome sequencing dataset.

### C++ bwa-mem2 Reference Implementation
```bash
/tmp/bwa-mem2-diag/
```

## Running Tests

```bash
# Rust implementation
./target/release/ferrous-align mem -v 3 \
  /home/jkane/Genomics/Reference/b38/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna \
  /home/jkane/Genomics/HG002/test_1read.fq

# C++ reference (for comparison)
/tmp/bwa-mem2-diag/bwa-mem2 mem -v 3 \
  /home/jkane/Genomics/Reference/b38/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna \
  /home/jkane/Genomics/HG002/test_1read.fq
```

**Note**: Index loading takes ~30 seconds for the 9GB human genome index files.

## Status

**Active Work**: Investigating SMEM generation algorithm to match C++ bwa-mem2 behavior
- Forward extension with k/l swapping for reverse complement BWT
- Cumulative sum calculation for BWT interval endpoints
- Maintaining BWT interval invariants during extension

**Recent Findings** (Session 2025-11-16):

### Critical Discoveries:
1. **SMEM `l` field encoding**: C++ uses `l = count[3-a]` (reverse complement), NOT `l = count[a+1]`
   - This breaks the traditional BWT invariant `s = l - k`
   - The `l` field encodes reverse complement BWT information

2. **backward_ext() cumulative sum**: C++ uses cumulative sum for `l[]` computation:
   ```cpp
   l[3] = smem.l + sentinel_offset;
   l[2] = l[3] + s[3];
   l[1] = l[2] + s[2];
   l[0] = l[1] + s[1];
   ```
   NOT the simple formula `l[b] = k[b] + s[b]`

3. **Two-Phase SMEM Algorithm**: C++ uses bidirectional search, Rust was only doing backward:
   - **Phase 1 (Forward Extension)**: Start at position x, extend forward (j from x+1 to readlength)
     - Uses k/l swap + backwardExt(3-a) to simulate forward extension
     - Collects intermediate SMEMs in array
     - Reverses the array
   - **Phase 2 (Backward Search)**: Extend backward (j from x-1 to 0)
     - Uses regular backwardExt(a) with literal base
     - Outputs final bidirectional SMEMs

4. **Result**: Rust was generating 4740 SMEMs vs C++ generating ~4 long SMEMs
   - Missing the bidirectional constraint

### Fixes Applied:
- ✅ Fixed SMEM initialization (`l = count[3-a]`)
- ✅ Fixed backward_ext() cumulative sum
- ✅ Added forward_ext() helper function
- ✅ Rewrote main SMEM generation loop for two-phase algorithm
  - Forward extension phase: collects 12-13 intermediate SMEMs
  - Backward extension phase: should extend SMEMs further

### Current Status (Debugging Phase - Session 2025-11-16 continued):

**CRITICAL BUG FOUND**: Forward extension stops too early!
- C++ bwa-mem2: Generates SMEMs with len=96, 51, 19, 19 (extends to j=95)
- Rust: Generates NO SMEMs >= 19 (stops at j=11)

**Root Cause**: Forward extension stopping condition
- Initial SMEM (x=0): k=0, l=4381983819, s=1817861263
- Forward extension j=1: k=1344759810, l=1344759809, s=473101453
  - **Note**: k > l is OK! The `l` field encodes reverse complement BWT info, NOT interval endpoint
  - The `s` field represents interval size/occurrence count
- s values decrease: 1817861263 → 473101453 → ... → 1429 → **433**
- Stops at j=11 because s=433 < min_intv=500
- **But C++ extends to j=95 with final s=1!**

### CRITICAL BUGS FIXED (Session 2025-11-16):

**Bug 1: Wrong min_intv during SMEM generation**
- **Problem**: Used `min_intv = max_occ` (500), causing forward extension to stop at j=11
- **Fix**: Use `min_intv = 1` to match C++ (bwamem.cpp:661)
- **Impact**: Now extends to j=95+ like C++, generating long SMEMs

**Bug 2: Wrong occurrence count in SMEM filtering**
- **Problem**: Used `occurrences = l - k` (invalid, since l encodes reverse complement BWT)
- **Fix**: Use `occurrences = s` (the actual interval size)
- **Impact**: SMEMs now pass filtering correctly

**Results after fixes**:
- Generated: 950 SMEMs → filtered to 566 unique (was 0 before!)
- Using 10 seeds for alignment
- Chaining: produces 1 chain
- SAM output IS produced (with `-T 0`) but alignment is WRONG

### Investigation Session 2025-11-16 (continued) - Reference Data Validation

**CRITICAL FINDING**: All SMEMs have s != l - k (verified via C++ comparison)
- C++ bwa-mem2 SMEM 0: k=1499663434, l=5175475687, **s=1** (l-k=3675812253)
- Rust SMEM 0 (before use): k=1847695904, l=4159962409, **s=1** (l-k=2312266505)
- The `l` field does NOT represent interval endpoint; it encodes reverse complement BWT info
- The `s` field is the TRUE occurrence count from BWT occ tables
- Fixed logging bug at align.rs:1093 that was displaying `l - k` instead of `smem.s`

**Current Problem**: Wrong alignment despite correct SMEM generation
- Expected (C++ bwa-mem2): chr7:67600394, CIGAR=148M, AS=143, NM=1
- Actual (Rust): chr6:148708668, CIGAR=148I, score too low to pass -T 30 threshold
- All 10 SMEMs show "NO PERFECT MATCH FOUND" when checking reference
- SMEM BWT intervals are correct (s values match C++ - all have s=1)
- **Hypothesis**: Suffix array reconstruction or .pac file reading is corrupted

**Verified Correct**:
- ✅ .ann file parsing (chr7 offset = 1232004303, len = 159345973)
- ✅ SMEM generation algorithm (two-phase bidirectional search)
- ✅ SMEM filtering (uses smem.s, not l - k)

### CRITICAL BUG FIXED (Session 2025-11-16 continued) - l2 Array Off-By-One

**Root Cause Found**: Step-by-step comparison of Rust vs C++ extension traces revealed off-by-one error

**The Bug**:
- Rust was loading l2[] array directly from index file without adjustment
- C++ adds +1 to all count[] values after loading (FMI_search.cpp:435)
- This caused initial SMEM k/l values to differ by 1:
  - Rust: k=0, l=4381983819 → after swap → k=4381983819, l=0
  - C++:  k=1, l=4381983820 → after swap → k=4381983820, l=1

**The Fix** (src/mem.rs:78-82):
```rust
// CRITICAL: Match C++ bwa-mem2 behavior - add 1 to all count values
// See FMI_search.cpp:435 - this is required for correct SMEM generation
for i in 0..5 {
    bwt.l2[i] += 1;
}
```

**Results After Fix**:
- ✅ Extension traces now match C++ EXACTLY
- ✅ Generated 370 SMEMs → filtered to 8 unique (was 4740 → 81 before)
- ✅ Read maps to **chr7:67600444** (expected chr7:67600394, within 50bp!)
- ❌ CIGAR still shows 148I (should be 148M with 1 mismatch)
- ❌ SMEMs still show "NO PERFECT MATCH FOUND" when checking reference

**Next Steps**: Investigate SA reconstruction or .pac reading - SMEMs are generated correctly but don't verify against reference
