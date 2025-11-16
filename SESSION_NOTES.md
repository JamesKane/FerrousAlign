# Session Notes - Index Compatibility Investigation

## Date: 2025-11-16 (Continuation)

### Session Summary

This session focused on debugging why SMEMs generated from the C++ bwa-mem2 GRCh38 index don't match the reference, while synthetic indices work perfectly.

---

## Issues Fixed

### 1. ✅ SA Length Calculation (FIXED)

**Problem**: Our SA length calculation differed from C++ bwa-mem2 by 1 element.

**Root Cause**:
- C++: `sa_len = (ref_seq_len >> SA_COMPX) + 1` where SA_COMPX = 3
- Rust (old): `sa_len = (bwt.seq_len + sa_intv - 1) / sa_intv`

These formulas are off by 1:
- C++: `(seq_len / 8) + 1`
- Rust: `(seq_len + 7) / 8`

**Fix Applied** (src/mem.rs:103):
```rust
// C++ uses: ((ref_seq_len >> SA_COMPX) + 1)
// which equals: (ref_seq_len / 8) + 1
let sa_len = (bwt.seq_len >> sa_compx) + 1;
```

**Result**: SA array now loads with correct size matching C++ format.

---

### 2. ✅ ONE_HOT_MASK_ARRAY Initialization (FIXED)

**Problem**: Mask array initialization differed slightly from C++ implementation.

**Root Cause**:
- Our loop: `for i in 1..=64` (65 elements, calculated array[1])
- C++: 64 elements, explicitly sets `array[1] = base`, then loops from i=2

**Fix Applied** (src/align.rs:61-69):
```rust
lazy_static::lazy_static! {
    static ref ONE_HOT_MASK_ARRAY: Vec<u64> = {
        let mut array = vec![0u64; 64]; // Size 64 to match C++
        let base = 0x8000000000000000u64;
        array[1] = base;  // Explicitly set like C++ does
        for i in 2..64 {
            array[i] = (array[i - 1] >> 1) | base;
        }
        array
    };
}
```

**Result**: Mask array now exactly matches C++ initialization pattern.

---

### 3. ✅ Reverse Complement SMEM Query Base Selection (CRITICAL FIX)

**Problem**: When comparing RC SMEMs with reference, we were using FORWARD strand query bases instead of reverse complement bases.

**Root Cause**: SMEMs generated from reverse complement search (`is_rev_comp: true`) have query positions [m, n] referring to the RC query, but we were always extracting bases from the forward strand query.

**Fix Applied** (src/align.rs:856-890):
```rust
// Prepare reverse complement for RC SMEMs
let mut query_segment_encoded_rc: Vec<u8> = query_segment_encoded.iter()
    .map(|&b| b ^ 3)  // A<->T (0<->3), C<->G (1<->2)
    .collect();
query_segment_encoded_rc.reverse();

// In SMEM loop:
// CRITICAL FIX: Use correct query orientation based on is_rev_comp flag
let query_for_smem = if smem.is_rev_comp {
    &query_segment_encoded_rc
} else {
    &query_segment_encoded
};
let smem_query_bases = &query_for_smem[smem.m as usize..=(smem.n as usize).min(query_for_smem.len()-1)];
```

**Result**: RC SMEMs now correctly compare against RC query bases. Match quality improved from ~50% to ~55%.

---

## Validation Results

### Synthetic Test (400bp reference)
- ✅ **Rust-built index**: 100% perfect SMEM matches
- ✅ **C++ bwa-mem2 index**: 100% perfect SMEM matches
- **Conclusion**: Index loader is fully compatible with C++ bwa-mem2 format

### GRCh38 Test (3GB reference)
- ❌ **Still failing**: Best match 11/20 bases (55%) vs expected 20/20 (100%)
- **Index source**: Confirmed built with C++ bwa-mem2 (not our Rust indexer)
- **C++ bwa-mem2 alignment**: Successfully aligns to chr7:67600394 with `148M` CIGAR
- **Our alignment**: Fails with chr6:148708668 and `148I` CIGAR (all insertions)

---

## Current Hypothesis

The bug is **scale-dependent** - it only manifests with large, complex genomes:

### Evidence
1. ✅ Small synthetic genomes (400bp) work perfectly
2. ✅ C++ indices work on synthetic data
3. ✅ SA values load correctly (verified: SA[0]=6,199,845,082 for GRCh38)
4. ✅ File format is compatible
5. ❌ Large real genome (3GB) fails

### Possible Remaining Causes
1. **BWT character reconstruction** - Issue with `get_bwt_base_from_cp_occ()` at large positions
2. **Checkpoint boundary handling** - Edge case at CP_BLOCK_SIZE boundaries
3. **Reference sequence reading** - `.pac` file reading issue for large files
4. **Integer overflow/precision** - Subtle arithmetic issue with large position values
5. **Occurrence counting** - Bug in `get_occ()` or `popcount64()` with large bitmasks

---

## Files Modified

### Core Fixes
- **src/mem.rs:103** - SA length calculation
- **src/align.rs:61-69** - ONE_HOT_MASK_ARRAY initialization
- **src/align.rs:856-890** - RC SMEM query base selection

### Debug Logging Added
- **src/mem.rs:131-138** - SA value verification on load
- **src/align.rs:866** - Added is_rev_comp to SMEM debug output

---

## Next Session Priorities

### Immediate
1. **Trace BWT character reconstruction** - Add detailed logging to `get_bwt_base_from_cp_occ()`
2. **Verify occurrence counting** - Check `get_occ()` and `popcount64()` with real data
3. **Compare with C++ at specific position** - Pick one failing SMEM and trace through C++ vs Rust step-by-step

### If Above Fails
4. **Test intermediate genome sizes** - Try 100MB, 500MB, 1GB references to find scale threshold
5. **Hex dump comparison** - Compare cp_occ blocks between synthetic and GRCh38 indices
6. **Reference (.pac) validation** - Verify reference bases are read correctly at SMEM positions

---

## Key Insights

1. **Index format compatibility is confirmed** - Our loader correctly reads C++ bwa-mem2 indices
2. **Small differences matter** - Off-by-one in SA length or wrong query strand breaks everything
3. **Scale reveals bugs** - Synthetic tests passed but real data exposed the RC query bug
4. **Debugging strategy** - Building incremental tests (synthetic → real) was essential for isolating issues

---

## Performance Notes

- **Synthetic alignment**: < 0.01 sec (works correctly)
- **GRCh38 alignment**: ~0.10 sec for 1 read (produces incorrect results)
- **Index loading**: ~2-3 seconds for GRCh38 (3GB index)

---

## Test Commands

### Synthetic Test (Validates Fixes)
```bash
# Build synthetic index with C++ bwa-mem2
/home/jkane/Applications/bwa-mem2/bwa-mem2 index test_cpp_ref.fasta

# Test alignment
./target/release/ferrous-align mem -T 0 -v 4 test_cpp_ref.fasta test_query_cpp.fastq
```

### GRCh38 Test (Shows Remaining Issue)
```bash
./target/release/ferrous-align mem -T 0 -v 4 \
  /home/jkane/Genomics/Reference/b38/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna \
  /home/jkane/Genomics/HG002/test_1read.fq \
  > output.sam 2> debug.log

# Check for perfect matches
grep "PERFECT MATCH" debug.log
```

---

## References

### C++ bwa-mem2 Code Locations
- **SA length**: FMI_search.cpp:258-276 (`(ref_seq_len >> SA_COMPX) + 1`)
- **Mask array**: FMI_search.cpp:load_index() (64 elements)
- **BWT reconstruction**: FMI_search.cpp:1134-1143 (get_bwt character extraction)
- **Occurrence counting**: FMI_search.h:72 (GET_OCC macro)

### Session History
- **Session 1 (2025-11-16 morning)**: Fixed BWT interval bug, chromosome position bug
- **Session 2 (2025-11-16 afternoon)**: Index compatibility investigation, SA/mask/RC fixes

---

## Session Metrics

- **Duration**: ~4 hours
- **Commits**: Pending (3 critical fixes ready)
- **Issues Fixed**: 3 (SA length, mask array, RC query bases)
- **Issues Remaining**: 1 (scale-dependent SMEM mismatch)
- **Lines Changed**: ~50
- **Debug Lines Added**: ~30
- **Test Cases Created**: 2 (synthetic reference tests)
