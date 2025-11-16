# SMEM Generation Rewrite - Progress Summary
**Date**: 2025-11-16
**Status**: Major rewrite completed, debugging in progress

## What Was Accomplished

### 1. Root Cause Analysis ✅
Identified three critical differences between C++ bwa-mem2 and Rust implementation:

1. **SMEM `l` field encoding**: C++ uses `l = count[3-a]` (reverse complement), not `l = count[a+1]`
2. **backward_ext() cumulative sum**: C++ uses cascading cumulative sum for `l[]` values:
   ```cpp
   l[3] = smem.l + sentinel_offset;
   l[2] = l[3] + s[3];
   l[1] = l[2] + s[2];
   l[0] = l[1] + s[1];
   ```
3. **Two-phase bidirectional algorithm**: C++ uses forward + backward extension, Rust was only doing backward

### 2. Core Fixes Implemented ✅

#### Fixed SMEM Initialization
```rust
// BEFORE: l = bwa_idx.bwt.l2[(current_base_code + 1) as usize]
// AFTER:  l = bwa_idx.bwt.l2[(3 - current_base_code) as usize]
```

#### Rewrote backward_ext()
- Implemented cumulative sum approach matching C++ exactly
- C++ reference: FMI_search.cpp lines 1025-1052

#### Added forward_ext() Helper
```rust
// Forward extension = k/l swap + backwardExt(3-a) + k/l swap back
pub fn forward_ext(bwa_idx: &BwaIndex, smem: SMEM, a: u8) -> SMEM
```

#### Complete SMEM Generation Rewrite
- Rewrote main loop to implement C++ two-phase algorithm
- Phase 1: Forward extension (extends `n` forward, collects intermediate SMEMs)
- Phase 2: Backward search (extends `m` backward, generates final bidirectional SMEMs)
- Implemented for both forward and reverse complement strands
- ~200 lines of new code following C++ specification exactly

## Current Status

### What's Working ✅
- Forward extension phase: Correctly collecting 12-13 intermediate SMEMs per position
- SMEM interval sizes above threshold: s=1429, 1112, 870 (all > min_intv=500)
- Code compiles without errors

### What's Not Working ⚠️
- **SMEM lengths too short**: Getting length 11, need >= 19 (min_seed_len parameter)
- **Zero SMEMs output**: All SMEMs filtered due to length < min_seed_len
- **Backward phase not extending enough**: Should extend `m` backward to create longer SMEMs

### Analysis
The algorithm works in two directions:
- **Forward extension**: Extends `n` forward by ~11 positions (until interval drops below threshold)
- **Backward extension**: Should extend `m` backward by many more positions (up to 96bp total based on C++ behavior)

The issue is that the backward phase isn't producing the long extensions seen in C++ bwa-mem2.

## Next Steps

### Immediate (Debug backward phase)
1. **Add detailed backward phase logging**:
   - Log each backward extension step
   - Track how `m` is being updated
   - Monitor when/why backward search stops

2. **Compare with C++ behavior**:
   - Run C++ with diagnostic output
   - Compare backward extension iterations
   - Identify where Rust deviates

3. **Potential issues to investigate**:
   - Loop iteration logic (C++ lines 607-649)
   - SMEM selection from prev_array
   - Output conditions (when SMEMs are saved)
   - Edge case handling for position x=0

### After Debugging
1. Fix backward phase to generate proper length SMEMs
2. Test with real data: should get ~4 long SMEMs (up to 96bp)
3. Verify alignment output matches C++ bwa-mem2
4. Remove debug logging
5. Performance testing

## Test Command
```bash
./target/release/ferrous-align mem -v 4 \
  /home/jkane/Genomics/Reference/b38/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna \
  /home/jkane/Genomics/HG002/test_1read.fq 2>&1 | grep -E "(Position|OUTPUT)"
```

## Files Modified
- `src/align.rs`: Complete SMEM generation rewrite (~200 lines)
- `SESSION_NOTES.md`: Documented findings and current status
- Added helper function `forward_ext()`
- Rewrote `backward_ext()` with cumulative sum approach

## Key Insights
1. The `l` field in SMEM is NOT a standard BWT interval endpoint - it encodes reverse complement information
2. Traditional BWT invariant `s = l - k` does NOT hold in bwa-mem2's implementation
3. Forward extension is implemented via k/l swap + complement base trick
4. Bidirectional search is critical - forward extends n, backward extends m
5. The "super maximal" in SMEM comes from the bidirectional constraint

## References
- C++ code: `/tmp/bwa-mem2-diag/src/FMI_search.cpp`
- Function: `getSMEMsOnePosOneThread` (lines 496-670)
- Backward extension: `backwardExt` (lines 1025-1052)
