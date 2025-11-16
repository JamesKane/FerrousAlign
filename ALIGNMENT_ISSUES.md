# Real-World Alignment Issues - Investigation Log

## Session: 2025-11-16 (Continued from BWT fix)

### Issues Found and Fixed

#### 1. ✅ Chromosome Position Calculation (FIXED)
**Problem**: Alignments were reporting global concatenated positions instead of chromosome-relative positions.

**Example**: Reported `chr1:1209906992` when chr1 is only 248MB long.

**Root Cause**: Code was:
- Always using first chromosome name (`bwa_idx.bns.anns[0].name`)
- Using global position without converting to chromosome-relative

**Fix Applied** (`src/align.rs:958-970`):
```rust
// Convert global position to chromosome-specific position
let global_pos = first_chain.ref_start as i64;
let (pos_f, _is_rev_depos) = bwa_idx.bns.bns_depos(global_pos);
let rid = bwa_idx.bns.bns_pos2rid(pos_f);

let (ref_name, ref_id, chr_pos) = if rid >= 0 && (rid as usize) < bwa_idx.bns.anns.len() {
    let ann = &bwa_idx.bns.anns[rid as usize];
    let chr_relative_pos = pos_f - ann.offset as i64;
    (ann.name.clone(), rid as usize, chr_relative_pos as u64)
} else {
    // fallback...
};
```

**Result**: Now correctly reports `chr6:148708668` ✅

#### 2. ⚠️ SMEM-to-Reference Position Mismatch (INVESTIGATION ONGOING)

**Problem**: Query sequences don't match reference sequences at the positions indicated by SMEMs.

**Evidence**:
```
SMEM 0: m=19, n=39, len=21
- Query at positions 19-39: [1, 0, 0, 0, 2, 2, 2, 0, 0, 0]
- Ref at SA position:        [0, 2, 0, 0, 0, 0, 0, 2, 2, 2]
                              ❌ First base doesn't match!
```

**Attempted Fix** (`src/align.rs:889-894`):
```rust
// Adjust reference position backwards by query_pos
// Since SMEM matched at query position m, but we're aligning from position 0
let ref_start_for_full_query = if seed.query_pos as u64 > ref_pos {
    0
} else {
    ref_pos - seed.query_pos as u64
};
```

**Status**: Partial - adjustment logic correct in theory, but sequences still don't match.

**Hypotheses**:
1. **get_sa_entry() offset issue**: Maybe SA positions need adjustment
2. **Backward search semantics**: BWT interval might point to pattern end, not start
3. **Strand confusion**: Reverse complement handling might be incorrect
4. **SMEM coordinates**: m/n might represent different positions than assumed

### Current Alignment Results

**Test**: Single read from HG002 (148bp)

**Before all fixes**:
- 0 alignments (all SMEMs filtered due to invalid intervals)

**After BWT fix**:
- 1 alignment produced
- Position: `chr6:148708668` (valid)
- CIGAR: `148I` (all insertions - INCORRECT)
- Score: < 30 (filtered by default threshold)

**After position fixes**:
- Same results, CIGAR still incorrect

**With 1000 reads, -T 0**:
- 998/1000 alignments (99.8%)
- But quality unknown due to CIGAR issues

### Debug Output Analysis

**SMEM Generation** (working correctly now):
```
Generated 4740 SMEMs, filtered to 81 unique
Using 10 of 81 filtered SMEMs for alignment
Found 10 seeds, 10 alignment jobs
```

**Seed Extension**:
```
Extended 10 seeds, 10 CIGARs produced
Chaining produced 1 chains
```

**Sequence Comparison** (showing the problem):
```
Seed 0: query_first_10=[0, 3, 1, 1, 0, 0, 1, 1, 1, 0]
Seed 0: target_first_10=[1, 3, 0, 1, 1, 1, 0, 0, 3, 3]
                         ❌ Mismatch at multiple positions!
```

### Next Steps

1. **Investigate get_sa_entry() semantics**
   - Add logging to show SA values before and after reconstruction
   - Compare with C++ bwa-mem2 get_sa_entry implementation
   - Check if positions need +1/-1 adjustment

2. **Verify SMEM backward search**
   - Check if BWT interval [k,l) points to start or end of match
   - Verify m/n coordinates are query start/end (not reversed)
   - Test with known simple example (e.g., "AAAA" pattern)

3. **Check reverse complement handling**
   - Verify is_rev_comp flag is set correctly
   - Check if RC sequences need special position calculation
   - Ensure query orientation matches reference orientation

4. **Create minimal reproducible test**
   - Generate synthetic reference with known pattern
   - Search for exact match
   - Verify SA position points to correct location

### Test Commands

```bash
# Build
cargo build --release

# Test single read with debug output
./target/release/ferrous-align mem -T 0 -v 4 \
  /home/jkane/Genomics/Reference/b38/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna \
  /home/jkane/Genomics/HG002/test_1read.fq \
  2>&1 | grep "SMEM\|Seed\|query_first\|target_first"

# Test 1K reads
./target/release/ferrous-align mem -T 0 \
  /home/jkane/Genomics/Reference/b38/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna \
  /home/jkane/Genomics/HG002/test_1k.fq \
  2>/dev/null | grep -v "^@" | wc -l
```

### Files Modified

- `src/align.rs`:
  - Lines 958-970: Chromosome position calculation fix
  - Lines 859-862: SMEM debug logging
  - Lines 889-908: Reference segment position adjustment
  - Lines 900-905: Sequence comparison debug logging

### References

- BWT/SA reconstruction: `src/align.rs:1174-1242` (get_sa_entry)
- BNS position helpers: `src/bntseq.rs:327-336` (bns_depos), `339-360` (bns_pos2rid)
- SMEM generation: `src/align.rs:590-675` (backward search loops)

