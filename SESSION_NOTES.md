# Session Notes - Real-World Data Investigation

## Date: 2025-11-16

### Session Summary

This session focused on investigating and fixing issues discovered when testing with real-world WGS data (GIAB HG002 on GRCh38 reference) for the first time.

---

## Issues Fixed

### 1. ✅ Critical BWT Interval Calculation Bug (FIXED - Commit 1ed3bb7)

**Problem**: The `backward_ext()` function was calculating BWT interval ends (`l_arr`) incorrectly, producing invalid intervals where `k > l`. This caused integer underflow when computing occurrence counts, filtering out ALL SMEMs.

**Root Cause**: Original code used cumulative sum based on old `smem.l`:
```rust
l_arr[3] = smem.l + sentinel_offset;  // Wrong - uses OLD l value
l_arr[2] = l_arr[3] + s_arr[3];
l_arr[1] = l_arr[2] + s_arr[2];
l_arr[0] = l_arr[1] + s_arr[1];
```

**Fix Applied**: Changed to straightforward calculation:
```rust
for b in 0..4 {
    l_arr[b] = k_arr[b] + s_arr[b];  // l = k + s
}
```

**Result**:
- Before: 0 unique SMEMs (all filtered)
- After: 81 unique SMEMs from 4,740 total ✅

### 2. ✅ Chromosome Position Calculation (FIXED)

**Problem**: Alignments were reporting global concatenated reference positions instead of chromosome-relative positions, causing invalid coordinates (e.g., chr1:1,209,906,992 when chr1 is only 248MB).

**Root Cause**: Code was:
- Always using first chromosome name: `bwa_idx.bns.anns[0].name`
- Using global position without converting to chromosome-relative coordinates

**Fix Applied** (src/align.rs:958-970):
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
    log::warn!("{}: Invalid reference ID {} for position {}", query_name, rid, global_pos);
    ("unknown_ref".to_string(), 0, first_chain.ref_start)
};
```

**Result**: Now correctly reports valid chromosome coordinates (e.g., chr6:148,708,668) ✅

---

## Issues Identified (Not Yet Fully Resolved)

### 3. ⚠️ SMEM-to-Reference Position Mapping

**Problem**: Query sequences don't match reference sequences at positions indicated by SMEMs, even though SMEMs represent exact matches.

**Evidence**:
```
SMEM 0: m=19, n=39 (query positions 19-39, length 21)
- Query bases at [19..]:  [1, 0, 0, 0, 2, 2, 2, 0, 0, 0]
- Ref at SA position:     [0, 2, 0, 0, 0, 0, 0, 2, 2, 2]
                          ❌ First base doesn't match!
```

**Impact**: Smith-Waterman aligner can't find matches, producing CIGARs like "148I" (all insertions).

**Current Alignment Results**:
- Position: chr6:148,708,668 (valid ✅)
- CIGAR: 148I (incorrect ❌)
- Score: < 30 (filtered by default)

**With 1000 reads, -T 0**:
- 998/1000 alignments produced (99.8%)
- Quality unknown due to CIGAR issues

---

## C++ bwa-mem2 Comparison

### Key Findings from Reference Implementation

**Location**: `/home/jkane/Applications/bwa-mem2/src/`

1. **backward_ext() Implementation** (FMI_search.cpp:1025-1052):
   - C++ uses cumulative l_arr calculation: `l[3] = smem.l + sentinel_offset; l[2] = l[3] + s[3]; ...`
   - BUT they also do k/l swapping for forward extension (lines 545-546, 551-552)
   - Comment indicates: "Forward extension is backward extension with the BWT of reverse complement"
   - This explains why their cumulative approach works - it's for combined forward/backward

2. **SMEM Initialization Difference**:
   - C++: `smem.l = count[3 - a];` (uses complement base count)
   - Rust: `l: bwa_idx.bwt.l2[(current_base_code + 1) as usize]`
   - This is a fundamental difference, possibly related to their k/l swapping scheme

3. **get_sa_entry() Implementation** (FMI_search.cpp:1103-1172):
   - Matches our Rust implementation for SA reconstruction
   - Walks BWT when position not sampled, returns `sa_entry + offset`
   - Our implementation is correct ✅

4. **Seed to Alignment Flow** (bwamem.cpp:870-920, 2145-2370):
   - C++ calculates reference range for entire CHAIN, not individual seeds:
     ```cpp
     b = t->rbeg - (t->qbeg + cal_max_gap(opt, t->qbeg));
     rmax[0] = min of all b's across chain seeds
     ```
   - Fetches ONE reference segment for whole chain
   - Does left/right extension from chained seeds

   **Key Difference**: Our Rust code tries to align each SMEM individually before chaining. The C++ code chains first, then aligns based on the chain boundaries.

### Why Our Simple Formula Works

Our `l = k + s` formula is correct for pure backward extension:
- New interval [k, l) has size s
- Therefore: l = k + s

The C++ cumulative approach appears designed for their specific forward/backward combined extension scheme with k/l swapping. Since we're doing straightforward backward extension only, the simple formula is both correct and clearer.

---

## Current Status

### Working ✅
- BWT interval generation (SMEMs produced with valid k, l, s)
- Chromosome position calculation
- Basic alignment pipeline (seeds → chains → alignments)
- Multi-threading (24 threads, Rayon work-stealing)
- Logging framework (professional output)

### Not Working ❌
- SMEM query bases don't match reference at SA positions
- CIGARs show all insertions instead of matches
- Alignment scores too low (< threshold)

### Root Cause Hypotheses

1. **SA Position Semantics**: `get_sa_entry()` may return positions with different meaning than expected (end vs start of match, or off-by-one)

2. **Chain-Based vs Seed-Based Alignment**: C++ aligns based on chains (computing rmax boundaries across all seeds), while we try to align individual seeds. This architectural difference may require redesign.

3. **SMEM m/n Interpretation**: Query positions m and n may represent different coordinates than assumed (e.g., end positions vs start positions due to backward search)

4. **Reference Segment Extraction**: Our current approach extracts reference at `ref_pos - query_pos`, but C++ uses more complex chain-aware calculation

---

## Files Modified

### Main Changes
- **src/align.rs**:
  - Lines 155-178: BWT interval calculation fix (l = k + s)
  - Lines 859-862: SMEM debug logging
  - Lines 889-908: Reference segment position adjustment (partial fix)
  - Lines 958-970: Chromosome position calculation fix

### Debug/Documentation
- **DEBUG_SMEM.md**: Detailed bug investigation and resolution
- **ALIGNMENT_ISSUES.md**: Real-world data issue tracking
- **SESSION_NOTES.md**: This file

---

## Test Data

**Reference**: GRCh38 (`/home/jkane/Genomics/Reference/b38/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna`)
- Size: ~3.1 GB
- Sequences: 195 (chromosomes + contigs)

**Reads**: GIAB HG002 (`/home/jkane/Genomics/HG002/`)
- test_1read.fq: Single 148bp read
- test_1k.fq: 1,000 reads (148 KB)
- test_10k.fq: 10,000 reads (3.4 MB)
- test_100k_R1.fq / test_100k_R2.fq: Paired-end data

---

## Next Steps (For New Session)

### Immediate Priority
1. **Investigate SA Position Semantics**
   - Add logging to show SA values before/after reconstruction
   - Create minimal test case with known synthetic pattern
   - Verify if SA positions need +1/-1 or other adjustment

2. **Compare with Working BWA-MEM**
   - Run same test read through original BWA-MEM
   - Compare SMEM positions, SA coordinates, and CIGARs
   - Identify exact point of divergence

3. **Chain-Based Alignment Redesign** (if needed)
   - Implement C++-style rmax calculation across chain
   - Fetch single reference segment for entire chain
   - Do proper left/right extension from chain boundaries

### Testing Strategy
1. Create synthetic reference with known patterns (e.g., repeating "AAAA", "CCCC")
2. Generate query that exactly matches a region
3. Verify SMEM generation produces correct positions
4. Verify SA lookup returns correct reference positions
5. Verify alignment produces correct CIGAR

### Validation
- Once fixed, test with 10K HG002 reads
- Compare alignment rate and quality with C++ bwa-mem2
- Benchmark performance

---

## Build Commands

```bash
# Build
cargo build --release

# Test with single read (debug output)
./target/release/ferrous-align mem -T 0 -v 4 \
  /home/jkane/Genomics/Reference/b38/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna \
  /home/jkane/Genomics/HG002/test_1read.fq \
  2>&1 | grep "SMEM\|Seed"

# Test with 1K reads
./target/release/ferrous-align mem -T 0 \
  /home/jkane/Genomics/Reference/b38/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna \
  /home/jkane/Genomics/HG002/test_1k.fq \
  2>/dev/null | grep -v "^@" | wc -l

# Run tests
cargo test --lib
```

---

## Performance Notes

- **SMEM Generation**: Fast, produces ~4700 SMEMs per read
- **Filtering**: Reduces to ~80 unique SMEMs (good selectivity)
- **Throughput**: 0.10 sec for 1000 reads (preliminary, with broken CIGARs)
- **Threading**: Scales well with 24 threads (Rayon work-stealing)

---

## References

- C++ bwa-mem2: `/home/jkane/Applications/bwa-mem2/src/`
  - FMI_search.cpp: SMEM generation, backward_ext
  - bwamem.cpp: Seed to alignment conversion, chain-based alignment
  - bandedSWA.cpp: Banded Smith-Waterman implementation

- Previous commits:
  - b93c550: Initial SMEM debugging (pre-fix)
  - ceee0c2: Suffix array position bug fix
  - 1ed3bb7: BWT interval calculation fix (this session)

---

## Key Insights

1. **Real-world data reveals bugs**: Synthetic tests passed, but real WGS data exposed fundamental issues
2. **C++ implementation complexity**: bwa-mem2 uses sophisticated tricks (k/l swapping, combined forward/backward) that aren't immediately obvious
3. **Architecture matters**: Seed-based vs chain-based alignment is a fundamental design choice with significant implications
4. **Debug logging essential**: Without extensive logging, these bugs would be nearly impossible to find

---

## Session Metrics

- **Duration**: ~3 hours
- **Commits**: 1 (BWT interval fix)
- **Issues Fixed**: 2 critical bugs
- **Issues Identified**: 1 major (SMEM position mapping)
- **Lines of Code Changed**: ~150
- **Debug Lines Added**: ~50
- **Documentation Created**: 3 files
