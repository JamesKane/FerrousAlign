# Critical Issues Summary - MUST READ

## ✅ RESOLVED: AVX-512 Crash (2025-12-02)

### The Problem
**AVX-512 kernel crashed with SIGSEGV** during mate rescue alignment when processing sequences exceeding workspace capacity.

**Root Cause**: Misaligned buffer allocation
- Workspace pre-allocated for 128bp sequences (insufficient for modern 150bp PE reads)
- Fallback used `vec![0u8]` which provides only ~48-byte alignment
- AVX-512 `_mm512_store_si512` requires 64-byte alignment → **CRASH**

### The Fix (commit e763e4a)
1. Added `allocate_aligned_buffer()` helper using `std::alloc::alloc` with 64-byte Layout
2. Increased `KSW_MAX_SEQ_LEN` from 128 to 512 bytes
3. Added unit tests validating alignment requirements

**Status**: ✅ **FIXED** - validated on 10K and 100K read datasets

---

## 🔴 CRITICAL: Missing Mate Validation

### The Problem
**Current code DOES NOT validate** that R1 and R2 files have the same number of reads per batch.

**Location**: `src/pipelines/linear/paired/paired_end.rs:431-442`

**Impact**: If R1 and R2 files have different read counts (due to truncation, sequencing errors, or corruption), the code will:
- Silently mis-pair reads (R1[i] paired with wrong R2[j])
- Produce corrupt insert size statistics
- Generate scientifically invalid alignments
- **NO WARNING OR ERROR ISSUED**

### The Fix (30 lines of code)

Add after line 421 in `paired_end.rs`:

```rust
// CRITICAL VALIDATION: Ensure R1 and R2 have same number of reads
if batch1.len() != batch2.len() {
    return Err(anyhow::anyhow!(
        "Paired-end read count mismatch in batch {}: R1={} reads, R2={} reads. \
         Paired-end FASTQ files must have exactly the same number of reads. \
         Common causes: truncated file, missing reads, or mismatched file pairs.",
        batch_num,
        batch1.len(),
        batch2.len()
    ));
}

// Check EOF synchronization
if batch1.is_empty() && !batch2.is_empty() {
    return Err(anyhow::anyhow!(
        "R1 file ended but R2 has {} reads remaining. Files are not properly paired.",
        batch2.len()
    ));
}
if !batch1.is_empty() && batch2.is_empty() {
    return Err(anyhow::anyhow!(
        "R2 file ended but R1 has {} reads remaining. Files are not properly paired.",
        batch1.len()
    ));
}
```

**Performance impact**: ZERO (just 3 integer comparisons)

**Priority**: 🔴 **MUST FIX IN v0.7.1** (hotfix within 1 week)

---

## 🟡 IMPORTANT: No Interleaved FASTQ Support

### The Problem
**FerrousAlign does NOT support interleaved FASTQ files** (single file with alternating R1/R2 reads).

This is a standard format output by many sequencing pipelines and supported by BWA-MEM.

**Current behavior**: Treats interleaved file as single-end → **WRONG RESULTS, NO WARNING**

### Example Failure
```bash
# User has interleaved FASTQ (common format)
ferrous-align mem ref.fa interleaved.fq > out.sam

# Expected: Paired-end alignment with proper pair flags
# Actual: Single-end alignment, no pairing information, WRONG OUTPUT
```

### The Fix (250 lines of code)

Implement `InterleavedFastqReader` that:
1. Auto-detects interleaved format by checking read name patterns
2. De-interleaves into separate R1/R2 batches
3. Validates pairing

**Priority**: 🟡 **SHOULD FIX IN v0.8.0** (within 3 weeks)

---

## Failure Scenarios

### Scenario 1: Truncated R2 File
```
R1: 10,000 reads
R2: 9,998 reads (disk full during write)

Current behavior: ❌ SILENT MIS-PAIRING
Fixed behavior: ✅ Error: "read count mismatch"
```

### Scenario 2: Missing Read Mid-Stream
```
R1: [read1, read2, MISSING, read4, read5]
R2: [read1, read2, read3,   read4, read5]

Current behavior: ❌ read4 <-> read3 (WRONG MATES!)
Fixed behavior: ✅ Error: "read count mismatch"
```

### Scenario 3: Interleaved FASTQ
```
Input: interleaved.fq (10K pairs in single file)

Current behavior: ❌ Treats as 10K single-end reads
Fixed behavior: ✅ Detects format, de-interleaves, pairs correctly
```

---

## Comparison with BWA-MEM2

| Feature | BWA-MEM2 | FerrousAlign v0.7.0 | Status |
|---------|----------|---------------------|--------|
| Batch size validation | ✅ Errors immediately | ❌ Silent failure | 🔴 Critical |
| EOF sync check | ✅ Errors immediately | ❌ Silent truncation | 🔴 Critical |
| Interleaved FASTQ | ✅ Auto-detects | ❌ Not supported | 🟡 Important |
| Read name verification | ✅ Optional | ❌ Not implemented | 🟢 Nice-to-have |

---

## Recommended Actions

### Immediate (v0.7.1 Hotfix - 1 week)
1. ✅ Add batch size validation (Solution 1)
2. ✅ Add EOF synchronization check
3. ✅ Add integration tests
4. ✅ Update CLAUDE.md with known limitation

### Short-term (v0.8.0 - 3 weeks)
5. ✅ Implement interleaved FASTQ support (Solution 2)
6. ✅ Auto-detection in CLI
7. ✅ Comprehensive testing

### Long-term (v0.9.0+ - Optional)
8. ⚪ Add read name verification (Solution 3)
9. ⚪ Performance optimization

---

## Testing

### Test Suite to Add

```bash
# Test 1: Normal paired-end (should pass)
./ferrous-align mem ref.fa R1.fq R2.fq > out.sam

# Test 2: Truncated R2 (should error)
head -n 400 R2.fq > R2_truncated.fq
./ferrous-align mem ref.fa R1.fq R2_truncated.fq > out.sam
# Expected: Error about read count mismatch

# Test 3: EOF mismatch (should error)
# (R1 has 1000 reads, R2 has 1005 reads)
./ferrous-align mem ref.fa R1.fq R2_extra.fq > out.sam
# Expected: Error about remaining reads

# Test 4: Interleaved (should auto-detect and work)
./ferrous-align mem ref.fa interleaved.fq > out.sam
# Expected: Paired-end output with proper flags
```

---

## Documentation Updates

### Update CLAUDE.md

Add to "Known Limitations" section:

```markdown
## Known Limitations (v0.7.0-alpha)

### Paired-End Processing
- ❌ **CRITICAL**: No validation that R1/R2 files have equal read counts
  - Can produce incorrect results if files are mismatched
  - **Workaround**: Validate file counts before running: `wc -l R1.fq R2.fq`
  - **Fix**: v0.7.1 will add validation and error immediately on mismatch

- ❌ **No interleaved FASTQ support**
  - Single file with alternating R1/R2 reads not supported
  - **Workaround**: De-interleave first using `seqtk split` or similar
  - **Fix**: v0.8.0 will add auto-detection and support
```

---

## References

- Full analysis: `Interleaved_FASTQ_and_Missing_Mates_Analysis.md`
- Code locations: `src/pipelines/linear/paired/paired_end.rs:422-442`
- BWA-MEM2 reference: `src/fastmap.cpp:1024-1032` (validation)
- BWA-MEM2 reference: `src/fastmap.cpp:866-890` (interleaved detection)

---

## Risk Assessment

| Issue | Severity | Likelihood | Impact | Priority |
|-------|----------|------------|--------|----------|
| Batch size mismatch | 🔴 Critical | Medium | Scientific validity compromised | P0 |
| EOF desync | 🔴 Critical | Low | Silent data loss | P0 |
| No interleaved support | 🟡 Important | High | User confusion, wrong results | P1 |
| No name verification | 🟢 Minor | Low | Catches swapped files | P2 |

**Overall Risk**: 🔴 **HIGH** - Can produce scientifically invalid results without warning

**Mitigation**: Implement batch size validation in v0.7.1 hotfix (1 week timeline)
