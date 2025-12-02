# Interleaved FASTQ and Missing Mates Analysis

## Executive Summary

FerrousAlign currently **does NOT support interleaved FASTQ** format and has **NO error handling for missing mates** in paired-end mode. These are **critical omissions** that can lead to:

1. **Silent data corruption**: Mis-pairing reads when mate counts differ
2. **Incorrect alignment statistics**: Wrong insert size distribution
3. **Downstream pipeline failures**: SAM files with invalid mate information
4. **Hard-to-debug errors**: Symptoms appear far from root cause

**Severity**: 🔴 **HIGH** - Can produce scientifically invalid results without warning

**Status**: ⚠️ **UNHANDLED** in v0.7.0-alpha

---

## Current Implementation Analysis

### Paired-End Processing Flow (src/pipelines/linear/paired/paired_end.rs)

```rust
// Lines 408-421: Read batches from R1 and R2 files
let batch1 = match reader1.read_batch(opt.batch_size) {
    Ok(b) => b,
    Err(e) => {
        log::error!("Error reading batch from read1: {e}");
        break;  // ❌ EXITS LOOP, LEAVES R2 UNREAD
    }
};
let batch2 = match reader2.read_batch(opt.batch_size) {
    Ok(b) => b,
    Err(e) => {
        log::error!("Error reading batch from read2: {e}");
        break;  // ❌ EXITS LOOP, LEAVES R1 UNREAD
    }
};

// Lines 424-427: EOF check
if batch1.names.is_empty() {
    log::debug!("[Main] EOF reached after {batch_num} batches");
    break;  // ❌ NO CHECK IF batch2 IS ALSO EMPTY
}

// Lines 431-442: Assume equal batch sizes
let batch_size = batch1.names.len();  // ❌ ASSUMES batch1.len() == batch2.len()
let batch_bp: usize = batch1.read_boundaries.iter().map(|(_, len)| *len).sum::<usize>()
    + batch2.read_boundaries.iter().map(|(_, len)| *len).sum::<usize>();
total_reads += batch_size * 2;  // ❌ ASSUMES PERFECT PAIRING
```

### Critical Issues

#### Issue 1: No Batch Size Validation
**Location**: `paired_end.rs:431-442`

**Problem**: Code assumes `batch1.len() == batch2.len()` without verification.

**Failure Mode**:
```
R1 file: 1000 reads
R2 file: 999 reads (one missing due to sequencing error)

Batch 1:
  batch1 = 500 reads
  batch2 = 500 reads  ✅ OK

Batch 2:
  batch1 = 500 reads
  batch2 = 499 reads  ❌ SILENT MISMATCH

Result:
  - R1[999] pairs with R2[498] (WRONG MATE!)
  - R1[500] is unpaired
  - Insert size stats are corrupted
  - No warning issued
```

#### Issue 2: No Interleaved FASTQ Support
**Location**: Nowhere (feature missing)

**Problem**: Cannot read paired reads from single interleaved file.

**Industry Standard Format** (CASAVA 1.8+):
```
@INSTRUMENT:RUN:FLOWCELL:LANE:TILE:X:Y 1:N:0:BARCODE
ACGTACGT...
+
IIIIIIII...
@INSTRUMENT:RUN:FLOWCELL:LANE:TILE:X:Y 2:N:0:BARCODE
TGCATGCA...
+
JJJJJJJJ...
```

**Note**: Read name suffix changes from ` 1:` to ` 2:` for mates.

**Current Behavior**:
- Interleaved file provided → treated as single-end
- All reads processed, but pairing information LOST
- No error or warning

#### Issue 3: No Missing Mate Detection
**Location**: `paired_end.rs:262-285` (bootstrap), `467-484` (main loop)

**Problem**: Parallel processing of R1 and R2 assumes perfect synchronization.

**Code**:
```rust
let (soa_result1, soa_result2) = rayon::join(
    || process_batch_parallel(&first_batch1, ...),  // R1
    || process_batch_parallel(&first_batch2, ...),  // R2
);
```

**Assumption**: `soa_result1[i]` and `soa_result2[i]` are mates.

**Reality**: If batch sizes differ, indices become misaligned!

#### Issue 4: Silent EOF Mismatch
**Location**: `paired_end.rs:424-427`

**Problem**: Only checks if R1 is empty, ignores R2.

**Failure Scenario**:
```
R1 file: 10,000 reads (normal)
R2 file: 10,005 reads (5 extra due to pipeline error)

Loop iteration N:
  batch1 = read_batch(500) → empty (EOF)
  batch2 = read_batch(500) → 5 reads remaining

  if batch1.names.is_empty() {
      break;  // ❌ EXITS, LEAVING 5 READS IN R2 UNPROCESSED
  }
```

**Result**: 5 reads silently dropped, no warning.

---

## Failure Modes and Edge Cases

### Catastrophic Failure Mode 1: Off-by-One Mis-pairing

**Scenario**: One read missing from R1 file mid-stream

```
R1 file:
  read1
  read2
  [MISSING - sequencing error]
  read4
  read5

R2 file:
  read1
  read2
  read3
  read4
  read5

Processing:
  Batch 1 (size=3):
    R1: [read1, read2, read4]
    R2: [read1, read2, read3]

    Pairs formed:
      read1 <-> read1  ✅ CORRECT
      read2 <-> read2  ✅ CORRECT
      read4 <-> read3  ❌ WRONG MATES!

  Batch 2 (size=2):
    R1: [read5]
    R2: [read4, read5]

    Batch size mismatch!
    IF checked: Would detect error
    IF unchecked: read5 <-> read4  ❌ WRONG MATES!
```

**Impact**:
- **Insert size corruption**: All subsequent pairs have wrong template lengths
- **Mapping quality degradation**: Discordant pairs marked as proper pairs
- **Structural variant calling failure**: False positives from mis-paired reads
- **Scientific validity compromised**: Results cannot be trusted

### Catastrophic Failure Mode 2: Batch Boundary Desync

**Scenario**: File truncation during sequencing

```
R1 file: 10,000 reads
R2 file: 9,998 reads (last 2 truncated by disk full error)

Batch processing (batch_size=500):
  Batches 1-19: ✅ OK (500 pairs each)

  Batch 20:
    R1: 500 reads (9500-9999)
    R2: 498 reads (9500-9997)

    batch1.len() = 500
    batch2.len() = 498

    ❌ NO VALIDATION: Treats as 500 pairs

    Pairings:
      R1[9500] <-> R2[9500]  ✅
      R1[9501] <-> R2[9501]  ✅
      ...
      R1[9997] <-> R2[9997]  ✅
      R1[9998] <-> R2[???]   ❌ OUT OF BOUNDS!
      R1[9999] <-> R2[???]   ❌ OUT OF BOUNDS!
```

**Result**: Index out of bounds panic OR garbage data pairing.

### Edge Case 1: Interleaved FASTQ with Standard Tool

**User Command**:
```bash
# User has interleaved FASTQ (common output from older pipelines)
ferrous-align mem ref.fa interleaved.fq > out.sam
```

**Expected Behavior** (BWA-MEM):
- Detects interleaved format
- Pairs reads automatically
- Sets proper pair flags

**Actual Behavior** (FerrousAlign):
- Treats as single-end
- All reads processed independently
- NO PAIRING FLAGS SET
- User gets WRONG RESULTS without warning!

### Edge Case 2: Gzipped Interleaved FASTQ

**User Command**:
```bash
ferrous-align mem ref.fa interleaved.fq.gz > out.sam
```

**Expected**: Automatic detection + decompression + pairing

**Actual**: Decompresses but treats as single-end → SILENT FAILURE

### Edge Case 3: Mixed Batch Sizes at EOF

**Scenario**: R1 and R2 have slightly different counts

```
R1: 100,512 reads
R2: 100,510 reads (2 missing)

With batch_size=500:
  Batches 1-201: ✅ OK
  Batch 202:
    R1: 12 reads
    R2: 10 reads

    Current code:
      batch_size = batch1.names.len() = 12
      total_reads += batch_size * 2 = 24  ❌ WRONG! Only 22 reads
```

**Impact**: Incorrect statistics reported to user.

---

## Comparison with BWA-MEM2

### BWA-MEM2 Handling (Reference Implementation)

**Interleaved FASTQ Detection** (src/fastmap.cpp:866-890):
```cpp
// Detect interleaved format by checking read name suffixes
bool is_interleaved = false;
if (nargs == 1) {  // Only one file provided
    // Read first two records
    kseq_t *ks1 = kseq_init(fp);
    kseq_t *ks2 = kseq_init(fp);

    if (kseq_read(ks1) >= 0 && kseq_read(ks2) >= 0) {
        // Check if names match with /1 and /2 suffixes
        if (strcmp(ks1->name.s, ks2->name.s) == 0 ||
            (strlen(ks1->name.s) > 2 &&
             strlen(ks2->name.s) > 2 &&
             ks1->name.s[strlen(ks1->name.s)-2] == '/' &&
             ks2->name.s[strlen(ks2->name.s)-2] == '/' &&
             ks1->name.s[strlen(ks1->name.s)-1] == '1' &&
             ks2->name.s[strlen(ks2->name.s)-1] == '2')) {
            is_interleaved = true;
            fprintf(stderr, "[M::main] Interleaved FASTQ detected\n");
        }
    }
}
```

**Batch Size Validation** (src/fastmap.cpp:1024-1032):
```cpp
// After reading paired batches
if (nseq1 != nseq2) {
    fprintf(stderr, "[E::main] Read count mismatch: R1=%d, R2=%d\n",
            nseq1, nseq2);
    fprintf(stderr, "[E::main] Paired-end files must have equal read counts\n");
    exit(EXIT_FAILURE);  // HARD ERROR: Cannot continue
}
```

**Key Difference**: BWA-MEM2 **fails fast** with clear error message, preventing garbage output.

---

## Proposed Solutions

### Solution 1: Batch Size Validation (CRITICAL - Must implement)

**Location**: `paired_end.rs:422` (after reading batches)

**Code**:
```rust
// After reading both batches
let batch1 = reader1.read_batch(opt.batch_size)?;
let batch2 = reader2.read_batch(opt.batch_size)?;

// CRITICAL VALIDATION
if batch1.len() != batch2.len() {
    return Err(anyhow::anyhow!(
        "Paired-end read count mismatch in batch {}: R1={} reads, R2={} reads. \
         Paired-end FASTQ files must have exactly the same number of reads in the same order. \
         Common causes: truncated file, missing reads, or mismatched file pairs.",
        batch_num,
        batch1.len(),
        batch2.len()
    ));
}

// Also check EOF synchronization
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

**Benefits**:
- Fails fast with clear error message
- Prevents mis-pairing
- Matches BWA-MEM2 behavior
- **ZERO performance impact** (just integer comparison)

**Drawbacks**:
- None (this is strictly better than silent corruption)

**Priority**: 🔴 **CRITICAL** - Implement immediately

---

### Solution 2: Interleaved FASTQ Support (IMPORTANT - Should implement)

**Implementation Strategy**:

#### Step 1: Add Interleaved Reader

**New file**: `src/core/io/interleaved_reader.rs`

```rust
/// Reader for interleaved FASTQ (alternating R1/R2 in single file)
pub struct InterleavedFastqReader {
    inner: SoaFastqReader,
}

impl InterleavedFastqReader {
    pub fn new(path: &str) -> io::Result<Self> {
        Ok(Self {
            inner: SoaFastqReader::new(path)?,
        })
    }

    /// Read a batch of PAIRS from interleaved file
    /// Returns (batch1, batch2) where batch1[i] pairs with batch2[i]
    pub fn read_batch_paired(&mut self, batch_size: usize)
        -> io::Result<(SoAReadBatch, SoAReadBatch)>
    {
        // Read 2×batch_size reads
        let full_batch = self.inner.read_batch(batch_size * 2)?;

        if full_batch.is_empty() {
            return Ok((SoAReadBatch::new(), SoAReadBatch::new()));
        }

        // Validate interleaved pairing
        self.validate_interleaved_pairs(&full_batch)?;

        // De-interleave: even indices → R1, odd indices → R2
        let (batch1, batch2) = self.deinterleave(full_batch)?;

        Ok((batch1, batch2))
    }

    fn validate_interleaved_pairs(&self, batch: &SoAReadBatch) -> io::Result<()> {
        if batch.len() % 2 != 0 {
            return Err(io::Error::other(format!(
                "Interleaved FASTQ has odd number of reads ({}). \
                 Interleaved format requires pairs (even count).",
                batch.len()
            )));
        }

        // Check that consecutive reads are mates (optional but recommended)
        for i in (0..batch.len()).step_by(2) {
            let name1 = &batch.names[i];
            let name2 = &batch.names[i + 1];

            if !Self::are_mate_names(name1, name2) {
                log::warn!(
                    "Potential pairing issue: '{}' and '{}' do not appear to be mates",
                    name1, name2
                );
            }
        }

        Ok(())
    }

    fn are_mate_names(name1: &str, name2: &str) -> bool {
        // Check CASAVA 1.8+ format: "NAME 1:..." and "NAME 2:..."
        if let Some((prefix1, suffix1)) = name1.rsplit_once(' ') {
            if let Some((prefix2, suffix2)) = name2.rsplit_once(' ') {
                if prefix1 == prefix2 &&
                   suffix1.starts_with("1:") &&
                   suffix2.starts_with("2:") {
                    return true;
                }
            }
        }

        // Check legacy /1 and /2 format
        if name1.ends_with("/1") && name2.ends_with("/2") {
            if name1[..name1.len()-2] == name2[..name2.len()-2] {
                return true;
            }
        }

        false
    }

    fn deinterleave(&self, batch: SoAReadBatch) -> io::Result<(SoAReadBatch, SoAReadBatch)> {
        let num_pairs = batch.len() / 2;

        let mut batch1 = SoAReadBatch::new();
        let mut batch2 = SoAReadBatch::new();

        batch1.names.reserve(num_pairs);
        batch2.names.reserve(num_pairs);
        batch1.read_boundaries.reserve(num_pairs);
        batch2.read_boundaries.reserve(num_pairs);

        for i in 0..num_pairs {
            let r1_idx = i * 2;
            let r2_idx = i * 2 + 1;

            // Copy R1
            batch1.names.push(batch.names[r1_idx].clone());
            let (r1_start, r1_len) = batch.read_boundaries[r1_idx];
            let r1_seq = &batch.seqs[r1_start..r1_start + r1_len];
            let r1_qual = &batch.quals[r1_start..r1_start + r1_len];
            let r1_offset = batch1.seqs.len();
            batch1.seqs.extend_from_slice(r1_seq);
            batch1.quals.extend_from_slice(r1_qual);
            batch1.read_boundaries.push((r1_offset, r1_len));

            // Copy R2
            batch2.names.push(batch.names[r2_idx].clone());
            let (r2_start, r2_len) = batch.read_boundaries[r2_idx];
            let r2_seq = &batch.seqs[r2_start..r2_start + r2_len];
            let r2_qual = &batch.quals[r2_start..r2_start + r2_len];
            let r2_offset = batch2.seqs.len();
            batch2.seqs.extend_from_slice(r2_seq);
            batch2.quals.extend_from_slice(r2_qual);
            batch2.read_boundaries.push((r2_offset, r2_len));
        }

        Ok((batch1, batch2))
    }
}
```

#### Step 2: Auto-Detection in CLI

**Location**: `mem.rs:main_mem`

```rust
// After parsing CLI options
let is_interleaved = if opts.reads.len() == 1 {
    // Single file: check if interleaved
    detect_interleaved_format(&opts.reads[0])?
} else {
    false
};

if is_interleaved {
    log::info!("Detected interleaved FASTQ format");
    process_interleaved_paired_end(&bwa_idx, &opts.reads[0], &mut writer, &opt, &compute_ctx);
} else if opts.reads.len() == 2 {
    process_paired_end(&bwa_idx, opts.reads[0], opts.reads[1], &mut writer, &opt, &compute_ctx);
} else {
    process_single_end(&bwa_idx, &opts.reads, &mut writer, &opt, &compute_ctx);
}
```

#### Step 3: Detection Function

```rust
fn detect_interleaved_format(path: &Path) -> Result<bool> {
    use bio::io::fastq::Reader;

    let mut reader = SoaFastqReader::new(path.to_str().unwrap())?;
    let sample = reader.read_batch(2)?;

    if sample.len() < 2 {
        return Ok(false); // Not enough reads to determine
    }

    // Check if first two reads look like mates
    let name1 = &sample.names[0];
    let name2 = &sample.names[1];

    Ok(InterleavedFastqReader::are_mate_names(name1, name2))
}
```

**Benefits**:
- Automatic detection (user-friendly)
- Handles common format (CASAVA 1.8+)
- Validates pairing
- Matches BWA-MEM behavior

**Drawbacks**:
- Adds code complexity (~200 lines)
- Requires testing with various formats

**Priority**: 🟡 **IMPORTANT** - Implement in v0.8.0

---

### Solution 3: Read Name Verification (OPTIONAL - Nice to have)

**Goal**: Detect when R1 and R2 files are mismatched by comparing read names.

**Location**: `paired_end.rs:431` (after batch size check)

```rust
// After validating batch sizes match
if batch1.len() != batch2.len() {
    return Err(...);  // From Solution 1
}

// Optional: Verify read names match (catches swapped files)
for i in 0..batch1.len() {
    let name1 = strip_mate_suffix(&batch1.names[i]);
    let name2 = strip_mate_suffix(&batch2.names[i]);

    if name1 != name2 {
        return Err(anyhow::anyhow!(
            "Read name mismatch at pair {}: R1='{}' vs R2='{}'. \
             This usually indicates mismatched or incorrectly ordered FASTQ files.",
            i,
            batch1.names[i],
            batch2.names[i]
        ));
    }
}

fn strip_mate_suffix(name: &str) -> &str {
    // Remove /1, /2, or CASAVA 1:/.../2:... suffixes
    if let Some(pos) = name.rfind(' ') {
        &name[..pos]  // Strip " 1:N:0:..." suffix
    } else if name.ends_with("/1") || name.ends_with("/2") {
        &name[..name.len()-2]  // Strip /1 or /2
    } else {
        name
    }
}
```

**Benefits**:
- Catches swapped R1/R2 files
- Catches completely mismatched file pairs
- Prevents mis-pairing early

**Drawbacks**:
- Performance cost: O(n) string comparisons per batch
- May false-positive on unconventional naming schemes

**Priority**: 🟢 **OPTIONAL** - Consider for v0.9.0

---

## Implementation Roadmap

### Phase 1: Critical Fixes (v0.7.1 - Hotfix)

**Timeline**: 1 week

**Changes**:
1. ✅ Add batch size validation (Solution 1)
2. ✅ Add EOF synchronization check (Solution 1)
3. ✅ Add integration test for mismatched files
4. ✅ Update error messages with helpful diagnostics

**Testing**:
```bash
# Test 1: Normal paired-end (should pass)
./ferrous-align mem ref.fa R1.fq R2.fq > out.sam

# Test 2: Truncated R2 (should error with clear message)
head -n 4000 R2.fq > R2_truncated.fq
./ferrous-align mem ref.fa R1.fq R2_truncated.fq > out.sam
# Expected: Error message about read count mismatch

# Test 3: Swapped files (should still work but detect if name check added)
./ferrous-align mem ref.fa R2.fq R1.fq > out.sam
# Expected: Warning about reversed order
```

**Acceptance Criteria**:
- ✅ Batch size mismatch → immediate error with helpful message
- ✅ EOF mismatch → immediate error
- ✅ No performance regression on normal files

---

### Phase 2: Interleaved Support (v0.8.0)

**Timeline**: 2-3 weeks

**Changes**:
1. ✅ Implement `InterleavedFastqReader`
2. ✅ Add auto-detection in CLI
3. ✅ Add `process_interleaved_paired_end()` function
4. ✅ Update documentation

**Testing**:
```bash
# Test 1: Auto-detect interleaved
./ferrous-align mem ref.fa interleaved.fq > out.sam
# Expected: "Detected interleaved FASTQ" message

# Test 2: Gzipped interleaved
./ferrous-align mem ref.fa interleaved.fq.gz > out.sam

# Test 3: Malformed interleaved (odd read count)
# Create file with 1001 reads (odd number)
./ferrous-align mem ref.fa malformed_interleaved.fq > out.sam
# Expected: Error about odd read count

# Test 4: Validate output matches BWA-MEM2
bwa mem ref.fa interleaved.fq > bwa_out.sam
./ferrous-align mem ref.fa interleaved.fq > ferrous_out.sam
# Compare alignment results
```

**Acceptance Criteria**:
- ✅ Automatically detects interleaved format
- ✅ De-interleaves correctly
- ✅ Validates pairing
- ✅ Output matches BWA-MEM2

---

### Phase 3: Enhanced Validation (v0.9.0 - Optional)

**Timeline**: 1 week

**Changes**:
1. ✅ Add read name verification (Solution 3)
2. ✅ Add CLI flag `--strict-pairing` to enable/disable
3. ✅ Performance optimization (sample-based checking)

**Testing**:
```bash
# Test 1: Swapped R1/R2
./ferrous-align mem --strict-pairing ref.fa R2.fq R1.fq > out.sam
# Expected: Error about reversed files

# Test 2: Completely mismatched files
./ferrous-align mem --strict-pairing ref.fa sampleA_R1.fq sampleB_R2.fq > out.sam
# Expected: Error about name mismatch
```

**Acceptance Criteria**:
- ✅ Detects swapped files
- ✅ Detects mismatched samples
- ✅ Performance impact <5% with checking enabled
- ✅ Can be disabled for non-standard naming schemes

---

## Backwards Compatibility

### Breaking Changes
**None** - All changes are additive or improve error handling.

### Behavior Changes
1. **Previously silent failures now error out** (GOOD!)
   - Mismatched batch sizes: was silent corruption → now immediate error
   - EOF mismatch: was silent truncation → now immediate error

2. **New feature: Interleaved FASTQ**
   - Single FASTQ file now interpreted as interleaved if format detected
   - May affect users who incorrectly pass single file in paired mode
   - Mitigation: Check with user before assuming interleaved

### Migration Guide for Users

**If you get "read count mismatch" error**:
```
Error: Paired-end read count mismatch in batch 42: R1=500 reads, R2=498 reads
```

**Diagnosis**:
1. Check file integrity: `wc -l R1.fq R2.fq` (should be same)
2. Check for truncation: `tail R1.fq R2.fq`
3. Check for format issues: `head -n 100 R1.fq R2.fq`

**Fix**:
- If files are corrupted: Re-download or regenerate
- If one file is shorter: Truncate longer file to match (NOT RECOMMENDED)
- If reads are genuinely unpaired: Use single-end mode on each file separately

---

## Testing Strategy

### Unit Tests

```rust
#[test]
fn test_batch_size_mismatch_detection() {
    // Create mock readers with mismatched counts
    let batch1 = create_test_batch(100);
    let batch2 = create_test_batch(99);

    let result = validate_batch_pairing(&batch1, &batch2);

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("mismatch"));
}

#[test]
fn test_interleaved_deinterleaving() {
    let interleaved = create_interleaved_batch(50); // 100 reads total

    let (batch1, batch2) = deinterleave(interleaved).unwrap();

    assert_eq!(batch1.len(), 50);
    assert_eq!(batch2.len(), 50);
    assert_eq!(batch1.names[0], "read1/1");
    assert_eq!(batch2.names[0], "read1/2");
}
```

### Integration Tests

```bash
#!/bin/bash
# tests/test_missing_mates.sh

set -e

REF="test_data/ref.fa"

# Test 1: Normal paired-end (should pass)
echo "Test 1: Normal paired-end"
./target/release/ferrous-align mem $REF test_data/R1.fq test_data/R2.fq > /dev/null
echo "✅ PASS"

# Test 2: Truncated R2 (should fail)
echo "Test 2: Truncated R2"
head -n 400 test_data/R2.fq > /tmp/R2_truncated.fq
if ./target/release/ferrous-align mem $REF test_data/R1.fq /tmp/R2_truncated.fq > /dev/null 2>&1; then
    echo "❌ FAIL: Should have detected mismatch"
    exit 1
else
    echo "✅ PASS: Correctly detected mismatch"
fi

# Test 3: Interleaved (should pass)
echo "Test 3: Interleaved FASTQ"
./target/release/ferrous-align mem $REF test_data/interleaved.fq > /dev/null
echo "✅ PASS"

echo "All tests passed!"
```

---

## Performance Impact

### Solution 1 (Batch Size Validation)
- **Cost**: 2 integer comparisons per batch
- **Frequency**: Once per 500K reads
- **Impact**: <0.001% (negligible)

### Solution 2 (Interleaved Support)
- **Cost**: De-interleaving overhead (memory copies)
- **Frequency**: Once per batch
- **Impact**: ~2-3% for interleaved files, 0% for standard paired files

### Solution 3 (Name Verification)
- **Cost**: O(n) string comparisons per batch
- **Frequency**: Once per 500K pairs = 1M reads
- **Impact**: ~5-10% if enabled (why it should be optional)

---

## Recommendations

### Immediate Action Items (v0.7.1 Hotfix)

1. **CRITICAL**: Implement Solution 1 (batch size validation)
   - Lines to add: ~30
   - Testing: 1 day
   - Review: 1 day
   - **Risk**: Low (strictly improves behavior)

2. **Document limitation** in CLAUDE.md:
   ```markdown
   ## Known Limitations (v0.7.0)
   - ❌ No interleaved FASTQ support (use separate R1/R2 files)
   - ❌ Paired-end files must have identical read counts
   ```

### Medium-Term (v0.8.0)

3. **IMPORTANT**: Implement Solution 2 (interleaved support)
   - New file: `interleaved_reader.rs` (~250 lines)
   - Integration: ~100 lines in `mem.rs` and `paired_end.rs`
   - Testing: Create interleaved test datasets
   - **Risk**: Medium (new code path)

### Long-Term (v0.9.0+)

4. **OPTIONAL**: Implement Solution 3 (name verification)
   - Add as opt-in feature via `--strict-pairing` flag
   - Document performance impact
   - **Risk**: Low (optional feature)

---

## Conclusion

FerrousAlign currently has **two critical gaps**:

1. ❌ **No validation** that paired-end files have equal read counts
2. ❌ **No support** for interleaved FASTQ format

These gaps can lead to **scientifically invalid results** that are hard to detect. The fixes are straightforward:

- **Solution 1** (batch validation): 30 lines of code, zero performance impact, **MUST implement**
- **Solution 2** (interleaved support): 250 lines of code, ~2% overhead for interleaved files, **SHOULD implement**
- **Solution 3** (name verification): 50 lines of code, ~5% overhead, **OPTIONAL**

**Recommended timeline**:
- v0.7.1 (1 week): Critical validation fixes
- v0.8.0 (3 weeks): Interleaved FASTQ support
- v0.9.0 (optional): Enhanced validation

---

## References

- BWA-MEM2 source: `src/fastmap.cpp` (interleaved detection, validation)
- FASTQ format spec: https://www.ncbi.nlm.nih.gov/sra/docs/submitformats/#fastq-files
- CASAVA 1.8+ read naming: https://support.illumina.com/help/BaseSpace_OLH_009008/Content/Source/Informatics/BS/NamingConvention_FASTQ-files-swBS.htm
- SAM spec v1.6: https://samtools.github.io/hts-specs/SAMv1.pdf (pairing flags)
