# Minimal Reproduction Case Analysis

**Read Pair**: HISEQ1:18:H8VC6ADXX:1:1101:1150:14380

## Expected Behavior (C++ bwa-mem2)

### R1 (First in Pair, Reverse Strand)
```
Flag: 81 (0x51) = paired + proper_pair + reverse + first_in_pair
RNAME: chrX
POS: 126402382
MAPQ: 60
CIGAR: 148M
RNEXT: = (same chromosome)
PNEXT: 126402200 (mate position)
TLEN: -330 (negative = upstream of mate)
MC:Z: 148M (mate CIGAR)
NM:i: 1 (one mismatch)
AS:i: 146 (alignment score)
```

### R2 (Second in Pair, Forward Strand)
```
Flag: 161 (0xA1) = paired + proper_pair + mate_reverse + second_in_pair
RNAME: chrX
POS: 126402200
MAPQ: 60
CIGAR: 148M
RNEXT: = (same chromosome)
PNEXT: 126402382 (mate position)
TLEN: 330 (positive = downstream of mate)
MC:Z: 148M (mate CIGAR)
NM:i: 1 (one mismatch)
AS:i: 143 (alignment score)
```

**Summary**: Both reads map to chrX, 182bp apart (126402200 to 126402382), insert size ~330bp, proper pair.

## Actual Behavior (Rust ferrous-align)

### R1 (First in Pair) ❌ BUGGY
```
Flag: 69 (0x45) = paired + UNMAPPED + first_in_pair
RNAME: chrX ⚠️ Should be "*" if unmapped!
POS: 126402200
MAPQ: 0
CIGAR: * (unmapped)
RNEXT: = ⚠️ Contradictory with unmapped!
PNEXT: 126402200
TLEN: 0
MC:Z: 148M ⚠️ Contradictory - says mate IS mapped!
```

### R2 (Second in Pair) ⚠️ PARTIALLY CORRECT
```
Flag: 137 (0x89) = paired + mate_unmapped + second_in_pair
RNAME: chrX ✓
POS: 126402200 ✓
MAPQ: 60 ✓ (High confidence!)
CIGAR: 148M ✓ (Successfully aligned!)
RNEXT: * ❌ Should be "=" (mate is on same chr)
PNEXT: 0 ❌ Should be 126402382 (mate position)
TLEN: 0 ❌ Should be 330 or -330
MC:Z: * ❌ Should be "148M" (mate CIGAR)
```

## Key Findings

### 1. ✅ Alignment Algorithm Works
**R2 successfully aligned**: CIGAR=148M, MAPQ=60, correct position (chrX:126402200)
- This proves FM-Index search, seed generation, and Smith-Waterman are working
- The alignment matches C++ position exactly

### 2. ❌ R1 Alignment Missing
**R1 should align to chrX:126402382** (per C++ reference)
- Rust incorrectly marks it as unmapped (flag 0x4)
- But has contradictory fields suggesting alignment exists somewhere:
  - RNAME=chrX (not "*")
  - MC:Z:148M (knows mate is mapped with 148M CIGAR)
  - RNEXT="=" (knows mate is on same chromosome)

**Hypotheses**:
- a) R1 alignment was generated but discarded during pairing
- b) R1 alignment failed score threshold (opt.t)
- c) R1 alignment exists but wasn't selected as primary

### 3. ❌ Mate Pairing Completely Broken
**R2 doesn't know about R1**:
- Flag 0x8 (mate_unmapped) set incorrectly
- RNEXT="*", PNEXT=0, TLEN=0 (missing mate info)
- MC:Z:"*" (missing mate CIGAR)
- No proper_pair flag (0x2)

**This is the critical bug**: Even though R2 is aligned, the pairing logic failed to:
1. Link R2 back to R1's alignment
2. Set proper_pair flags
3. Calculate insert size (TLEN)
4. Populate mate fields (RNEXT, PNEXT, MC:Z)

### 4. ⚠️ Contradictory SAM Fields
**R1 has incompatible flags**:
- Flag 0x4 (unmapped) → RNAME should be "*", POS should be mate's position
- But: RNAME=chrX, RNEXT="=", MC:Z:148M
- This violates SAM spec and suggests incomplete state management

## Root Cause Hypothesis

Based on evidence, the bug is likely in **`src/mem.rs::process_read_pairs()`**:

### Stage 1: Seed Generation & Alignment ✅ WORKING
- Both R1 and R2 generate seeds (SMEMs)
- Smith-Waterman alignment runs successfully
- R2 produces alignment: chrX:126402200, 148M, score 143-146

### Stage 2: Mate Pairing ❌ FAILING
- Alignments from R1 and R2 should be matched/scored as pairs
- Insert size should be calculated
- Proper_pair flag (0x2) should be set based on insert size distribution
- Mate information (RNEXT, PNEXT, MC, TLEN) should be populated

**Evidence of failure**:
- R1 has empty/filtered alignment list → defaults to unmapped
- R2 has alignment but no mate linkage
- No proper_pair flags
- Insert size calculation didn't run (TLEN=0)

### Stage 3: SAM Output ⚠️ INCONSISTENT
- R1 output shows contradictory fields (unmapped flag + position data)
- Suggests unmapped reads are using mate's position for RNAME/POS
- But CIGAR is correctly set to "*" (from our earlier fix)

## Debugging Plan

### 1. Add Debug Logging for This Read
Instrument code with:
```rust
#[cfg(feature = "debug-logging")]
if read_name.contains("1150:14380") {
    log::debug!("[DEBUG_READ] Stage: {}, State: {:?}", stage, state);
}
```

### 2. Key Questions to Answer
- [ ] Does R1 generate any alignments? (Check after `align_read_pair()`)
- [ ] Are R1 alignments filtered out? (Check opt.t threshold)
- [ ] How many alignments per read before pairing? (R1: ?, R2: ?)
- [ ] What happens during mate rescue? (Is R1 rescued based on R2?)
- [ ] Why is RNAME=chrX for R1 if unmapped? (Check unmapped read output logic)

### 3. Compare with C++ State
Add matching logs to C++ bwa-mem2:
- Number of SMEMs for R1 and R2
- Number of alignments after extension for R1 and R2
- Insert size estimation results
- Mate rescue attempts
- Final alignment selection for each read

### 4. Files for Investigation
- `src/mem.rs:382-906` - `process_read_pairs()` main loop
- `src/mem.rs` - Mate rescue logic
- `src/mem.rs` - Insert size calculation
- `src/mem.rs` - Proper pair determination
- `src/align.rs` - Individual read alignment (to verify it's working)

## Success Criteria

After fixes, Rust output should match C++:
```
R1: Flag=81, chrX:126402382, 148M, RNEXT:=, PNEXT:126402200, TLEN:-330, MC:Z:148M
R2: Flag=161, chrX:126402200, 148M, RNEXT:=, PNEXT:126402382, TLEN:330, MC:Z:148M
```
