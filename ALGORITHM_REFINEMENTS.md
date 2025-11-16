# Algorithm Refinements Plan

This document tracks algorithmic features that are **parsed as CLI parameters** but not yet **fully implemented in the alignment pipeline**. These refinements improve alignment quality and C++ bwa-mem2 compatibility but are not blockers for basic functionality.

---

## 🔴 High Priority: Core Algorithm Consistency

These items affect alignment correctness and must be addressed for full C++ bwa-mem2 compatibility.

### 1. Re-seeding (`-r` split_factor)
- **Current Status**: Parameter parsed and stored (`opt.split_factor = 1.5`), but not used in seed generation
- **Impact**: **HIGH** - Affects sensitivity for long reads and repetitive sequences
- **C++ Behavior**: Long MEMs (> min_seed_len * split_factor) are split into shorter seeds for better chaining
- **Location**: `src/align.rs` - `backward_search()` and SMEM extraction
- **Implementation**:
  ```rust
  // After finding a MEM of length L:
  if L > opt.min_seed_len * opt.split_factor {
      // Generate additional seeds at interval opt.split_factor * opt.min_seed_len
      // This helps chain through repetitive regions
  }
  ```
- **Estimated Effort**: 4-6 hours
- **Testing**: Compare SMEM counts and chain quality with C++ version

### 2. Split-width 3rd Round Seeding (`-y`)
- **Current Status**: Parameter parsed (`opt.max_mem_intv = 20`), not used
- **Impact**: **MEDIUM-HIGH** - Affects rescue of poorly mapped reads
- **C++ Behavior**: After initial seeding, perform 3rd round with relaxed occurrence threshold
- **Location**: `src/align.rs` - Add 3rd seeding pass
- **Implementation**:
  ```rust
  // After initial SMEMs and chains:
  if alignments.is_empty() || best_score < threshold {
      // Re-seed with max_occ = opt.max_mem_intv (20 instead of 500)
      // This catches more repetitive seeds for difficult reads
  }
  ```
- **Estimated Effort**: 3-4 hours
- **Testing**: Test on difficult reads with repetitive sequences

### 3. Chain Dropping (`-D` drop_ratio)
- **Current Status**: Parameter parsed (`opt.drop_ratio = 0.50`), not used
- **Impact**: **HIGH** - Affects multi-mapping and alignment quality
- **C++ Behavior**: Drop chains < drop_ratio * best_chain_score when chains overlap
- **Location**: `src/align.rs:622` - `chain_seeds()` function
- **Implementation**:
  ```rust
  // After scoring all chains:
  let max_score = chains.iter().map(|c| c.score).max().unwrap_or(0);
  let threshold = (max_score as f32 * opt.drop_ratio) as i32;

  // Remove chains that:
  // 1. Score < threshold
  // 2. Overlap with higher-scoring chains
  chains.retain(|c| c.score >= threshold && !overlaps_with_better(c, &chains));
  ```
- **Estimated Effort**: 4-5 hours
- **Testing**: Compare number of reported alignments with C++ version

### 4. Mate Rescue Enhancements
- **Current Status**: Basic mate rescue implemented, but controls not wired up
- **Impact**: **MEDIUM-HIGH** - Affects paired-end alignment completeness
- **Missing Controls**:
  - `-m` max_matesw: Currently hardcoded to 1 round (C++ default: 50)
  - `-S` skip_mate_rescue: Flag parsed but not used
  - `-P` skip_pairing: Flag parsed but not used
- **Location**: `src/mem.rs:512-570` - mate rescue logic
- **Implementation**:
  ```rust
  // Gate mate rescue based on flags:
  if !opt.skip_mate_rescue {
      for round in 0..opt.max_matesw {
          // Perform mate rescue...
          if rescued_count == 0 { break; } // Early exit if no rescues
      }
  }
  ```
- **Estimated Effort**: 2-3 hours
- **Testing**: Verify rescue counts match C++ with various `-m` values

---

## 🟡 Medium Priority: Output Compatibility

These items affect SAM output format compatibility with downstream tools.

### 5. Clipping Penalties in Scoring
- **Current Status**: Parameters parsed (`opt.pen_clip5`, `opt.pen_clip3`), not used in scoring
- **Impact**: **MEDIUM** - Affects alignment scores and soft-clipping decisions
- **C++ Behavior**: Penalties applied during alignment extension to penalize end clipping
- **Location**: `src/banded_swa.rs` - Smith-Waterman scoring
- **Implementation**:
  ```rust
  // In banded_swa.rs scoring:
  // Penalize soft clips at 5' end by opt.pen_clip5
  // Penalize soft clips at 3' end by opt.pen_clip3
  // This discourages excessive clipping
  ```
- **Estimated Effort**: 3-4 hours
- **Testing**: Compare CIGAR strings with C++ version, especially clip lengths

### 6. XA Tag Support (`-h`)
- **Current Status**: Parameter parsed (`opt.max_xa_hits = 5`, `opt.max_xa_hits_alt = 200`), not used
- **Impact**: **MEDIUM** - Important for variant callers that use alternative alignments
- **C++ Behavior**: Report alternative alignments in XA tag when score > 80% of max
- **Location**: SAM output in `src/mem.rs` or `src/align.rs`
- **Implementation**:
  ```rust
  // After selecting primary alignment:
  let max_score = alignments.iter().map(|a| a.score).max().unwrap_or(0);
  let xa_threshold = (max_score as f32 * opt.xa_drop_ratio) as i32;

  let xa_alns: Vec<_> = alignments.iter()
      .filter(|a| a.score >= xa_threshold && a != primary)
      .take(opt.max_xa_hits as usize)
      .collect();

  if !xa_alns.is_empty() {
      alignment.tags.push(("XA".to_string(), format_xa_tag(&xa_alns)));
  }
  ```
- **Estimated Effort**: 4-5 hours
- **Testing**: Verify XA tag format matches SAM spec and C++ output

### 7. Output Formatting Flags
- **Current Status**: Flags parsed but not implemented
- **Impact**: **LOW-MEDIUM** - Affects specific use cases
- **Flags**:
  - `-a`: Output all alignments (not just primary)
  - `-C`: Append FASTA/FASTQ comment to SAM
  - `-V`: Output reference header in XR tag
  - `-Y`: Soft clip supplementary alignments
  - `-M`: Mark shorter split hits as secondary
  - `-q`: Don't modify mapQ of supplementary alignments
- **Location**: SAM output formatting
- **Estimated Effort**: 3-4 hours total
- **Testing**: Verify SAM flags and tags match C++ output

---

## 🟢 Low Priority: Advanced Features

These items have minimal impact on standard workflows but are needed for edge cases.

### 8. Smart Pairing (`-p`)
- **Current Status**: Flag parsed (`opt.smart_pairing`), not used
- **Impact**: **LOW** - Only affects interleaved FASTQ input
- **C++ Behavior**: Treats single interleaved file as paired-end (ignoring second file)
- **Location**: Input handling in `src/mem.rs`
- **Estimated Effort**: 2-3 hours
- **Testing**: Test with interleaved FASTQ files

### 9. ALT Contig Handling (`-j`)
- **Current Status**: Flag parsed (`opt.treat_alt_as_primary`), not used
- **Impact**: **LOW** - Only relevant for human reference with ALT contigs
- **C++ Behavior**: Skip reading `.alt` file and treat ALT contigs as regular sequences
- **Location**: Index loading in `src/bwa_index.rs`
- **Estimated Effort**: 2-3 hours
- **Testing**: Test with human reference GRCh38 + ALT contigs

### 10. Split Alignment Coordinate Selection (`-5`)
- **Current Status**: Flag parsed (`opt.smallest_coord_primary`), not used
- **Impact**: **LOW** - Affects which split alignment is marked primary
- **C++ Behavior**: Mark alignment with smallest coordinate as primary (default: highest score)
- **Location**: Alignment selection logic
- **Estimated Effort**: 1-2 hours
- **Testing**: Verify primary/secondary flags in split alignments

---

## 🎯 Priority Decision Matrix

| Item | Impact on Accuracy | Impact on Compatibility | Effort | Priority |
|------|-------------------|------------------------|--------|----------|
| Re-seeding | High | High | 4-6h | 🔴 Critical |
| Chain dropping | High | High | 4-5h | 🔴 Critical |
| Split-width seeding | Medium | Medium | 3-4h | 🔴 High |
| Mate rescue controls | Medium | Medium | 2-3h | 🔴 High |
| Clipping penalties | Medium | Medium | 3-4h | 🟡 Medium |
| XA tag | Low | High | 4-5h | 🟡 Medium |
| Output flags | Low | Medium | 3-4h | 🟡 Medium |
| Smart pairing | Low | Low | 2-3h | 🟢 Low |
| ALT handling | Low | Low | 2-3h | 🟢 Low |
| Split coord select | Low | Low | 1-2h | 🟢 Low |

**Total Estimated Effort**: 28-39 hours (3.5-5 weeks)

---

## ✅ Validation Strategy

For each refinement, validate consistency with C++ bwa-mem2:

1. **Unit Tests**: Test parameter parsing and edge cases
2. **Integration Tests**: Compare output on standard datasets:
   - Human genome chr22 (medium complexity)
   - Repetitive sequences (test re-seeding)
   - Paired-end data (test mate rescue)
3. **Compatibility Tests**: Run identical commands on both versions:
   ```bash
   # Rust version
   ./ferrous-align mem -t 8 ref.idx r1.fq r2.fq > rust.sam

   # C++ version
   ./bwa-mem2 mem -t 8 ref.idx r1.fq r2.fq > cpp.sam

   # Compare (should be identical after refinements)
   diff <(grep -v '^@PG' rust.sam) <(grep -v '^@PG' cpp.sam)
   ```
4. **Performance Benchmarks**: Ensure no regression in speed
5. **Downstream Tool Tests**: Verify SAM works with samtools, GATK, etc.

---

## 📋 Implementation Notes

### Current Status (v0.5.0)
All parameters listed above are:
- ✅ Defined in `MemOpt` struct (`src/mem_opt.rs`)
- ✅ Parsed from command-line arguments (`src/main.rs`)
- ✅ Passed through the alignment pipeline
- ⏳ **Not yet used in algorithmic decisions** (this document tracks those gaps)

### What Works Today
- Basic seeding with `-k` (min seed length) and `-c` (max occurrences)
- Simple chain scoring (without dropping)
- Single-round mate rescue
- Standard scoring without clipping penalties
- Primary alignment selection only (no XA tags)

### Integration Points
When implementing these features, wire them into:
- `src/align.rs`: Re-seeding, chain dropping, 3rd round seeding
- `src/mem.rs`: Mate rescue controls, output flags
- `src/banded_swa.rs`: Clipping penalties
- SAM output: XA tags, formatting flags

---

*This refinement plan tracks the gap between CLI parameter acceptance and full algorithmic implementation. For overall project status, see TODO.md and CLAUDE.md.*
