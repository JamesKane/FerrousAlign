# FerrousAlign Pipeline Flow Diagram

## Overview

This document provides a comprehensive flow diagram from disk read through single-end and paired-end processing to SAM output. This serves as a reference for understanding the current architecture and as a foundation for refactoring into cleaner abstractions.

---

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          CLI Entry Point                              │
│                          (main.rs)                                    │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
                              ├─► Commands::Index → bwa_index()
                              │
                              └─► Commands::Mem
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     Alignment Orchestration                           │
│                     (mem.rs:main_mem)                                 │
│                                                                        │
│  1. Parse CLI options → MemOpt                                        │
│  2. Initialize compute backend (CPU SIMD/GPU/NPU)                     │
│  3. Load BWA index (FM-Index, BWT, SA, PAC)                           │
│  4. Write SAM header (@HD, @SQ, @PG, @RG)                             │
│  5. Dispatch: Single-End or Paired-End                                │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
    ┌───────────────────────┐   ┌───────────────────────┐
    │  Single-End Pipeline  │   │  Paired-End Pipeline  │
    └───────────────────────┘   └───────────────────────┘
```

---

## Single-End Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│                         STAGE 0: I/O LAYER                            │
│                     (single_end.rs:process_single_end)                │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  For each FASTQ file:                                                 │
│    • SoaFastqReader reads batch of N reads                            │
│    • Batch size = (10MB × n_threads) / avg_read_len                   │
│    • Typical: 500K reads with 16 threads                              │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       STAGE 1: COMPUTATION                            │
│                  (batch_extension/process_sub_batch)                  │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
        ┌───────────────────────┴────────────────────┐
        │                                            │
        ▼                                            ▼
┌────────────────────┐                    ┌──────────────────────┐
│ Parallel Chunking  │                    │  Per-Read Pipeline   │
│                    │                    │  (SoA-batched)       │
│ Split batch into   │───────────────────▶│                      │
│ thread-sized       │                    │  1. Seeding          │
│ chunks (Rayon)     │                    │  2. Chaining         │
│                    │                    │  3. Extension        │
│ Chunk size:        │                    │  4. Finalization     │
│   batch_size /     │                    │                      │
│   n_threads        │                    │  (detailed below)    │
└────────────────────┘                    └──────────┬───────────┘
                                                     │
                                                     ▼
                                          ┌──────────────────────┐
                                          │ SoAAlignmentResult   │
                                          │ (alignments + stats) │
                                          └──────────┬───────────┘
                                                     │
                                                     ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         STAGE 2: OUTPUT LAYER                         │
│                     (sam_output/write_sam_records_soa)                │
│                                                                        │
│  For each read:                                                       │
│    • Select alignments (primary + secondaries up to -h limit)        │
│    • Format SAM record (flag, CIGAR, tags)                           │
│    • Write via AsyncChannelWriter (overlapped I/O)                   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Paired-End Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│                         STAGE 0: I/O LAYER                            │
│                   (paired_end.rs:process_paired_end)                  │
└───────────────────────────────────┬──────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│  PHASE 1: Bootstrap Insert Size (First Batch)                        │
│    • Read 512 pairs (small batch for quick start)                    │
│    • Process alignments                                               │
│    • bootstrap_insert_size_stats_soa()                                │
│      - Compute mean, std, percentiles                                 │
│      - Determine proper pair boundaries                               │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  PHASE 2: Main Processing Loop (500K pairs/batch)                    │
│    • SoaFastqReader reads R1 + R2 batches                            │
│    • Interleave into unified batch: [R1[0], R2[0], R1[1], R2[1]...]  │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       STAGE 1: COMPUTATION                            │
│                  (batch_extension/process_sub_batch)                  │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
        ┌───────────────────────┴────────────────────┐
        │                                            │
        ▼                                            ▼
┌────────────────────┐                    ┌──────────────────────┐
│ Parallel Chunking  │                    │  Per-Read Pipeline   │
│                    │                    │  (SoA-batched)       │
│ Split batch into   │───────────────────▶│                      │
│ thread-sized       │                    │  1. Seeding          │
│ chunks (Rayon)     │                    │  2. Chaining         │
│                    │                    │  3. Extension        │
│                    │                    │  4. Finalization     │
│                    │                    │                      │
│                    │                    │  (detailed below)    │
└────────────────────┘                    └──────────┬───────────┘
                                                     │
                                                     ▼
                                          ┌──────────────────────┐
                                          │ SoAAlignmentResult   │
                                          │ (R1 + R2 alignments) │
                                          └──────────┬───────────┘
                                                     │
                                                     ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       STAGE 2: PAIRING LOGIC                          │
│                      (paired/pairing.rs)                              │
│                                                                        │
│  1. De-interleave: split alignments into R1/R2                        │
│  2. pair_alignments_soa(): Score all R1×R2 combinations               │
│     • Proper pairs: score = AS_R1 + AS_R2 - pen_unpaired             │
│     • Discordant: score = AS_R1 + AS_R2                              │
│  3. mate_rescue_soa(): Smith-Waterman rescue for unmapped mates       │
│  4. finalize_pairs_soa(): Select best pair, mark flags                │
│     • Set PAIRED, PROPER_PAIR, MREVERSE, RNEXT, PNEXT, TLEN          │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         STAGE 3: OUTPUT LAYER                         │
│                     (sam_output/write_sam_records_soa)                │
│                                                                        │
│  For each pair:                                                       │
│    • Write R1 record (with mate info)                                 │
│    • Write R2 record (with mate info)                                 │
│    • AsyncChannelWriter (overlapped I/O)                              │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Per-Read Alignment Pipeline (SoA-Batched)

This is the common computational core used by both single-end and paired-end modes.

```
┌──────────────────────────────────────────────────────────────────────┐
│                   INPUT: SoAReadBatch                                 │
│                                                                        │
│  • Sequences: concatenated bytes [ACGT...]                            │
│  • Qualities: concatenated bytes [IIII...]                            │
│  • Names: concatenated strings                                        │
│  • Boundaries: (offset, length) per read                              │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        PHASE 1: SEEDING                               │
│                    (seeding.rs:find_seeds_batch)                      │
│                                                                        │
│  For each read:                                                       │
│    1. Encode sequence (A=0, C=1, G=2, T=3)                            │
│    2. SMEM extraction (3 passes):                                     │
│       • Pass 1: Initial SMEMs (backward search on FM-Index)           │
│       • Pass 2: Re-seeding from middle (chimeric detection)           │
│       • Pass 3: Forward-only seeding (fill gaps)                      │
│    3. Filter by min_seed_len and max_occ                              │
│    4. Convert SMEMs → Seeds (sample SA entries)                       │
│                                                                        │
│  Output: SoASeedBatch (concatenated seeds)                            │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       PHASE 2: CHAINING                               │
│                  (chaining.rs:chain_seeds_batch)                      │
│                                                                        │
│  For each read:                                                       │
│    1. O(n²) DP chaining (compatible seeds)                            │
│       • Same strand & chromosome                                      │
│       • Monotonic: ref_pos[j] > ref_pos[i]                            │
│       • Not too far: gap < max_chain_gap                              │
│    2. Score chains: sum(seed_len) - gap_penalties                     │
│    3. Filter chains:                                                  │
│       • score >= min_chain_weight                                     │
│       • not redundant (drop_ratio filter)                             │
│                                                                        │
│  Output: SoAChainBatch (concatenated chains)                          │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      PHASE 3: EXTENSION                               │
│              (batch_extension/align_regions_batch)                    │
│                                                                        │
│  For each chain:                                                      │
│    1. Determine extension boundaries:                                 │
│       • Left: min(seed.ref_pos) - query_start                         │
│       • Right: max(seed.ref_pos + seed.len) + (query_len - qe)        │
│    2. Extract reference region from PAC                               │
│    3. Smith-Waterman alignment:                                       │
│       • Vertical SoA (banded_swa/) for long reads                     │
│       • Horizontal SIMD batching (kswv_*) for short extensions        │
│       • AVX-512 (64-lane) > AVX2 (32-lane) > SSE2 (16-lane)          │
│    4. Compute alignment score, boundaries (qb, qe, rb, re)            │
│                                                                        │
│  Output: SoA alignment scores + boundaries                            │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    PHASE 4: FINALIZATION                              │
│                (finalization.rs:finalize_candidates)                  │
│                                                                        │
│  For each alignment:                                                  │
│    1. Generate CIGAR (traceback from DP matrix)                       │
│    2. Compute MD tag (reference-query differences)                    │
│    3. Compute NM tag (edit distance)                                  │
│    4. Filter by min_score threshold (opt.t)                           │
│    5. Remove redundant alignments (mask_level)                        │
│    6. Mark secondary/supplementary:                                   │
│       • Primary: highest score                                        │
│       • Secondary: overlaps primary on query (0x100)                  │
│       • Supplementary: disjoint from primary (0x800)                  │
│    7. Compute MAPQ (Phred-scaled mapping quality)                     │
│                                                                        │
│  Output: Vec<Alignment> with CIGAR, MD, NM, flags, MAPQ              │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     OUTPUT: SoAAlignmentResult                        │
│                                                                        │
│  Per read:                                                            │
│    • Vec<Alignment> (primary + secondaries)                           │
│    • Original query name, sequence, quality                           │
│    • Statistics (time, bases processed)                               │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Data Structures Flow

### Input Layer
```
FASTQ file on disk
    ↓
SoaFastqReader (buffered, gzip-aware)
    ↓
SoAReadBatch {
    sequences: Vec<u8>,        // Concatenated: "ACGT...TGCA..."
    qualities: Vec<u8>,        // Concatenated: "IIII...JJJJ..."
    names: Vec<String>,
    read_boundaries: Vec<(usize, usize)>,  // (offset, len) per read
}
```

### Seeding Layer
```
SoAReadBatch
    ↓
find_seeds_batch()
    ↓
SoASeedBatch {
    seeds: Vec<Seed>,                // Concatenated seeds
    read_seed_boundaries: Vec<(usize, usize)>,  // Seeds per read
    encoded_queries: Vec<u8>,        // Encoded sequences
}

Seed {
    query_pos: i32,
    ref_pos: u64,
    len: i32,
    is_rev: bool,
    interval_size: u64,
    rid: i32,
}
```

### Chaining Layer
```
SoASeedBatch
    ↓
chain_seeds_batch()
    ↓
SoAChainBatch {
    chains: Vec<Chain>,              // Concatenated chains
    read_chain_boundaries: Vec<(usize, usize)>,  // Chains per read
}

Chain {
    score: i32,
    seeds: Vec<usize>,               // Indices into SoASeedBatch
    query_start: i32,
    query_end: i32,
    ref_start: u64,
    ref_end: u64,
    is_rev: bool,
    weight: i32,
    frac_rep: f32,
    rid: i32,
}
```

### Extension Layer
```
SoAChainBatch
    ↓
align_regions_batch()
    ↓
AlignmentRegion {
    qb: i32,                         // Query begin
    qe: i32,                         // Query end
    rb: u64,                         // Reference begin
    re: u64,                         // Reference end
    score: i32,
    is_rev: bool,
    chr_pos: u64,                    // Chromosome position
    ref_name: String,
    rid: i32,
    seedcov: i32,
    frac_rep: f32,
}
```

### Finalization Layer
```
Vec<AlignmentRegion>
    ↓
finalize_candidates()
    ↓
Vec<Alignment> {
    query_name: String,
    flag: u16,                       // SAM flags
    ref_name: String,
    pos: u64,
    mapq: u8,
    score: i32,
    cigar: Vec<(u8, i32)>,          // (op, len)
    rnext: String,                   // Mate reference
    pnext: u64,                      // Mate position
    tlen: i64,                       // Template length
    seq: String,
    qual: String,
    tags: Vec<(String, String)>,    // AS, NM, MD, etc.
}
```

### Output Layer
```
SoAAlignmentResult {
    alignments: Vec<Vec<Alignment>>,  // Per read
    names: Vec<String>,
    sequences: Vec<Vec<u8>>,
    qualities: Vec<String>,
}
    ↓
write_sam_records_soa()
    ↓
SAM text output (via AsyncChannelWriter)
```

---

## Key Decision Points

### 1. Single-End vs Paired-End
- **Location**: `mem.rs:main_mem:237-259`
- **Condition**: `opts.reads.len() == 2`
- **Branches**: `process_paired_end()` vs `process_single_end()`

### 2. Batch Size Selection
- **Single-End**: `(10MB × n_threads) / avg_read_len` (src/pipelines/linear/single_end.rs:169-170)
- **Paired-End**: 512 pairs for bootstrap, 500K pairs for main (src/pipelines/linear/paired/paired_end.rs:43, opt.batch_size)

### 3. SoA vs AoS Processing
- **Current**: 100% SoA pipeline (AoS paths removed)
- **SoA advantages**: SIMD efficiency, cache locality, reduced allocations

### 4. Parallel Chunking
- **Location**: `process_batch_parallel()` in both `single_end.rs` and `paired_end.rs`
- **Strategy**: Split batch into `n_threads` chunks, process via Rayon
- **Chunk size**: `(batch_size + n_threads - 1) / n_threads`

### 5. SIMD Engine Selection
- **Detection**: `detect_optimal_backend()` in `compute/mod.rs`
- **Hierarchy**: AVX-512 > AVX2 > SSE2 (x86_64), NEON (aarch64)
- **Runtime**: Dynamic dispatch based on CPU features

### 6. Alignment Selection
- **Single-End**: Primary + secondaries up to `-h` limit (default 5)
- **Paired-End**: Best proper pair > best discordant > single-end

---

## Critical Performance Paths

### Hot Paths (>50% CPU time)
1. **Smith-Waterman kernels** (`kswv_avx2.rs`, `kswv_avx512.rs`)
   - Horizontal SIMD batching (8/16/32 alignments parallel)
   - 8-bit saturating arithmetic
   - Vectorized max tracking

2. **FM-Index backward search** (`seeding.rs:generate_smems_for_strand`)
   - BWT lookups (cache-critical)
   - SA sampling (memory bandwidth)

3. **Seed chaining** (`chaining.rs:chain_seeds`)
   - O(n²) DP (with early pruning)
   - Frequent allocations (opportunity for pooling)

### Memory-Critical Paths
1. **PAC data access** (2-bit packed reference)
   - Hot: ~50-100MB working set
   - Already mmap'd and resident

2. **BWT index** (L2 cache sensitive)
   - Sequential scans during seeding
   - Prefetch opportunities

3. **Batch buffers** (allocate once, reuse)
   - SoAReadBatch: 10-50MB per batch
   - Thread-local workspaces (avoid contention)

---

## Future Abstraction Targets

### Ideal Module Structure
```
src/
├── core/
│   ├── alignment/         # SIMD kernels (existing, good)
│   ├── compute/           # Backend abstraction (existing, good)
│   └── io/                # I/O primitives (existing, good)
│
├── pipelines/
│   └── linear/
│       ├── index/         # Index loading (existing)
│       ├── orchestrator.rs   # NEW: Main loop abstraction
│       ├── stages/        # NEW: Stage implementations
│       │   ├── loading.rs     # Stage 0: Read loading
│       │   ├── seeding.rs     # Stage 1: SMEM extraction
│       │   ├── chaining.rs    # Stage 2: Seed chaining
│       │   ├── extension.rs   # Stage 3: SW alignment
│       │   └── finalization.rs # Stage 4: CIGAR, MD, NM
│       └── modes/         # NEW: Mode-specific logic
│           ├── single_end.rs  # Single-end coordinator
│           └── paired_end.rs  # Paired-end coordinator
│
└── main.rs                # CLI entry (existing)
```

### Clean Abstractions
1. **Pipeline Trait**: Define stage interface
2. **Batch Processor**: Abstract parallel chunking
3. **Alignment Selector**: Unify single/paired selection logic
4. **Output Formatter**: Separate SAM formatting from selection

---

## References

- C++ bwa-mem2: `FastMap::main_mem_thread()` (src/fastmap.cpp:947)
- Original BWA-MEM: `mem_process_seqs()` (bwa/bwamem.c:1549)
- SAM spec v1.6: https://samtools.github.io/hts-specs/SAMv1.pdf
