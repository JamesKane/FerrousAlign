# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**FerrousAlign** (`ferrous-align`) is a Rust port of [bwa-mem2](https://github.com/bwa-mem2/bwa-mem2), a high-performance bioinformatics tool for aligning DNA sequencing reads to reference genomes using the Burrows-Wheeler Transform. This implementation targets performance parity with the C++ version (1.3-3.1x faster than original BWA-MEM) while providing Rust's safety and maintainability benefits.

### NOTICE: The C++ reference implementation's behavior and file formats are the technical specification.  Any deviation is a critical bug, which blocks all downstream tasks.  We must match the behavior even if that means rewriting the code.

**Project Status**: v0.7.0-alpha (feature/core-rearch branch) - Structure-of-Arrays (SoA) Architecture Migration Complete!
- ✅ Full SoA pipeline from FASTQ reading through SAM output
- ✅ SIMD-friendly memory layout for batch processing
- ✅ Thread-local workspace allocation with buffer reuse
- ✅ Unified alignment kernel dispatch (banded_swa + kswv)
- 🔄 Integration testing in progress to validate against v0.6.0 baseline
- 🎯 Target: Maintain GATK parity while improving memory efficiency and batch parallelism

**Previous Stable Version**: v0.6.0 - GATK Parity Achieved! Full GATK ValidateSamFile parity with BWA-MEM2 on 4M HG002 read pairs. Properly paired rate EXCEEDS target (97.71% vs 97.10%), CIGAR errors eliminated (0), NM errors at parity (2,708 vs 2,343).

## Build Commands

```bash
# Build in release mode (required for reasonable performance)
cargo build --release

# Build with native CPU optimizations (recommended)
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Build with AVX-512 support (requires nightly Rust, experimental)
# Note: AVX-512 intrinsics are currently unstable in Rust
cargo +nightly build --release --features avx512

# Run the binary
./target/release/ferrous-align --help
```

### Testing

```bash
# Run all unit tests (embedded in src/*_test.rs files)
cargo test --lib

# Run specific module tests
cargo test --lib bwt::tests
cargo test --lib bntseq::tests

# Run all integration tests (tests/*.rs)
cargo test --test '*'

# Run specific integration test
cargo test --test integration_test
cargo test --test paired_end_integration_test

# Run with output (useful for debugging)
cargo test -- --nocapture

# Run benchmarks
cargo bench

# Run specific benchmark
cargo bench --bench simd_benchmarks

# Test threading edge cases (thread count validation)
./target/release/ferrous-align mem -t 0 test.idx test.fq  # Should warn and use 1
./target/release/ferrous-align mem -t 1000 test.idx test.fq  # Should warn and cap
```

**Note**: Some tests require test data in `test_data/` and will gracefully skip if files are missing. The large test file `test_data/HG03576.final.cram` (~200 MB) is not required for basic testing.

**Thread Validation Testing**:
- Invalid thread counts (< 1) are auto-corrected to 1 with warning
- Excessive thread counts (> 2×CPU) are capped with warning
- Test script available: See commit message for Session 25

### Code Quality

```bash
# Format code (always run before committing)
cargo fmt

# Check for common issues
cargo clippy

# Check formatting without modifying
cargo fmt -- --check
```

## Architecture

### Entry Points and Command Flow

**CLI Entry** (`src/main.rs`):
- Uses `clap` for argument parsing
- Two subcommands: `index` and `mem`
- `index`: Calls `bwa_index::bwa_index()` to build FM-Index
- `mem`: Calls `mem::main_mem()` to align reads

**Library Modules** (`src/lib.rs`) - Structure-of-Arrays (SoA) Architecture:
```
src/
├── core/                           # Reusable reference-agnostic components (SoA-native)
│   ├── alignment/                  # SIMD alignment kernels with SoA support
│   │   ├── banded_swa/             # Banded Smith-Waterman (vertical SIMD)
│   │   │   ├── engines.rs          # ISA dispatch (SSE/AVX2/AVX-512)
│   │   │   ├── kernel.rs           # Main u8 scoring kernel
│   │   │   ├── kernel_i16.rs       # i16 scoring for overflow safety
│   │   │   └── scalar/             # Scalar fallback implementations
│   │   ├── kswv/                   # Horizontal SIMD batching (KSW variant)
│   │   │   ├── mod.rs              # Shared macros and adapters
│   │   │   └── shared.rs           # Reusable SoA buffers
│   │   ├── kswv_sse_neon.rs        # 128-bit horizontal kernel
│   │   ├── kswv_avx2.rs            # 256-bit horizontal kernel (x86_64)
│   │   ├── kswv_avx512.rs          # 512-bit horizontal kernel (experimental)
│   │   ├── shared_types.rs         # SoA carriers (SwSoA, KswSoA, AlignJob)
│   │   ├── workspace.rs            # Thread-local buffer pools
│   │   └── ...                     # CIGAR, edit distance, utils
│   ├── compute/                    # SIMD abstraction + backend detection
│   ├── io/                         # I/O with SoA support
│   │   ├── soa_readers.rs          # SoAReadBatch, SoAFastqReader
│   │   ├── fastq_reader.rs         # Traditional AoS reader (retained)
│   │   └── sam_output.rs           # SAM writer (SoA-aware)
│   └── utils/                      # General utilities
├── pipelines/
│   ├── linear/                     # BWA-MEM linear alignment pipeline (SoA-native)
│   │   ├── batch_extension/        # Batched extension orchestration (SoA)
│   │   │   ├── types.rs            # ExtensionJobBatch, SoAAlignmentResult
│   │   │   ├── collect_soa.rs      # SoA job collection
│   │   │   ├── orchestration_soa.rs# SoA batch processing
│   │   │   ├── finalize_soa.rs     # SoA result finalization
│   │   │   └── dispatch.rs         # Kernel routing
│   │   ├── index/                  # FM-Index, BWT, suffix array
│   │   ├── paired/                 # Paired-end processing
│   │   ├── seeding.rs              # SMEM extraction (SoA batch support)
│   │   ├── chaining.rs             # Seed chaining (SoA batch support)
│   │   ├── pipeline.rs             # Main alignment entry point
│   │   ├── finalization.rs         # Alignment finalization
│   │   ├── single_end.rs           # Single-end processing
│   │   └── mem.rs                  # CLI entry point
│   └── graph/                      # Future pangenome pipeline (placeholder)
└── defaults.rs
```

**Core Modules** (`src/core/`) - SoA-Native Design:
- **`core/alignment/`**: SIMD alignment kernels with Structure-of-Arrays support
  - `banded_swa/`: Vertical SIMD Smith-Waterman (processes multiple query positions in parallel)
  - `kswv_*.rs`: Horizontal SIMD Smith-Waterman (processes multiple alignments in parallel)
  - `shared_types.rs`: SoA carriers (`SwSoA`, `KswSoA`, `AlignJob`) and kernel config bundles
  - `workspace.rs`: Thread-local buffer pools for zero-allocation batch processing
  - `cigar.rs`, `edit_distance.rs`: CIGAR and MD tag utilities
- **`core/compute/`**: SIMD abstraction layer with SSE/AVX2/AVX-512/NEON backends
- **`core/io/`**: I/O with SoA batch support
  - `soa_readers.rs`: `SoAReadBatch`, `SoAFastqReader` for batch-oriented reading
  - `fastq_reader.rs`: Traditional AoS reader (retained for compatibility)
  - `sam_output.rs`: SAM writer with SoA-aware bulk output
- **`core/utils/`**: General-purpose utilities (timing, hashing, binary I/O)

**Linear Pipeline** (`src/pipelines/linear/`) - SoA Batch Processing:
- **`index/`**: `bntseq.rs`, `bwa_index.rs`, `bwt.rs`, `fm_index.rs` - FM-Index and reference handling
- **`paired/`**: Paired-end processing with mate rescue
  - `paired_end.rs`: Main paired-end processing loop
  - `pairing.rs`: Proper pairing logic and orientation scoring
  - `insert_size.rs`: Insert size distribution inference
  - `mate_rescue.rs`: Re-alignment using mate location hints
- **`batch_extension/`**: SoA-native batched extension module (<500 lines per file)
  - `types.rs`: `ExtensionJobBatch`, `SoAAlignmentResult`, job descriptors
  - `collect_soa.rs`: Collect alignment jobs from seeds/chains into SoA batches
  - `orchestration_soa.rs`: Batch processing orchestration with kernel dispatch
  - `finalize_soa.rs`: Distribute results back to per-read alignments
  - `dispatch.rs`: Unified kernel routing (banded_swa vs kswv, ISA selection)
- **Pipeline stages**:
  - `seeding.rs`: SMEM extraction with `SoASeedBatch` support (1902 lines - to be split)
  - `chaining.rs`: Seed chaining with `SoAChainBatch` support (1009 lines - to be split)
  - `pipeline.rs`: Main alignment orchestration (1235 lines - to be split)
  - `finalization.rs`: Alignment finalization and filtering (1707 lines - to be split)
  - `region.rs`: Chain-to-region mapping (1598 lines - to be split)
- **Entry points**: `mem.rs`, `mem_opt.rs`, `single_end.rs`

**Graph Pipeline** (`src/pipelines/graph/`):
- Placeholder for future pangenome graph alignment (see roadmap)

**Core Alignment Kernels** (`src/core/alignment/`) - SoA-Native Design:
```rust
// ============================================================================
// SIMD Kernel Modules
// ============================================================================
pub mod banded_swa;       // Vertical SIMD Smith-Waterman (processes query positions)
  ├── engines.rs          // ISA dispatch: SSE/NEON → AVX2 → AVX-512
  ├── kernel.rs           // Main u8 scoring kernel (overflow-checked)
  ├── kernel_i16.rs       // i16 scoring for long/high-scoring alignments
  ├── scalar/             // Scalar fallback (directional + implementation)
  └── shared.rs           // SoA providers and kernel utilities

pub mod kswv_batch;       // Horizontal SIMD batching infrastructure
pub mod kswv_sse_neon;    // 128-bit horizontal kernel (16 alignments/batch)
pub mod kswv_avx2;        // 256-bit horizontal kernel (32 alignments/batch, x86_64)
pub mod kswv_avx512;      // 512-bit horizontal kernel (64 alignments/batch, experimental)
pub mod kswv;             // Shared macros and SoA adapters

// ============================================================================
// Shared Infrastructure
// ============================================================================
pub mod shared_types;     // SoA carriers and kernel config
  ├── AlignJob            // Per-alignment metadata (AoS format)
  ├── SwSoA / SwSoA16     // SoA carriers for banded_swa (u8/i16 scoring)
  ├── KswSoA              // SoA carrier for kswv (horizontal batching)
  ├── KernelConfig        // Gap penalties, banding, scoring matrix
  └── Traits: SoAProvider, WorkspaceArena  // Buffer reuse contracts

pub mod workspace;        // Thread-local buffer pools (zero-allocation)
  ├── AlignmentWorkspace  // DP matrices, traceback, SMEM buffers
  └── with_workspace()    // RAII accessor for thread-local storage

// ============================================================================
// Supporting Modules
// ============================================================================
pub mod cigar;            // CIGAR string manipulation
pub mod edit_distance;    // NM/MD tag computation
pub mod ksw_affine_gap;   // Affine-gap Smith-Waterman fallback (scalar)
pub mod utils;            // Base encoding, scoring matrices
```

**Key SoA Design Patterns**:
1. **Transposition**: `AlignJob` (AoS) → `SwSoA`/`KswSoA` (SoA) → Results (AoS)
2. **Workspace Reuse**: Thread-local buffers sized for max batch, eliminating allocations
3. **Unified Dispatch**: `batch_extension::dispatch` routes to best kernel/ISA combo
4. **Type Safety**: SoA carriers prevent length mismatches via `SwSoA::lanes` field

**External Dependencies**:
- `log` + `env_logger`: Structured logging with verbosity control (Session 26)
- `rayon`: Multi-threading with work-stealing scheduler (Session 25)
- `clap`: Command-line argument parsing
- `bio`: Bioinformatics utilities (suffix array, FASTQ parsing - Session 27)
- `flate2`: Gzip compression support for FASTQ files (Session 27)

### Alignment Pipeline (Four-Stage SoA Architecture)

The pipeline implements a Structure-of-Arrays (SoA) design for SIMD-friendly batch processing. Entry point: `pipeline::align_read()` (per-read) or batch variants (`find_seeds_batch`, `chain_seeds_batch`, etc.).

**Design Philosophy: SoA End-to-End**
- Reads are processed in batches of 512 (configurable via `--batch-size`)
- Data flows through the pipeline in SoA layout for cache-friendly access
- SIMD kernels process multiple alignments in parallel (horizontal batching)
- Thread-local workspaces eliminate per-batch allocations

**Stage 1: Seeding** (`pipeline::find_seeds()` / `find_seeds_batch()`):
- Extracts SMEMs (Supermaximal Exact Matches) via FM-Index backward search
- Searches both forward and reverse-complement strands
- Re-seeding pass for long unique SMEMs (chimeric detection)
- 3rd round seeding for highly repetitive regions (`max_mem_intv`)
- Filters seeds by minimum length and maximum occurrence count
- Converts BWT intervals to reference positions via suffix array
- **SoA Output**: `SoASeedBatch` - contiguous seed arrays per read

**Stage 2: Chaining** (`chaining::chain_seeds()` / `chain_seeds_batch()`):
- Groups compatible seeds using O(n²) dynamic programming
- Scores chains based on seed lengths and gaps
- Filters chains by score threshold and overlap
- **SoA Output**: `SoAChainBatch` - chain indices and metadata per read

**Stage 3: Extension** (`batch_extension` module):
- **3a. Collection** (`collect_soa::collect_extension_jobs_batch_soa()`):
  - Extracts reference sequences for each seed extension
  - Creates `ExtensionJobBatch` with SoA layout: contiguous query/ref buffers
  - Metadata: read_idx, chain_idx, seed_idx, direction, offsets
- **3b. Execution** (`orchestration_soa::process_sub_batch_internal_soa()`):
  - Routes jobs to appropriate kernel via `dispatch::route_batch_soa()`
  - **Kernel Selection**:
    - `banded_swa`: Vertical SIMD (processes query positions in parallel)
    - `kswv`: Horizontal SIMD (processes multiple alignments in parallel)
  - **ISA Selection**: SSE/NEON (128-bit) → AVX2 (256-bit) → AVX-512 (512-bit)
  - Returns `Vec<BatchExtensionResult>` with scores, CIGAR, alignment bounds
- **3c. Finalization** (`finalize_soa::finalize_alignments_soa()`):
  - Distributes results back to per-read `SoAAlignmentResult`
  - Merges left/right extension CIGARs with seed
  - Converts FM-index positions to chromosome coordinates

**Stage 4: Finalization** (`finalization::finalize_alignments()`):
- Computes MD tag and exact NM (edit distance)
- Filters by score threshold, removes redundant alignments
- Marks secondary/supplementary alignments, calculates MAPQ
- Generates XA/SA tags for alternative alignments
- Produces unmapped record if no alignments survive
- **SoA-aware**: Accepts `SoAAlignmentResult` and outputs standard `Alignment`

**Paired-End Processing** (`paired_end::process_read_pairs()`):
1. **Batch Reading**: `SoAFastqReader` loads R1/R2 batches in SoA layout
2. **Per-batch Processing**: Align all reads in batch independently
3. **Insert Size Estimation**: From concordant pairs across entire batch
4. **Mate Rescue**: Re-align using mate's location (batch-aware)
5. **Pair Re-scoring**: Based on insert size distribution
6. **Primary/Secondary Marking**: Per-pair alignment selection
7. **SAM Output**: Bulk write with proper pair flags

### SIMD Abstraction Layer

**Design Philosophy**: Write-once, run-anywhere SIMD code

**Platform Detection** (`src/core/compute/simd_abstraction/mod.rs`):
```rust
#[cfg(target_arch = "x86_64")]
// Native x86 intrinsics via <immintrin.h>

#[cfg(target_arch = "aarch64")]
// Native ARM NEON via <arm_neon.h>
```

**Type System**:
- `__m128i`: Native type on x86_64, transparent wrapper struct on ARM
- Conversion methods: `as_s16()`, `from_s16()`, `as_u8()` for ARM signedness casting
- Zero-cost abstractions via `#[repr(transparent)]`

**Critical Operations** (30+ intrinsics):
- **Arithmetic**: `_mm_add_epi16`, `_mm_sub_epi16`, `_mm_mullo_epi16`
- **Comparison**: `_mm_max_epi16`, `_mm_min_epi16`, `_mm_cmpeq_epi8`
- **Bitwise**: `_mm_and_si128`, `_mm_or_si128`, `_mm_andnot_si128`
- **Shifts**: `_mm_slli_epi16`, `_mm_srli_si128` (element vs byte shifts)
- **Load/Store**: `_mm_load_si128`, `_mm_storeu_si128`

**Performance Hotspot** (`popcount64()` in `src/pipelines/linear/index/fm_index.rs`):
- x86_64: Uses `_popcnt64` hardware instruction
- ARM: Uses NEON `vcnt` + pairwise additions (`vpaddl`)
- Called millions of times during FM-Index backward search

**SIMD Width** (Runtime Detection):
- **128-bit SSE/NEON** (baseline): 16-way parallelism for 8-bit operations
- **256-bit AVX2** (x86_64 only): 32-way parallelism (2x speedup, 1.8-2.2x real-world)
- **512-bit AVX-512** (x86_64, requires `--features avx512`): 64-way parallelism (4x theoretical, 2.5-3.0x real-world)
- Runtime detection automatically selects best available SIMD engine
- Banded Smith-Waterman processes 16/32/64 sequences in parallel (depending on SIMD width)
- Threshold: Switches to batched SIMD when ≥16 alignment jobs available

### Heterogeneous Compute Abstraction

**Status**: ✅ **Integration points implemented** (Session 30)

The codebase provides clean integration points for adding GPU (Metal/CUDA/ROCm) and NPU (ANE/ONNX) acceleration. Currently, GPU and NPU backends are **NO-OPs** that fall back to CPU SIMD.

**Module Structure** (`src/core/compute/`):
```
src/core/compute/
├── mod.rs              # ComputeBackend enum, detection, ComputeContext
├── encoding.rs         # EncodingStrategy (Classic 2-bit vs ONE-HOT for NPU)
└── simd_abstraction/   # SIMD engine implementations (SSE/AVX2/AVX-512/NEON)
```

**ComputeBackend Enum** (`compute/mod.rs`):
```rust
pub enum ComputeBackend {
    CpuSimd(SimdEngineType),  // Active: SSE/AVX2/AVX-512/NEON
    Gpu,                       // NO-OP: Falls back to CpuSimd
    Npu,                       // NO-OP: Falls back to CpuSimd
}
```

**EncodingStrategy Enum** (`compute/encoding.rs`):
```rust
pub enum EncodingStrategy {
    Classic,   // 2-bit encoding: A=0, C=1, G=2, T=3 (active)
    OneHot,    // 4-channel: A=[1,0,0,0], etc. (NO-OP, for NPU)
}
```

**Integration Points** (marked with `HETEROGENEOUS COMPUTE` comment blocks):

| Location | File | Function | Purpose |
|----------|------|----------|---------|
| Entry | `src/pipelines/linear/mem.rs` | `main_mem()` | Backend detection |
| Dispatch | `src/pipelines/linear/mem.rs` | `main_mem()` | Pass context to processing |
| Single-end | `src/pipelines/linear/single_end.rs` | `process_single_end()` | Accept compute context |
| Paired-end | `src/pipelines/linear/paired/paired_end.rs` | `process_paired_end()` | Accept compute context |
| Pipeline | `src/pipelines/linear/pipeline.rs` | `align_read()` | Main alignment entry |
| Extension | `src/pipelines/linear/pipeline.rs` | `extend_chains_to_alignments()` | Stage 3 dispatch |
| Backend Switch | `src/pipelines/linear/pipeline.rs` | `extend_chains_to_alignments()` | Route to backend |
| Extension Docs | `src/pipelines/linear/batch_extension.rs` | Module header | GPU/NPU implementation guide |

**Adding a New Backend** (e.g., Metal GPU):

1. **Update `core/compute/mod.rs`**:
   - Remove NO-OP fallback in `effective_backend()` for your variant
   - Add detection logic in `detect_optimal_backend()`

2. **Update `pipelines/linear/pipeline.rs`**:
   ```rust
   ComputeBackend::Gpu => {
       execute_gpu_alignments(&gpu_context, &sw_params, &alignment_jobs)
   }
   ```

3. **Implement backend kernel** (new file, e.g., `core/compute/gpu.rs`):
   ```rust
   pub fn execute_gpu_alignments(
       ctx: &GpuContext,
       sw_params: &BandedPairWiseSW,
       jobs: &[AlignmentJob],
   ) -> Vec<(i32, Vec<(u8, i32)>, Vec<u8>, Vec<u8>)>
   ```

4. **For NPU seed pre-filtering**:
   - Use `EncodingStrategy::OneHot` in `find_seeds()` (pipelines/linear/pipeline.rs)
   - Implement seed classifier model
   - Filter seeds before building alignment jobs

**Performance Thresholds**:
- GPU dispatch overhead: ~20-50μs
- SW kernel CPU time: ~1-2μs per alignment
- GPU threshold: batch_size >= 1024 to amortize overhead

### Key Data Structures

**SoA Batch Carriers** (`src/core/io/soa_readers.rs`, `src/pipelines/linear/seeding.rs`, `src/pipelines/linear/chaining.rs`):
```rust
// FASTQ batch in SoA layout
pub struct SoAReadBatch {                // core/io/soa_readers.rs
    pub names: Vec<String>,              // Read names (kept as strings)
    pub sequences: Vec<u8>,              // Concatenated sequences (2-bit encoded)
    pub qualities: Vec<u8>,              // Concatenated quality scores
    pub seq_offsets: Vec<usize>,         // Offset into sequences buffer
    pub qual_offsets: Vec<usize>,        // Offset into qualities buffer
    pub lengths: Vec<usize>,             // Per-read sequence lengths
    pub count: usize,                    // Number of reads in batch
}

// Seed batch in SoA layout
pub struct SoASeedBatch {                // pipelines/linear/seeding.rs
    pub seeds: Vec<Seed>,                // Concatenated seeds for all reads
    pub seed_offsets: Vec<usize>,        // Offset into seeds array per read
    pub seed_counts: Vec<usize>,         // Number of seeds per read
    pub encoded_queries: Vec<Vec<u8>>,   // Encoded query sequences (kept per-read)
    pub encoded_queries_rc: Vec<Vec<u8>>,// Reverse complement queries
    pub read_count: usize,               // Number of reads
}

// Chain batch in SoA layout
pub struct SoAChainBatch {               // pipelines/linear/chaining.rs
    pub chains: Vec<Chain>,              // Concatenated chains for all reads
    pub chain_offsets: Vec<usize>,       // Offset into chains array per read
    pub chain_counts: Vec<usize>,        // Number of chains per read
    pub read_count: usize,               // Number of reads
}
```

**Extension Job Batch** (`src/pipelines/linear/batch_extension/types.rs`):
```rust
// Batched extension jobs in SoA layout
pub struct ExtensionJobBatch {
    pub jobs: Vec<BatchedExtensionJob>,  // Job metadata (read_idx, chain_idx, etc.)
    pub query_buf: Vec<u8>,              // Concatenated query sequences (2-bit)
    pub ref_buf: Vec<u8>,                // Concatenated reference sequences (2-bit)
}

// Result of batched extensions
pub struct SoAAlignmentResult {          // batch_extension/types.rs
    pub read_idx: usize,                 // Index in original batch
    pub alignments: Vec<Alignment>,      // Per-read alignments
    pub query_seq: Vec<u8>,              // Original query (for finalization)
    pub query_name: String,              // Read name
}
```

**FM-Index Tier** (`src/pipelines/linear/index/fm_index.rs`, `src/pipelines/linear/index/bwt.rs`):
```rust
pub struct SMEM {                        // pipelines/linear/seeding.rs
    pub query_start: i32,                // Start position in query (0-based, inclusive)
    pub query_end: i32,                  // End position in query (0-based, exclusive)
    pub bwt_interval_start: u64,         // Start of BWT interval in suffix array
    pub bwt_interval_end: u64,           // End of BWT interval
    pub interval_size: u64,              // Occurrence count
    pub is_reverse_complement: bool,     // Forward or reverse strand
}

pub struct CpOcc {                       // src/pipelines/linear/index/fm_index.rs
    cp_count: [i64; 4],                  // Cumulative base counts at checkpoint
    one_hot_bwt_str: [u64; 4],           // One-hot encoded BWT for popcount
}

pub struct Bwt {                         // src/pipelines/linear/index/bwt.rs
    pub bwt_str: Vec<u8>,                // 2-bit packed BWT (4 bases per byte)
    pub sa_samples: Vec<u64>,            // Sampled suffix array
    pub cp_occ: Vec<CpOcc>,              // Checkpoints every 64 bases
    pub cumulative_count: [u64; 5],      // L2 array for FM-Index
}
```

**Alignment Tier** (`pipelines/linear/seeding.rs`, `pipelines/linear/chaining.rs`, `pipelines/linear/finalization.rs`):
```rust
pub struct Seed {                        // pipelines/linear/seeding.rs
    pub query_pos: i32,                  // Position in query sequence
    pub ref_pos: u64,                    // Position in reference genome
    pub len: i32,                        // Seed length
    pub is_rev: bool,                    // Strand orientation
    pub interval_size: u64,              // BWT interval size
    pub rid: i32,                        // Reference sequence ID (-1 if spans boundaries)
}

pub struct Chain {                       // pipelines/linear/chaining.rs
    pub seeds: Vec<usize>,               // Indices into sorted seeds array
    pub score: i32,                      // Chain score
    pub weight: i32,                     // Sum of seed lengths
    pub is_rev: bool,                    // Strand orientation
    pub frac_rep: f32,                   // Fraction of repetitive seeds
}

pub struct Alignment {                   // pipelines/linear/finalization.rs
    pub query_name: String,
    pub flag: u16,                       // SAM flag bits
    pub ref_name: String,
    pub ref_id: usize,                   // Reference sequence ID
    pub pos: u64,                        // 0-based leftmost position
    pub mapq: u8,                        // Mapping quality
    pub score: i32,                      // Alignment score
    pub cigar: Vec<(u8, i32)>,           // CIGAR operations
    pub tags: Vec<(String, String)>,     // SAM tags (AS, NM, MD, XA, SA, etc.)
    // Internal fields for selection (not in SAM output)
    pub(crate) query_start: i32,
    pub(crate) query_end: i32,
    pub(crate) seed_coverage: i32,
    pub(crate) hash: u64,
    pub(crate) frac_rep: f32,
}
```

**Reference Tier** (`src/pipelines/linear/index/bntseq.rs`):
```rust
pub struct BntSeq {
    l_pac: u64,              // Total packed sequence length
    n_seqs: i32,             // Number of sequences (chromosomes)
    anns: Vec<BntAnn1>,      // Per-sequence annotations
    ambs: Vec<BntAmb1>,      // Ambiguous base positions
    pac: Vec<u8>             // 2-bit packed reference (4 bases/byte)
}

pub struct BntAnn1 {
    offset: i64,    // Offset in packed sequence
    len: i32,       // Sequence length
    name: String,   // Sequence name (e.g., "chr1")
}
```

### Memory Layout and Performance

**Index Files** (human genome, ~3.1 GB):
- `.bwt.2bit.64`: BWT string (~770 MB, 2-bit packed)
- `.sa`: Sampled suffix array (~390 MB, every 8th position)
- `.pac`: Packed reference (~770 MB, 2-bit packed)
- `.ann`, `.amb`: Metadata (~5 MB)

**Runtime Memory** (~2.4 GB loaded index):
- BWT + checkpoints: ~1.5 GB
- Suffix array samples: ~400 MB
- Reference sequence: ~500 MB

**Allocation Strategy**:
- No per-thread memory pools (unlike C++ version)
- Lazy vector growth for seeds and alignment jobs
- RAII and Rust ownership eliminates explicit memory management
- **Potential issue**: Paired-end mode buffers all alignments in memory (`Vec<(Vec<Alignment>, Vec<Alignment>)>`) - marked as TODO for streaming mode

### Threading Model

**Status**: ✅ **Multi-threaded alignment implemented** (Session 25 - 2025-11-15)

**Architecture**: Batched parallel processing using Rayon work-stealing scheduler

**Pipeline Stages** (matching C++ bwa-mem2 pattern):
1. **Stage 0 (Sequential)**: Read FASTQ/FASTA in batches of 512 reads
2. **Stage 1 (Parallel)**: Process batch using Rayon's `par_iter()`
   - Each read aligned independently in parallel
   - Shared `Arc<BwaIndex>` for read-only FM-index access
   - Automatic load balancing via work-stealing
3. **Stage 2 (Sequential)**: Write SAM output in order (maintains deterministic output)

**Key Differences from C++ bwa-mem2**:

| Aspect | C++ bwa-mem2 | Rust FerrousAlign |
|--------|-------------|-------------------|
| Threading | pthreads + mutex/condvar | Rayon work-stealing |
| Pipeline workers | 2 threads (default) | N threads (configurable via `-t`) |
| Batch size | 512 reads | 512 reads (matching) |
| Memory pools | Per-thread pre-allocation | On-demand allocation |
| Index sharing | Raw pointer sharing | `Arc<BwaIndex>` (thread-safe) |
| Synchronization | Manual mutex/condvar | Lock-free (read-only index) |

**Thread Count Configuration**:
```bash
# Default: all CPU cores
./target/release/ferrous-align mem ref.idx reads.fq

# Specify thread count
./target/release/ferrous-align mem -t 8 ref.idx reads.fq
```

**Validation** (matching C++ fastmap.cpp:674, 810):
- Minimum: 1 thread (values < 1 auto-corrected with warning)
- Maximum: 2× CPU cores (values > 2× capped with warning)
- Default: `num_cpus::get()`

**Performance Characteristics**:
- Linear scaling up to memory bandwidth limits
- Better load balancing than static partitioning (work-stealing)
- Lower synchronization overhead than pthread mutex/condvar
- More memory efficient (single shared index vs per-thread copies)

**Implementation Files**:
- `src/pipelines/linear/single_end.rs`: Single-end batched processing with Rayon
- `src/pipelines/linear/paired/paired_end.rs`: Paired-end batched processing with Rayon
- `src/pipelines/linear/mem.rs`: Entry point (`main_mem()`), dispatches to single/paired-end
- `src/main.rs`: CLI parsing, logger initialization, thread validation
- `src/core/io/fastq_reader.rs`: FASTQ I/O wrapper using bio::io::fastq with gzip support

### Logging and Statistics (Session 26)

**Status**: ✅ **Professional logging framework implemented**

**Logging Framework** (`log` + `env_logger` crates):
- Verbosity levels matching C++ bwa-mem2 convention:
  - `-v 1`: ERROR only (quiet mode - only SAM output)
  - `-v 2`: WARN + ERROR
  - `-v 3`: INFO + WARN + ERROR (default)
  - `-v 4+`: DEBUG + INFO + WARN + ERROR
- Clean output format with level prefixes: `[INFO ]`, `[WARN ]`, `[ERROR]`, `[DEBUG]`
- RUST_LOG environment variable support for advanced users
- No timestamps or module names (cleaner for bioinformatics tools)

**Initialization** (per-command):
```rust
// In main.rs, verbosity mapping:
let log_level = match verbosity {
    v if v <= 1 => log::LevelFilter::Error,
    2 => log::LevelFilter::Warn,
    3 => log::LevelFilter::Info,
    _ => log::LevelFilter::Debug,
};

env_logger::Builder::from_default_env()
    .filter_level(log_level)
    .format_timestamp(None)
    .format_target(false)
    .init();
```

**Statistics Tracking**:
- Per-batch reporting: `Read N sequences (M bp)`
  - Tracks sequence count and total base pairs per batch
  - Matches C++ bwa-mem2 format
- Processing summary: `Processed N reads (M bp) in X.XX sec`
  - Overall statistics with wall time
  - Both single-end and paired-end modes
- Paired-end detailed statistics:
  - Candidate pairs by orientation (FF/FR/RF/RR)
  - Insert size analysis with percentiles
  - Proper pair bounds
  - Mate rescue results

**Implementation Files**:
- `src/main.rs`: Logger initialization, verbosity mapping
- `src/pipelines/linear/single_end.rs`: Single-end with statistics tracking
- `src/pipelines/linear/paired/paired_end.rs`: Paired-end with statistics tracking

**Usage Examples**:
```bash
# Quiet mode (only SAM output)
./target/release/ferrous-align mem -v 1 ref.idx reads.fq > output.sam

# Normal mode (default - shows key progress)
./target/release/ferrous-align mem ref.idx reads.fq > output.sam

# Gzipped FASTQ input (auto-detected by .gz extension - Session 27)
./target/release/ferrous-align mem ref.idx reads.fq.gz > output.sam

# Debug mode (detailed batch processing)
./target/release/ferrous-align mem -v 4 ref.idx reads.fq > output.sam

# Override with RUST_LOG
RUST_LOG=debug ./target/release/ferrous-align mem ref.idx reads.fq > output.sam
```

## Important Code Patterns

### Platform-Specific SIMD

Always use the abstraction layer, never raw intrinsics:

```rust
// GOOD: Portable across x86_64 and ARM
use crate::compute::simd_abstraction::*;
let sum = _mm_add_epi16(a, b);

// BAD: x86-only
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::_mm_add_epi16;
```

### Reverse Complement Handling

Reads are always processed in both orientations:

```rust
// Search forward strand
let fwd_smems = backward_search(query, ...);

// Search reverse complement
let rev_query: Vec<u8> = query.iter().rev()
    .map(|&b| b ^ 3) // A↔T, C↔G (2-bit encoding)
    .collect();
let rev_smems = backward_search(&rev_query, ...);
```

### BWT Interval Size Filtering

Seed filtering based on occurrence count (avoid repetitive regions):

```rust
const MAX_SEED_OCCURRENCE: u64 = 500;
if smem.l - smem.k > MAX_SEED_OCCURRENCE {
    continue; // Skip over-represented seeds
}
```

### CIGAR String Generation

Generated in `banded_swa.rs` from Smith-Waterman traceback:

```rust
// Format: e.g., "50M2I48M" (50 matches, 2 insertions, 48 matches)
// Operations: M (match/mismatch), I (insertion), D (deletion), S (soft clip)
let cigar = result.cigar_string;
```

## Testing Architecture

**Unit Tests** (296+ tests in source files):
- `src/pipelines/linear/index/bwt_test.rs`: FM-Index backward search, checkpoint calculation, popcount64
- `src/core/io/fastq_reader_test.rs`: FASTQ parsing with bio::io::fastq, gzip support
- `src/pipelines/linear/index/bntseq.rs` (inline tests): Reference loading and ambiguous base handling
- `src/core/alignment/banded_swa.rs` (inline tests): Smith-Waterman tests including SIMD batched variants
- `src/pipelines/linear/pipeline.rs` (inline tests): Pipeline stage tests
- `src/pipelines/linear/mem_opt.rs` (inline tests): Command-line parameter parsing tests

**Integration Tests** (`tests/`):
- `integration_test.rs`: End-to-end index build + single-end alignment
- `paired_end_integration_test.rs`: Insert size calculation, mate rescue
- `complex_integration_test.rs`: Multi-reference genomes, edge cases
- `complex_alignment_tests.rs`: CIGAR validation, strand orientation

**Golden Reads Parity Tests** (`tests/golden_reads/`):
- 10K read pairs from HG002 WGS for regression testing
- `baseline_ferrous.sam`: Frozen ferrous-align output (parity target)
- `baseline_bwamem2.sam`: BWA-MEM2 reference for comparison
- Used for validating pipeline refactoring (see `dev_notes/PIPELINE_FRONTLOADING_PLAN.md`)
- Data files excluded from git; regenerate locally with commands in README.md

**Test Data**:
- Small synthetic references in integration tests
- Tests gracefully skip if data missing: `if !path.exists() { return; }`
- Large test data (HG002, etc.) stored in `/home/jkane/Genomics/`

**Benchmarks** (`benches/simd_benchmarks.rs`):
- Compares scalar vs batched SIMD Smith-Waterman
- Tests sequence lengths: 50bp, 100bp, 200bp
- Run with `cargo bench`

## Compatibility Notes

**Index Format**:
- Matches bwa-mem2 v2.0+ (post-October 2020)
- Files: `.bwt.2bit.64`, `.sa`, `.pac`, `.ann`, `.amb`
- Incompatible with older bwa-mem2 indices (pre-commit #4b59796)

**SAM Output** (Session 26):
- Compatible with bwa-mem v0.7.17
- Complete headers: @HD, @SQ, @PG with auto-updating metadata
- @PG header uses `env!("CARGO_PKG_NAME")` and `env!("CARGO_PKG_VERSION")` for future-proof renaming
- Includes MC (mate CIGAR) tag for paired-end
- Standard flags: 0x1 (paired), 0x2 (proper pair), 0x10 (reverse), etc.
- Proper STDOUT/STDERR separation (SAM on stdout, diagnostics on stderr)

**Platform Support**:
- ✅ macOS x86_64, arm64 (Apple Silicon with native NEON)
- ✅ Linux x86_64, aarch64
- ⚠️ Windows (untested, likely works with minor path handling changes)

**SIMD Support by Platform**:
- x86_64: SSE2 (baseline), AVX2 (automatic), AVX-512 (opt-in via `--features avx512`)
- aarch64: NEON (baseline, equivalent to SSE2)
- AVX2 CPUs: Intel Haswell+ (2013), AMD Excavator+ (2015)
- AVX-512 CPUs: Intel Skylake-X+ (2017), AMD Zen 4+ (2022, Ryzen 7000/9000)

## Performance Characteristics

**Indexing** (human genome, 3.1 GB FASTA):
- Time: ~45 minutes (single-threaded SAIS)
- Memory: ~28 GB peak (8-9x genome size)
- Output: ~3.1 GB index files

**Alignment** (WGS reads, 30x coverage):
- Speed: ~85-95% of C++ bwa-mem2 (room for optimization)
- Memory: ~2.4 GB (loaded index)
- Bottlenecks: Suffix array reconstruction, seed chaining

**SIMD Impact**:
- SSE/NEON (128-bit): Baseline performance (16-way parallelism)
- AVX2 (256-bit): 1.8-2.2x speedup over SSE (32-way parallelism)
- AVX-512 (512-bit): 2.5-3.0x speedup over SSE (64-way, experimental)
- Threshold: Need ≥16 alignments to amortize SIMD setup cost
- Runtime detection automatically uses best available SIMD engine

## Known Issues and TODOs

**Completed Features (v0.6.0)**:
1. ✅ Multi-threading for read processing (Session 25 - Rayon batched processing)
2. ✅ Logging framework and statistics (Session 26 - log + env_logger)
3. ✅ Complete SAM headers (Session 26 - @HD, @SQ, @PG)
4. ✅ bio::io::fastq migration with gzip support (Session 27 - native .fq.gz handling)
5. ✅ AVX2 SIMD support (Session 28 - 256-bit vectors, 32-way parallelism, automatic detection)
6. ✅ AVX-512 SIMD support (Session 28 - 512-bit vectors, 64-way parallelism, feature-gated)
7. ✅ SMEM generation algorithm (Session 29 - complete rewrite to match C++ bwa-mem2)
8. ✅ Index building format compatibility (Session 29 - .pac file format, ambiguous bases, BWT construction)
9. ✅ Adaptive batch sizing and SIMD routing (Session 29 - performance optimizations)
10. ✅ Golden reads parity test infrastructure (tests/golden_reads/ - 10K HG002 pairs)
11. ✅ Heterogeneous compute abstraction (Session 30 - GPU/NPU integration points, NO-OP placeholders)
12. ✅ GATK ValidateSamFile parity (97.71% proper pairing vs BWA-MEM2's 97.10%)

**New Features (v0.7.0-alpha - SoA Migration)**:
1. ✅ SoA-native FASTQ reader (`SoAReadBatch`, `SoAFastqReader`)
2. ✅ SoA batch processing through entire pipeline (seeding → chaining → extension → finalization)
3. ✅ Thread-local workspace allocation (`AlignmentWorkspace` with buffer reuse)
4. ✅ Unified kernel dispatch (`batch_extension::dispatch` - banded_swa + kswv routing)
5. ✅ Modular file structure (<500 lines per file in `batch_extension/`)
6. ✅ SoA carriers for SIMD kernels (`SwSoA`, `KswSoA`, `AlignJob`)
7. ✅ Vertical + horizontal SIMD kernel support (banded_swa + kswv)

**Integration Testing (v0.7.0-alpha - Current Focus)**:
🔄 **In Progress**: Validate SoA pipeline against v0.6.0 baseline
- Run full test suite (unit tests, integration tests, benchmarks)
- Verify GATK parity maintained on 4M read dataset
- Profile memory usage (expect reduction due to workspace reuse)
- Benchmark throughput (expect improvement from better cache utilization)
- Test all SIMD variants (SSE/NEON, AVX2, AVX-512)

**Known File Size Issues** (files exceeding 500-line target):
- `seeding.rs`: 1902 lines (to be split into submodules)
- `finalization.rs`: 1707 lines (to be split into submodules)
- `region.rs`: 1598 lines (to be split into submodules)
- `pipeline.rs`: 1235 lines (to be split into submodules)
- `chaining.rs`: 1009 lines (to be split into submodules)
- `mem_opt.rs`: 893 lines (configuration, low priority for splitting)

**Pending Optimizations** (post-v0.7.0):
1. ⏳ Faster suffix array reconstruction (cache recent lookups)
2. ⏳ Streaming mode for paired-end (avoid buffering all alignments)
3. ⏳ Vectorize backward_ext() for FM-Index search
4. ⏳ Stabilize AVX-512 support (waiting on Rust compiler stabilization)
5. ⏳ Split large files into <500 line modules (improve maintainability)

**AVX-512 Support** (Experimental):
- Status: ✅ Implemented but gated behind `--features avx512`
- Reason: AVX-512 intrinsics are unstable in stable Rust compiler
- To enable: Use nightly Rust with `cargo +nightly build --release --features avx512`
- Hardware: Requires Intel Skylake-X+ or AMD Zen 4+ (Ryzen 7000/9000)
- Performance: Expected 2.5-3.0x speedup over SSE baseline
- Note: Will be enabled by default once Rust stabilizes AVX-512 intrinsics

**Future Enhancements**:
- Apple Acceleration framework integration (vDSP for matrix ops)
- Memory-mapped index loading (reduce startup time)
- GPU acceleration via Metal (Apple Silicon) or CUDA - **Integration points ready in `src/core/compute/`**
- NPU seed pre-filtering via ANE/ONNX - **Integration points ready, ONE-HOT encoding in `src/core/compute/encoding.rs`**
- Pangenome graph alignment (`src/pipelines/graph/`) - **Placeholder created for future implementation**

## Structure-of-Arrays (SoA) Design Philosophy

**Why SoA?** (v0.7.0 Architecture)

The v0.7.0 release migrates from Array-of-Structures (AoS) to Structure-of-Arrays (SoA) throughout the pipeline. This is a fundamental memory layout change that improves SIMD efficiency and cache utilization.

**AoS vs SoA Example**:
```rust
// AoS (v0.6.0): Poor cache locality, scattered loads
struct Read {
    name: String,
    sequence: Vec<u8>,
    quality: Vec<u8>,
}
let reads: Vec<Read> = ...;
for read in &reads {
    process(read.sequence);  // Cache miss per iteration
}

// SoA (v0.7.0): Sequential access, cache-friendly
struct ReadBatch {
    sequences: Vec<u8>,       // All sequences concatenated
    seq_offsets: Vec<usize>,  // Slice indices
    lengths: Vec<usize>,
}
let batch: ReadBatch = ...;
for i in 0..batch.count {
    let seq = &batch.sequences[batch.seq_offsets[i]..][..batch.lengths[i]];
    process(seq);  // Sequential memory access
}
```

**SoA Benefits**:
1. **SIMD Efficiency**: Kernels process `lanes` alignments in parallel with aligned loads
2. **Cache Utilization**: Sequential memory access reduces cache misses by ~40%
3. **Zero-Copy**: Thread-local workspaces eliminate per-batch allocations
4. **Batch Parallelism**: Horizontal SIMD (kswv) requires SoA layout

**Performance Targets** (v0.7.0 vs v0.6.0):
- Memory footprint: -15% (workspace reuse)
- Throughput: +10-20% (better cache utilization)
- GATK parity: Maintained (97.71% proper pairing)

## Development Workflow

**Active Branch**: `feature/core-rearch` (SoA migration)

**Current Development Plans**:
- **SoA Integration Testing** (Current): Validate v0.7.0-alpha against v0.6.0 baseline
  - Run full test suite to catch regressions
  - Benchmark performance on HG002 4M read dataset
  - Verify GATK parity maintained
  - Profile memory usage improvements

**Making Changes**:
1. Work in `feature/core-rearch` branch for SoA-related changes
2. Run `cargo fmt` before each commit
3. Run `cargo clippy` and address warnings
4. **File size limit**: Keep files under 500 lines (split into submodules if needed)
5. Add unit tests for new utilities
6. Add integration tests for new pipeline stages
7. Run `cargo test` to ensure no regressions
8. Benchmark critical path changes with `cargo bench`
9. For pipeline changes: verify against v0.6.0 baseline output
10. **SoA-specific**: Ensure new modules use SoA batch processing where appropriate

**Debugging Tips**:
- Use `cargo test -- --nocapture` to see println! output
- Enable debug assertions: `RUSTFLAGS="-C debug-assertions=on" cargo build --release`
- Profile with `cargo instruments` on macOS or `perf` on Linux
- Check SIMD codegen: `cargo rustc --release -- --emit asm`

**Common Development Tasks**:
- **Add new SIMD intrinsic**: Update `src/core/compute/simd_abstraction/mod.rs` with both x86_64 and aarch64 implementations
- **Modify alignment scoring**: Edit `src/core/alignment/shared_types.rs` (`KernelConfig`, `ScoringMatrix`)
- **Add new alignment kernel**: Implement in `src/core/alignment/`, integrate via `batch_extension::dispatch`
- **Change index format**: Update serialization in `src/pipelines/linear/index/bwa_index.rs` and `src/pipelines/linear/index/bwt.rs`
- **Add SAM tags**: Modify output formatting in `src/core/io/sam_output.rs` or `src/pipelines/linear/finalization.rs`
- **Add SoA batch processing**: Create new `SoA*Batch` struct, implement batch collection/distribution
- **Split large file**: Extract related functions into submodule (e.g., `seeding/` directory with `mod.rs`, `smem.rs`, `convert.rs`)

## References

This Rust port is based on:
- [bwa-mem2](https://github.com/bwa-mem2/bwa-mem2) - C++ optimized version
- [bwa](https://github.com/lh3/bwa) - Original algorithm by Heng Li

**Citation**: Vasimuddin Md, Sanchit Misra, Heng Li, Srinivas Aluru. "Efficient Architecture-Aware Acceleration of BWA-MEM for Multicore Systems." IEEE IPDPS 2019.
