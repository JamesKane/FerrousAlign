# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**FerrousAlign** (`ferrous-align`) is a Rust port of [bwa-mem2](https://github.com/bwa-mem2/bwa-mem2), a high-performance bioinformatics tool for aligning DNA sequencing reads to reference genomes using the Burrows-Wheeler Transform. This implementation targets performance parity with the C++ version (1.3-3.1x faster than original BWA-MEM) while providing Rust's safety and maintainability benefits.

### NOTICE: The C++ reference implementation's behavior and file formats are the technical specification.  Any deviation is a critical bug, which blocks all downstream tasks.  We must match the behavior even if that means rewriting the code.

**Project Status**: v0.5.0 (~65% complete), ~85-95% of C++ bwa-mem2 performance, working single-end and paired-end alignment with complete SAM output. Session 29 achieved major milestone: **Rust-built indices now produce identical results to C++ bwa-mem2**. Recent sessions focused on alignment quality (MAPQ, supplementary marking, proper pairing).

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

**Library Modules** (`src/lib.rs`):
```
pub mod alignment;        // Core alignment pipeline (seeding, chaining, extension, finalization)
pub mod bntseq;           // Reference sequence handling
pub mod bwa_index;        // Index building (uses bio crate for suffix array)
pub mod bwt;              // BWT data structure
pub mod compute;          // Heterogeneous compute abstraction (CPU SIMD/GPU/NPU integration points)
pub mod fastq_reader;     // FASTQ parsing using bio::io::fastq (query reads with gzip support)
pub mod fm_index;         // FM-Index operations (BWT search, occurrence counting)
pub mod index;            // Index management (BwaIndex loading/dumping)
pub mod insert_size;      // Insert size statistics for paired-end
pub mod kseq;             // FASTA parsing (used for reference genomes during indexing)
pub mod mate_rescue;      // Mate rescue using Smith-Waterman
pub mod mem;              // High-level alignment logic (includes logging/stats)
pub mod mem_opt;          // Command-line options and parameters
pub mod paired_end;       // Paired-end read processing
pub mod pairing;          // Paired-end alignment scoring
pub mod sam_output;       // SAM output formatting and flag management
pub mod simd;             // SIMD engine detection and dispatch
pub mod simd_abstraction; // Cross-platform SIMD layer (SSE2/AVX2/AVX-512/NEON)
pub mod single_end;       // Single-end read processing
pub mod utils;            // I/O and bit manipulation utilities
```

**Alignment Submodules** (`src/alignment/`):
```
pub mod banded_swa;       // SIMD Smith-Waterman (128-bit baseline)
pub mod banded_swa_avx2;  // AVX2 256-bit Smith-Waterman (x86_64 only)
pub mod banded_swa_avx512;// AVX-512 512-bit Smith-Waterman (feature-gated)
pub mod chaining;         // Seed chaining with O(n²) DP
pub mod extension;        // Banded alignment extension jobs
pub mod finalization;     // Alignment selection, MAPQ, secondary marking
pub mod ksw_affine_gap;   // Affine-gap Smith-Waterman fallback
pub mod pipeline;         // Main alignment pipeline (generate_seeds -> align_read)
pub mod seeding;          // SMEM generation from FM-Index
pub mod utils;            // Base encoding, scoring matrices
```

**External Dependencies**:
- `log` + `env_logger`: Structured logging with verbosity control (Session 26)
- `rayon`: Multi-threading with work-stealing scheduler (Session 25)
- `clap`: Command-line argument parsing
- `bio`: Bioinformatics utilities (suffix array, FASTQ parsing - Session 27)
- `flate2`: Gzip compression support for FASTQ files (Session 27)

### Alignment Pipeline (Four-Stage Architecture)

The pipeline implements a multi-kernel design matching the C++ version. Entry point: `alignment::pipeline::align_read()`.

**Stage 1: Seeding** (`alignment::pipeline::find_seeds()`):
- Extracts SMEMs (Supermaximal Exact Matches) via FM-Index backward search
- Searches both forward and reverse-complement strands
- Re-seeding pass for long unique SMEMs (chimeric detection)
- 3rd round seeding for highly repetitive regions (`max_mem_intv`)
- Filters seeds by minimum length and maximum occurrence count
- Converts BWT intervals to reference positions via suffix array

**Stage 2: Chaining** (`alignment::pipeline::build_and_filter_chains()`):
- Groups compatible seeds using O(n²) dynamic programming (`chaining::chain_seeds()`)
- Scores chains based on seed lengths and gaps
- Filters chains by score threshold and overlap

**Stage 3: Extension** (`alignment::pipeline::extend_chains_to_alignments()`):
- Creates alignment jobs for left/right extension around each seed
- Executes banded Smith-Waterman via adaptive routing:
  - **SIMD mode**: Routes to 128/256/512-bit engine based on CPU
  - **Scalar fallback**: For pathological alignments with ksw_extend2
- Generates CIGAR strings for aligned regions

**Stage 4: Finalization** (`alignment::pipeline::finalize_alignments()`):
- Merges left/right extension CIGARs with seed
- Converts FM-index positions to chromosome coordinates
- Computes MD tag and exact NM (edit distance)
- Filters by score threshold, removes redundant alignments
- Marks secondary/supplementary alignments, calculates MAPQ
- Generates XA/SA tags for alternative alignments
- Produces unmapped record if no alignments survive

**Paired-End Processing** (`paired_end::process_read_pairs()`):
1. Insert size estimation from concordant pairs
2. Mate rescue (re-align using mate's location)
3. Pair re-scoring based on insert size distribution
4. Primary/secondary alignment marking
5. SAM output with proper pair flags

### SIMD Abstraction Layer

**Design Philosophy**: Write-once, run-anywhere SIMD code

**Platform Detection** (`simd_abstraction.rs`):
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

**Performance Hotspot** (`popcount64()` in `align.rs`):
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

**Module Structure** (`src/compute/`):
```
src/compute/
├── mod.rs          # ComputeBackend enum, detection, ComputeContext
└── encoding.rs     # EncodingStrategy (Classic 2-bit vs ONE-HOT for NPU)
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
| Entry | `src/mem.rs:10-33` | `main_mem()` | Backend detection |
| Dispatch | `src/mem.rs:230-241` | `main_mem()` | Pass context to processing |
| Single-end | `src/single_end.rs:35-50` | `process_single_end()` | Accept compute context |
| Paired-end | `src/paired_end.rs:161-176` | `process_paired_end()` | Accept compute context |
| Pipeline | `src/alignment/pipeline.rs:95-114` | `align_read()` | Main alignment entry |
| Extension | `src/alignment/pipeline.rs:494-517` | `extend_chains_to_alignments()` | Stage 3 dispatch |
| Backend Switch | `src/alignment/pipeline.rs:641-682` | `extend_chains_to_alignments()` | Route to backend |
| Extension Docs | `src/alignment/extension.rs:1-44` | Module header | GPU/NPU implementation guide |

**Adding a New Backend** (e.g., Metal GPU):

1. **Update `compute/mod.rs`**:
   - Remove NO-OP fallback in `effective_backend()` for your variant
   - Add detection logic in `detect_optimal_backend()`

2. **Update `alignment/pipeline.rs`** (line ~659):
   ```rust
   ComputeBackend::Gpu => {
       execute_gpu_alignments(&gpu_context, &sw_params, &alignment_jobs)
   }
   ```

3. **Implement backend kernel** (new file, e.g., `compute/gpu.rs`):
   ```rust
   pub fn execute_gpu_alignments(
       ctx: &GpuContext,
       sw_params: &BandedPairWiseSW,
       jobs: &[AlignmentJob],
   ) -> Vec<(i32, Vec<(u8, i32)>, Vec<u8>, Vec<u8>)>
   ```

4. **For NPU seed pre-filtering**:
   - Use `EncodingStrategy::OneHot` in `find_seeds()` (pipeline.rs:127)
   - Implement seed classifier model
   - Filter seeds before building alignment jobs

**Performance Thresholds**:
- GPU dispatch overhead: ~20-50μs
- SW kernel CPU time: ~1-2μs per alignment
- GPU threshold: batch_size >= 1024 to amortize overhead

### Key Data Structures

**FM-Index Tier** (`fm_index.rs`, `bwt.rs`):
```rust
pub struct SMEM {                        // alignment/seeding.rs
    pub query_start: i32,                // Start position in query (0-based, inclusive)
    pub query_end: i32,                  // End position in query (0-based, exclusive)
    pub bwt_interval_start: u64,         // Start of BWT interval in suffix array
    pub bwt_interval_end: u64,           // End of BWT interval
    pub interval_size: u64,              // Occurrence count
    pub is_reverse_complement: bool,     // Forward or reverse strand
}

pub struct CpOcc {                       // fm_index.rs
    cp_count: [i64; 4],                  // Cumulative base counts at checkpoint
    one_hot_bwt_str: [u64; 4],           // One-hot encoded BWT for popcount
}

pub struct Bwt {                         // bwt.rs
    pub bwt_str: Vec<u8>,                // 2-bit packed BWT (4 bases per byte)
    pub sa_samples: Vec<u64>,            // Sampled suffix array
    pub cp_occ: Vec<CpOcc>,              // Checkpoints every 64 bases
    pub cumulative_count: [u64; 5],      // L2 array for FM-Index
}
```

**Alignment Tier** (`alignment/seeding.rs`, `alignment/chaining.rs`, `alignment/finalization.rs`):
```rust
pub struct Seed {                        // alignment/seeding.rs
    pub query_pos: i32,                  // Position in query sequence
    pub ref_pos: u64,                    // Position in reference genome
    pub len: i32,                        // Seed length
    pub is_rev: bool,                    // Strand orientation
    pub interval_size: u64,              // BWT interval size
    pub rid: i32,                        // Reference sequence ID (-1 if spans boundaries)
}

pub struct Chain {                       // alignment/chaining.rs
    pub seeds: Vec<usize>,               // Indices into sorted seeds array
    pub score: i32,                      // Chain score
    pub weight: i32,                     // Sum of seed lengths
    pub is_rev: bool,                    // Strand orientation
    pub frac_rep: f32,                   // Fraction of repetitive seeds
}

pub struct Alignment {                   // alignment/finalization.rs
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

**Reference Tier** (`bntseq.rs`):
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
- `src/single_end.rs`: Single-end batched processing with Rayon
- `src/paired_end.rs`: Paired-end batched processing with Rayon
- `src/mem.rs`: Entry point (`main_mem()`), dispatches to single/paired-end
- `src/main.rs`: CLI parsing, logger initialization, thread validation
- `src/fastq_reader.rs`: FASTQ I/O wrapper using bio::io::fastq with gzip support

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
- `src/single_end.rs`: Single-end with statistics tracking
- `src/paired_end.rs`: Paired-end with statistics tracking

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
use crate::simd_abstraction::*;
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

**Unit Tests** (98+ tests in source files):
- `bwt_test.rs`: FM-Index backward search, checkpoint calculation, popcount64
- `kseq_test.rs`: FASTA parsing edge cases (used for reference genomes)
- `fastq_reader_test.rs`: FASTQ parsing with bio::io::fastq, gzip support
- `bntseq_test.rs`: Reference loading and ambiguous base handling
- `alignment/banded_swa`: Smith-Waterman tests including SIMD batched variants
- `alignment/pipeline`: Pipeline stage tests
- `mem_opt`: Command-line parameter parsing tests

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

**Completed Features**:
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

**Remaining Optimizations**:
1. Faster suffix array reconstruction (cache recent lookups)
2. Streaming mode for paired-end (avoid buffering all alignments)
3. Vectorize backward_ext() for FM-Index search
4. Stabilize AVX-512 support (waiting on Rust compiler stabilization)

**Remaining Correctness Issues**:
- **Proper pairing rate gap**: 90.28% vs BWA-MEM2's 97.11% on HG002 10K test (6.83% gap)
  - Root cause: Insert size estimation or pair scoring differences
  - Tracked in `tests/golden_reads/README.md`
- Secondary alignment marking not fully matching bwa-mem2 behavior
- Some edge cases in CIGAR generation for complex indels

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
- GPU acceleration via Metal (Apple Silicon) or CUDA - **Integration points ready in `src/compute/`**
- NPU seed pre-filtering via ANE/ONNX - **Integration points ready, ONE-HOT encoding in `src/compute/encoding.rs`**

## Development Workflow

**Active Development Plans**:
- **Pipeline Front-Loading** (`dev_notes/PIPELINE_FRONTLOADING_PLAN.md`): Refactoring to move computations earlier in the alignment pipeline, making finalization thinner. Phase 0 (golden reads parity tests) complete.

**Making Changes**:
1. Create feature branch from `main`
2. Run `cargo fmt` before each commit
3. Run `cargo clippy` and address warnings
4. Add unit tests for new utilities
5. Add integration tests for new pipeline stages
6. Run `cargo test` to ensure no regressions
7. Benchmark critical path changes with `cargo bench`
8. For pipeline changes: verify against `tests/golden_reads/baseline_ferrous.sam`

**Debugging Tips**:
- Use `cargo test -- --nocapture` to see println! output
- Enable debug assertions: `RUSTFLAGS="-C debug-assertions=on" cargo build --release`
- Profile with `cargo instruments` on macOS or `perf` on Linux
- Check SIMD codegen: `cargo rustc --release -- --emit asm`

**Common Development Tasks**:
- Add new SIMD intrinsic: Update `simd_abstraction.rs` with both x86_64 and aarch64 implementations
- Modify alignment scoring: Edit constants in `align.rs` and `banded_swa.rs`
- Change index format: Update serialization in `bwa_index.rs` and `bwt.rs`
- Add SAM tags: Modify output formatting in `align.rs`

## References

This Rust port is based on:
- [bwa-mem2](https://github.com/bwa-mem2/bwa-mem2) - C++ optimized version
- [bwa](https://github.com/lh3/bwa) - Original algorithm by Heng Li

**Citation**: Vasimuddin Md, Sanchit Misra, Heng Li, Srinivas Aluru. "Efficient Architecture-Aware Acceleration of BWA-MEM for Multicore Systems." IEEE IPDPS 2019.
