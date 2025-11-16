# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**FerrousAlign** (`ferrous-align`) is a Rust port of [bwa-mem2](https://github.com/bwa-mem2/bwa-mem2), a high-performance bioinformatics tool for aligning DNA sequencing reads to reference genomes using the Burrows-Wheeler Transform. This implementation targets performance parity with the C++ version (1.3-3.1x faster than original BWA-MEM) while providing Rust's safety and maintainability benefits.

**Project Status**: v0.5.0 (~50% complete), ~85-95% of C++ bwa-mem2 performance, working single-end and paired-end alignment with complete SAM output including proper headers and logging.

## Build Commands

```bash
# Build in release mode (required for reasonable performance)
cargo build --release

# Build with native CPU optimizations (recommended)
RUSTFLAGS="-C target-cpu=native" cargo build --release

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
pub mod bwt;              // BWT data structure and FM-Index operations
pub mod bntseq;           // Reference sequence handling
pub mod kseq;             // FASTA parsing (used for reference genomes during indexing)
pub mod fastq_reader;     // FASTQ parsing using bio::io::fastq (query reads with gzip support)
pub mod align;            // Core alignment pipeline
pub mod mem;              // High-level alignment logic (includes logging/stats)
pub mod banded_swa;       // SIMD Smith-Waterman
pub mod utils;            // I/O and bit manipulation utilities
pub mod bwa_index;        // Index building (uses bio crate for suffix array)
pub mod simd_abstraction; // Cross-platform SIMD layer
pub mod mem_opt;          // Command-line options and parameters
```

**External Dependencies**:
- `log` + `env_logger`: Structured logging with verbosity control (Session 26)
- `rayon`: Multi-threading with work-stealing scheduler (Session 25)
- `clap`: Command-line argument parsing
- `bio`: Bioinformatics utilities (suffix array, FASTQ parsing - Session 27)
- `flate2`: Gzip compression support for FASTQ files (Session 27)

### Alignment Pipeline (Five-Stage Architecture)

The pipeline implements a multi-kernel design matching the C++ version:

**Stage 1: Seed Extraction** (`align::generate_seeds()`):
- Extracts SMEMs (Supermaximal Exact Matches) via FM-Index backward search
- Searches both forward and reverse-complement strands
- Filters seeds by minimum length and maximum occurrence count
- Returns `Vec<SMEM>` with BWT intervals [k, l)

**Stage 2: Suffix Array Reconstruction** (`get_sa_entry()`):
- Converts BWT positions to reference genome positions
- Uses sampled suffix array (typically every 8th position)
- Includes infinite loop detection for safety

**Stage 3: Seed Extension** (`execute_batched_alignments()` or `execute_scalar_alignments()`):
- Extends seeds using banded Smith-Waterman alignment
- **Batched SIMD mode**: When ≥16 alignment jobs, uses 8-way or 16-way SIMD parallelism
- **Scalar mode**: Falls back to non-SIMD for small batches
- Generates CIGAR strings for aligned regions
- Returns `Vec<Alignment>` with scores and CIGARs

**Stage 4: Seed Chaining** (`chain_seeds()`):
- Groups compatible seeds using O(n²) dynamic programming
- Scores chains based on seed lengths and gaps
- Identifies best chain per read

**Stage 5: SAM Output**:
- Formats alignments as SAM records
- Includes CIGAR strings, flags, mapping quality, and metadata
- For paired-end: adds insert size, mate information, and MC (mate CIGAR) tag

**Paired-End Extension** (`mem::process_read_pairs()`):
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

**SIMD Width**:
- 128-bit vectors (8 x i16 or 16 x i8)
- Banded Smith-Waterman processes 8 or 16 sequences in parallel
- Threshold: Switches to batched SIMD when ≥16 alignment jobs available

### Key Data Structures

**FM-Index Tier** (`align.rs`, `bwt.rs`):
```rust
pub struct SMEM {
    m: i32,           // Start position in query
    n: i32,           // End position in query
    k: u64,           // Start of BWT interval
    l: u64,           // End of BWT interval
    is_rev_comp: bool // Forward or reverse strand
}

pub struct CpOcc {
    cp_count: [i64; 4],        // Cumulative base counts at checkpoint
    one_hot_bwt_str: [u64; 4]  // One-hot encoded BWT for popcount
}

pub struct Bwt {
    bwt_str: Vec<u8>,      // 2-bit packed BWT (4 bases per byte)
    sa_samples: Vec<u64>,   // Sampled suffix array
    cp_occ: Vec<CpOcc>,     // Checkpoints every 64 bases
}
```

**Alignment Tier** (`align.rs`, `banded_swa.rs`):
```rust
pub struct Seed {
    query_pos: i32,  // Position in query sequence
    ref_pos: u64,    // Position in reference genome
    len: i32,        // Seed length
    is_rev: bool     // Strand orientation
}

pub struct Chain {
    seeds: Vec<Seed>,
    score: i32,
    query_start: i32,
    query_end: i32
}

pub struct Alignment {
    ref_pos: u64,
    query_start: i32,
    query_end: i32,
    cigar: String,
    score: i32,
    mapq: u8,
    is_reverse: bool
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
- `src/mem.rs:275-379`: Single-end batched processing (uses FastqReader - Session 27)
- `src/mem.rs:382-906`: Paired-end batched processing (uses FastqReader - Session 27)
- `src/main.rs:206-289`: Logger initialization and thread validation
- `src/fastq_reader.rs`: FASTQ I/O wrapper using bio::io::fastq (Session 27)

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
- `src/main.rs:206-289`: Logger initialization, verbosity mapping
- `src/mem.rs:275-379`: Single-end with statistics tracking
- `src/mem.rs:382-906`: Paired-end with statistics tracking
- Converted ~35 `eprintln!` statements to structured logging

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

**Unit Tests** (98 tests in source files):
- `bwt_test.rs`: FM-Index backward search, checkpoint calculation, popcount64
- `kseq_test.rs`: FASTA parsing edge cases (used for reference genomes)
- `fastq_reader_test.rs`: FASTQ parsing with bio::io::fastq, gzip support (Session 27)
- `bntseq_test.rs`: Reference loading and ambiguous base handling
- `utils_test.rs`: Bit manipulation, file I/O
- `banded_swa` module: Smith-Waterman tests including SIMD batched variants
- `align` module: Alignment pipeline tests
- `mem_opt` module: Command-line parameter parsing tests

**Integration Tests** (`tests/`):
- `integration_test.rs`: End-to-end index build + single-end alignment
- `paired_end_integration_test.rs`: Insert size calculation, mate rescue
- `complex_integration_test.rs`: Multi-reference genomes, edge cases
- `complex_alignment_tests.rs`: CIGAR validation, strand orientation

**Test Data**:
- Small synthetic references in integration tests
- Tests gracefully skip if data missing: `if !path.exists() { return; }`

**Benchmarks** (`benches/simd_benchmarks.rs`):
- Compares scalar vs batched SIMD Smith-Waterman
- Tests sequence lengths: 50bp, 100bp, 200bp
- Run with `cargo bench` (requires nightly Rust for criterion)

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
- Batched mode: 3-4x speedup over scalar Smith-Waterman
- Threshold: Need ≥16 alignments to amortize SIMD setup cost

## Known Issues and TODOs

**Completed Features**:
1. ✅ Multi-threading for read processing (Session 25 - Rayon batched processing)
2. ✅ Logging framework and statistics (Session 26 - log + env_logger)
3. ✅ Complete SAM headers (Session 26 - @HD, @SQ, @PG)
4. ✅ bio::io::fastq migration with gzip support (Session 27 - native .fq.gz handling)

**Remaining Optimizations**:
1. Faster suffix array reconstruction (cache recent lookups)
2. Streaming mode for paired-end (avoid buffering all alignments)
3. Runtime SIMD dispatch (detect AVX2/AVX-512 on x86_64)
4. Vectorize backward_ext() for FM-Index search

**Correctness Issues**:
- Paired-end insert size distribution may differ slightly from C++ version
- Secondary alignment marking not fully matching bwa-mem2 behavior
- Some edge cases in CIGAR generation for complex indels

**Future Enhancements**:
- Apple Acceleration framework integration (vDSP for matrix ops)
- Memory-mapped index loading (reduce startup time)
- GPU acceleration via Metal (Apple Silicon) or CUDA

## Development Workflow

**Making Changes**:
1. Create feature branch from `main`
2. Run `cargo fmt` before each commit
3. Run `cargo clippy` and address warnings
4. Add unit tests for new utilities
5. Add integration tests for new pipeline stages
6. Run `cargo test` to ensure no regressions
7. Benchmark critical path changes with `cargo bench`

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
