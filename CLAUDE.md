# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Overview

**FerrousAlign** (`ferrous-align`) is a Rust port of [bwa-mem2](https://github.com/bwa-mem2/bwa-mem2), a high-performance DNA sequence aligner using the Burrows-Wheeler Transform. Targets performance parity with C++ (1.3-3.1x faster than original BWA-MEM) with Rust's safety benefits.

**Critical**: The C++ bwa-mem2 behavior and file formats are the technical specification. Any deviation is a critical bug.

**Status**: v0.7.0-alpha (feature/core-rearch) - SoA architecture migration complete, integration testing in progress.

## Build Commands

```bash
# Release build (required for reasonable performance)
cargo build --release

# With native CPU optimizations (recommended)
RUSTFLAGS="-C target-cpu=native" cargo build --release

# AVX-512 (nightly, experimental)
cargo +nightly build --release --features avx512

# Run
./target/release/ferrous-align --help
```

### Testing

```bash
cargo test --lib                    # Unit tests
cargo test --test '*'               # Integration tests
cargo test -- --nocapture           # With output
cargo bench                         # Benchmarks
```

### Code Quality

```bash
cargo fmt                           # Format (always before committing)
cargo clippy                        # Lint
```

## Architecture

### Module Structure

```
src/
├── core/                           # Reference-agnostic components (SoA-native)
│   ├── alignment/                  # SIMD alignment kernels
│   │   ├── banded_swa/             # Vertical SIMD Smith-Waterman
│   │   ├── kswv_*.rs               # Horizontal SIMD batching
│   │   ├── shared_types.rs         # SoA carriers (SwSoA, KswSoA, AlignJob)
│   │   └── workspace.rs            # Thread-local buffer pools
│   ├── compute/                    # SIMD abstraction (SSE/AVX2/AVX-512/NEON)
│   └── io/                         # SoA-aware I/O (SoAReadBatch, SAM output)
├── pipelines/
│   └── linear/                     # BWA-MEM pipeline (SoA-native)
│       ├── batch_extension/        # Batched extension orchestration
│       ├── index/                  # FM-Index, BWT, suffix array
│       ├── paired/                 # Paired-end processing
│       ├── seeding.rs              # SMEM extraction
│       ├── chaining.rs             # Seed chaining
│       ├── pipeline.rs             # Main alignment entry
│       └── finalization.rs         # CIGAR, MD tags, filtering
└── main.rs                         # CLI entry point
```

### Pipeline Stages

1. **Seeding**: SMEM extraction via FM-Index backward search
2. **Chaining**: Group compatible seeds via DP scoring
3. **Extension**: Batched Smith-Waterman (SoA layout, SIMD kernels)
4. **Finalization**: MD tags, NM computation, MAPQ, filtering

### SIMD Engine Hierarchy

- **128-bit** (baseline): SSE2 (x86_64), NEON (aarch64) - 16 lanes i8
- **256-bit**: AVX2 (x86_64 auto-detected) - 32 lanes i8
- **512-bit**: AVX-512 (feature-gated, nightly) - 64 lanes i8

Runtime detection selects the best available engine.

### Key Data Structures

```rust
// SoA batch for reads
struct SoAReadBatch {
    sequences: Vec<u8>,          // Concatenated
    seq_offsets: Vec<usize>,     // Per-read boundaries
    lengths: Vec<usize>,
}

// SoA carrier for alignment kernels
struct SwSoA<'a> {
    query_soa: &'a [u8],         // Interleaved: pos * lanes + lane
    target_soa: &'a [u8],
    lanes: usize,
}
```

## Development Guidelines

### Code Patterns

**SIMD**: Always use the abstraction layer:
```rust
use crate::compute::simd_abstraction::*;
let sum = _mm_add_epi16(a, b);  // Works on x86 and ARM
```

**SoA Layout**: Hot paths use Structure-of-Arrays for cache efficiency:
```rust
// SoA interleaving: query_soa[pos * LANES + lane]
```

### Making Changes

1. Work in `feature/core-rearch` for SoA-related changes
2. Run `cargo fmt` before each commit
3. Keep files under 500 lines (split into submodules if needed)
4. Run `cargo test` to ensure no regressions
5. For pipeline changes: verify against baseline output

### File Size Targets

Files exceeding 500-line target (pending split):
- `seeding.rs`: 1902 lines
- `finalization.rs`: 1707 lines
- `region.rs`: 1598 lines

## Roadmaps and Design Documents

Detailed design documents are in `documents/`:

| Document | Purpose |
|----------|---------|
| `RedesignStrategy.md` | SIMD kernel unification and module split plan |
| `SOA_End_to_End.md` | End-to-end SoA pipeline design (PR1-PR4 complete) |
| `SOA_Transition.md` | SoA migration checklist and acceptance criteria |
| `ARM_SVE_SME_Roadmap.md` | ARM SVE/SME support (post-1.x) |
| `RISCV_RVV_Roadmap.md` | RISC-V Vector support (experimental, post-1.x) |
| `Metal_GPU_Acceleration_Design.md` | GPU acceleration via Metal (post-1.x) |
| `NPU_Seed_Filter_Design.md` | NPU seed pre-filtering (post-1.x) |
| `Learned_Index_SA_Lookup_Design.md` | Sapling-style SA acceleration (post-1.x) |

## Compatibility

**Index Format**: Matches bwa-mem2 v2.0+ (`.bwt.2bit.64`, `.sa`, `.pac`, `.ann`, `.amb`)

**SAM Output**: Compatible with bwa-mem v0.7.17, GATK ValidateSamFile parity achieved

**Platforms**: macOS (x86_64, arm64), Linux (x86_64, aarch64)

## Known Limitations

### Paired-End Processing

#### ✅ FIXED in v0.7.1: Batch Size Validation
- **Previous**: No validation that R1/R2 files have equal read counts per batch
  - Could produce incorrect results if files were mismatched
  - Silent mis-pairing of reads
- **Current**: Strict validation with clear error messages
  - Detects truncated files immediately
  - Verifies EOF synchronization
  - Fails fast to prevent corrupt output

#### ❌ No Interleaved FASTQ Support (Planned for v0.8.0)
- Single file with alternating R1/R2 reads not supported
- **Workaround**: De-interleave first using `seqtk split` or similar tools:
  ```bash
  # De-interleave a single FASTQ file
  seqtk seq -1 interleaved.fq > R1.fq
  seqtk seq -2 interleaved.fq > R2.fq
  ferrous-align mem ref.fa R1.fq R2.fq > out.sam
  ```
- **Status**: Feature request tracked for v0.8.0

### File Integrity Requirements

**CRITICAL**: Paired-end FASTQ files must have:
1. **Equal read counts** (e.g., if R1 has 1M reads, R2 must have exactly 1M)
2. **Same read order** (R1[i] must be the mate of R2[i])
3. **No truncation** (both files must be complete)

**To verify before running**:
```bash
# Check line counts (should be identical)
wc -l R1.fq R2.fq

# For gzipped files
zcat R1.fq.gz | wc -l
zcat R2.fq.gz | wc -l
```

**Common causes of mismatches**:
- Disk full during sequencing output
- Incomplete file transfer (e.g., scp interrupted)
- Corrupted or truncated downloads
- Mismatched file pairs from different samples

## References

- [bwa-mem2](https://github.com/bwa-mem2/bwa-mem2) - C++ reference implementation
- [bwa](https://github.com/lh3/bwa) - Original algorithm by Heng Li
