# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Overview

**FerrousAlign** (`ferrous-align`) is a Rust port of [bwa-mem2](https://github.com/bwa-mem2/bwa-mem2), a high-performance DNA sequence aligner using the Burrows-Wheeler Transform. Targets performance parity with C++ (1.3-3.1x faster than original BWA-MEM) with Rust's safety benefits.

**Critical**: The C++ bwa-mem2 behavior and file formats are the technical specification. Any deviation is a critical bug.

**Status**: v0.8.0-alpha (feature/pipeline-structure) - Stage-based pipeline architecture complete, optimization work in progress.

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
│   └── linear/                     # BWA-MEM pipeline (stage-based architecture)
│       ├── orchestrator/           # Pipeline coordination (NEW in v0.8.0)
│       │   ├── single_end.rs       # Pure SoA orchestrator (314 lines)
│       │   ├── paired_end/         # Hybrid AoS/SoA orchestrator (423+242 lines)
│       │   └── conversions.rs      # AoS↔SoA transitions (137 lines)
│       ├── stages/                 # Stage wrappers (NEW in v0.8.0)
│       │   ├── loading/            # Stage 0: Read loading
│       │   ├── seeding/            # Stage 1: SMEM extraction (190 lines wrapper)
│       │   ├── chaining/           # Stage 2: Seed chaining (218 lines wrapper)
│       │   ├── extension/          # Stage 3: SW alignment (276 lines wrapper)
│       │   └── finalization/       # Stage 4: CIGAR/MD/NM (324 lines wrapper)
│       ├── batch_extension/        # Batched extension implementation
│       ├── index/                  # FM-Index, BWT, suffix array
│       ├── paired/                 # Paired-end logic (pairing, mate rescue)
│       ├── seeding/                # SMEM implementation (1929 lines - legacy)
│       ├── chaining/               # Chaining implementation (1116 lines - legacy)
│       ├── region/                 # Extension implementation (1598 lines - legacy)
│       ├── finalization/           # Finalization implementation (1704 lines - legacy)
│       └── mem.rs                  # CLI entry point (uses orchestrators)
└── main.rs                         # Main entry point
```

### Pipeline Stages

The v0.8.0 refactor introduced a stage-based architecture with explicit orchestration:

**Single-End Pipeline** (Pure SoA):
1. **Loading**: Read FASTQ into SoA batches
2. **Seeding**: SMEM extraction via FM-Index backward search
3. **Chaining**: Group compatible seeds via DP scoring
4. **Extension**: Batched Smith-Waterman (SoA layout, SIMD kernels)
5. **Finalization**: CIGAR, MD tags, NM computation, MAPQ, filtering
6. **Output**: Write SAM records directly from SoA

**Paired-End Pipeline** (Hybrid AoS/SoA):
1. **Loading**: Read FASTQ R1/R2 into SoA batches
2. **Seeding**: SMEM extraction (SoA) for both R1 and R2
3. **Chaining**: Seed chaining (SoA) for both R1 and R2
4. **Extension**: SW alignment (SoA) for both R1 and R2
5. **Finalization**: CIGAR/MD/NM (SoA) for both R1 and R2
6. **SoA→AoS**: Convert to AoS for pairing logic
7. **Pairing**: Associate R1/R2 alignments (AoS - required for correctness)
8. **AoS→SoA**: Convert back to SoA for mate rescue
9. **Mate Rescue**: Rescue unmapped mates using SIMD (SoA)
10. **SoA→AoS**: Convert to AoS for output
11. **Output**: Write SAM records with proper pairing flags

**Key Insight**: Pure SoA pairing caused 96% duplicate reads in v0.7.0. The hybrid architecture
is mandatory for correctness - pairing logic requires per-read indexing that SoA cannot provide.

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

1. Work in `feature/pipeline-structure` for v0.8.0 optimization work
2. Run `cargo fmt` before each commit
3. Keep NEW files under 500 lines (enforced for merge gating)
4. Run `cargo test` to ensure no regressions
5. For pipeline changes: verify against baseline output (10K HG002 dataset)
6. For performance changes: benchmark before/after with `hyperfine`

### File Size Targets

**v0.8.0 Achievement**: All NEW modules under 500 lines ✅
- All `orchestrator/` modules: 137-423 lines each
- All `stages/` modules: 190-324 lines each
- Total: ~1,223 lines of dead code eliminated

**Legacy files** (wrapped by stages, full split deferred):
- `seeding/` modules: ~1929 lines total (wrapped by `stages/seeding/`)
- `finalization/` modules: ~1704 lines total (wrapped by `stages/finalization/`)
- `region/` modules: ~1598 lines total (wrapped by `stages/extension/`)
- `chaining/` modules: ~1116 lines total (wrapped by `stages/chaining/`)

**Note**: The wrapper approach creates thin stage modules that delegate to existing
implementations. Full file splitting of legacy code is deferred to minimize risk.

## Roadmaps and Design Documents

Detailed design documents are in `documents/`:

| Document | Purpose | Status |
|----------|---------|--------|
| `v0.8.0_Completion_Plan.md` | Plan for completing v0.8.0 (pairing, perf, memory, threading) | 📋 Current |
| `Pipeline_Restructure_v0.8_Plan.md` | Stage-based architecture implementation | ✅ Complete |
| `SOA_End_to_End.md` | End-to-end SoA pipeline design with hybrid discovery | ✅ Complete |
| `SOA_Transition.md` | SoA migration checklist and acceptance criteria | ✅ Complete |
| `RedesignStrategy.md` | SIMD kernel unification and module split plan | 📚 Reference |
| `ARM_SVE_SME_Roadmap.md` | ARM SVE/SME support (post-1.x) | 🔮 Future |
| `RISCV_RVV_Roadmap.md` | RISC-V Vector support (experimental, post-1.x) | 🔮 Future |
| `Metal_GPU_Acceleration_Design.md` | GPU acceleration via Metal (post-1.x) | 🔮 Future |
| `NPU_Seed_Filter_Design.md` | NPU seed pre-filtering (post-1.x) | 🔮 Future |
| `Learned_Index_SA_Lookup_Design.md` | Sapling-style SA acceleration (post-1.x) | 🔮 Future |

## v0.8.0 Architecture Improvements

### Stage-Based Pipeline (Completed)

The v0.8.0 refactor introduced a clean separation between pipeline coordination (orchestrators)
and algorithm implementation (stages):

**Benefits**:
- **Modularity**: Each stage is independently testable and swappable
- **Maintainability**: All new modules under 500 lines (merge gating requirement)
- **Debuggability**: Stage boundaries provide natural instrumentation points
- **Performance**: 15% improvement vs v0.7.0 from reduced overhead

**Architecture**:
```rust
// PipelineStage trait - all stages implement this
pub trait PipelineStage {
    type Input;
    type Output;
    fn process(&self, input: Self::Input, ctx: &StageContext) -> Result<Self::Output>;
    fn name(&self) -> &str;
}

// Orchestrators coordinate stage execution
pub trait PipelineOrchestrator {
    fn run(&mut self, files: &[PathBuf], output: &mut dyn Write) -> Result<PipelineStatistics>;
}
```

**Hybrid AoS/SoA Discovery**:
- Pure SoA pairing caused 96% duplicate reads (v0.7.0 bug)
- Solution: Explicit representation transitions at stage boundaries
- SoA for compute-heavy stages (SIMD alignment, mate rescue)
- AoS for logic-heavy stages (pairing, output)
- ~2% conversion overhead (acceptable for correctness)

### Remaining v0.8.0 Work

See `documents/v0.8.0_Completion_Plan.md` for detailed implementation plan:

1. **Pairing Accuracy** - Close 3pp gap (94.14% → 97%+)
2. **Performance** - Reach 85-90% of BWA-MEM2 throughput (currently ~79%)
3. **Memory** - Reduce peak usage (~32 GB → 24 GB)
4. **Threading** - Improve core utilization

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

#### ❌ No Interleaved FASTQ Support (Deferred to v0.9.0)
- Single file with alternating R1/R2 reads not supported
- **Workaround**: De-interleave first using `seqtk split` or similar tools:
  ```bash
  # De-interleave a single FASTQ file
  seqtk seq -1 interleaved.fq > R1.fq
  seqtk seq -2 interleaved.fq > R2.fq
  ferrous-align mem ref.fa R1.fq R2.fq > out.sam
  ```
- **Status**: Feature request deferred to v0.9.0 (v0.8.0 focuses on performance/accuracy)

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
