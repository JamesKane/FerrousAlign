# FerrousAlign

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)

A Rust implementation of [bwa-mem2](https://github.com/bwa-mem2/bwa-mem2), a fast and accurate aligner for DNA sequencing reads.

## ⚠️ Experimental Software - Not Production Ready

**WARNING: This is pre-1.0 experimental software under active development.**

- **Do NOT use in production environments** or for clinical/diagnostic purposes
- **Alignment algorithms may contain bugs** and produce incorrect results
- **Output has not been fully validated** against bwa-mem2 reference implementation
- **API and command-line interface may change** without notice
- **Intended for research, development, and testing purposes only**

For production workloads, please use the stable [bwa-mem2](https://github.com/bwa-mem2/bwa-mem2) C++ implementation until FerrousAlign reaches v1.0.

## Overview

**FerrousAlign** (`ferrous-align`) is a high-performance reimplementation of BWA-MEM2 in Rust, targeting performance parity with the C++ version while providing memory safety and modern language features.

### What Works

**Current Version**: v0.5.0

- ✅ **Index Building**: Create BWA-MEM2 compatible indices from FASTA files
- ✅ **Single-End Alignment**: Align single-end reads to reference genomes
- ✅ **Paired-End Alignment**: Align paired-end reads with insert size inference
- ✅ **Multi-Threading**: Parallel processing using all CPU cores
- ✅ **Gzip Support**: Read compressed FASTQ files (.fq.gz) natively
- ✅ **SAM Output**: Standard SAM format with complete headers
- ✅ **Platform Support**: macOS (Intel/Apple Silicon), Linux (x86_64/ARM64)

### What's Missing

- ⚠️ **Index Compatibility**: Can build indices but **NOT YET VALIDATED** for production use
- ⚠️ **Algorithm Refinements**: Some advanced features partially implemented (re-seeding, chain dropping)
- ⚠️ **Performance**: Currently 85-95% of C++ bwa-mem2 speed
- ⚠️ **Validation**: Output not fully tested against real-world datasets
- 🔴 **CRITICAL: Memory Scaling Issue**: Paired-end alignment requires **ALL reads in memory** before output
  - **4M read pairs = 20.7 GB RAM**
  - **30x WGS (450M pairs) = 2.3 TB RAM** (impractical for most systems)
  - **Use only for small-scale experiments** until streaming architecture is implemented (planned for v0.6.0)
  - See [Known Limitations](#known-limitations) below for details

## Installation

### Prerequisites

- Rust 2024 edition or later (install from [rustup.rs](https://rustup.rs))

### Building from Source

```bash
# Clone the repository
git clone https://github.com/JamesKane/ferrous-align.git
cd ferrous-align

# Build in release mode (required for good performance)
cargo build --release

# The binary will be at ./target/release/ferrous-align
./target/release/ferrous-align --help
```

### Performance Optimization

For best performance, build with native CPU optimizations:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

## Quick Start

### 1. Index a Reference Genome

```bash
# Create an index from a FASTA file
./target/release/ferrous-align index reference.fa

# This creates several index files:
# - reference.fa.bwt.2bit.64 (BWT index)
# - reference.fa.pac (packed reference)
# - reference.fa.ann (annotations)
# - reference.fa.amb (ambiguous bases)
```

**Memory requirement**: ~28× the reference genome size (e.g., 84 GB for human genome)

### 2. Align Reads

**Single-End:**
```bash
./target/release/ferrous-align mem reference.fa reads.fq > output.sam
```

**Paired-End:**
```bash
./target/release/ferrous-align mem reference.fa read1.fq read2.fq > output.sam
```

**With Gzip Compression:**
```bash
./target/release/ferrous-align mem reference.fa reads.fq.gz > output.sam
```

**Multi-Threading:**
```bash
# Use 8 threads
./target/release/ferrous-align mem -t 8 reference.fa reads.fq > output.sam

# Use all available CPU cores (default)
./target/release/ferrous-align mem reference.fa reads.fq > output.sam
```

### 3. Common Options

```
-t INT    Number of threads [default: all cores]
-o FILE   Output SAM file [default: stdout]
-k INT    Minimum seed length [19]
-w INT    Band width for alignment [100]
-r FLOAT  Re-seed trigger [1.5]
-c INT    Max seed occurrences [500]
-A INT    Match score [1]
-B INT    Mismatch penalty [4]
-O INT    Gap open penalty [6]
-E INT    Gap extension penalty [1]
-v INT    Verbosity level [3]
          1=quiet, 2=warnings, 3=info, 4+=debug
```

Run `./target/release/ferrous-align mem --help` for all options.

## Example Workflow

```bash
# 1. Download a reference genome (e.g., E. coli)
wget https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2_genomic.fna.gz
gunzip GCF_000005845.2_ASM584v2_genomic.fna.gz

# 2. Build index
./target/release/ferrous-align index GCF_000005845.2_ASM584v2_genomic.fna

# 3. Align reads (example with paired-end reads)
./target/release/ferrous-align mem \
    -t 8 \
    GCF_000005845.2_ASM584v2_genomic.fna \
    reads_R1.fq.gz \
    reads_R2.fq.gz \
    > alignments.sam

# 4. Convert to BAM and sort (requires samtools)
samtools view -bS alignments.sam | samtools sort -o alignments.sorted.bam
samtools index alignments.sorted.bam
```

## Performance

**Threading:**
- Automatically uses all CPU cores
- Thread count validated: minimum 1, maximum 2× CPU count
- Efficient work-stealing parallelism via Rayon

**SIMD Acceleration:**
- Automatic CPU feature detection
- x86_64: SSE4.1 (baseline), AVX2 (automatic)
- ARM64: NEON intrinsics (Apple Silicon native)

**Typical Performance:**
- Small genomes (< 10 Mb): Near-instant indexing, ~1-10K reads/sec alignment
- Bacterial genomes (~5 Mb): ~1 minute indexing, ~5-20K reads/sec alignment
- Human genome (~3 Gb): ~45 minutes indexing, ~1-5K reads/sec alignment
- Currently 85-95% of C++ bwa-mem2 speed

## Compatibility

### Input Formats
- ✅ FASTA reference genomes (`.fa`, `.fasta`, `.fna`)
- ✅ FASTQ reads (`.fq`, `.fastq`)
- ✅ Gzip-compressed FASTQ (`.fq.gz`, `.fastq.gz`)

### Output Formats
- ✅ SAM format (compatible with SAMtools, Picard, GATK, etc.)
- ❌ BAM/CRAM direct output (use `samtools view` to convert)

### Index Format
- ✅ Matches bwa-mem2 v2.0+ format
- ⚠️ Indices built by FerrousAlign are **EXPERIMENTAL** - not validated for production
- ✅ Can read indices created by C++ bwa-mem2

### Platform Support
- ✅ macOS (Intel x86_64, Apple Silicon ARM64)
- ✅ Linux (x86_64, ARM64/aarch64)
- ⚠️ Windows (untested, may work with minor adjustments)

## Known Limitations

### 🔴 CRITICAL: Memory Scaling for Paired-End Alignment

**Issue**: The current paired-end implementation stores ALL alignments in memory before writing output.

**Impact**:
- **Small datasets (< 1M pairs):** Works fine, memory usage acceptable
- **Medium datasets (1-10M pairs):** Requires 5-50 GB RAM, may cause swapping
- **Large datasets (> 10M pairs):** Requires 50+ GB RAM
- **30x WGS (450M pairs):** Requires **2.3 TB RAM** - **WILL NOT RUN** on typical servers

**Memory scaling**: ~5.2 bytes per read, or ~10.4 bytes per paired-end read pair

**Examples**:
```
Dataset Size          Memory Required
────────────────────────────────────
100K pairs            ~1 GB
1M pairs              ~10 GB
4M pairs              ~20.7 GB
30M pairs             ~156 GB
450M pairs (30x WGS)  ~2.3 TB  ❌ IMPRACTICAL
```

**Why this happens**:
1. All read pairs are aligned and stored in memory
2. Insert size statistics are calculated from all pairs
3. Mate rescue is performed on all pairs
4. Finally, all results are written to SAM output
5. No incremental output until processing completes

**Workarounds**:
- ✅ Use single-end mode (`mem` with one FASTQ file) - no memory issue
- ✅ Split large datasets into smaller chunks (< 1M pairs each)
- ✅ Use C++ bwa-mem2 for large-scale WGS until this is fixed

**Status**:
- Design for streaming architecture complete (see `dev_notes/fix-read-flushing.md`)
- Fix planned for **v0.6.0** (estimated 4-5 hours implementation)
- Will reduce memory to **constant ~200 MB** regardless of dataset size

**DO NOT ATTEMPT** to run whole-genome sequencing datasets (> 10M pairs) until this is resolved.

---

## Troubleshooting

### Index Building Fails with "Out of Memory"
- Indexing requires ~28× reference size in RAM
- For human genome: need ~85 GB RAM
- Solution: Use C++ bwa-mem2 to build index, then use with FerrousAlign

### Alignment is Slow
- Make sure you built with `--release` flag
- Use `-t` to specify thread count: `-t 8` for 8 threads
- Build with native optimizations: `RUSTFLAGS="-C target-cpu=native" cargo build --release`

### "Thread count validation" Warnings
- Thread counts < 1 are set to 1 automatically
- Thread counts > 2× CPU cores are capped automatically
- This is normal and safe

### Paired-End Alignment Runs Out of Memory or is Killed
- See [Known Limitations](#known-limitations) - paired-end mode requires all reads in memory
- **4M pairs = 20.7 GB, 30x WGS = 2.3 TB** (impractical)
- Solutions:
  - Use single-end mode (no memory issue)
  - Split dataset into smaller chunks (< 1M pairs)
  - Use C++ bwa-mem2 for large datasets
  - Wait for v0.6.0 streaming architecture fix

### SAM Output Shows Only Headers for Minutes
- This is expected with current architecture
- All alignments are processed in memory, then written at once
- No incremental output until processing completes
- Will be fixed in v0.6.0 with streaming architecture

### SAM Output Looks Wrong
- **This is experimental software** - output may not be production-quality
- Compare against C++ bwa-mem2 output before using results
- Report issues on GitHub with sample data

## Getting Help

- **Documentation**: See [CLAUDE.md](CLAUDE.md) for developer documentation
- **Issues**: Report bugs at [GitHub Issues](https://github.com/JamesKane/ferrous-align/issues)
- **Performance**: See [PERFORMANCE.md](PERFORMANCE.md) for benchmarking details

## Development

For developers interested in contributing or understanding the internals:

- **Developer Guide**: See [CLAUDE.md](CLAUDE.md) for architecture, code patterns, and testing
- **Code Style**: Run `cargo fmt` before committing
- **Testing**: Run `cargo test` for unit/integration tests
- **Benchmarks**: Run `cargo bench` for performance benchmarks

## Project Status

### Recent Progress (November 17, 2025)

**Critical Bugs Fixed:**
- ✅ **Catastrophic PAC file I/O bug** - was loading 740 MB file 2,934× per batch (2.1 TB I/O)
  - Fix: Load PAC once, pass as parameter → **780x speedup** for mate rescue
  - Validation: 4M pairs now complete in 4m31s (was hanging indefinitely)
- ✅ **SIMD routing disabled** - all jobs using scalar instead of AVX2
  - Fix: Remove flawed length-based heuristic → **100% SIMD utilization**
  - Impact: Unlocks 2-3x AVX2 speedup potential

**Major Fixes (Previous Sessions):**
- ✅ Fixed SMEM (Supermaximal Exact Match) generation algorithm
- ✅ Fixed index building to match C++ bwa-mem2 format
- ✅ Fixed ambiguous base handling in reference genomes
- ✅ Fixed BWT interval calculations
- ✅ Fixed suffix array reconstruction

**Performance Improvements:**
- ✅ Vector capacity pre-allocation - 3.1x parallelism improvement
- ✅ BGZIP parallel decompression - 5x I/O improvement
- ✅ Adaptive batch sizing for alignment jobs

**Critical Issue Discovered:**
- 🔴 **Memory scaling architecture** - paired-end requires ALL reads in memory
  - 4M pairs = 20.7 GB, 30x WGS = 2.3 TB (impractical)
  - Design for streaming fix complete, implementation planned for v0.6.0
  - See [Known Limitations](#known-limitations) for details

### Roadmap

**v0.6.0** (Next Release - High Priority)
- [ ] **🔴 CRITICAL: Streaming architecture for paired-end alignment** (fixes 2.3 TB memory issue)
  - Memory usage: 20.7 GB → 200 MB constant
  - Enables 30x WGS on typical servers
  - Incremental SAM output for progress monitoring
- [ ] Full algorithm refinement implementation (re-seeding, chain dropping)
- [ ] Comprehensive validation against C++ bwa-mem2 on real datasets

**v0.7.0 - v0.8.0** (3-6 months)
- [ ] Performance optimization to match or exceed C++ version
- [ ] Threading optimization for better core utilization
- [ ] Improved error handling and diagnostics

**v0.9.0 - v1.0.0** (6-12 months)
- [ ] 100% feature parity with C++ bwa-mem2
- [ ] Production-ready index building
- [ ] BAM/CRAM output support
- [ ] Extensive real-world validation
- [ ] Performance matching C++ bwa-mem2

**v2.0.0+** (Long-term)
- [ ] GPU acceleration (Metal/CUDA/ROCm)
- [ ] Advanced alignment algorithms
- [ ] Cloud-native optimizations

## License

MIT License - see [LICENSE.md](LICENSE.md)

## Citation

This Rust implementation is based on the bwa-mem2 algorithm. If you use FerrousAlign, please cite the original paper:

Vasimuddin Md, Sanchit Misra, Heng Li, Srinivas Aluru.
**Efficient Architecture-Aware Acceleration of BWA-MEM for Multicore Systems.**
*IEEE Parallel and Distributed Processing Symposium (IPDPS), 2019.*
[doi:10.1109/IPDPS.2019.00041](https://doi.org/10.1109/IPDPS.2019.00041)

## Acknowledgments

Based on:
- **bwa-mem2** by Vasimuddin Md, Sanchit Misra, and contributors at Intel Parallel Computing Lab
- **bwa** (original algorithm) by Heng Li (@lh3)

---

**Remember**: This is experimental software. Always validate results against the reference C++ implementation before using in any important analysis.
