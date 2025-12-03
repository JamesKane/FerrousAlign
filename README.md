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

**Current Version**: v0.7.0 - SoA Architecture & Performance Improvements!

- ✅ **Index Building**: Create BWA-MEM2 compatible indices from FASTA files
- ✅ **Single-End Alignment**: Align single-end reads to reference genomes (100% SoA pipeline)
- ✅ **Paired-End Alignment**: Align paired-end reads with hybrid AoS/SoA architecture
- ✅ **Multi-Threading**: Parallel processing using all CPU cores
- ✅ **Gzip Support**: Read compressed FASTQ files (.fq.gz) natively
- ✅ **SAM Output**: Standard SAM format with complete headers
- ✅ **Platform Support**: macOS (Intel/Apple Silicon), Linux (x86_64/ARM64)
- ✅ **GATK4 Compatibility**: Near parity with BWA-MEM2 (94.14% properly paired vs 97.11%)

### What's Missing

- ⚠️ **Index Compatibility**: Can build indices but **NOT YET VALIDATED** for production use
- ⚠️ **Algorithm Refinements**: Some advanced features partially implemented (re-seeding, chain dropping)
- ⚠️ **Pairing Accuracy**: 94.14% properly paired vs BWA-MEM2's 97.11% (3pp gap, acceptable for alpha)
- ⚠️ **Validation**: Output validated on HG002 10K Golden dataset but needs broader testing

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

### SIMD Acceleration

The project includes a portable SIMD backend with automatic CPU feature detection:

- **x86_64**: SSE4.1 (baseline), AVX2 (automatic), AVX-512 (opt-in via `--features avx512`)
- **ARM64/Apple Silicon**: NEON (128-bit) - fully tested and enabled by default

You can control SIMD usage via an environment variable at runtime:

```bash
# Disable SIMD explicitly (use scalar fallback)
FERROUS_ALIGN_SIMD=0 ./target/release/ferrous-align mem ...

# Enable SIMD explicitly (default behavior)
FERROUS_ALIGN_SIMD=1 ./target/release/ferrous-align mem ...
```

**Apple Silicon Performance (M3 Max, 16 threads)**:
- 100K reads: 5.6s alignment, 36K reads/sec throughput
- 10K reads: 0.95s alignment, 21K reads/sec throughput
- NEON achieves higher throughput than x86 AVX2 due to efficient Apple Silicon microarchitecture

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
- x86_64: SSE4.1 (baseline), AVX2 (automatic), AVX-512 (feature-gated)
- ARM64: NEON intrinsics - fully tested on Apple Silicon (M3 Max)

**Typical Performance:**
- Small genomes (< 10 Mb): Near-instant indexing, ~1-10K reads/sec alignment
- Bacterial genomes (~5 Mb): ~1 minute indexing, ~5-20K reads/sec alignment
- Human genome (~3 Gb): ~45 minutes indexing, ~1-5K reads/sec alignment
- x86_64 (Ryzen 9 7900X): ~79% of BWA-MEM2 speed on 4M read pairs
- Apple Silicon (M3 Max): 36K reads/sec on 100K HG002 pairs (no BWA-MEM2 comparison available)

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

### ⚠️ Paired-End Pairing Accuracy (v0.7.0)

**Status**: Near parity with BWA-MEM2, acceptable for alpha release

**v0.7.0 Benchmark Results (10K HG002 Golden Reads)**:
| Metric | BWA-MEM2 | FerrousAlign | Delta | Status |
|--------|----------|--------------|-------|--------|
| Properly paired | 97.11% | 94.14% | -2.97pp | ⚠️ ACCEPTABLE |
| Mapping rate | 99.50% | 98.66% | -0.84pp | ✅ OK |
| Mate diff chr | 1.51% | 1.90% | +0.39pp | ✅ OK |
| Singletons | 0.30% | 1.05% | +0.75pp | ⚠️ MINOR |
| Duplicate reads | 0% | 0% | 0 | ✅ FIXED |

**Architecture**: Hybrid AoS/SoA required for correctness
- Pure SoA pairing caused 96% duplicate reads (critical bug, now fixed)
- Hybrid approach: SoA for alignment/rescue, AoS for pairing/output
- 3pp gap in proper pairing acceptable for alpha; defer to v0.8.0

**Compatibility**:
- ✅ **SAM Format**: All required tags (AS, XS, NM, MD, XA, MC)
- ✅ **Zero Duplicates**: Hybrid architecture eliminates duplicate bug
- ⚠️ **GATK Validation**: Minor differences from BWA-MEM2 (acceptable for research)

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

### Recent Progress (December 3, 2025)

**v0.7.0 - SoA Architecture Complete!**
- ✅ **End-to-End SoA Pipeline** - Zero AoS conversions for single-end reads
  - SoA-aware I/O layer (SoaFastqReader)
  - SoA seeding, chaining, extension, and output
  - Performance improvements from reduced allocations
- ✅ **Hybrid AoS/SoA for Paired-End** - Critical bug fix
  - Pure SoA pairing caused 96% duplicate reads (now fixed)
  - Hybrid: SoA for alignment/rescue, AoS for pairing/output
  - 94.14% properly paired rate (3pp gap acceptable for alpha)
- ✅ **Zero Duplicate Reads** - Correctness fix for paired-end output
  - Fixed indexing bug in SoA pairing logic
  - All reads appear exactly once in output
- ✅ **Architecture Documentation** - Comprehensive design docs
  - Hybrid architecture discovery documented
  - Pipeline flow diagrams updated
  - Future refactoring plans account for hybrid requirements

**v0.6.0 - GATK Parity (Previous Release)**
- ✅ GATK ValidateSamFile parity with BWA-MEM2
- ✅ All required SAM tags (AS, XS, NM, MD, XA, MC)
- ✅ Comprehensive bounds checking (254 tests passing)
- ✅ BaseRecalibrator, HaplotypeCaller, MarkDuplicates ready

**Performance Profile:**
- Single-end: 100% SoA pipeline, minimal overhead
- Paired-end: ~2% conversion overhead (hybrid AoS/SoA)
- Memory usage: Streaming architecture, bounded allocations
- Threading: Full CPU utilization via Rayon work-stealing

### Roadmap

**v0.7.0** (Current Release) ✅ COMPLETE
- [✅] **End-to-End SoA Pipeline** - Zero AoS conversions for single-end reads
- [✅] **Hybrid AoS/SoA for Paired-End** - Critical bug fix (96% duplicates → 0%)
- [✅] **Zero Duplicate Reads** - Correctness fix for paired-end output
- [✅] **Architecture Documentation** - Comprehensive design docs for hybrid approach
- [✅] **Performance Improvement** - ~79% of BWA-MEM2 speed on 4M HG002 (up from v0.6.0)

**v0.8.0** (Next Release)
- [ ] **Pairing Accuracy** - Close 3pp gap (94.14% → 97%+ properly paired)
- [ ] **Performance optimization** - Target 85-90% of BWA-MEM2 throughput
- [ ] **Memory optimization** - Reduce peak usage (~32 GB → 24 GB target)
- [ ] **Threading optimization** - Better core utilization

**v0.9.0**
- [ ] Algorithm refinements (re-seeding, chain dropping)
- [ ] Production-ready index building validation
- [ ] Broader dataset validation beyond HG002

**v1.0.0** (Production Ready)
- [ ] 100% feature parity with C++ bwa-mem2
- [ ] BAM/CRAM output support
- [ ] Performance matching C++ bwa-mem2
- [ ] Extensive real-world validation

**v2.0.0+** (Long-term)
- [ ] GPU acceleration (Metal/CUDA/ROCm)
- [ ] Advanced alignment algorithms

**v3.0.0** (~1 year out)
- [ ] Pangenome graph alignment - align reads to reference graphs (GFA format)
- [ ] Support for population-level variation (SNPs, indels, structural variants)
- [ ] Graph index construction from VCF + reference FASTA
- [ ] Compatible with minigraph/vg pangenome workflows

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
