# v0.8.0 Baseline Performance Summary

**Date**: December 6, 2025
**Branch**: `feature/pipeline-structure`
**Commit**: `3768049` (feat: Parallelize full pipeline per chunk with AoS merging)
**Dataset**: HG002 10K paired-end reads (20K total reads, 2.96 Mbases)
**Reference**: CHM13v2.0 (human genome, ~6.2 Gb)
**Hardware**: 16-thread system

---

## Performance Baseline (10K Reads)

### Timing Results

| Run | Wall Time | Throughput (reads/sec) | Throughput (Mbases/sec) |
|-----|-----------|------------------------|-------------------------|
| Run 1 | 0.83s | 24,152 | 3.57 |

**Internal metrics** (from logs):
- Application reports: 24,152 reads/sec, 3.57 Mbases/sec (0.83s wall time)
- Pipeline CPU time: 8.08s
- CPU utilization: ~970% (parallel processing working!)

### Comparison to Previous v0.8.0 Baseline

| Metric | Old v0.8.0 | New (Parallelized) | Change |
|--------|------------|-------------------|---------|
| Wall time (10K reads) | ~1.1s | 0.83s | **-25%** |
| Throughput | ~18K reads/sec | 24K reads/sec | **+33%** |
| CPU utilization | 206% (~2 cores) | ~970% (~10 cores) | **+4.7x** |

---

## Accuracy Baseline

### Flagstat Comparison (10K Paired-End Reads)

| Metric | BWA-MEM2 | FerrousAlign | Delta |
|--------|----------|--------------|-------|
| Total records | 20,069 | 20,031 | -38 |
| Primary reads | 20,000 | 19,998 | -2 |
| Supplementary | 69 | 33 | -36 |
| Mapped | 99.49% | 99.53% | +0.04pp |
| Primary mapped | 99.49% | 99.53% | +0.04pp |
| **Properly paired** | **98.10%** | **95.14%** | **-2.96pp** |
| With mate mapped | 19,838 | 19,838 | 0 |
| Singletons | 0.30% | 0.33% | +0.03pp |
| Diff chr | 164 | 434 | +270 |

### Concordance Analysis

| Metric | Value |
|--------|-------|
| Total reads compared | 19,998 |
| Concordant | 18,770 (93.86%) |
| Discordant | 1,228 (6.14%) |
| Missing in test | 2 |

**Status**: 93.86% concordance with BWA-MEM2

---

## Memory Baseline

### Peak Memory Usage

From `/usr/bin/time -v`:
- **Maximum resident set size**: ~20.7 GB (10K reads)
- Previous baseline (100K): 23.3 GB

---

## Key Changes Since Last Baseline

### Parallelization Improvements (December 6, 2025)

1. **Full Pipeline Per-Chunk**: Moved finalization into `process_single_chunk()` so entire pipeline runs in parallel
2. **AoS Merging**: Trivially correct R1/R2 index synchronization using simple extend()
3. **Parallel Secondary Marking**: Used `par_iter_mut()` for secondary alignment marking

### Commits

1. `feat: Parallelize full pipeline per chunk with AoS merging`
2. `fix: Convert qlen/tlen to i16 for SoAInputs16 in benchmarks`
3. `fix: Update integration test for filter_chains_batch signature change`

---

## v0.8.0 Status

### Completed

- [x] Stage-based pipeline architecture
- [x] Pipeline parallelization (finalization in parallel chunks)
- [x] AoS/SoA hybrid for correctness
- [x] All tests passing (239 lib tests)

### Remaining

- [ ] Pairing accuracy gap (93.86% → 97%+ target)
- [ ] Performance tuning for larger datasets

---

## Files in This Directory

- `bwa_mem2.sam` - BWA-MEM2 output for golden 10K reads
- `ferrous.sam` - FerrousAlign output for golden 10K reads
- `bwa_flagstat.txt` - BWA-MEM2 flagstat
- `ferrous_flagstat.txt` - FerrousAlign flagstat

---

**Document Version**: 2.0
**Updated**: December 6, 2025
**Author**: Claude Code
