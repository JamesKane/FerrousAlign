# Session 37: NEON Performance Profiling Plan

**Date**: 2025-11-28

## 1. Summary of Findings

Following the implementation of optimized NEON versions of `movemask_epi8` and `shuffle_epi8` in `src/core/compute/simd_abstraction/engine128.rs`, end-to-end benchmarks were conducted using a larger dataset (`test_data/HG002_local`).

The results showed a **~3.36%** improvement in throughput, which, while positive, is not the significant performance gain that was expected.

*   **Unoptimized Throughput:** ~275,232 reads/sec
*   **Optimized Throughput:** ~284,495 reads/sec

This suggests that the `movemask_epi8` and `shuffle_epi8` functions, while inefficient in their original form, are not the primary performance bottlenecks in the overall alignment process. The bottleneck likely lies elsewhere in the `banded_swa` alignment kernel.

A panic that was occurring with sequence lengths > 128bp was also fixed as part of this investigation.

## 2. Next Steps: Detailed Profiling with Instruments

To identify the true performance hotspots, a more detailed CPU profile is required. The recommended approach is to use Apple's **Instruments** `Time Profiler`.

### 2.1. Profiling Instructions

1.  **Open Instruments**: Launch from `/Applications/Xcode.app/Contents/Applications/Instruments.app`.
2.  **Choose Template**: Select the "Time Profiler" template.
3.  **Configure Target**:
    *   In the profiling window, click the "Target" dropdown and select "Choose Target...".
    *   Navigate to the `target/release/ferrous-align` executable in the project directory.
4.  **Set Arguments**:
    *   In the "Arguments" field, enter:
        ```
        mem -t 16 test_data/chrM.fna /Users/jkane/RustroverProjects/FerrousAlign/test_data/HG002_local/read1.fastq.gz /Users/jkane/RustroverProjects/FerrousAlign/test_data/HG002_local/read2.fastq.gz > /dev/null
        ```
5.  **Set Environment Variables**:
    *   Add a new environment variable:
        *   **Name**: `FERROUS_ALIGN_FORCE_SSE`
        *   **Value**: `1`
6.  **Record**: Click the red "Record" button to start profiling.
7.  **Save**: Once the process finishes, save the trace document (`.trace` file).

### 2.2. Analysis

The saved trace file should be analyzed to identify the functions within the `ferrous_align::core::alignment::banded_swa` and `ferrous_align::core::compute` modules that have the highest CPU usage. This will guide the next round of optimizations.
