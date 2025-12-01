# End-to-End Structure-of-Arrays (SoA) Pipeline

## Objective
This document outlines the architectural changes required to make the FerrousAlign pipeline a zero-overhead, end-to-end Structure-of-Arrays (SoA) data processing system. The goal is to stream data from disk directly into SoA buffers, process it through all pipeline stages in SoA format, and write the final output from SoA data, eliminating all AoS-to-SoA conversion overhead.

## Current State
The alignment pipeline has been successfully refactored to use SoA as its internal data representation for the core alignment kernels (`banded_swa` and `kswv`). However, the data is still read from disk and processed in the initial pipeline stages (I/O, seeding, chaining) in an Array-of-Structures (AoS) format. The conversion to SoA happens just before the alignment kernels are called, which introduces a performance overhead.

## Proposed End-to-End SoA Architecture
The new architecture will be SoA-native from end to end. This will involve changes in the I/O, seeding, chaining, and output layers of the pipeline.

### 1. I/O Layer: SoA-aware Readers
The current I/O layer reads data from FASTQ/BAM files into collections of records (structs), which is an AoS representation. We will introduce new readers that parse these files directly into SoA buffers.

- **`SoAReadBatch`**: A new struct will be defined to hold a batch of reads in SoA format.
  ```rust
  struct SoAReadBatch {
      // Sequence data for all reads in the batch, stored contiguously.
      seqs: Vec<u8>,
      // Quality scores for all reads.
      quals: Vec<u8>,
      // Read names.
      names: Vec<String>,
      // Offsets and lengths to delineate individual reads within the contiguous buffers.
      read_boundaries: Vec<(usize, usize)>, // (start_offset, length)
  }
  ```
- **Custom Parsers**: We will implement new parsers in `src/core/io` for FASTQ and BAM formats that populate the `SoAReadBatch` directly. This will avoid the creation of intermediate `Read` objects for each read.

### 2. Seeding and Chaining on SoA Data
The seeding and chaining stages will be refactored to work directly with the `SoAReadBatch`.

- **Seeding (`find_seeds`)**: The `find_seeds` function will be modified to take `SoAReadBatch` as input. The SMEM generation process will be adapted to work with the SoA data layout.
- **Chaining (`build_and_filter_chains`)**: The chaining algorithm will also be updated to consume seeds generated from the SoA data, producing chains of SoA-native seeds.

### 3. Alignment Kernels
The core alignment kernels (`banded_swa` and `kswv`) are already SoA-native. They will now receive their data directly from the SoA-native chaining stage, without any conversion overhead.

### 4. Output Layer: SoA-aware Writers
The final stage of the pipeline is to write the alignments to a SAM/BAM file. The current implementation converts the results back to an AoS format (a list of `Alignment` structs) before writing. We will implement a new SoA-aware writer.

- **`SoAAlignmentResult`**: A new result struct will be created to hold the alignment results in SoA format.
- **SAM/BAM Writer**: A new writer in `src/core/io/sam_output.rs` will be implemented to take `SoAAlignmentResult` and write the SAM/BAM records directly, avoiding the final conversion back to AoS.

## Phased Implementation Plan

### PR1: SoA-aware I/O Layer
- **Objective**: Read FASTQ/BAM files directly into SoA buffers.
- **Tasks**:
  - Implement the `SoAReadBatch` struct.
  - Create new FASTQ and BAM readers in `src/core/io` that populate `SoAReadBatch`.
  - Add unit tests for the new readers to ensure correctness.

### PR2: Update Seeding and Chaining to Consume SoA
- **Objective**: Adapt the seeding and chaining stages to work with SoA data.
- **Tasks**:
  - Refactor `find_seeds` to accept `SoAReadBatch` and generate seeds in an SoA-friendly format.
  - Refactor `build_and_filter_chains` to consume SoA seeds.
  - Update integration tests to verify the correctness of the SoA-native seeding and chaining.

### PR3: End-to-End SoA Integration (up to alignment)
- **Objective**: Connect the SoA-native I/O, seeding, and chaining stages with the existing SoA alignment kernels.
- **Tasks**:
  - Modify the main pipeline function (`process_batch_cross_read`) to use the new SoA-native components.
  - Remove the AoS-to-SoA conversion layer before the alignment kernels.
  - Ensure that data flows from disk to the alignment kernels in SoA format without intermediate AoS representation.

### PR4: SoA-aware Output Layer
- **Objective**: Write SAM/BAM records directly from SoA alignment results.
- **Tasks**:
  - Define the `SoAAlignmentResult` struct.
  - Implement a new SAM/BAM writer in `src/core/io/sam_output.rs` that consumes `SoAAlignmentResult`.
  - Update the main pipeline to use the new SoA-aware writer.

### PR5: Final Cleanup and Performance Validation
- **Objective**: Remove all remaining legacy AoS code paths and validate the performance of the end-to-end SoA pipeline.
- **Tasks**:
  - Remove the old AoS-based I/O, seeding, and chaining functions.
  - Run benchmarks to compare the performance of the new end-to-end SoA pipeline with the previous version.
  - Update `PERFORMANCE.md` with the new benchmark results.
  - Ensure all tests pass and the `size_guard.sh` script is satisfied.

## Risks and Mitigations

- **Complexity of I/O**: Writing high-performance, SoA-aware parsers for FASTQ and BAM can be complex.
  - **Mitigation**: We can start with a simpler, correct implementation and optimize it in later stages. We can also investigate existing libraries that might support SoA-style reading.

- **Performance Regressions**: While the goal is to improve performance, the refactoring could introduce unexpected bottlenecks.
  - **Mitigation**: We will benchmark each PR to closely monitor performance and identify any regressions early. The existing benchmark suite will be adapted to test the new SoA paths.

- **Data Locality**: While SoA is generally better for SIMD, some pipeline stages might benefit from the data locality of AoS.
  - **Mitigation**: We will analyze the performance of each stage and, if necessary, consider hybrid AoS/SoA approaches for specific parts of the pipeline where it makes sense. However, the default will be a pure SoA approach.
