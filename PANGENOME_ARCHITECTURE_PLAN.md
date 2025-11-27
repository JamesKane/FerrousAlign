# Pangenome Aligner (v3.x) Architecture Plan

## 1. Executive Summary

This document outlines the architectural strategy for incorporating a pangenome alignment variant into the FerrousAlign pipeline. The goal is to support emerging graph-based reference models while maximizing the reuse of existing, battle-tested code.

The analysis concludes that this is a feasible and highly beneficial evolution for the software. A significant portion of the existing codebase, including the most computationally intensive components, can be reused. The core, SIMD-accelerated alignment kernels are agnostic to the reference structure and represent a major reusable asset.

The primary development effort will be concentrated on replacing the linear reference model (BWT/FM-Index) with a new graph-based framework. This involves implementing new modules for graph indexing, seed discovery (seeding), and seed grouping (chaining) that can operate on a graph topology.

## 2. Proposed Architecture Revision

The transition to a dual linear/graph alignment architecture can be achieved by abstracting the key stages of the alignment pipeline. The existing high-performance components will be preserved while new, graph-specific modules are introduced.

### 2.1. Reusable Components (Little to No Modification)

These modules are considered stable and can be consumed directly by the new pangenome pipeline.

*   **`src/io` (`fastq_reader.rs`, `sam_output.rs`):** Standardized FASTQ/SAM I/O handling is fully reusable.
*   **`src/compute/simd_abstraction`:** The hardware-agnostic SIMD engine is fundamental for performance and is directly applicable to sequence operations in a graph context.
*   **Core Alignment Kernels (`src/alignment/banded_swa*.rs`, `ksw*.rs`):** The core Smith-Waterman-Gotoh (SWG) alignment functions are the computational heart of the aligner. They operate on sequence pairs and are fully reusable for aligning reads to paths extracted from the graph.
*   **Alignment Utilities (`src/alignment/cigar.rs`, `edit_distance.rs`):** Generic utilities for creating CIGAR strings and calculating edit distance remain essential.

### 2.2. Components to be Adapted (Requires Refactoring)

These modules contain conceptually reusable logic but are currently implemented with assumptions about a linear reference. They must be refactored to support both linear and graph coordinate systems.

*   **`src/alignment/paired`:** The high-level logic for paired-end rescue and insert size estimation is valid. However, all code implementing distance calculations and region definition must be abstracted to handle graph-based metrics (e.g., path distance) instead of simple integer offsets.
*   **`src/alignment/pipeline.rs`, `src/alignment/finalization.rs`:** The overall pipeline structure and finalization steps will serve as a template. They will need to be generalized to accommodate different data structures from the new chaining and alignment stages.

### 2.3. Components to be Replaced (Requires New Implementation)

These modules are fundamentally tied to the BWA-MEM algorithm and its linear reference model. They will be preserved for the existing pipeline but bypassed by a new set of modules for pangenome alignment.

*   **`src/index` (BWT/FM-Index):** **To be replaced** with a dedicated graph indexing solution capable of handling structures like GFA. This new index will be responsible for providing the data needed for graph-based seeding.
*   **`src/alignment/seeding.rs`:** **To be replaced** with a new seeding module (e.g., using minimizers or strobemers) that can efficiently query the graph index to find seed locations across multiple paths.
*   **`src/alignment/chaining.rs`:** **To be replaced** with a graph-aware chaining algorithm. This is a critical component that must correctly score and group seeds across the complex topology of a pangenome graph (e.g., using a colinear chaining on a DAG algorithm).
*   **BWA-MEM Specifics (`src/alignment/mem.rs`, `coordinates.rs`, `region.rs`):** These files are specific to the linear alignment workflow and will not be used in the pangenome pipeline.

## 3. Proposed File Organization

To support multiple alignment pipelines (the existing linear aligner and the new pangenome graph aligner) concurrently, the `src` directory should be reorganized. This change isolates pipeline-specific logic while promoting the sharing of common, high-performance components.

The proposed structure introduces a `core` module for reusable logic and a `pipelines` module containing implementations for each alignment strategy.

```
src/
├── core/
│   ├── alignment/          # Reusable core alignment kernels
│   │   ├── banded_swa.rs
│   │   ├── ksw_affine_gap.rs
│   │   └── ... (other swa*.rs and ksw*.rs files)
│   ├── compute/            # Reusable computation logic
│   │   └── simd_abstraction/ # The entire SIMD engine module
│   ├── io/                 # Reusable I/O (FASTQ/SAM)
│   │   ├── fastq_reader.rs
│   │   └── sam_output.rs
│   └── utils/              # Project-wide utilities
│       ├── cigar.rs
│       └── edit_distance.rs
│
├── pipelines/
│   ├── linear/             # Existing linear alignment pipeline (BWA-MEM)
│   │   ├── index/            # BWT/FM-Index specific to linear alignment
│   │   ├── seeding.rs
│   │   ├── chaining.rs
│   │   ├── coordinates.rs
│   │   └── mem.rs
│   │
│   └── graph/              # New pangenome graph alignment pipeline
│       ├── index/            # New graph index implementation (e.g., GFA-based)
│       ├── seeding.rs        # New graph seeding implementation
│       ├── chaining.rs       # New graph chaining implementation
│       └── path_traverse.rs  # Example of graph-specific logic
│
├── lib.rs                  # Main library entry point
└── main.rs                 # Binary entry point
```

**Rationale:**

*   **Isolation:** This structure clearly separates the concerns of the linear and graph aligners. A developer working on the graph pipeline can do so without modifying the production linear aligner.
*   **Reusability:** All genuinely reusable components are promoted to the `src/core` module, making them easily accessible to any pipeline and preventing code duplication.
*   **Extensibility:** This layout makes it straightforward to add other alignment pipelines in the future (e.g., a `long_read` pipeline) by simply adding a new subdirectory to `src/pipelines`.
