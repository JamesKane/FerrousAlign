# Pipeline Restructure v0.8 Plan

## Executive Summary

This document updates the Main Loop Abstraction Proposal based on critical architectural findings discovered during v0.7.0 development. The original proposal assumed a pure SoA (Structure-of-Arrays) pipeline, but practical implementation revealed that **paired-end processing fundamentally requires AoS (Array-of-Structures) for correctness**.

**Key Finding**: Pure SoA pairing causes a 96% duplicate read rate due to index boundary corruption. The solution is a **hybrid AoS/SoA architecture** that preserves SoA's SIMD benefits for compute-heavy stages while using AoS for logic-heavy stages.

**Status**: Planning (updated 2025-12-03)
**Target**: v0.8.0

---

## v0.7.0 Architectural Discoveries

### 1. Hybrid AoS/SoA Architecture is Mandatory

**Discovery**: During paired-end integration, pure SoA pairing produced 96% duplicate reads in output.

**Root Cause**: SoA flattens all alignments into contiguous arrays with offset boundaries. During pairing:
- Each read can have multiple alignments (primary + secondary/supplementary)
- Pairing logic must associate R1[i] alignments with R2[i] alignments
- SoA layout loses per-read boundaries during index calculations
- Results in systematic mis-pairing

**Solution**: Hybrid pipeline with representation transitions:

```
FASTQ → SoA Alignment → AoS Pairing → SoA Mate Rescue → AoS Output → SAM
        [SIMD batching]  [correct]    [SIMD batching]   [correct]
```

**Performance Impact**: ~2% conversion overhead (acceptable for correctness)

### 2. Concordance Gap Requires Investigation

**Current State** (from `dev_notes/SoA_PIPELINE_CONCORDANCE_FINDINGS.md`):
- 81% concordance vs BWA-MEM2 (target: >98%)
- 94.44% properly paired (fixed from 48%)
- 0 supplementary alignments (vs 104 in baseline)

**Suspected Causes** (in priority order):
1. Seeding stage - Different SMEM generation
2. Chaining stage - Different chain selection/tie-breaking
3. Extension stage - Different extension boundaries

**Implication for Refactoring**: The refactoring MUST preserve exact behavior of current stages. Any changes to stage internals must be done AFTER the architectural refactoring, not during.

### 3. Current File Size Violations

| File | Lines | Target | Priority |
|------|-------|--------|----------|
| `seeding.rs` | 1902 | 500 | High |
| `finalization.rs` | 1707 | 500 | High |
| `region.rs` | 1598 | 500 | High |
| `pipeline.rs` | 1247 | 800 | Medium |
| `chaining.rs` | 823 | 500 | Medium |
| `paired_end.rs` | 600 | 400 | High |

---

## Updated Architecture

### Design Principles

1. **Hybrid-First**: Explicitly support AoS↔SoA transitions at stage boundaries
2. **Zero Regression**: Bit-for-bit output match during migration
3. **Concordance Neutral**: Do NOT change algorithm behavior during refactoring
4. **Deferred Optimization**: Concordance debugging happens AFTER structure is clean

### Core Trait Definitions

```rust
/// A pipeline stage that transforms data between representations
pub trait PipelineStage {
    type Input;
    type Output;

    fn process(&self, input: Self::Input, ctx: &StageContext) -> Result<Self::Output, StageError>;
    fn name(&self) -> &str;
}

/// Marker traits for data representation
pub trait SoALayout {}
pub trait AoSLayout {}

/// Conversion traits for hybrid transitions
pub trait IntoSoA<T: SoALayout> {
    fn into_soa(self) -> T;
}

pub trait IntoAoS<T: AoSLayout> {
    fn into_aos(self) -> T;
}
```

### Pipeline Flow with Representation Markers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SINGLE-END PIPELINE                                 │
│                                                                              │
│  [SoA] Loading → [SoA] Seeding → [SoA] Chaining → [SoA] Extension           │
│                                                            ↓                 │
│                                              [SoA] Finalization → [SoA] SAM  │
│                                                                              │
│  All stages use SoA - no conversions needed                                 │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          PAIRED-END PIPELINE                                 │
│                                                                              │
│  [SoA] Loading → [SoA] Seeding → [SoA] Chaining → [SoA] Extension           │
│                                                            ↓                 │
│                                              [SoA] Finalization              │
│                                                            ↓                 │
│                                              [SoA→AoS] CONVERT               │
│                                                            ↓                 │
│                                              [AoS] Pairing                   │
│                                                            ↓                 │
│                                              [AoS→SoA] CONVERT               │
│                                                            ↓                 │
│                                              [SoA] Mate Rescue               │
│                                                            ↓                 │
│                                              [SoA→AoS] CONVERT               │
│                                                            ↓                 │
│                                              [AoS] Output → SAM              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Target Module Structure

```
src/pipelines/linear/
├── orchestrator/                   # Main loop coordination
│   ├── mod.rs                      # PipelineOrchestrator trait + factory
│   ├── single_end.rs               # SE orchestrator (pure SoA)
│   ├── paired_end.rs               # PE orchestrator (hybrid AoS/SoA)
│   ├── statistics.rs               # Stats aggregation
│   └── conversions.rs              # NEW: AoS↔SoA conversion utilities
│
├── stages/                         # Pipeline stages (representation-aware)
│   ├── mod.rs                      # PipelineStage trait + StageContext
│   ├── loading/                    # Stage 0: Read loading
│   │   ├── mod.rs                  # LoadingStage wrapper
│   │   └── soa_reader.rs           # SoA batch loading
│   ├── seeding/                    # Stage 1: SMEM extraction
│   │   ├── mod.rs                  # SeedingStage wrapper (~200 lines)
│   │   ├── smem.rs                 # Core SMEM algorithm (~800 lines)
│   │   ├── reseeding.rs            # Chimeric detection (~300 lines)
│   │   ├── forward_only.rs         # 3rd round seeding (~400 lines)
│   │   └── seed_convert.rs         # SMEM→Seed conversion (~200 lines)
│   ├── chaining/                   # Stage 2: Seed chaining
│   │   ├── mod.rs                  # ChainingStage wrapper (~200 lines)
│   │   ├── dp_chain.rs             # O(n²) DP chaining (~400 lines)
│   │   └── filter.rs               # Chain filtering (~223 lines)
│   ├── extension/                  # Stage 3: SW alignment
│   │   ├── mod.rs                  # ExtensionStage wrapper (~200 lines)
│   │   ├── region.rs               # Region extension (~800 lines)
│   │   └── batch.rs                # Batch orchestration (~600 lines)
│   └── finalization/               # Stage 4: CIGAR/MD/NM
│       ├── mod.rs                  # FinalizationStage wrapper (~200 lines)
│       ├── cigar.rs                # CIGAR generation (~500 lines)
│       ├── md_tag.rs               # MD tag computation (~400 lines)
│       ├── mapq.rs                 # MAPQ calculation (~300 lines)
│       └── flags.rs                # SAM flag handling (~300 lines)
│
├── modes/                          # Mode-specific logic (hybrid-aware)
│   ├── mod.rs                      # Mode selection
│   ├── single_end/                 # SE-specific
│   │   ├── mod.rs
│   │   └── selection.rs            # Primary alignment selection
│   └── paired_end/                 # PE-specific (hybrid AoS/SoA)
│       ├── mod.rs
│       ├── insert_size.rs          # Insert size statistics
│       ├── pairing_aos.rs          # EXISTING: AoS pairing (CORRECT)
│       ├── mate_rescue.rs          # SoA mate rescue wrapper
│       └── conversions.rs          # PE-specific AoS↔SoA helpers
│
├── index/                          # NO CHANGE
├── mem.rs                          # Entry point (updated to use orchestrators)
├── mem_opt.rs                      # NO CHANGE
└── batch_extension/                # EXISTING: Move to stages/extension/
```

---

## Implementation Milestones

### Phase 0: Preparation (1 day)

**Goal**: Establish foundation without changing behavior

**Tasks**:
1. Create directory structure:
   ```bash
   mkdir -p src/pipelines/linear/{orchestrator,stages,modes}
   mkdir -p src/pipelines/linear/stages/{loading,seeding,chaining,extension,finalization}
   mkdir -p src/pipelines/linear/modes/{single_end,paired_end}
   ```

2. Create empty `mod.rs` files with placeholder exports

3. Run `cargo build` to verify no compilation errors

4. Run `cargo test` to establish baseline

**Acceptance Criteria**:
- [ ] All directories created
- [ ] Empty modules compile
- [ ] All existing tests pass
- [ ] No changes to any existing logic

---

### Phase 1: Core Abstractions (2-3 days)

**Goal**: Define traits and shared types WITHOUT connecting to existing code

**Files to Create**:

#### 1.1 `stages/mod.rs` - Stage Trait Definition
```rust
//! Pipeline stage abstraction layer
//!
//! This module defines the `PipelineStage` trait that all pipeline stages implement.
//! Stages transform data from one type to another, with explicit support for
//! both SoA (Structure-of-Arrays) and AoS (Array-of-Structures) layouts.

use crate::pipelines::linear::index::BwaIndex;
use crate::pipelines::linear::mem_opt::MemOpt;
use crate::core::compute::ComputeContext;
use thiserror::Error;

/// A pipeline stage that transforms input data to output data
pub trait PipelineStage {
    type Input;
    type Output;

    /// Process a batch of data through this stage
    fn process(&self, input: Self::Input, ctx: &StageContext) -> Result<Self::Output, StageError>;

    /// Human-readable stage name (for logging/profiling)
    fn name(&self) -> &str;

    /// Optional: Validate input before processing
    fn validate(&self, _input: &Self::Input) -> Result<(), StageError> {
        Ok(())
    }
}

/// Context passed to each stage (immutable during batch processing)
pub struct StageContext<'a> {
    pub index: &'a BwaIndex,
    pub options: &'a MemOpt,
    pub compute_ctx: &'a ComputeContext,
    pub batch_id: u64,
}

/// Errors that can occur during stage processing
#[derive(Debug, Error)]
pub enum StageError {
    #[error("Seeding failed: {0}")]
    Seeding(String),

    #[error("Chaining failed: {0}")]
    Chaining(String),

    #[error("Extension failed: {0}")]
    Extension(String),

    #[error("Finalization failed: {0}")]
    Finalization(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Empty batch")]
    EmptyBatch,
}

// Submodule declarations (empty stubs initially)
pub mod loading;
pub mod seeding;
pub mod chaining;
pub mod extension;
pub mod finalization;
```

#### 1.2 `orchestrator/mod.rs` - Orchestrator Trait Definition
```rust
//! Pipeline orchestration layer
//!
//! Orchestrators coordinate the execution of pipeline stages, handling:
//! - Batch loading and iteration
//! - Stage sequencing
//! - AoS/SoA representation transitions (for paired-end)
//! - Statistics aggregation
//! - Output writing

use std::io::Write;
use std::path::PathBuf;
use thiserror::Error;

/// High-level pipeline orchestrator
pub trait PipelineOrchestrator {
    /// Run the complete pipeline on input files
    fn run(
        &mut self,
        input_files: &[PathBuf],
        output: &mut dyn Write,
    ) -> Result<PipelineStatistics, OrchestratorError>;

    /// Get the pipeline mode
    fn mode(&self) -> PipelineMode;
}

/// Pipeline execution mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineMode {
    SingleEnd,
    PairedEnd,
}

/// Aggregate statistics from pipeline execution
#[derive(Debug, Clone, Default)]
pub struct PipelineStatistics {
    pub total_reads: usize,
    pub total_bases: usize,
    pub total_alignments: usize,
    pub batches_processed: usize,
    pub wall_time_secs: f64,
}

impl PipelineStatistics {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Errors that can occur during orchestration
#[derive(Debug, Error)]
pub enum OrchestratorError {
    #[error("Stage error: {0}")]
    Stage(#[from] super::stages::StageError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

pub mod single_end;
pub mod paired_end;
pub mod statistics;
pub mod conversions;
```

#### 1.3 `orchestrator/conversions.rs` - AoS/SoA Conversion Utilities
```rust
//! AoS/SoA conversion utilities for hybrid pipeline
//!
//! The paired-end pipeline requires transitioning between data representations:
//! - SoA for compute-heavy stages (SIMD alignment, mate rescue)
//! - AoS for logic-heavy stages (pairing, output)
//!
//! These conversions are explicitly modeled as pipeline operations.

use crate::pipelines::linear::finalization::Alignment;
use crate::core::io::sam_output::SoAAlignmentResult;

/// Convert SoA alignment results to AoS for pairing
///
/// This is required because pairing logic needs per-read indexing that
/// SoA's flattened structure cannot provide correctly.
pub fn soa_to_aos_for_pairing(soa: &SoAAlignmentResult) -> Vec<Vec<Alignment>> {
    // Extract per-read alignments from SoA structure
    // Returns Vec<Vec<Alignment>> where outer vec is per-read, inner vec is per-alignment
    todo!("Wrap existing conversion logic")
}

/// Convert AoS alignments back to SoA for mate rescue
///
/// After pairing is complete, we convert back to SoA to leverage
/// SIMD batching for the compute-heavy mate rescue stage.
pub fn aos_to_soa_for_rescue(aos: &[Vec<Alignment>]) -> SoAAlignmentResult {
    todo!("Wrap existing conversion logic")
}

/// Convert SoA to AoS for final SAM output
///
/// SAM output requires per-alignment iteration which works better with AoS.
pub fn soa_to_aos_for_output(soa: &SoAAlignmentResult) -> Vec<Alignment> {
    todo!("Wrap existing conversion logic")
}
```

**Acceptance Criteria**:
- [ ] All trait definitions compile
- [ ] No changes to existing code paths
- [ ] Unit tests for trait object safety
- [ ] Documentation complete

---

### Phase 2: Seeding Stage Split (2-3 days)

**Goal**: Split `seeding.rs` (1902 lines) into coherent submodules

**Strategy**: WRAPPER-FIRST approach
1. Create new files that re-export existing functions
2. Gradually move code to new locations
3. Keep old `seeding.rs` as a re-export facade during transition

**Target Structure**:
```
stages/seeding/
├── mod.rs              # SeedingStage impl + re-exports (~200 lines)
├── smem.rs             # SMEM algorithm (~800 lines)
├── reseeding.rs        # Re-seeding logic (~300 lines)
├── forward_only.rs     # 3rd round seeding (~400 lines)
└── seed_convert.rs     # SMEM→Seed conversion (~200 lines)
```

**Migration Steps**:

1. **Create `smem.rs`**: Move SMEM generation functions
   - `generate_smems_for_strand()`
   - `generate_smems_from_position()`
   - `backward_search()`
   - Helper functions

2. **Create `reseeding.rs`**: Move re-seeding logic
   - Re-seeding loop (current lines ~140-192)
   - `collect_reseed_candidates()`
   - `compute_split_len()`

3. **Create `forward_only.rs`**: Move forward-only seeding
   - `forward_only_seed_strategy()`
   - Related helpers

4. **Create `seed_convert.rs`**: Move SMEM→Seed conversion
   - `get_sa_entries()`
   - SMEM to Seed conversion loop

5. **Create stage wrapper `mod.rs`**:
```rust
pub struct SeedingStage;

impl PipelineStage for SeedingStage {
    type Input = SoAReadBatch;
    type Output = SoASeedBatch;

    fn process(&self, batch: Self::Input, ctx: &StageContext) -> Result<Self::Output, StageError> {
        // Delegate to existing find_seeds_batch
        super::super::seeding::find_seeds_batch(ctx.index, &batch, ctx.options)
            .map_err(|e| StageError::Seeding(e.to_string()))
    }

    fn name(&self) -> &str {
        "Seeding"
    }
}
```

6. **Update old `seeding.rs`**: Facade that re-exports from new locations

**Acceptance Criteria**:
- [ ] All tests pass after split
- [ ] No behavioral changes (verified by golden dataset)
- [ ] Each file < 500 lines
- [ ] `SeedingStage` compiles and works

---

### Phase 3: Finalization Stage Split (2-3 days)

**Goal**: Split `finalization.rs` (1707 lines) into coherent submodules

**Target Structure**:
```
stages/finalization/
├── mod.rs              # FinalizationStage impl (~200 lines)
├── cigar.rs            # CIGAR generation (~500 lines)
├── md_tag.rs           # MD tag computation (~400 lines)
├── mapq.rs             # MAPQ calculation (~300 lines)
└── flags.rs            # SAM flag handling (~300 lines)
```

**Migration Steps**:

1. **Create `cigar.rs`**: Move CIGAR generation
   - `generate_cigar_from_region()`
   - `traceback_dp_matrix()`
   - `compress_cigar()`
   - CIGAR operation helpers

2. **Create `md_tag.rs`**: Move MD tag computation
   - `compute_md_tag()`
   - Reference extraction helpers

3. **Create `mapq.rs`**: Move MAPQ calculation
   - `calculate_mapq()`
   - Phred score helpers

4. **Create `flags.rs`**: Move SAM flag handling
   - `mark_secondary_alignments()`
   - `sam_flags` constants/module
   - `compute_overlap()`

5. **Create stage wrapper**

**Acceptance Criteria**:
- [ ] All tests pass after split
- [ ] No behavioral changes
- [ ] Each file < 500 lines
- [ ] `FinalizationStage` compiles and works

---

### Phase 4: Extension and Chaining Splits (2-3 days)

**Goal**: Split `region.rs` (1598 lines) and `chaining.rs` (823 lines)

#### 4.1 Extension Stage
```
stages/extension/
├── mod.rs              # ExtensionStage impl (~200 lines)
├── region.rs           # Region extension logic (~800 lines)
└── batch.rs            # Batch orchestration (~600 lines)
```

#### 4.2 Chaining Stage
```
stages/chaining/
├── mod.rs              # ChainingStage impl (~200 lines)
├── dp_chain.rs         # O(n²) DP chaining (~400 lines)
└── filter.rs           # Chain filtering (~223 lines)
```

**Acceptance Criteria**:
- [ ] All tests pass
- [ ] No behavioral changes
- [ ] Each file < 500 lines
- [ ] Both stage wrappers work

---

### Phase 5: Orchestrator Implementation (3-4 days)

**Goal**: Implement SingleEndOrchestrator and PairedEndOrchestrator

#### 5.1 Single-End Orchestrator
```rust
// orchestrator/single_end.rs
pub struct SingleEndOrchestrator<'a> {
    ctx: StageContext<'a>,
    seeder: SeedingStage,
    chainer: ChainingStage,
    extender: ExtensionStage,
    finalizer: FinalizationStage,
    stats: PipelineStatistics,
}

impl PipelineOrchestrator for SingleEndOrchestrator<'_> {
    fn run(&mut self, files: &[PathBuf], output: &mut dyn Write) -> Result<PipelineStatistics, OrchestratorError> {
        for file in files {
            let mut reader = SoaFastqReader::new(file)?;

            while let Some(batch) = reader.next_batch()? {
                // Pure SoA pipeline - no conversions
                let seeds = self.seeder.process(batch.clone(), &self.ctx)?;
                let chains = self.chainer.process(seeds, &self.ctx)?;
                let regions = self.extender.process(chains, &self.ctx)?;
                let alignments = self.finalizer.process(regions, &self.ctx)?;

                write_sam_records_soa(&alignments, output, self.ctx.options)?;
                self.stats.update(&alignments);
            }
        }

        Ok(self.stats.clone())
    }

    fn mode(&self) -> PipelineMode {
        PipelineMode::SingleEnd
    }
}
```

#### 5.2 Paired-End Orchestrator (Hybrid)
```rust
// orchestrator/paired_end.rs
pub struct PairedEndOrchestrator<'a> {
    ctx: StageContext<'a>,
    // Same stages as single-end
    seeder: SeedingStage,
    chainer: ChainingStage,
    extender: ExtensionStage,
    finalizer: FinalizationStage,
    // Paired-end specific (hybrid AoS/SoA)
    insert_stats: Option<InsertSizeStats>,
    stats: PipelineStatistics,
}

impl PipelineOrchestrator for PairedEndOrchestrator<'_> {
    fn run(&mut self, files: &[PathBuf], output: &mut dyn Write) -> Result<PipelineStatistics, OrchestratorError> {
        assert_eq!(files.len(), 2, "Paired-end requires exactly 2 files");

        let mut r1 = SoaFastqReader::new(&files[0])?;
        let mut r2 = SoaFastqReader::new(&files[1])?;

        // Bootstrap insert size
        self.bootstrap_insert_size(&mut r1, &mut r2)?;

        // Main processing loop
        while let Some((batch1, batch2)) = load_paired_batch(&mut r1, &mut r2)? {
            // === SoA STAGES ===
            let seeds1 = self.seeder.process(batch1, &self.ctx)?;
            let seeds2 = self.seeder.process(batch2, &self.ctx)?;
            let chains1 = self.chainer.process(seeds1, &self.ctx)?;
            let chains2 = self.chainer.process(seeds2, &self.ctx)?;
            let regions1 = self.extender.process(chains1, &self.ctx)?;
            let regions2 = self.extender.process(chains2, &self.ctx)?;
            let alignments1 = self.finalizer.process(regions1, &self.ctx)?;
            let alignments2 = self.finalizer.process(regions2, &self.ctx)?;

            // === SoA → AoS CONVERSION (for pairing) ===
            let aos_pairs = conversions::soa_to_aos_for_pairing(&alignments1, &alignments2);

            // === AoS STAGE: Pairing (MUST be AoS for correctness) ===
            let paired = pair_alignments_aos(&aos_pairs, self.insert_stats.as_ref())?;

            // === AoS → SoA CONVERSION (for mate rescue) ===
            let soa_for_rescue = conversions::aos_to_soa_for_rescue(&paired);

            // === SoA STAGE: Mate Rescue ===
            let rescued = mate_rescue_soa(&soa_for_rescue, &self.ctx)?;

            // === SoA → AoS CONVERSION (for output) ===
            let aos_output = conversions::soa_to_aos_for_output(&rescued);

            // === AoS OUTPUT ===
            write_sam_records_aos(&aos_output, output, self.ctx.options)?;

            self.stats.update(&rescued);
        }

        Ok(self.stats.clone())
    }

    fn mode(&self) -> PipelineMode {
        PipelineMode::PairedEnd
    }
}
```

**Acceptance Criteria**:
- [ ] Both orchestrators compile
- [ ] Integration tests pass
- [ ] Output matches existing pipeline byte-for-byte
- [ ] Hybrid transitions are explicit and documented

---

### Phase 6: Integration (2 days)

**Goal**: Wire orchestrators into `mem.rs` entry point

#### 6.1 Update `mem.rs`
```rust
// Before:
if opts.reads.len() == 2 {
    process_paired_end(&bwa_idx, ...);
} else {
    process_single_end(&bwa_idx, ...);
}

// After:
use orchestrator::{PipelineOrchestrator, SingleEndOrchestrator, PairedEndOrchestrator};

let stats = if opts.reads.len() == 2 {
    let mut orch = PairedEndOrchestrator::new(&bwa_idx, &opt, &compute_ctx);
    orch.run(&opts.reads, &mut writer)?
} else {
    let mut orch = SingleEndOrchestrator::new(&bwa_idx, &opt, &compute_ctx);
    orch.run(&opts.reads, &mut writer)?
};

log::info!("Processed {} reads in {:.2}s", stats.total_reads, stats.wall_time_secs);
```

#### 6.2 Deprecate Old Entry Points
```rust
#[deprecated(since = "0.8.0", note = "Use SingleEndOrchestrator instead")]
pub fn process_single_end(...) {
    let mut orch = SingleEndOrchestrator::new(...);
    orch.run(...).expect("Pipeline failed");
}
```

**Acceptance Criteria**:
- [ ] `cargo build --release` succeeds
- [ ] All tests pass
- [ ] CLI works identically to before
- [ ] Deprecation warnings appear for old API

---

### Phase 7: Validation (2-3 days)

**Goal**: Verify refactoring has zero behavioral impact

#### 7.1 Golden Dataset Testing
```bash
# Build baseline (current v0.7.0)
git stash
cargo build --release
./target/release/ferrous-align mem -t 16 $REF R1.fq R2.fq > baseline.sam

# Build refactored
git stash pop
cargo build --release
./target/release/ferrous-align mem -t 16 $REF R1.fq R2.fq > refactored.sam

# Compare
diff baseline.sam refactored.sam  # MUST be empty
```

#### 7.2 Performance Regression Testing
```bash
# 10K reads - no regression allowed
hyperfine --warmup 3 --runs 10 \
    'v0.7.0/ferrous-align mem $REF 10k_R1.fq 10k_R2.fq' \
    'v0.8.0/ferrous-align mem $REF 10k_R1.fq 10k_R2.fq'

# 4M reads - ±5% acceptable
/usr/bin/time -v ./target/release/ferrous-align mem -t 16 $REF 4M_R1.fq 4M_R2.fq > /dev/null
```

#### 7.3 Memory Profiling
```bash
# Check for leaks or increased memory usage
valgrind --tool=massif ./target/release/ferrous-align mem $REF R1.fq R2.fq > /dev/null
```

**Acceptance Criteria**:
- [ ] Bit-for-bit identical SAM output
- [ ] No performance regression >5%
- [ ] No memory usage increase >10%
- [ ] GATK ValidateSamFile passes

---

## Risk Assessment

### Blocking Issues

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Behavioral regression | Low | Critical | Golden dataset testing at each phase |
| Performance regression | Medium | High | Benchmark after each phase |
| Compilation errors during split | Medium | Low | Incremental moves with builds |
| Concordance changes | Low | Critical | Do NOT change algorithm logic |

### Rollback Triggers

1. SAM output differs from baseline
2. Performance regression >10%
3. Memory usage increase >20%
4. Test failures that can't be fixed in 2 days

### Rollback Plan

```bash
# If critical issues found:
git checkout v0.7.0
# Analyze failure
# Adjust plan
# Re-attempt in next cycle
```

---

## Post-Refactoring Priorities

### Immediate (v0.8.x)
1. **Concordance Debugging**: Investigate 81% → target 98%
   - Now easier to debug individual stages
   - Can add stage-level logging without cluttering monolithic files

### Future (v0.9.0+)
2. **GPU Extension Backend**: Swap `ExtensionStage` implementation
3. **Stage-Level Profiling**: Add timing decorators
4. **Custom Pipelines**: Fast mode (fewer SMEM passes)

---

## Summary

The v0.7.0 findings fundamentally change the Main Loop Abstraction approach:

| Original Proposal | Updated Plan |
|-------------------|--------------|
| Pure SoA pipeline | Hybrid AoS/SoA with explicit transitions |
| All stages same layout | Layout-appropriate per stage |
| Simple stage chaining | Conversion stages at boundaries |
| Implicit representations | Marker traits for compile-time safety |

**Total Timeline**: 14-18 working days (~3 weeks)

**Critical Success Factor**: Zero behavioral regression. The refactoring is purely structural - concordance debugging is a separate, subsequent effort.

---

## References

- `Main_Loop_Abstraction_Proposal.md` - Original design (superseded by this document)
- `Module_Reorganization_Plan.md` - Detailed file splitting plan
- `SOA_End_to_End.md` - SoA architecture with hybrid discovery
- `dev_notes/SoA_PIPELINE_CONCORDANCE_FINDINGS.md` - Concordance issues
- C++ bwa-mem2: `src/fastmap.cpp` - Reference implementation

---

*Document Version: 1.0*
*Last Updated: 2025-12-03*
*Author: Claude Code + Human Review*
