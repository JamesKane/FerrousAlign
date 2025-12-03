# Main Loop Abstraction Proposal

## Executive Summary

This document proposes a refactoring of FerrousAlign's pipeline into a clean, modular architecture with clear separation of concerns. The goal is to replace the current sprawling `process_single_end()` and `process_paired_end()` functions with a unified orchestrator that coordinates well-defined stages.

**Status**: Proposal (v0.8.0 refactoring target)

**⚠️ Important**: The hybrid AoS/SoA architecture discovery (v0.7.0) impacts this proposal. Paired-end pairing **requires** AoS for correctness, while compute-heavy stages benefit from SoA. The orchestrator must support seamless AoS↔SoA conversions. See `SOA_End_to_End.md` for details.

---

## Problems with Current Architecture

### 1. Code Duplication
- `process_single_end()` and `process_paired_end()` duplicate:
  - Batch loading logic
  - Parallel chunking
  - Statistics tracking
  - Error handling
- Changes must be applied to both paths

### 2. Mixed Concerns
- I/O, computation, and output logic interleaved
- Difficult to test individual stages in isolation
- Hard to swap implementations (e.g., GPU extension)

### 3. Large Functions
- `process_single_end()`: ~400 lines (src/pipelines/linear/single_end.rs)
- `process_paired_end()`: ~600 lines (src/pipelines/linear/paired/paired_end.rs)
- Violates single responsibility principle

### 4. Hidden Dependencies
- Implicit state flow between stages
- Hard to trace data transformations
- Difficult to optimize individual stages

---

## Proposed Architecture

### High-Level Design

```
┌──────────────────────────────────────────────────────────────────────┐
│                          Orchestrator                                 │
│                   (Main Loop Abstraction)                             │
│                                                                        │
│  Responsibilities:                                                    │
│    • Load batches from disk                                           │
│    • Dispatch to stage pipeline                                       │
│    • Aggregate statistics                                             │
│    • Handle errors and logging                                        │
│    • Write output                                                     │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        Stage Pipeline                                 │
│                    (Composable Stages)                                │
│                                                                        │
│  Stage 0: Loading   → SoAReadBatch                                    │
│  Stage 1: Seeding   → SoASeedBatch                                    │
│  Stage 2: Chaining  → SoAChainBatch                                   │
│  Stage 3: Extension → SoAAlignmentRegions                             │
│  Stage 4: Finalization → SoAAlignmentResult                           │
└──────────────────────────────────────────────────────────────────────┘
```

### Key Principles

1. **Single Responsibility**: Each module does ONE thing well
2. **Explicit Data Flow**: Clear input/output types for each stage
3. **Testability**: Mock any stage for unit testing
4. **Extensibility**: Easy to add GPU/NPU backends
5. **Performance**: Zero-cost abstractions (trait dispatch only at batch boundaries)

---

## Module Structure

### Proposed File Layout

```
src/pipelines/linear/
├── mod.rs                      # Public API
├── orchestrator/               # NEW: Main loop coordination
│   ├── mod.rs                  # Orchestrator trait + implementation
│   ├── single_end.rs           # Single-end orchestrator
│   ├── paired_end.rs           # Paired-end orchestrator
│   └── statistics.rs           # Stats tracking
│
├── stages/                     # NEW: Pipeline stages (refactored from existing)
│   ├── mod.rs                  # Stage trait definition
│   ├── loading.rs              # Stage 0: Read loading (wraps soa_readers)
│   ├── seeding.rs              # Stage 1: SMEM extraction (refactored)
│   ├── chaining.rs             # Stage 2: Seed chaining (refactored)
│   ├── extension.rs            # Stage 3: SW alignment (refactored)
│   └── finalization.rs         # Stage 4: CIGAR/MD/NM (refactored)
│
├── modes/                      # Mode-specific logic
│   ├── mod.rs
│   ├── single_end.rs           # Single-end selection (from sam_output)
│   └── paired_end.rs           # Paired-end pairing logic (from paired/)
│
├── index/                      # Existing (no change)
├── mem_opt.rs                  # Existing (no change)
├── mem.rs                      # Existing (minor refactor for orchestrator)
│
└── batch_extension/            # Existing (no structural change)
    ├── mod.rs
    └── ...
```

---

## Core Abstractions

### 1. Stage Trait

```rust
/// A pipeline stage that transforms data from type `In` to type `Out`
pub trait PipelineStage<In, Out> {
    /// Process a batch through this stage
    fn process(&self, input: In, ctx: &StageContext) -> Result<Out, StageError>;

    /// Stage name (for logging)
    fn name(&self) -> &str;

    /// Optional: validate input before processing
    fn validate(&self, input: &In) -> Result<(), StageError> {
        Ok(())
    }
}

/// Context passed to each stage (read-only)
pub struct StageContext<'a> {
    pub index: &'a BwaIndex,
    pub options: &'a MemOpt,
    pub compute_ctx: &'a ComputeContext,
    pub batch_id: u64,
}

/// Stage-level errors
#[derive(Debug, thiserror::Error)]
pub enum StageError {
    #[error("Seeding failed: {0}")]
    SeedingError(String),

    #[error("Extension failed: {0}")]
    ExtensionError(String),

    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    // ... other variants
}
```

### 2. Orchestrator Trait

```rust
/// High-level orchestrator for the alignment pipeline
pub trait PipelineOrchestrator {
    /// Run the full pipeline on input files
    fn run(
        &mut self,
        input_files: &[PathBuf],
        output: &mut dyn Write,
    ) -> Result<PipelineStatistics, OrchestratorError>;

    /// Get pipeline mode (single-end, paired-end)
    fn mode(&self) -> PipelineMode;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineMode {
    SingleEnd,
    PairedEnd,
}

/// Aggregate pipeline statistics
pub struct PipelineStatistics {
    pub total_reads: usize,
    pub total_bases: usize,
    pub total_alignments: usize,
    pub wall_time_secs: f64,
    pub cpu_time_secs: f64,
    pub throughput_mbases_per_sec: f64,
}
```

### 3. Batch Processor

```rust
/// Generic batch processor with parallel chunking
pub struct ParallelBatchProcessor<In, Out> {
    num_threads: usize,
    chunk_strategy: ChunkStrategy,
}

#[derive(Debug, Clone, Copy)]
pub enum ChunkStrategy {
    /// Fixed chunk size
    FixedSize(usize),

    /// Divide evenly among threads
    DivideEvenly,

    /// Dynamic work stealing
    Dynamic,
}

impl<In, Out> ParallelBatchProcessor<In, Out>
where
    In: Sync + Send + Sliceable,
    Out: Send + Mergeable,
{
    /// Process a batch using parallel chunks
    pub fn process_parallel<F>(
        &self,
        batch: &In,
        process_fn: F,
    ) -> Out
    where
        F: Fn(&In::Slice) -> Out + Sync + Send,
    {
        // Implementation: split, rayon::par_iter, merge
        todo!()
    }
}

/// Types that can be sliced for parallel processing
pub trait Sliceable {
    type Slice;
    fn slice(&self, start: usize, end: usize) -> Self::Slice;
    fn len(&self) -> usize;
}

/// Types that can merge results from parallel chunks
pub trait Mergeable {
    fn merge_all(chunks: Vec<Self>) -> Self;
}
```

---

## Concrete Implementations

### Single-End Orchestrator

```rust
pub struct SingleEndOrchestrator<'a> {
    index: &'a BwaIndex,
    options: &'a MemOpt,
    compute_ctx: &'a ComputeContext,

    // Stage implementations
    loader: LoadingStage,
    seeder: SeedingStage,
    chainer: ChainingStage,
    extender: ExtensionStage,
    finalizer: FinalizationStage,

    // Output selection
    selector: SingleEndSelector,

    // Statistics
    stats: PipelineStatistics,
}

impl<'a> SingleEndOrchestrator<'a> {
    pub fn new(
        index: &'a BwaIndex,
        options: &'a MemOpt,
        compute_ctx: &'a ComputeContext,
    ) -> Self {
        // Initialize stages with shared context
        todo!()
    }

    /// Process a single batch through all stages
    fn process_batch(&mut self, file: &mut SoaFastqReader) -> Result<SoAAlignmentResult> {
        // Stage 0: Load batch
        let ctx = StageContext {
            index: self.index,
            options: self.options,
            compute_ctx: self.compute_ctx,
            batch_id: self.stats.batches_processed,
        };

        let batch = self.loader.process(file, &ctx)?;

        // Early return if EOF
        if batch.is_empty() {
            return Ok(SoAAlignmentResult::new());
        }

        // Stage 1: Seeding (SoA batch)
        let seeds = self.seeder.process(batch.clone(), &ctx)?;

        // Stage 2: Chaining
        let chains = self.chainer.process(seeds, &ctx)?;

        // Stage 3: Extension (parallel)
        let regions = self.extender.process(chains, &ctx)?;

        // Stage 4: Finalization
        let alignments = self.finalizer.process(regions, &ctx)?;

        Ok(alignments)
    }
}

impl<'a> PipelineOrchestrator for SingleEndOrchestrator<'a> {
    fn run(
        &mut self,
        input_files: &[PathBuf],
        output: &mut dyn Write,
    ) -> Result<PipelineStatistics> {
        for file_path in input_files {
            let mut reader = SoaFastqReader::new(file_path)?;

            loop {
                // Process one batch
                let result = self.process_batch(&mut reader)?;

                if result.is_empty() {
                    break; // EOF
                }

                // Select and write alignments
                self.selector.select_and_write(&result, output)?;

                // Update stats
                self.stats.update(&result);
            }
        }

        Ok(self.stats.clone())
    }

    fn mode(&self) -> PipelineMode {
        PipelineMode::SingleEnd
    }
}
```

### Paired-End Orchestrator

```rust
pub struct PairedEndOrchestrator<'a> {
    index: &'a BwaIndex,
    options: &'a MemOpt,
    compute_ctx: &'a ComputeContext,

    // Stage implementations (same as single-end)
    loader: LoadingStage,
    seeder: SeedingStage,
    chainer: ChainingStage,
    extender: ExtensionStage,
    finalizer: FinalizationStage,

    // Paired-end specific
    insert_stats: Option<InsertSizeStats>,
    pairing_engine: PairingEngine,
    mate_rescuer: MateRescuer,

    // Statistics
    stats: PipelineStatistics,
}

impl<'a> PairedEndOrchestrator<'a> {
    /// Phase 1: Bootstrap insert size from first batch
    fn bootstrap_insert_size(&mut self, r1: &mut SoaFastqReader, r2: &mut SoaFastqReader)
        -> Result<()>
    {
        // Load small batch (512 pairs)
        let batch = self.loader.load_paired(r1, r2, 512)?;

        // Run through stages
        let result = self.process_batch_core(&batch)?;

        // Compute insert size statistics
        self.insert_stats = Some(bootstrap_insert_size_stats_soa(&result, self.options));

        log::info!("Insert size: μ={:.1} σ={:.1}",
            self.insert_stats.as_ref().unwrap().mean,
            self.insert_stats.as_ref().unwrap().std);

        Ok(())
    }

    /// Phase 2: Main processing loop
    fn process_main(&mut self, r1: &mut SoaFastqReader, r2: &mut SoaFastqReader, output: &mut dyn Write)
        -> Result<()>
    {
        loop {
            // Load large batch (500K pairs)
            let batch = self.loader.load_paired(r1, r2, self.options.batch_size)?;

            if batch.is_empty() {
                break;
            }

            // Process through stages
            let mut result = self.process_batch_core(&batch)?;

            // Paired-end specific: pairing + mate rescue
            self.pairing_engine.pair_alignments(&mut result, self.insert_stats.as_ref())?;
            self.mate_rescuer.rescue_mates(&mut result, self.options)?;

            // Write paired output
            write_sam_records_soa(&result, output, self.options)?;

            // Update stats
            self.stats.update(&result);
        }

        Ok(())
    }

    /// Core batch processing (shared with bootstrap)
    fn process_batch_core(&self, batch: &SoAReadBatch) -> Result<SoAAlignmentResult> {
        let ctx = StageContext {
            index: self.index,
            options: self.options,
            compute_ctx: self.compute_ctx,
            batch_id: self.stats.batches_processed,
        };

        let seeds = self.seeder.process(batch.clone(), &ctx)?;
        let chains = self.chainer.process(seeds, &ctx)?;
        let regions = self.extender.process(chains, &ctx)?;
        let alignments = self.finalizer.process(regions, &ctx)?;

        Ok(alignments)
    }
}

impl<'a> PipelineOrchestrator for PairedEndOrchestrator<'a> {
    fn run(
        &mut self,
        input_files: &[PathBuf],
        output: &mut dyn Write,
    ) -> Result<PipelineStatistics> {
        assert_eq!(input_files.len(), 2, "Paired-end requires exactly 2 files");

        let mut r1 = SoaFastqReader::new(&input_files[0])?;
        let mut r2 = SoaFastqReader::new(&input_files[1])?;

        // Phase 1: Bootstrap
        self.bootstrap_insert_size(&mut r1, &mut r2)?;

        // Phase 2: Main processing
        self.process_main(&mut r1, &mut r2, output)?;

        Ok(self.stats.clone())
    }

    fn mode(&self) -> PipelineMode {
        PipelineMode::PairedEnd
    }
}
```

---

## Stage Implementations

### Example: Seeding Stage

```rust
pub struct SeedingStage;

impl PipelineStage<SoAReadBatch, SoASeedBatch> for SeedingStage {
    fn process(&self, batch: SoAReadBatch, ctx: &StageContext) -> Result<SoASeedBatch> {
        // Wrap existing find_seeds_batch function
        let (seeds, encoded_queries, encoded_queries_rc) =
            find_seeds_batch(ctx.index, &batch, ctx.options);

        Ok(SoASeedBatch {
            seeds,
            read_seed_boundaries: compute_boundaries(&seeds, &batch),
            encoded_queries,
            encoded_queries_rc,
        })
    }

    fn name(&self) -> &str {
        "Seeding"
    }

    fn validate(&self, batch: &SoAReadBatch) -> Result<()> {
        if batch.is_empty() {
            return Err(StageError::SeedingError("Empty batch".to_string()));
        }
        Ok(())
    }
}
```

### Example: Extension Stage with Parallel Processing

```rust
pub struct ExtensionStage {
    parallel_processor: ParallelBatchProcessor<SoAChainBatch, SoAAlignmentRegions>,
}

impl PipelineStage<SoAChainBatch, SoAAlignmentRegions> for ExtensionStage {
    fn process(&self, chains: SoAChainBatch, ctx: &StageContext) -> Result<SoAAlignmentRegions> {
        // Use parallel processor for chunked execution
        let result = self.parallel_processor.process_parallel(&chains, |chunk| {
            // Call existing extension function on chunk
            align_regions_batch(chunk, ctx.index, ctx.options, ctx.compute_ctx.backend)
        });

        Ok(result)
    }

    fn name(&self) -> &str {
        "Extension"
    }
}
```

---

## Integration with Existing Code

### Migration Strategy

#### Phase 1: Extract Stages (No Behavior Change)
1. Move seeding logic to `stages/seeding.rs` (wrapper only)
2. Move chaining logic to `stages/chaining.rs` (wrapper only)
3. Move extension logic to `stages/extension.rs` (wrapper only)
4. Move finalization logic to `stages/finalization.rs` (wrapper only)
5. Keep existing functions intact, add thin wrapper

**Goal**: No behavior change, just organizational

#### Phase 2: Implement Orchestrators
1. Create `orchestrator/single_end.rs` using stage wrappers
2. Create `orchestrator/paired_end.rs` using stage wrappers
3. Update `mem.rs` to use orchestrators
4. Run full test suite, verify byte-for-byte output match

**Goal**: Same output, cleaner structure

#### Phase 3: Refactor Stage Internals (Optional)
1. Optimize individual stages independently
2. Add GPU/NPU backends to extension stage
3. Improve error handling and logging
4. Add stage-level benchmarks

**Goal**: Incremental improvements without breaking abstraction

### Backwards Compatibility

```rust
// Old API (deprecated but functional)
pub fn process_single_end(
    index: &BwaIndex,
    files: &[String],
    writer: &mut dyn Write,
    opt: &MemOpt,
    ctx: &ComputeContext,
) {
    // Delegate to new orchestrator
    let mut orchestrator = SingleEndOrchestrator::new(index, opt, ctx);
    let paths: Vec<PathBuf> = files.iter().map(PathBuf::from).collect();
    orchestrator.run(&paths, writer).expect("Pipeline failed");
}

// New API
pub fn run_pipeline(
    mode: PipelineMode,
    index: &BwaIndex,
    files: &[PathBuf],
    writer: &mut dyn Write,
    opt: &MemOpt,
    ctx: &ComputeContext,
) -> Result<PipelineStatistics> {
    match mode {
        PipelineMode::SingleEnd => {
            let mut orch = SingleEndOrchestrator::new(index, opt, ctx);
            orch.run(files, writer)
        }
        PipelineMode::PairedEnd => {
            let mut orch = PairedEndOrchestrator::new(index, opt, ctx);
            orch.run(files, writer)
        }
    }
}
```

---

## Benefits

### 1. Testability
```rust
#[test]
fn test_seeding_stage_basic() {
    let index = load_test_index();
    let batch = create_test_batch();
    let ctx = StageContext::test_default(&index);

    let seeder = SeedingStage;
    let result = seeder.process(batch, &ctx).unwrap();

    assert_eq!(result.seeds.len(), 42); // Expected seed count
}
```

### 2. Modularity
```rust
// Easy to swap implementations
let extension_stage: Box<dyn PipelineStage<SoAChainBatch, SoAAlignmentRegions>> =
    if ctx.compute_ctx.backend == ComputeBackend::Gpu {
        Box::new(GpuExtensionStage::new())
    } else {
        Box::new(CpuExtensionStage::new())
    };
```

### 3. Clear Data Flow
```rust
// Explicit types at each stage boundary
LoadingStage    : () → SoAReadBatch
SeedingStage    : SoAReadBatch → SoASeedBatch
ChainingStage   : SoASeedBatch → SoAChainBatch
ExtensionStage  : SoAChainBatch → SoAAlignmentRegions
FinalizationStage : SoAAlignmentRegions → SoAAlignmentResult
```

### 4. Composability
```rust
// Compose stages into custom pipelines
let fast_pipeline = Pipeline::new()
    .with_stage(LoadingStage)
    .with_stage(FastSeedingStage)  // Fewer passes
    .with_stage(GreedyChainingStage) // Skip redundancy filter
    .with_stage(ExtensionStage)
    .with_stage(FinalizationStage);

let accurate_pipeline = Pipeline::new()
    .with_stage(LoadingStage)
    .with_stage(ExhaustiveSeedingStage) // All 3 passes
    .with_stage(OptimalChainingStage)   // Full DP
    .with_stage(ExtensionStage)
    .with_stage(FinalizationStage);
```

---

## Performance Considerations

### Zero-Cost Abstractions
- **Stage trait dispatch**: Only at batch boundaries (~1 call per 500K reads)
- **No heap allocations** in hot paths (reuse buffers)
- **Monomorphization**: Compiler optimizes away trait overhead

### Benchmarking Strategy
```rust
// Stage-level benchmarks
#[bench]
fn bench_seeding_stage_10k_reads(b: &mut Bencher) {
    let batch = load_benchmark_batch();
    let ctx = create_benchmark_context();
    let seeder = SeedingStage;

    b.iter(|| {
        seeder.process(batch.clone(), &ctx)
    });
}

// Pipeline-level benchmarks
#[bench]
fn bench_full_pipeline_single_end(b: &mut Bencher) {
    let mut orch = SingleEndOrchestrator::new(...);

    b.iter(|| {
        orch.run(test_files(), &mut io::sink())
    });
}
```

### Profiling Integration
```rust
impl<In, Out> PipelineStage<In, Out> for TimedStage<In, Out> {
    fn process(&self, input: In, ctx: &StageContext) -> Result<Out> {
        let start = Instant::now();
        let result = self.inner.process(input, ctx)?;
        let elapsed = start.elapsed();

        log::debug!("{} took {:?}", self.inner.name(), elapsed);

        Ok(result)
    }
}
```

---

## Open Questions

1. **Should stages be stateful or stateless?**
   - Stateless: Easier to test, thread-safe
   - Stateful: Can cache thread-local buffers
   - **Proposal**: Stateless stages + separate `WorkspacePool` for buffers

2. **How to handle errors in parallel chunks?**
   - Fail fast (current behavior)
   - Collect all errors and return batch result
   - **Proposal**: Fail fast for consistency with C++ bwa-mem2

3. **Should orchestrator own stages or accept them as parameters?**
   - Own: Simpler API
   - Accept: More flexible for testing
   - **Proposal**: Own by default, `with_stage()` builder for custom pipelines

4. **How to expose stage-level metrics?**
   - Per-batch statistics (time, throughput)
   - Aggregated statistics
   - **Proposal**: Both via `StageMetrics` struct

---

## Implementation Checklist

### Milestone 1: Core Abstractions (1-2 weeks)
- [ ] Define `PipelineStage` trait
- [ ] Define `PipelineOrchestrator` trait
- [ ] Implement `StageContext` and `StageError`
- [ ] Implement `ParallelBatchProcessor` generic
- [ ] Add unit tests for abstractions

### Milestone 2: Stage Wrappers (1 week)
- [ ] Extract `LoadingStage` (wrap `SoaFastqReader`)
- [ ] Extract `SeedingStage` (wrap `find_seeds_batch`)
- [ ] Extract `ChainingStage` (wrap `chain_seeds_batch`)
- [ ] Extract `ExtensionStage` (wrap `align_regions_batch`)
- [ ] Extract `FinalizationStage` (wrap `finalize_candidates`)
- [ ] Add integration tests per stage

### Milestone 3: Orchestrators (1-2 weeks)
- [ ] Implement `SingleEndOrchestrator`
- [ ] Implement `PairedEndOrchestrator`
- [ ] Update `mem.rs` to use orchestrators
- [ ] Run golden dataset comparison (ensure bit-exact output)
- [ ] Performance regression testing

### Milestone 4: Documentation & Polish (1 week)
- [ ] Update CLAUDE.md with new architecture
- [ ] Add rustdoc for all public APIs
- [ ] Create migration guide for external users
- [ ] Benchmark before/after refactoring
- [ ] Add examples to documentation

**Total Estimate**: 4-6 weeks for complete refactoring

---

## Conclusion

This refactoring will:
1. **Improve maintainability**: Clear separation of concerns
2. **Enable testing**: Mock individual stages
3. **Facilitate GPU/NPU integration**: Swap extension stage implementation
4. **Reduce duplication**: Unified orchestrator logic
5. **Maintain performance**: Zero-cost abstractions

The migration can be done incrementally without breaking existing functionality, with each milestone delivering value independently.

---

## References

- **C++ bwa-mem2 architecture**: `src/fastmap.cpp` (monolithic but performant)
- **Rust async runtimes**: Tokio's staged architecture for inspiration
- **Game engine ECS patterns**: Data-oriented stage design
- **Scientific workflow systems**: CWL, Nextflow (declarative pipelines)
