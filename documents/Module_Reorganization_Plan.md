# Module Reorganization Plan

## Executive Summary

This document provides a concrete plan for reorganizing FerrousAlign's codebase to implement the main loop abstraction proposed in `Main_Loop_Abstraction_Proposal.md`. It includes file-by-file refactoring steps, dependency analysis, and a phased migration strategy.

**Target Version**: v0.8.0 (post-SoA integration)

**Status**: Planning (pending v0.7.0 completion)

---

## Current Module Structure (v0.7.0-alpha)

```
src/
├── core/
│   ├── alignment/                 # SIMD kernels (good structure)
│   │   ├── banded_swa/
│   │   ├── kswv_avx2.rs
│   │   ├── kswv_avx512.rs
│   │   ├── kswv_batch.rs
│   │   ├── shared_types.rs
│   │   └── workspace.rs
│   ├── compute/                   # Backend abstraction (good)
│   │   ├── mod.rs
│   │   └── simd_abstraction/
│   └── io/                        # I/O primitives (good)
│       ├── async_writer.rs
│       ├── fasta_reader.rs
│       ├── fastq_reader.rs
│       ├── sam_output.rs
│       └── soa_readers.rs
│
├── pipelines/
│   └── linear/
│       ├── batch_extension/       # Extension orchestration
│       │   ├── mod.rs
│       │   ├── types.rs
│       │   └── ...
│       ├── index/                 # FM-Index, BWT, SA
│       │   ├── bns.rs
│       │   ├── bwa_index.rs
│       │   ├── bwt.rs
│       │   └── ...
│       ├── paired/                # Paired-end logic
│       │   ├── insert_size.rs
│       │   ├── mate_rescue.rs
│       │   ├── paired_end.rs     # 600 lines - NEEDS REFACTORING
│       │   └── pairing.rs
│       ├── chaining.rs            # 823 lines
│       ├── finalization.rs        # 1707 lines - NEEDS SPLITTING
│       ├── mem.rs                 # 264 lines
│       ├── mem_opt.rs             # Options struct
│       ├── pipeline.rs            # 1247 lines
│       ├── region.rs              # 1598 lines - NEEDS SPLITTING
│       ├── seeding.rs             # 1902 lines - NEEDS SPLITTING
│       └── single_end.rs          # 400 lines - NEEDS REFACTORING
│
├── utils.rs
├── lib.rs
└── main.rs
```

### File Size Analysis

| File | Lines | Target | Status | Priority |
|------|-------|--------|--------|----------|
| `seeding.rs` | 1902 | 500 | ⚠️ Split | High |
| `finalization.rs` | 1707 | 500 | ⚠️ Split | High |
| `region.rs` | 1598 | 500 | ⚠️ Split | High |
| `pipeline.rs` | 1247 | 800 | ⚠️ Refactor | Medium |
| `chaining.rs` | 823 | 500 | ⚠️ Split | Medium |
| `paired_end.rs` | 600 | 400 | ⚠️ Refactor | High |
| `single_end.rs` | 400 | 400 | ✅ OK | Low |

---

## Target Module Structure (v0.8.0)

```
src/
├── core/                          # Reference-agnostic (NO CHANGE)
│   ├── alignment/
│   ├── compute/
│   └── io/
│
├── pipelines/
│   └── linear/
│       ├── index/                 # Index loading (NO CHANGE)
│       │
│       ├── orchestrator/          # 🆕 Main loop coordination
│       │   ├── mod.rs             # Public API + PipelineOrchestrator trait
│       │   ├── single_end.rs      # Single-end orchestrator
│       │   ├── paired_end.rs      # Paired-end orchestrator
│       │   └── statistics.rs      # Stats aggregation
│       │
│       ├── stages/                # 🆕 Pipeline stages (refactored)
│       │   ├── mod.rs             # PipelineStage trait
│       │   ├── loading.rs         # Stage 0: Read loading
│       │   ├── seeding/           # Stage 1: SMEM extraction
│       │   │   ├── mod.rs         # Public API
│       │   │   ├── smem.rs        # SMEM algorithm
│       │   │   ├── reseeding.rs   # Chimeric detection
│       │   │   ├── forward_only.rs # 3rd round seeding
│       │   │   └── seed_convert.rs # SMEM → Seed conversion
│       │   ├── chaining/          # Stage 2: Seed chaining
│       │   │   ├── mod.rs
│       │   │   ├── dp_chain.rs    # O(n²) DP chaining
│       │   │   └── filter.rs      # Chain filtering
│       │   ├── extension/         # Stage 3: SW alignment
│       │   │   ├── mod.rs
│       │   │   ├── region.rs      # Region extension
│       │   │   └── batch.rs       # Batch orchestration
│       │   └── finalization/      # Stage 4: CIGAR/MD/NM
│       │       ├── mod.rs
│       │       ├── cigar.rs       # CIGAR generation
│       │       ├── md_tag.rs      # MD tag computation
│       │       ├── mapq.rs        # MAPQ calculation
│       │       └── flags.rs       # SAM flag handling
│       │
│       ├── modes/                 # 🆕 Mode-specific logic
│       │   ├── mod.rs
│       │   ├── single_end.rs      # SE selection (from sam_output)
│       │   └── paired_end/        # PE pairing (from paired/)
│       │       ├── mod.rs
│       │       ├── insert_size.rs # Insert size stats
│       │       ├── pairing.rs     # Pairing algorithm
│       │       └── mate_rescue.rs # Mate rescue
│       │
│       ├── mem.rs                 # Entry point (minor refactor)
│       └── mem_opt.rs             # Options (NO CHANGE)
│
├── utils.rs
├── lib.rs
└── main.rs
```

### Design Principles

1. **500-line target per file** (except trait definitions)
2. **Clear module boundaries** (one responsibility per module)
3. **Testable units** (mock any stage)
4. **Zero regression** (byte-for-byte output match during migration)

---

## Refactoring Roadmap

### Phase 1: Foundation (Week 1-2)

#### 1.1 Create Orchestrator Skeleton

**New files:**
- `src/pipelines/linear/orchestrator/mod.rs`
- `src/pipelines/linear/orchestrator/statistics.rs`
- `src/pipelines/linear/stages/mod.rs`

**Action:**
```bash
mkdir -p src/pipelines/linear/orchestrator
mkdir -p src/pipelines/linear/stages
mkdir -p src/pipelines/linear/modes
```

**Code:**
```rust
// src/pipelines/linear/orchestrator/mod.rs
pub trait PipelineOrchestrator {
    fn run(
        &mut self,
        input_files: &[PathBuf],
        output: &mut dyn Write,
    ) -> Result<PipelineStatistics>;

    fn mode(&self) -> PipelineMode;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineMode {
    SingleEnd,
    PairedEnd,
}

pub use statistics::PipelineStatistics;

pub mod statistics;
```

```rust
// src/pipelines/linear/stages/mod.rs
pub trait PipelineStage<In, Out> {
    fn process(&self, input: In, ctx: &StageContext) -> Result<Out, StageError>;
    fn name(&self) -> &str;
    fn validate(&self, input: &In) -> Result<(), StageError> {
        Ok(())
    }
}

pub struct StageContext<'a> {
    pub index: &'a BwaIndex,
    pub options: &'a MemOpt,
    pub compute_ctx: &'a ComputeContext,
    pub batch_id: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum StageError {
    #[error("Seeding failed: {0}")]
    SeedingError(String),
    #[error("Extension failed: {0}")]
    ExtensionError(String),
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
}

pub mod loading;
pub mod seeding;
pub mod chaining;
pub mod extension;
pub mod finalization;
```

**Testing:**
```rust
// tests/test_orchestrator_traits.rs
#[test]
fn test_stage_trait_compiles() {
    // Ensure trait is object-safe and compiles
    struct DummyStage;
    impl PipelineStage<(), ()> for DummyStage {
        fn process(&self, _: (), _: &StageContext) -> Result<(), StageError> {
            Ok(())
        }
        fn name(&self) -> &str { "dummy" }
    }
}
```

---

### Phase 2: Split Large Files (Week 2-3)

#### 2.1 Split `seeding.rs` (1902 lines → 4 files)

**Target structure:**
```
stages/seeding/
├── mod.rs              # ~200 lines (public API + stage trait impl)
├── smem.rs             # ~800 lines (SMEM algorithm, backward search)
├── reseeding.rs        # ~300 lines (re-seeding logic)
├── forward_only.rs     # ~400 lines (3rd round seeding)
└── seed_convert.rs     # ~200 lines (SMEM → Seed conversion)
```

**Migration steps:**

1. **Create module skeleton:**
```bash
mkdir -p src/pipelines/linear/stages/seeding
touch src/pipelines/linear/stages/seeding/{mod.rs,smem.rs,reseeding.rs,forward_only.rs,seed_convert.rs}
```

2. **Move SMEM core to `smem.rs`:**
   - `generate_smems_for_strand()` → `smem.rs`
   - `generate_smems_from_position()` → `smem.rs`
   - Helper functions: `backward_search()`, `compute_smem_interval()` → `smem.rs`

3. **Move re-seeding to `reseeding.rs`:**
   - Re-seeding loop (lines 140-192) → `reseeding.rs:collect_reseed_candidates()`
   - Split length calculation → `reseeding.rs:compute_split_len()`

4. **Move forward-only to `forward_only.rs`:**
   - `forward_only_seed_strategy()` → `forward_only.rs`

5. **Move seed conversion to `seed_convert.rs`:**
   - `get_sa_entries()` → `seed_convert.rs`
   - SMEM → Seed loop (lines 436-528) → `seed_convert.rs:convert_smems_to_seeds()`

6. **Create stage wrapper in `mod.rs`:**
```rust
// src/pipelines/linear/stages/seeding/mod.rs
use super::{PipelineStage, StageContext, StageError};

pub struct SeedingStage;

impl PipelineStage<SoAReadBatch, SoASeedBatch> for SeedingStage {
    fn process(&self, batch: SoAReadBatch, ctx: &StageContext) -> Result<SoASeedBatch, StageError> {
        // Call existing functions from submodules
        let smems = smem::generate_smems_batch(&batch, ctx.index, ctx.options)?;
        let reseeded = reseeding::apply_reseeding(smems, ctx.options)?;
        let forward_seeds = forward_only::apply_forward_seeding(&batch, &reseeded, ctx)?;
        let seeds = seed_convert::convert_to_seeds(&forward_seeds, ctx)?;

        Ok(seeds)
    }

    fn name(&self) -> &str {
        "Seeding"
    }
}

pub use smem::*;
pub use reseeding::*;
pub use forward_only::*;
pub use seed_convert::*;
```

7. **Update imports in existing code:**
```rust
// Before
use super::seeding::{find_seeds, Seed, SMEM};

// After
use super::stages::seeding::{find_seeds, Seed, SMEM};
```

8. **Verify with tests:**
```bash
cargo test --lib seeding
cargo test --test '*' | grep -i seed
```

---

#### 2.2 Split `finalization.rs` (1707 lines → 5 files)

**Target structure:**
```
stages/finalization/
├── mod.rs              # ~200 lines (public API + stage trait impl)
├── cigar.rs            # ~500 lines (CIGAR generation + traceback)
├── md_tag.rs           # ~400 lines (MD tag computation)
├── mapq.rs             # ~300 lines (MAPQ calculation)
└── flags.rs            # ~300 lines (SAM flags + secondary marking)
```

**Migration steps:**

1. **Move CIGAR generation to `cigar.rs`:**
   - `generate_cigar_from_region()` → `cigar.rs`
   - `traceback()` → `cigar.rs:traceback_dp_matrix()`
   - CIGAR simplification → `cigar.rs:compress_cigar()`

2. **Move MD tag to `md_tag.rs`:**
   - `compute_md_tag()` → `md_tag.rs`
   - Reference extraction helpers → `md_tag.rs`

3. **Move MAPQ to `mapq.rs`:**
   - `calculate_mapq()` → `mapq.rs`
   - Phred score helpers → `mapq.rs`

4. **Move flag handling to `flags.rs`:**
   - `mark_secondary_alignments()` → `flags.rs`
   - `sam_flags` constants → `flags.rs:SAM_FLAGS`
   - Overlap detection → `flags.rs:compute_overlap()`

5. **Create stage wrapper:**
```rust
// src/pipelines/linear/stages/finalization/mod.rs
impl PipelineStage<Vec<AlignmentRegion>, Vec<Alignment>> for FinalizationStage {
    fn process(&self, regions: Vec<AlignmentRegion>, ctx: &StageContext) -> Result<Vec<Alignment>, StageError> {
        let mut alignments = Vec::new();

        for region in regions {
            // Generate CIGAR
            let cigar = cigar::generate_cigar(&region, ctx)?;

            // Compute MD tag
            let md = md_tag::compute_md(&region, &cigar, ctx)?;

            // Build alignment
            let mut aln = Alignment::from_region(region, cigar, md);

            alignments.push(aln);
        }

        // Mark secondary/supplementary
        flags::mark_secondary_alignments(&mut alignments, ctx.options);

        // Compute MAPQ
        for aln in &mut alignments {
            aln.mapq = mapq::calculate_mapq(aln, ctx.options);
        }

        Ok(alignments)
    }

    fn name(&self) -> &str {
        "Finalization"
    }
}
```

---

#### 2.3 Split `region.rs` (1598 lines → 2 files)

**Target structure:**
```
stages/extension/
├── mod.rs              # ~200 lines (stage trait impl)
├── region.rs           # ~800 lines (AlignmentRegion type + extension logic)
└── batch.rs            # ~600 lines (Batch orchestration)
```

**Migration steps:**

1. **Keep core region logic in `region.rs`:**
   - `AlignmentRegion` struct → `region.rs`
   - `extend_chains_to_regions()` → `region.rs`
   - `generate_cigar_from_region()` → move to `finalization/cigar.rs`

2. **Extract batch orchestration to `batch.rs`:**
   - `align_regions_batch()` → `batch.rs`
   - Parallel chunking logic → `batch.rs`

3. **Create stage wrapper:**
```rust
// src/pipelines/linear/stages/extension/mod.rs
impl PipelineStage<SoAChainBatch, Vec<AlignmentRegion>> for ExtensionStage {
    fn process(&self, chains: SoAChainBatch, ctx: &StageContext) -> Result<Vec<AlignmentRegion>, StageError> {
        batch::align_regions_batch(chains, ctx)
    }

    fn name(&self) -> &str {
        "Extension"
    }
}
```

---

#### 2.4 Split `chaining.rs` (823 lines → 2 files)

**Target structure:**
```
stages/chaining/
├── mod.rs              # ~200 lines (stage trait impl)
├── dp_chain.rs         # ~400 lines (O(n²) DP chaining)
└── filter.rs           # ~223 lines (Chain filtering)
```

**Migration steps:**

1. **Move DP chaining to `dp_chain.rs`:**
   - `chain_seeds()` → `dp_chain.rs`
   - `chain_seeds_batch()` → `dp_chain.rs`

2. **Move filtering to `filter.rs`:**
   - `filter_chains()` → `filter.rs`
   - `filter_chains_batch()` → `filter.rs`

3. **Create stage wrapper:**
```rust
// src/pipelines/linear/stages/chaining/mod.rs
impl PipelineStage<SoASeedBatch, SoAChainBatch> for ChainingStage {
    fn process(&self, seeds: SoASeedBatch, ctx: &StageContext) -> Result<SoAChainBatch, StageError> {
        let chains = dp_chain::chain_seeds_batch(&seeds, ctx.options)?;
        let filtered = filter::filter_chains_batch(chains, &seeds, ctx.options)?;
        Ok(filtered)
    }

    fn name(&self) -> &str {
        "Chaining"
    }
}
```

---

### Phase 3: Implement Orchestrators (Week 4)

#### 3.1 Single-End Orchestrator

**File:** `src/pipelines/linear/orchestrator/single_end.rs`

**Code:**
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

    // Statistics
    stats: PipelineStatistics,
}

impl<'a> SingleEndOrchestrator<'a> {
    pub fn new(
        index: &'a BwaIndex,
        options: &'a MemOpt,
        compute_ctx: &'a ComputeContext,
    ) -> Self {
        Self {
            index,
            options,
            compute_ctx,
            loader: LoadingStage,
            seeder: SeedingStage,
            chainer: ChainingStage,
            extender: ExtensionStage,
            finalizer: FinalizationStage,
            stats: PipelineStatistics::new(),
        }
    }

    fn process_batch(&mut self, file: &mut SoaFastqReader) -> Result<SoAAlignmentResult> {
        let ctx = StageContext {
            index: self.index,
            options: self.options,
            compute_ctx: self.compute_ctx,
            batch_id: self.stats.batches_processed,
        };

        // Stage pipeline
        let batch = self.loader.process(file, &ctx)?;
        if batch.is_empty() {
            return Ok(SoAAlignmentResult::new());
        }

        let seeds = self.seeder.process(batch.clone(), &ctx)?;
        let chains = self.chainer.process(seeds, &ctx)?;
        let regions = self.extender.process(chains, &ctx)?;
        let alignments = self.finalizer.process(regions, &ctx)?;

        Ok(alignments)
    }
}

impl<'a> PipelineOrchestrator for SingleEndOrchestrator<'a> {
    fn run(&mut self, files: &[PathBuf], output: &mut dyn Write) -> Result<PipelineStatistics> {
        for file in files {
            let mut reader = SoaFastqReader::new(file)?;

            loop {
                let result = self.process_batch(&mut reader)?;
                if result.is_empty() {
                    break;
                }

                write_sam_records_soa(&result, output, self.options)?;
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

**Testing:**
```rust
#[test]
fn test_single_end_orchestrator_basic() {
    let index = load_test_index();
    let opt = MemOpt::default();
    let ctx = ComputeContext::default();

    let mut orch = SingleEndOrchestrator::new(&index, &opt, &ctx);
    let files = vec![PathBuf::from("test_data/test.fq")];
    let mut output = Vec::new();

    let stats = orch.run(&files, &mut output).unwrap();

    assert!(stats.total_reads > 0);
}
```

---

#### 3.2 Paired-End Orchestrator

**File:** `src/pipelines/linear/orchestrator/paired_end.rs`

**Code:**
```rust
pub struct PairedEndOrchestrator<'a> {
    index: &'a BwaIndex,
    options: &'a MemOpt,
    compute_ctx: &'a ComputeContext,

    // Stages (same as single-end)
    loader: LoadingStage,
    seeder: SeedingStage,
    chainer: ChainingStage,
    extender: ExtensionStage,
    finalizer: FinalizationStage,

    // Paired-end specific
    insert_stats: Option<InsertSizeStats>,
    pairing_engine: PairingEngine,
    mate_rescuer: MateRescuer,

    stats: PipelineStatistics,
}

impl<'a> PairedEndOrchestrator<'a> {
    fn bootstrap_insert_size(&mut self, r1: &mut SoaFastqReader, r2: &mut SoaFastqReader)
        -> Result<()>
    {
        // Load 512 pairs
        let batch = self.loader.load_paired(r1, r2, 512)?;

        // Process through stages
        let result = self.process_batch_core(&batch)?;

        // Bootstrap insert size
        self.insert_stats = Some(bootstrap_insert_size_stats_soa(&result, self.options));

        Ok(())
    }

    fn process_main(&mut self, r1: &mut SoaFastqReader, r2: &mut SoaFastqReader, output: &mut dyn Write)
        -> Result<()>
    {
        loop {
            let batch = self.loader.load_paired(r1, r2, self.options.batch_size)?;
            if batch.is_empty() {
                break;
            }

            let mut result = self.process_batch_core(&batch)?;

            // Pairing + mate rescue
            self.pairing_engine.pair_alignments(&mut result, self.insert_stats.as_ref())?;
            self.mate_rescuer.rescue_mates(&mut result, self.options)?;

            write_sam_records_soa(&result, output, self.options)?;
            self.stats.update(&result);
        }

        Ok(())
    }

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
    fn run(&mut self, files: &[PathBuf], output: &mut dyn Write) -> Result<PipelineStatistics> {
        assert_eq!(files.len(), 2);

        let mut r1 = SoaFastqReader::new(&files[0])?;
        let mut r2 = SoaFastqReader::new(&files[1])?;

        self.bootstrap_insert_size(&mut r1, &mut r2)?;
        self.process_main(&mut r1, &mut r2, output)?;

        Ok(self.stats.clone())
    }

    fn mode(&self) -> PipelineMode {
        PipelineMode::PairedEnd
    }
}
```

---

### Phase 4: Integration (Week 5)

#### 4.1 Update `mem.rs` Entry Point

**File:** `src/pipelines/linear/mem.rs`

**Changes:**
```rust
// Before (lines 237-259):
if opts.reads.len() == 2 {
    process_paired_end(&bwa_idx, ...);
} else {
    process_single_end(&bwa_idx, ...);
}

// After:
use super::orchestrator::{PipelineOrchestrator, SingleEndOrchestrator, PairedEndOrchestrator};

let stats = if opts.reads.len() == 2 {
    let mut orch = PairedEndOrchestrator::new(&bwa_idx, &opt, &compute_ctx);
    orch.run(&opts.reads, &mut writer)?
} else {
    let mut orch = SingleEndOrchestrator::new(&bwa_idx, &opt, &compute_ctx);
    orch.run(&opts.reads, &mut writer)?
};

log::info!("Processed {} reads in {:.2}s", stats.total_reads, stats.wall_time_secs);
```

#### 4.2 Deprecate Old Entry Points

**Files to mark as deprecated:**
- `src/pipelines/linear/single_end.rs:process_single_end()` → `#[deprecated]`
- `src/pipelines/linear/paired/paired_end.rs:process_paired_end()` → `#[deprecated]`

**Keep for backwards compatibility** (1-2 releases):
```rust
#[deprecated(since = "0.8.0", note = "Use SingleEndOrchestrator instead")]
pub fn process_single_end(...) {
    // Delegate to new orchestrator
    let mut orch = SingleEndOrchestrator::new(...);
    orch.run(...).expect("Pipeline failed");
}
```

---

### Phase 5: Validation (Week 6)

#### 5.1 Golden Dataset Testing

**Test suite:**
```bash
# Download golden reads (if not exists)
./scripts/download_golden_reads.sh

# Run baseline (old code)
git checkout v0.7.0
./target/release/ferrous-align mem -t 16 \
    tests/golden_reads/baseline_ref.fa \
    tests/golden_reads/golden_10k_R1.fq \
    tests/golden_reads/golden_10k_R2.fq \
    > /tmp/baseline_output.sam

# Run refactored (new code)
git checkout feature/orchestrator-refactor
cargo build --release
./target/release/ferrous-align mem -t 16 \
    tests/golden_reads/baseline_ref.fa \
    tests/golden_reads/golden_10k_R1.fq \
    tests/golden_reads/golden_10k_R2.fq \
    > /tmp/refactored_output.sam

# Compare (should be identical)
diff /tmp/baseline_output.sam /tmp/refactored_output.sam
```

**Acceptance criteria:**
- ✅ Bit-for-bit identical SAM output
- ✅ Same alignment count (samtools flagstat)
- ✅ Same MAPQ distribution
- ✅ Same CIGAR string distribution
- ✅ No performance regression (±5% acceptable)

#### 5.2 Performance Benchmarking

**Benchmark suite:**
```bash
# 10K reads (small)
hyperfine --warmup 3 --runs 10 \
    './target/release/ferrous-align mem REF R1 R2 > /dev/null'

# 100K reads (medium)
hyperfine --warmup 1 --runs 5 \
    './target/release/ferrous-align mem REF R1_100k R2_100k > /dev/null'

# 4M reads (large, full dataset)
/usr/bin/time -v \
    ./target/release/ferrous-align mem -t 16 REF R1_4M R2_4M > /dev/null
```

**Performance targets:**
- Small batch (10K): No regression
- Medium batch (100K): No regression
- Large batch (4M): ±5% acceptable (abstraction overhead)

---

## Dependency Graph

### Module Dependencies (After Refactoring)

```
main.rs
  └─► pipelines::linear::mem
        └─► orchestrator::{SingleEndOrchestrator, PairedEndOrchestrator}
              ├─► stages::loading
              ├─► stages::seeding
              ├─► stages::chaining
              ├─► stages::extension
              └─► stages::finalization

stages::*
  ├─► core::alignment (SIMD kernels)
  ├─► core::io (readers/writers)
  ├─► core::compute (backend abstraction)
  └─► pipelines::linear::index (FM-Index)

modes::paired_end
  ├─► stages::* (all stages)
  └─► paired::{insert_size, pairing, mate_rescue}
```

### Breaking Changes

**None** (backwards compatibility maintained via deprecated wrappers)

### API Stability

| Component | Stability | Notes |
|-----------|-----------|-------|
| `PipelineOrchestrator` | 🆕 Stable | New public API |
| `PipelineStage` | 🆕 Stable | New public API |
| `process_single_end()` | ⚠️ Deprecated | Remove in v0.9.0 |
| `process_paired_end()` | ⚠️ Deprecated | Remove in v0.9.0 |
| Core alignment kernels | ✅ Stable | No changes |
| Index loading | ✅ Stable | No changes |

---

## Migration Checklist

### Week 1-2: Foundation
- [ ] Create `orchestrator/` module structure
- [ ] Create `stages/` module structure
- [ ] Define `PipelineStage` trait
- [ ] Define `PipelineOrchestrator` trait
- [ ] Add unit tests for traits
- [ ] Update `CLAUDE.md` with new structure

### Week 2-3: File Splitting
- [ ] Split `seeding.rs` → `stages/seeding/`
  - [ ] Extract SMEM algorithm
  - [ ] Extract re-seeding logic
  - [ ] Extract forward-only seeding
  - [ ] Extract seed conversion
  - [ ] Verify all tests pass
- [ ] Split `finalization.rs` → `stages/finalization/`
  - [ ] Extract CIGAR generation
  - [ ] Extract MD tag computation
  - [ ] Extract MAPQ calculation
  - [ ] Extract flag handling
  - [ ] Verify all tests pass
- [ ] Split `region.rs` → `stages/extension/`
  - [ ] Extract batch orchestration
  - [ ] Verify all tests pass
- [ ] Split `chaining.rs` → `stages/chaining/`
  - [ ] Extract DP chaining
  - [ ] Extract filtering
  - [ ] Verify all tests pass

### Week 4: Orchestrators
- [ ] Implement `SingleEndOrchestrator`
- [ ] Implement `PairedEndOrchestrator`
- [ ] Add integration tests for orchestrators
- [ ] Benchmark orchestrators vs old code

### Week 5: Integration
- [ ] Update `mem.rs` to use orchestrators
- [ ] Mark old entry points as deprecated
- [ ] Update documentation
- [ ] Run full test suite

### Week 6: Validation
- [ ] Golden dataset testing (10K reads)
- [ ] Performance benchmarking (10K, 100K, 4M)
- [ ] Memory profiling (check for leaks)
- [ ] Code review
- [ ] Merge to main

---

## Rollback Plan

If critical issues are discovered during validation:

### Option 1: Quick Fix
- Fix bug in refactored code
- Re-run validation suite
- Continue with merge

### Option 2: Revert and Re-plan
- Revert to v0.7.0 baseline
- Analyze failure root cause
- Adjust refactoring strategy
- Re-attempt in next cycle

### Rollback Triggers
- SAM output differs from baseline (non-deterministic hash is OK)
- Performance regression >10%
- Memory usage increase >20%
- Test failures that can't be fixed in 2 days

---

## Success Metrics

### Code Quality
- ✅ All files <500 lines (except trait definitions)
- ✅ Clear module boundaries (one responsibility per module)
- ✅ 100% test coverage for orchestrator logic
- ✅ No clippy warnings

### Functionality
- ✅ Bit-for-bit identical output on golden dataset
- ✅ All existing tests pass
- ✅ No new bugs reported in first 2 weeks post-release

### Performance
- ✅ No regression on 10K read benchmark
- ✅ <5% regression on 4M read benchmark
- ✅ Memory usage unchanged (±10%)

### Maintainability
- ✅ New contributors can understand pipeline flow in <1 hour
- ✅ Easy to mock stages for testing
- ✅ Clear extension points for GPU/NPU backends

---

## Post-Refactoring Opportunities

Once the orchestrator abstraction is in place, these become much easier:

### 1. GPU Extension Backend
```rust
struct GpuExtensionStage {
    context: GpuContext,
}

impl PipelineStage<SoAChainBatch, Vec<AlignmentRegion>> for GpuExtensionStage {
    fn process(&self, chains: SoAChainBatch, ctx: &StageContext) -> Result<...> {
        // Offload to GPU via Metal/CUDA/ROCm
        self.context.align_batch_gpu(&chains)
    }
}
```

### 2. Incremental Output (Streaming)
```rust
struct StreamingOrchestrator {
    // Process batches on background thread
    // Write output as soon as available
}
```

### 3. Custom Pipelines
```rust
let fast_pipeline = Pipeline::builder()
    .with_stage(LoadingStage)
    .with_stage(FastSeedingStage)  // Fewer SMEM passes
    .with_stage(GreedyChainingStage)
    .with_stage(ExtensionStage)
    .with_stage(FinalizationStage)
    .build();
```

### 4. Stage-Level Profiling
```rust
let profiled_orch = SingleEndOrchestrator::new(...)
    .with_profiling(true);

// Automatic per-stage timing and throughput logging
```

---

## Conclusion

This refactoring will:
1. **Reduce complexity**: 500-line modules vs 1700-line monoliths
2. **Improve testability**: Mock any stage independently
3. **Enable extensibility**: Easy to add GPU/NPU backends
4. **Maintain compatibility**: Deprecated wrappers prevent breaking changes
5. **Zero regression**: Validation suite ensures identical output

**Timeline**: 6 weeks from start to merge

**Risk**: Low (incremental migration with validation at each step)

**Impact**: High (foundation for future GPU/NPU integration)

---

## References

- `Pipeline_Flow_Diagram.md` - Current architecture analysis
- `Main_Loop_Abstraction_Proposal.md` - Design philosophy
- `SOA_Transition.md` - SoA migration lessons learned
- C++ bwa-mem2: `src/fastmap.cpp` - Comparison baseline
