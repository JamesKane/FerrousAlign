# Legacy File Split Plan

**Status**: Planning
**Target**: All modules ≤500 lines (merge gating requirement)
**Created**: 2025-12-03

## Overview

Four legacy files exceed the 500-line limit and require splitting:

| File | Current Lines | Target Modules | Priority |
|------|---------------|----------------|----------|
| `seeding.rs` | 1929 | 5 modules | High |
| `finalization.rs` | 1704 | 6 modules | High |
| `region.rs` | 1598 | 5 modules | Medium |
| `chaining.rs` | 1116 | 4 modules | Medium |

**Total**: 6,347 lines → ~20 modules averaging ~300 lines each

---

## 1. finalization.rs (1704 lines → 6 modules)

### Current Structure Analysis

The file contains several distinct responsibilities:
- `Alignment` struct and SAM output methods (~200 lines)
- `sam_flags` module - SAM flag constants (~20 lines, already a submodule)
- `remove_redundant_alignments()` - Overlap detection and filtering (~300 lines)
- `mark_secondary_alignments()` - Secondary/supplementary marking (~300 lines)
- `calculate_mapq()` - MAPQ computation (~100 lines)
- `generate_xa_tags()`, `generate_sa_tags()` - Tag generation (~200 lines)
- Tests (~400 lines)

### Proposed Split

```
src/pipelines/linear/finalization/
├── mod.rs              (~100 lines) - Re-exports, FinalizationStage wrapper
├── alignment.rs        (~250 lines) - Alignment struct, SAM output methods
├── sam_flags.rs        (~30 lines)  - Flag constants (already exists inline)
├── redundancy.rs       (~350 lines) - remove_redundant_alignments, overlap detection
├── secondary.rs        (~350 lines) - mark_secondary_alignments, supplementary logic
├── mapq.rs             (~150 lines) - MAPQ calculation, mem_approx_mapq_se
└── tags.rs             (~250 lines) - XA/SA tag generation
```

**Tests**: Keep inline with each module (~50-80 lines each)

### Dependencies

```
mod.rs
  ├── alignment.rs (core struct)
  │     └── sam_flags.rs (constants)
  ├── redundancy.rs
  │     └── alignment.rs
  ├── secondary.rs
  │     └── alignment.rs
  ├── mapq.rs
  │     └── alignment.rs
  └── tags.rs
        └── alignment.rs
```

### Implementation Order

1. Create `finalization/` directory and `mod.rs`
2. Extract `sam_flags` (trivial, no dependencies)
3. Extract `Alignment` struct to `alignment.rs`
4. Extract `mapq.rs` (smallest, self-contained)
5. Extract `tags.rs` (self-contained)
6. Extract `redundancy.rs` (complex, needs careful testing)
7. Extract `secondary.rs` (complex, needs careful testing)
8. Update all imports in `stages/finalization/mod.rs`

---

## 2. seeding.rs (1929 lines → 5 modules)

### Current Structure Analysis

- `Seed` struct and basic types (~100 lines)
- `SoASeedBatch` struct with SoA layout (~200 lines)
- `SmemConfig` and configuration (~50 lines)
- `collect_seeds_batch()` - Main entry point (~150 lines)
- `smem_search()` and SMEM iteration (~400 lines)
- `bidirectional_extend()` - FM-index bidirectional search (~200 lines)
- Helper functions for interval operations (~200 lines)
- Seed filtering and deduplication (~150 lines)
- Tests (~400 lines)

### Proposed Split

```
src/pipelines/linear/seeding/
├── mod.rs              (~120 lines) - Re-exports, SeedingStage wrapper
├── types.rs            (~200 lines) - Seed, SmemConfig, SeedInterval
├── soa_batch.rs        (~250 lines) - SoASeedBatch struct and methods
├── smem.rs             (~450 lines) - SMEM search algorithm, iteration
├── bidirectional.rs    (~250 lines) - Bidirectional FM-index extension
└── collection.rs       (~350 lines) - collect_seeds_batch, filtering, dedup
```

**Tests**: Distribute with relevant modules

### Dependencies

```
mod.rs
  ├── types.rs (core structs)
  ├── soa_batch.rs
  │     └── types.rs
  ├── smem.rs
  │     ├── types.rs
  │     └── bidirectional.rs
  ├── bidirectional.rs
  │     └── types.rs (SeedInterval)
  └── collection.rs
        ├── types.rs
        ├── soa_batch.rs
        └── smem.rs
```

### Implementation Order

1. Create `seeding/` directory and `mod.rs`
2. Extract `types.rs` (Seed, SmemConfig, SeedInterval)
3. Extract `soa_batch.rs` (SoASeedBatch)
4. Extract `bidirectional.rs` (low-level FM-index ops)
5. Extract `smem.rs` (SMEM algorithm)
6. Extract `collection.rs` (batch collection, filtering)
7. Update imports in `stages/seeding/mod.rs`

---

## 3. region.rs (1598 lines → 5 modules)

### Current Structure Analysis

- `AlignmentRegion` struct - Deferred CIGAR representation (~150 lines)
- `RegionScore` and score types (~50 lines)
- `extend_chains_to_regions()` - Main extension dispatch (~350 lines)
- `generate_cigar_from_region()` - CIGAR regeneration (~200 lines)
- `merge_scores_to_regions()` - Score merging (~150 lines)
- SIMD scoring helpers, band width inference (~200 lines)
- Coordinate conversion utilities (~100 lines)
- Tests (~250 lines)

### Proposed Split

```
src/pipelines/linear/region/
├── mod.rs              (~100 lines) - Re-exports
├── types.rs            (~200 lines) - AlignmentRegion, RegionScore, enums
├── extension.rs        (~400 lines) - extend_chains_to_regions, dispatch
├── cigar.rs            (~250 lines) - generate_cigar_from_region
├── scoring.rs          (~300 lines) - merge_scores_to_regions, SIMD helpers
└── coordinates.rs      (~150 lines) - Coordinate conversion, band inference
```

### Dependencies

```
mod.rs
  ├── types.rs (core structs)
  ├── extension.rs
  │     ├── types.rs
  │     ├── scoring.rs
  │     └── coordinates.rs
  ├── cigar.rs
  │     ├── types.rs
  │     └── coordinates.rs
  ├── scoring.rs
  │     └── types.rs
  └── coordinates.rs
        └── types.rs
```

### Implementation Order

1. Create `region/` directory and `mod.rs`
2. Extract `types.rs` (AlignmentRegion, RegionScore)
3. Extract `coordinates.rs` (utility functions)
4. Extract `scoring.rs` (SIMD helpers, merge)
5. Extract `cigar.rs` (CIGAR generation)
6. Extract `extension.rs` (main dispatch)
7. Update imports throughout pipeline

---

## 4. chaining.rs (1116 lines → 4 modules)

### Current Structure Analysis

- `Chain` struct (~80 lines)
- `SoAChainBatch` struct (~100 lines)
- `chain_seeds()` - AoS B-tree chaining (~200 lines)
- `chain_seeds_batch()` - SoA batch chaining (~150 lines)
- `filter_chains()` / `filter_chains_batch()` - Chain filtering (~250 lines)
- `calculate_chain_weight()` / `_soa()` - Weight calculation (~100 lines)
- `test_and_merge()` / `_soa()` - Merge logic (~100 lines)
- Tests (~150 lines)

### Proposed Split

```
src/pipelines/linear/chaining/
├── mod.rs              (~100 lines) - Re-exports, ChainingStage wrapper
├── types.rs            (~200 lines) - Chain, SoAChainBatch structs
├── btree.rs            (~300 lines) - B-tree chaining algorithm (AoS + SoA)
├── filter.rs           (~300 lines) - filter_chains, filter_chains_batch
└── weight.rs           (~200 lines) - Weight calculation, test_and_merge
```

### Dependencies

```
mod.rs
  ├── types.rs (core structs)
  ├── btree.rs
  │     ├── types.rs
  │     └── weight.rs
  ├── filter.rs
  │     ├── types.rs
  │     └── weight.rs
  └── weight.rs
        └── types.rs
```

### Implementation Order

1. Create `chaining/` directory and `mod.rs`
2. Extract `types.rs` (Chain, SoAChainBatch)
3. Extract `weight.rs` (weight calculation, test_and_merge)
4. Extract `btree.rs` (B-tree chaining)
5. Extract `filter.rs` (chain filtering)
6. Update imports in `stages/chaining/mod.rs`

---

## Implementation Strategy

### Phase 1: Low-Risk Splits (chaining, seeding types)

Start with modules that have:
- Clear boundaries
- Minimal external dependencies
- Good test coverage

**Order**:
1. `chaining/` split (1116 lines, clearest boundaries)
2. `seeding/types.rs` and `seeding/soa_batch.rs` extraction

### Phase 2: Medium-Risk Splits (region, seeding algorithms)

**Order**:
3. `region/` split (coordinate and CIGAR generation are self-contained)
4. Complete `seeding/` split (smem algorithm is complex)

### Phase 3: High-Risk Splits (finalization)

The `finalization.rs` split is highest risk because:
- `Alignment` struct is used throughout the codebase
- Complex interdependencies with SAM output
- Secondary marking affects pairing correctness

**Order**:
5. Extract `finalization/alignment.rs` first (most critical)
6. Extract smaller modules (mapq, tags)
7. Extract complex modules (redundancy, secondary)

---

## Validation Checklist

For each split:

- [ ] `cargo build --release` succeeds
- [ ] `cargo test --lib` passes
- [ ] `cargo test --test '*'` passes (integration tests)
- [ ] Line counts verified: `wc -l src/pipelines/linear/<module>/*.rs`
- [ ] No new warnings from `cargo clippy`
- [ ] `cargo fmt` applied

### Regression Testing

After all splits complete:

```bash
# Golden test (must produce identical output)
./target/release/ferrous-align mem $REF tests/golden_reads/golden_10k_R1.fq tests/golden_reads/golden_10k_R2.fq > /tmp/split_test.sam
diff /tmp/split_test.sam tests/golden_reads/expected.sam

# Performance (should not regress >5%)
time ./target/release/ferrous-align mem -t 16 $REF tests/golden_reads/golden_10k_R1.fq tests/golden_reads/golden_10k_R2.fq > /dev/null
```

---

## Risk Assessment

| Module | Risk | Rationale |
|--------|------|-----------|
| `chaining/` | Low | Clear boundaries, isolated from SAM output |
| `seeding/` | Medium | SMEM algorithm is complex, but well-tested |
| `region/` | Medium | CIGAR generation critical for correctness |
| `finalization/` | High | `Alignment` struct used everywhere, SAM output |

---

## Timeline Estimate

**Note**: No time estimates per project guidelines. Work breakdown:

1. **chaining/** split: ~4 focused work sessions
2. **seeding/** split: ~5 focused work sessions
3. **region/** split: ~5 focused work sessions
4. **finalization/** split: ~6 focused work sessions

**Total**: ~20 work sessions + integration testing

---

## Post-Split File Size Summary

| Module | Est. Lines | Status |
|--------|------------|--------|
| `chaining/mod.rs` | ~100 | Pending |
| `chaining/types.rs` | ~200 | Pending |
| `chaining/btree.rs` | ~300 | Pending |
| `chaining/filter.rs` | ~300 | Pending |
| `chaining/weight.rs` | ~200 | Pending |
| `seeding/mod.rs` | ~120 | Pending |
| `seeding/types.rs` | ~200 | Pending |
| `seeding/soa_batch.rs` | ~250 | Pending |
| `seeding/smem.rs` | ~450 | Pending |
| `seeding/bidirectional.rs` | ~250 | Pending |
| `seeding/collection.rs` | ~350 | Pending |
| `region/mod.rs` | ~100 | Pending |
| `region/types.rs` | ~200 | Pending |
| `region/extension.rs` | ~400 | Pending |
| `region/cigar.rs` | ~250 | Pending |
| `region/scoring.rs` | ~300 | Pending |
| `region/coordinates.rs` | ~150 | Pending |
| `finalization/mod.rs` | ~100 | Pending |
| `finalization/alignment.rs` | ~250 | Pending |
| `finalization/sam_flags.rs` | ~30 | Pending |
| `finalization/redundancy.rs` | ~350 | Pending |
| `finalization/secondary.rs` | ~350 | Pending |
| `finalization/mapq.rs` | ~150 | Pending |
| `finalization/tags.rs` | ~250 | Pending |

**All modules ≤500 lines** ✓
