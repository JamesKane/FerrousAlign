# Architecture Documentation Index

This directory contains design documents for FerrousAlign's architecture refactoring. These documents are interconnected and should be read in sequence for a complete understanding.

---

## Document Overview

### 1. Pipeline Flow Diagram (`Pipeline_Flow_Diagram.md`)

**Purpose**: Comprehensive visualization of the current pipeline from disk to SAM output

**Key Content**:
- High-level architecture overview
- Detailed flow diagrams for single-end and paired-end modes
- Per-read alignment pipeline breakdown (seeding → chaining → extension → finalization)
- Data structure transformations at each stage
- Performance-critical path analysis
- Decision points and branching logic

**Who should read this**: Anyone wanting to understand how data flows through the system

**Status**: ✅ Complete (reflects v0.7.0-alpha)

---

### 2. Main Loop Abstraction Proposal (`Main_Loop_Abstraction_Proposal.md`)

**Purpose**: Design proposal for refactoring the pipeline into clean, modular abstractions

**Key Content**:
- Problems with current architecture (code duplication, mixed concerns)
- Proposed trait-based abstractions (`PipelineStage`, `PipelineOrchestrator`)
- Concrete implementations for single-end and paired-end modes
- Benefits: testability, modularity, extensibility
- Migration strategy (backwards compatible)
- Performance considerations (zero-cost abstractions)

**Who should read this**: Developers planning the refactoring, reviewers evaluating the approach

**Status**: ✅ Complete (proposal for v0.8.0)

---

### 3. Module Reorganization Plan (`Module_Reorganization_Plan.md`)

**Purpose**: Concrete, actionable plan for executing the refactoring

**Key Content**:
- File-by-file refactoring steps
- Current vs target module structure
- Phase-by-phase implementation roadmap (6 weeks)
- File splitting strategy for large files (seeding.rs, finalization.rs, etc.)
- Validation and testing requirements
- Rollback plan and success metrics

**Who should read this**: Developers implementing the refactoring

**Status**: ✅ Complete (planning for v0.8.0)

---

## Reading Order

### For Understanding Current Architecture
1. Start with `Pipeline_Flow_Diagram.md`
2. Focus on the "High-Level Architecture" and "Single-End Pipeline" sections
3. Trace a single read through the system

### For Planning the Refactoring
1. Read `Pipeline_Flow_Diagram.md` (understand current state)
2. Read `Main_Loop_Abstraction_Proposal.md` (understand target state)
3. Read `Module_Reorganization_Plan.md` (understand migration path)

### For Implementing the Refactoring
1. Use `Module_Reorganization_Plan.md` as your checklist
2. Reference `Main_Loop_Abstraction_Proposal.md` for design details
3. Use `Pipeline_Flow_Diagram.md` to understand what each stage does

---

## Quick Reference

### Key Abstractions

```rust
// Core trait for pipeline stages
trait PipelineStage<In, Out> {
    fn process(&self, input: In, ctx: &StageContext) -> Result<Out, StageError>;
    fn name(&self) -> &str;
}

// High-level orchestrator
trait PipelineOrchestrator {
    fn run(&mut self, files: &[PathBuf], output: &mut dyn Write)
        -> Result<PipelineStatistics>;
    fn mode(&self) -> PipelineMode;
}
```

### Pipeline Stages

| Stage | Input | Output | Purpose |
|-------|-------|--------|---------|
| Loading | FASTQ file | `SoAReadBatch` | Read batches from disk |
| Seeding | `SoAReadBatch` | `SoASeedBatch` | SMEM extraction via FM-Index |
| Chaining | `SoASeedBatch` | `SoAChainBatch` | O(n²) DP chaining |
| Extension | `SoAChainBatch` | `Vec<AlignmentRegion>` | Smith-Waterman alignment |
| Finalization | `Vec<AlignmentRegion>` | `Vec<Alignment>` | CIGAR, MD, NM, MAPQ |

### Module Structure (Target v0.8.0)

```
src/pipelines/linear/
├── orchestrator/          # Main loop coordination
│   ├── single_end.rs      # Single-end orchestrator
│   └── paired_end.rs      # Paired-end orchestrator
│
├── stages/                # Pipeline stages
│   ├── seeding/           # SMEM extraction (split from 1902 lines)
│   ├── chaining/          # Seed chaining (split from 823 lines)
│   ├── extension/         # SW alignment (split from 1598 lines)
│   └── finalization/      # CIGAR/MD/NM (split from 1707 lines)
│
└── modes/                 # Mode-specific logic
    ├── single_end.rs      # Selection logic
    └── paired_end/        # Pairing + mate rescue
```

---

## Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Foundation | Weeks 1-2 | Trait definitions, module skeleton |
| File Splitting | Weeks 2-3 | Split large files into logical modules |
| Orchestrators | Week 4 | Implement single-end and paired-end orchestrators |
| Integration | Week 5 | Update entry points, deprecate old API |
| Validation | Week 6 | Golden dataset testing, benchmarking |

**Total**: 6 weeks from start to merge

---

## Success Criteria

### Code Quality
- ✅ All files <500 lines (except trait definitions)
- ✅ Clear module boundaries
- ✅ 100% test coverage for orchestrator logic
- ✅ No clippy warnings

### Functionality
- ✅ Bit-for-bit identical output on golden dataset
- ✅ All existing tests pass
- ✅ No new bugs in first 2 weeks post-release

### Performance
- ✅ No regression on 10K read benchmark
- ✅ <5% regression on 4M read benchmark
- ✅ Memory usage unchanged (±10%)

---

## Related Documents

### Existing Architecture Docs
- `RedesignStrategy.md` - SIMD kernel unification plan
- `SOA_End_to_End.md` - SoA pipeline design (completed in v0.7.0, **updated for hybrid architecture**)
- `SOA_Transition.md` - SoA migration checklist

### Critical Architectural Discovery (v0.7.0)
**Hybrid AoS/SoA Architecture**: During paired-end integration, we discovered that **pure SoA pairing has a fundamental indexing bug** that causes 96% duplicate reads. The solution is a **hybrid architecture**:
- **SoA for alignment & mate rescue** (SIMD batching benefits)
- **AoS for pairing & output** (correct per-read indexing)

See `dev_notes/HYBRID_AOS_SOA_STATUS.md` for details. This impacts future pipeline designs.

### Future Roadmaps
- `ARM_SVE_SME_Roadmap.md` - ARM SIMD support
- `RISCV_RVV_Roadmap.md` - RISC-V Vector support
- `Metal_GPU_Acceleration_Design.md` - GPU acceleration
- `NPU_Seed_Filter_Design.md` - NPU pre-filtering
- `Learned_Index_SA_Lookup_Design.md` - Sapling-style SA acceleration

---

## Contributing

### Proposing Changes to These Docs
1. Create a new branch
2. Update the relevant document
3. Update this README if adding new docs
4. Submit PR with clear rationale

### Reporting Issues
If you find inaccuracies or outdated information:
1. File an issue with `[docs]` prefix
2. Reference specific document and section
3. Suggest correction if possible

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | 2025-12-02 | Initial creation: Pipeline flow, abstraction proposal, reorganization plan |

---

## Contact

For questions about these documents, see:
- `CLAUDE.md` - Project overview and development guidelines
- GitHub Issues - For bug reports and feature requests
