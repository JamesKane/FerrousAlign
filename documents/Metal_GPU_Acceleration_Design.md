# Metal GPU Acceleration for Smith-Waterman Alignment

## Overview

This document outlines the high-level design for accelerating Smith-Waterman (SW) alignment on Apple Silicon GPUs using the Metal compute API. This is targeted for **post-1.x** releases.

**Scope**: This document focuses on the Metal implementation. The architecture is designed to support CUDA and ROCm backends in the future, but those implementations are **out of scope** for this document.

## Executive Summary

| Aspect | Specification |
|--------|---------------|
| **Target Hardware** | Apple Silicon (M1/M2/M3/M4) |
| **API** | Metal 3 via `metal-rs` crate |
| **Memory Model** | Unified Memory (zero-copy) |
| **Parallelism Strategy** | Horizontal batching (inter-alignment) |
| **Batch Threshold** | >= 1024 alignments (amortize dispatch overhead) |
| **Expected Speedup** | 2-5x over NEON for large batches |
| **Priority** | Post-1.x (after SVE roadmap) |

## Multi-Backend GPU Architecture

### Design Goals

1. **Metal first**: Apple Silicon is the primary development platform
2. **Backend-agnostic trait**: Define a `GpuBackend` trait that Metal implements
3. **Future-proof**: CUDA/ROCm can implement the same trait later
4. **Minimal coupling**: Backend-specific code isolated in feature-gated modules

### Backend Comparison

| Aspect | Metal | CUDA | ROCm |
|--------|-------|------|------|
| **Scope** | In scope | Out of scope | Out of scope |
| **Memory Model** | Unified (zero-copy) | Discrete (explicit copy) | Discrete (explicit copy) |
| **Shader Language** | MSL | PTX/CUDA C | HIP |
| **Rust Crate** | `metal-rs` | `cuda-rs` / `cudarc` | `hip-rs` / `rocm-rs` |
| **Buffer Strategy** | `StorageModeShared` | Pinned + async copy | Pinned + async copy |

### GpuBackend Trait

The key abstraction enabling future CUDA/ROCm support:

```rust
// src/core/compute/gpu/mod.rs

/// Backend-agnostic GPU context trait.
///
/// Implementations:
/// - MetalBackend (in scope)
/// - CudaBackend (out of scope - future)
/// - RocmBackend (out of scope - future)
pub trait GpuBackend: Send + Sync {
    /// Backend identifier for logging/debugging
    fn name(&self) -> &'static str;

    /// Check if this backend is available on the current system
    fn is_available() -> bool where Self: Sized;

    /// Maximum batch size this backend can handle efficiently
    fn max_batch_size(&self) -> usize;

    /// Minimum batch size to amortize dispatch overhead
    fn min_batch_size(&self) -> usize;

    /// Execute Smith-Waterman alignment on a batch of jobs.
    ///
    /// # Arguments
    /// * `batch` - Pre-transposed SoA batch (see `GpuSwBatch`)
    /// * `config` - Alignment parameters (gaps, scoring matrix)
    ///
    /// # Returns
    /// * Scores and end positions for each alignment
    fn execute_sw_batch(
        &self,
        batch: &GpuSwBatch,
        config: &GpuSwConfig,
    ) -> GpuSwResults;

    /// Synchronize and wait for all pending GPU work to complete
    fn synchronize(&self);
}

/// Input batch in GPU-friendly SoA layout.
///
/// This struct is backend-agnostic. Each backend implementation
/// handles the actual buffer allocation and data transfer.
#[derive(Debug)]
pub struct GpuSwBatch<'a> {
    /// Query sequences in SoA layout: `queries[pos * batch_size + lane]`
    pub queries: &'a [u8],
    /// Target sequences in SoA layout: `targets[pos * batch_size + lane]`
    pub targets: &'a [u8],
    /// Per-alignment query lengths
    pub query_lens: &'a [i16],
    /// Per-alignment target lengths
    pub target_lens: &'a [i16],
    /// Number of alignments in batch
    pub batch_size: usize,
    /// Maximum query length (for buffer sizing)
    pub max_query_len: usize,
    /// Maximum target length (for buffer sizing)
    pub max_target_len: usize,
}

/// Alignment configuration (backend-agnostic).
#[derive(Debug, Clone)]
pub struct GpuSwConfig {
    pub gap_open: i32,
    pub gap_extend: i32,
    pub band_width: i32,
    /// 5x5 scoring matrix (A,C,G,T,N)
    pub scoring_matrix: [i8; 25],
}

/// Results from GPU alignment batch.
#[derive(Debug)]
pub struct GpuSwResults {
    /// Alignment scores
    pub scores: Vec<i32>,
    /// Query end positions (for traceback)
    pub query_ends: Vec<i32>,
    /// Target end positions (for traceback)
    pub target_ends: Vec<i32>,
}
```

### Module Structure

```
src/core/compute/
├── mod.rs                    # ComputeBackend enum
├── gpu/                      # NEW: GPU backend abstraction
│   ├── mod.rs                # GpuBackend trait, GpuSwBatch, GpuSwResults
│   ├── types.rs              # Shared types (GpuSwConfig, etc.)
│   └── dispatch.rs           # Backend selection logic
├── metal/                    # Metal implementation (IN SCOPE)
│   ├── mod.rs                # MetalBackend impl GpuBackend
│   ├── context.rs            # Device, queue, pipeline management
│   ├── buffers.rs            # Unified memory buffer pool
│   └── shaders/
│       └── sw_kernel.metal
├── cuda/                     # CUDA implementation (OUT OF SCOPE)
│   └── mod.rs                # Placeholder: CudaBackend impl GpuBackend
├── rocm/                     # ROCm implementation (OUT OF SCOPE)
│   └── mod.rs                # Placeholder: RocmBackend impl GpuBackend
├── encoding.rs
└── simd_abstraction/
```

### ComputeBackend Enum (Updated)

```rust
// src/core/compute/mod.rs

pub enum ComputeBackend {
    /// CPU SIMD (always available)
    CpuSimd(SimdEngineType),

    /// GPU acceleration via backend-agnostic trait
    #[cfg(feature = "gpu")]
    Gpu(Arc<dyn GpuBackend>),

    /// NPU for seed pre-filtering (future)
    Npu,
}

impl ComputeBackend {
    /// Detect optimal backend, preferring GPU when available and beneficial
    pub fn detect_optimal() -> Self {
        #[cfg(feature = "gpu")]
        {
            if let Some(gpu) = detect_gpu_backend() {
                return ComputeBackend::Gpu(gpu);
            }
        }

        ComputeBackend::CpuSimd(simd::detect_optimal_simd_engine())
    }
}

#[cfg(feature = "gpu")]
fn detect_gpu_backend() -> Option<Arc<dyn GpuBackend>> {
    // Priority order: Metal > CUDA > ROCm

    #[cfg(all(target_os = "macos", feature = "metal"))]
    if metal::MetalBackend::is_available() {
        return Some(Arc::new(metal::MetalBackend::new()?));
    }

    #[cfg(feature = "cuda")]
    if cuda::CudaBackend::is_available() {
        return Some(Arc::new(cuda::CudaBackend::new()?));
    }

    #[cfg(feature = "rocm")]
    if rocm::RocmBackend::is_available() {
        return Some(Arc::new(rocm::RocmBackend::new()?));
    }

    None
}
```

### Feature Flags

```toml
# Cargo.toml

[features]
# GPU umbrella feature
gpu = []

# Backend-specific features
metal = ["gpu", "metal-rs"]
cuda = ["gpu"]      # Add cuda-rs or cudarc when implemented
rocm = ["gpu"]      # Add hip-rs when implemented

[target.'cfg(target_os = "macos")'.dependencies]
metal-rs = { version = "0.29", optional = true }

# Future (out of scope):
# [target.'cfg(target_os = "linux")'.dependencies]
# cudarc = { version = "0.10", optional = true }
```

### Dispatch Integration

```rust
// src/pipelines/linear/batch_extension/dispatch.rs

pub fn dispatch_alignment_batch(
    ctx: &ComputeContext,
    jobs: &[AlignJob],
    config: &KernelConfig,
) -> Vec<BatchExtensionResult> {
    match &ctx.backend {
        ComputeBackend::CpuSimd(engine) => {
            // Existing SIMD path
            dispatch_cpu_simd(*engine, jobs, config)
        }

        #[cfg(feature = "gpu")]
        ComputeBackend::Gpu(backend) => {
            // Check batch size threshold
            if jobs.len() >= backend.min_batch_size() {
                dispatch_gpu(backend.as_ref(), jobs, config)
            } else {
                // Fall back to CPU for small batches
                let engine = simd::detect_optimal_simd_engine();
                dispatch_cpu_simd(engine, jobs, config)
            }
        }

        ComputeBackend::Npu => {
            // Fallback (NPU not implemented)
            let engine = simd::detect_optimal_simd_engine();
            dispatch_cpu_simd(engine, jobs, config)
        }
    }
}

#[cfg(feature = "gpu")]
fn dispatch_gpu(
    backend: &dyn GpuBackend,
    jobs: &[AlignJob],
    config: &KernelConfig,
) -> Vec<BatchExtensionResult> {
    // 1. Transpose jobs to SoA
    let batch = transpose_to_gpu_batch(jobs);

    // 2. Convert config
    let gpu_config = GpuSwConfig::from(config);

    // 3. Execute on GPU
    let results = backend.execute_sw_batch(&batch, &gpu_config);

    // 4. Convert results back
    convert_gpu_results(jobs, results)
}
```

### CUDA/ROCm Placeholder (Out of Scope)

For future contributors implementing CUDA or ROCm:

```rust
// src/core/compute/cuda/mod.rs
// STATUS: OUT OF SCOPE - Placeholder for future implementation

#[cfg(feature = "cuda")]
pub struct CudaBackend {
    // device: cuda::Device,
    // stream: cuda::Stream,
    // module: cuda::Module,
}

#[cfg(feature = "cuda")]
impl GpuBackend for CudaBackend {
    fn name(&self) -> &'static str { "CUDA" }

    fn is_available() -> bool {
        // TODO: Check for CUDA device
        // cuda::Device::count() > 0
        false
    }

    fn max_batch_size(&self) -> usize { 65536 }
    fn min_batch_size(&self) -> usize { 1024 }

    fn execute_sw_batch(&self, batch: &GpuSwBatch, config: &GpuSwConfig) -> GpuSwResults {
        // TODO: Implement CUDA kernel dispatch
        // Key differences from Metal:
        // 1. Explicit cudaMemcpy for data transfer
        // 2. Async streams for overlap
        // 3. PTX or CUDA C kernel
        unimplemented!("CUDA backend not yet implemented")
    }

    fn synchronize(&self) {
        // TODO: cudaDeviceSynchronize()
    }
}
```

### Memory Model Differences

Future CUDA/ROCm implementers should note:

```
Metal (Unified Memory)              CUDA/ROCm (Discrete Memory)
───────────────────────             ──────────────────────────

CPU Memory ─────────────┐           CPU Memory (Host)
          │             │                │
          │ (same RAM)  │           cudaMemcpyAsync()
          │             │                │
          ▼             │                ▼
GPU Access ─────────────┘           GPU Memory (Device)
                                         │
No explicit copy needed             Explicit copy required
StorageModeShared                   Pinned memory + async streams
```

For CUDA/ROCm, the `execute_sw_batch` implementation must:
1. Allocate device buffers (or use buffer pool)
2. Copy input data to device (async)
3. Launch kernel
4. Copy results back (async)
5. Synchronize

The `GpuSwBatch` struct provides CPU-side SoA data; backends handle transfer.

## Why Metal on Apple Silicon?

### Unified Memory Advantage

Unlike discrete GPUs (NVIDIA/AMD), Apple Silicon has **unified memory** where CPU and GPU share the same physical RAM. This eliminates:
- Explicit `cudaMemcpy()` / `clEnqueueWriteBuffer()` transfers
- Double-buffering overhead
- PCIe bandwidth bottleneck

Metal buffers created with `StorageModeShared` allow direct access from both CPU and GPU without synchronization overhead for read-only data.

### Apple Silicon GPU Characteristics

| Chip | GPU Cores | Memory BW | Theoretical FP32 | Notes |
|------|-----------|-----------|------------------|-------|
| M1 | 8 | 68 GB/s | 2.6 TFLOPS | Baseline |
| M1 Pro/Max | 16-32 | 200-400 GB/s | 5.2-10.4 TFLOPS | |
| M2 | 10 | 100 GB/s | 3.6 TFLOPS | |
| M3 | 10 | 100 GB/s | 4.1 TFLOPS | Dynamic caching |
| M3 Max | 40 | 400 GB/s | 16.4 TFLOPS | |
| M4 | 10 | 120 GB/s | ~4.5 TFLOPS | SME2 support |

## Parallelism Strategy

### Why Horizontal (Inter-Alignment) Batching

Smith-Waterman has **diagonal wavefront dependencies** that limit intra-alignment parallelism:

```
H[i,j] = max(H[i-1,j-1] + score, H[i-1,j] - gap, H[i,j-1] - gap)
```

Two parallelism strategies exist:

1. **Intra-alignment (anti-diagonal wavefront)**: Parallelize cells on same anti-diagonal
   - Poor GPU utilization (varying parallelism per diagonal)
   - High synchronization overhead between diagonals
   - Complex memory access patterns

2. **Inter-alignment (horizontal batching)**: Each thread handles one complete alignment
   - Uniform workload per thread
   - No inter-thread synchronization
   - Coalesced memory access when sequences are SoA-transposed
   - **This is the recommended approach**

### Batch Size Economics

| Batch Size | GPU Dispatch Overhead | SW Kernel Time | Overhead % |
|------------|----------------------|----------------|------------|
| 64 | ~30 μs | ~64 μs | 47% |
| 256 | ~30 μs | ~256 μs | 12% |
| 1024 | ~30 μs | ~1024 μs | 3% |
| 4096 | ~30 μs | ~4096 μs | <1% |

**Threshold**: GPU acceleration only beneficial when batch >= 1024 alignments.

## Metal Backend Implementation

This section covers the Metal-specific implementation details. The `GpuBackend` trait (defined above) provides the interface.

### MetalBackend Structure

```rust
// src/core/compute/metal/mod.rs
use metal::{Device, CommandQueue, Library, ComputePipelineState};
use super::gpu::{GpuBackend, GpuSwBatch, GpuSwConfig, GpuSwResults};

pub struct MetalBackend {
    device: Device,
    command_queue: CommandQueue,
    sw_pipeline: ComputePipelineState,
    buffer_pool: BufferPool,
}

impl MetalBackend {
    pub fn new() -> Option<Self> {
        let device = Device::system_default()?;
        let command_queue = device.new_command_queue();

        // Compile shaders at initialization
        let library = device.new_library_with_source(
            include_str!("shaders/sw_kernel.metal"),
            &metal::CompileOptions::new(),
        ).ok()?;

        let sw_function = library.get_function("smith_waterman_batch", None).ok()?;
        let sw_pipeline = device.new_compute_pipeline_state_with_function(&sw_function).ok()?;

        Some(Self {
            device,
            command_queue,
            sw_pipeline,
            buffer_pool: BufferPool::new(),
        })
    }
}

impl GpuBackend for MetalBackend {
    fn name(&self) -> &'static str { "Metal" }

    fn is_available() -> bool {
        Device::system_default().is_some()
    }

    fn max_batch_size(&self) -> usize { 16384 }
    fn min_batch_size(&self) -> usize { 1024 }

    fn execute_sw_batch(&self, batch: &GpuSwBatch, config: &GpuSwConfig) -> GpuSwResults {
        // See implementation below
        self.execute_batch_internal(batch, config)
    }

    fn synchronize(&self) {
        // Metal uses command buffer completion; no global sync needed
    }
}
```

### Buffer Management (Unified Memory)

```rust
// src/core/compute/metal/buffers.rs
use metal::{Buffer, MTLResourceOptions};

pub struct BufferPool {
    // Pre-allocated buffers for common sizes
    query_buffers: Vec<Buffer>,
    target_buffers: Vec<Buffer>,
    score_buffers: Vec<Buffer>,
    // Reuse tracking
    available: Vec<usize>,
}

impl BufferPool {
    /// Allocate buffer in unified memory (zero-copy)
    pub fn allocate(&mut self, device: &Device, size: usize) -> Buffer {
        device.new_buffer(
            size as u64,
            MTLResourceOptions::StorageModeShared,  // Unified memory!
        )
    }

    /// Get or create buffer for batch
    pub fn get_query_buffer(&mut self, device: &Device, batch_size: usize, max_len: usize) -> &Buffer {
        let required_size = batch_size * max_len;
        // Return existing or allocate new
        // ...
    }
}
```

## Metal Shader Design

### Smith-Waterman Compute Kernel

```metal
// src/core/compute/metal/shaders/sw_kernel.metal
#include <metal_stdlib>
using namespace metal;

// Scoring matrix (5x5 for A,C,G,T,N)
constant int8_t SCORING_MATRIX[25] = {
    // A   C   G   T   N
       1, -1, -1, -1,  0,  // A
      -1,  1, -1, -1,  0,  // C
      -1, -1,  1, -1,  0,  // G
      -1, -1, -1,  1,  0,  // T
       0,  0,  0,  0,  0   // N
};

struct AlignmentParams {
    int gap_open;
    int gap_extend;
    int band_width;
    int max_qlen;
    int max_tlen;
};

kernel void smith_waterman_batch(
    device const uint8_t* queries      [[buffer(0)]],  // SoA: batch_size * max_qlen
    device const uint8_t* targets      [[buffer(1)]],  // SoA: batch_size * max_tlen
    device const int16_t* qlens        [[buffer(2)]],  // Per-alignment query lengths
    device const int16_t* tlens        [[buffer(3)]],  // Per-alignment target lengths
    device int32_t* scores             [[buffer(4)]],  // Output scores
    device int32_t* query_ends         [[buffer(5)]],  // Output: query end position
    device int32_t* target_ends        [[buffer(6)]],  // Output: target end position
    constant AlignmentParams& params   [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    // Each thread handles one complete alignment
    int qlen = qlens[tid];
    int tlen = tlens[tid];

    // Pointers to this alignment's sequences
    device const uint8_t* query = queries + tid;
    device const uint8_t* target = targets + tid;

    // Thread-local DP storage (register pressure consideration)
    // For banded SW, only need 2 * band_width values
    int16_t H_prev[256];  // Adjust based on band_width
    int16_t H_curr[256];
    int16_t E[256];

    int max_score = 0;
    int max_i = 0, max_j = 0;

    // Standard banded Smith-Waterman DP
    for (int i = 0; i < qlen; i++) {
        // Determine band boundaries
        int j_start = max(0, i - params.band_width);
        int j_end = min(tlen, i + params.band_width + 1);

        uint8_t q_base = query[i * /* stride */];

        for (int j = j_start; j < j_end; j++) {
            uint8_t t_base = target[j * /* stride */];

            // Score lookup
            int match_score = SCORING_MATRIX[q_base * 5 + t_base];

            // DP recurrence
            int diag = H_prev[j-1] + match_score;
            int up = E[j];  // Already computed
            int left = H_curr[j-1] - params.gap_open - params.gap_extend;

            int h = max(0, max(diag, max(up, left)));
            H_curr[j] = h;

            // Update E for next row
            E[j] = max(H_curr[j] - params.gap_open, E[j] - params.gap_extend);

            // Track maximum
            if (h > max_score) {
                max_score = h;
                max_i = i;
                max_j = j;
            }
        }

        // Swap rows
        for (int j = j_start; j < j_end; j++) {
            H_prev[j] = H_curr[j];
        }
    }

    scores[tid] = max_score;
    query_ends[tid] = max_i;
    target_ends[tid] = max_j;
}
```

### Thread Group Configuration

```rust
// src/core/compute/metal/dispatch.rs
use metal::{MTLSize, ComputeCommandEncoder};

pub fn dispatch_sw_batch(
    encoder: &ComputeCommandEncoder,
    pipeline: &ComputePipelineState,
    batch_size: usize,
) {
    // One thread per alignment
    let threads_per_group = 256;  // Optimal for Apple Silicon
    let thread_groups = (batch_size + threads_per_group - 1) / threads_per_group;

    encoder.set_compute_pipeline_state(pipeline);
    encoder.dispatch_thread_groups(
        MTLSize::new(thread_groups as u64, 1, 1),
        MTLSize::new(threads_per_group as u64, 1, 1),
    );
}
```

## Data Flow

### SoA Layout for GPU

The existing SoA infrastructure (`SwSoA`, `KswSoA`) maps directly to Metal buffer layout:

```
CPU SoA Layout                    Metal Buffer Layout
─────────────────                 ──────────────────
query_soa: [u8]                   buffer(0): queries
  pos0: [q0, q1, q2, ...]           Same layout, zero-copy
  pos1: [q0, q1, q2, ...]
  ...

target_soa: [u8]                  buffer(1): targets
  pos0: [t0, t1, t2, ...]           Same layout, zero-copy
  ...
```

### Execution Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│                        GPU Execution Flow                             │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  1. Collect batch (CPU)                                              │
│     ├── AlignJob[] from extension pipeline                           │
│     └── Check batch_size >= gpu_batch_threshold                      │
│                                                                       │
│  2. Prepare SoA (CPU)                                                │
│     ├── Transpose AoS → SoA (existing infrastructure)               │
│     └── Write directly to Metal shared buffers (zero-copy)          │
│                                                                       │
│  3. Encode commands (CPU)                                            │
│     ├── Create command buffer                                        │
│     ├── Set pipeline state                                           │
│     ├── Bind buffers                                                 │
│     └── Dispatch thread groups                                       │
│                                                                       │
│  4. Execute (GPU)                                                    │
│     ├── Each thread: one alignment                                   │
│     ├── Thread-local DP in registers                                 │
│     └── Write scores to output buffer                                │
│                                                                       │
│  5. Synchronize (CPU)                                                │
│     ├── command_buffer.wait_until_completed()                        │
│     └── Read scores from shared buffer (zero-copy)                   │
│                                                                       │
│  6. Continue pipeline (CPU)                                          │
│     └── CIGAR generation, finalization (remain on CPU)              │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

## CIGAR Generation Strategy

CIGAR string generation requires traceback through the DP matrix, which is:
1. Sequential (cannot parallelize)
2. Memory-intensive (need to store full DP matrix)
3. Divergent (different alignments have different traceback paths)

**Recommended Approach**: GPU computes scores only; CPU generates CIGARs.

### Score-Only GPU Mode

For initial implementation:
1. GPU: Compute alignment scores and end positions
2. CPU: Re-run SW with traceback for top-scoring alignments only
3. Rationale: Most alignments are filtered by score; only ~10-20% need CIGAR

### Future: GPU Traceback

If CIGAR generation becomes bottleneck:
- Store compressed traceback (2 bits per cell: diag/up/left/stop)
- GPU-parallel traceback using bit manipulation
- Complexity: High; defer to future optimization

## Feature Gating

Feature flags are defined in the Multi-Backend GPU Architecture section above. Summary:

```toml
[features]
gpu = []                              # Umbrella feature
metal = ["gpu", "metal-rs"]           # Apple Silicon (IN SCOPE)
cuda = ["gpu"]                        # NVIDIA (OUT OF SCOPE)
rocm = ["gpu"]                        # AMD (OUT OF SCOPE)
```

The `ComputeBackend::Gpu(Arc<dyn GpuBackend>)` variant uses dynamic dispatch to support any backend implementing the `GpuBackend` trait.

## Performance Considerations

### Memory Bandwidth

Apple Silicon GPUs are bandwidth-limited for many workloads:

| Operation | Bandwidth Required | M3 Max Available |
|-----------|-------------------|------------------|
| Read queries | batch × qlen bytes | 400 GB/s |
| Read targets | batch × tlen bytes | 400 GB/s |
| Read scoring matrix | 25 bytes (cached) | - |
| Write scores | batch × 4 bytes | 400 GB/s |

For 4096 alignments of 150bp reads:
- Total data: 4096 × 150 × 2 + 4096 × 12 = ~1.3 MB
- At 400 GB/s: ~3 μs for memory transfers
- Compute-bound, not memory-bound

### Register Pressure

Metal shader registers are limited. For banded SW:
- H_prev[band]: 2 × band_width bytes
- H_curr[band]: 2 × band_width bytes
- E[band]: 2 × band_width bytes
- Total: 6 × band_width bytes per thread

With band_width = 100: 600 bytes/thread (fits in thread registers)

### Occupancy

Apple Silicon GPUs achieve best occupancy with:
- Thread group size: 256 or 512
- Threads per SIMD group: 32
- Register usage: < 128 per thread

## Testing Strategy

### Unit Tests

```rust
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn test_metal_sw_score_matches_cpu() {
    let metal_ctx = MetalContext::new().expect("Metal unavailable");
    let jobs = generate_test_alignments(1024);

    // Run on GPU
    let gpu_scores = metal::execute_sw_batch(&metal_ctx, &jobs, &config);

    // Run on CPU (reference)
    let cpu_scores = simd_banded_swa_batch(&jobs, &config);

    // Compare
    for (gpu, cpu) in gpu_scores.iter().zip(cpu_scores.iter()) {
        assert_eq!(gpu.score, cpu.score, "Score mismatch");
    }
}
```

### Benchmarks

```rust
#[cfg(all(target_os = "macos", feature = "metal"))]
fn bench_metal_vs_neon(c: &mut Criterion) {
    let metal_ctx = MetalContext::new().unwrap();
    let jobs = generate_test_alignments(4096);

    c.bench_function("metal_sw_4096", |b| {
        b.iter(|| metal::execute_sw_batch(&metal_ctx, &jobs, &config))
    });

    c.bench_function("neon_sw_4096", |b| {
        b.iter(|| simd_banded_swa_dispatch_soa::<16>(&jobs, &config))
    });
}
```

## Implementation Phases

### Phase 0: GPU Abstraction Layer (1 week)
- [ ] Create `src/core/compute/gpu/mod.rs` with `GpuBackend` trait
- [ ] Define `GpuSwBatch`, `GpuSwConfig`, `GpuSwResults` types
- [ ] Add `gpu` umbrella feature to Cargo.toml
- [ ] Update `ComputeBackend` enum to use `Arc<dyn GpuBackend>`
- [ ] Implement `detect_gpu_backend()` dispatch logic

### Phase 1: Metal Foundation (2 weeks)
- [ ] Add `metal` feature flag to Cargo.toml
- [ ] Create `src/core/compute/metal/mod.rs` with `MetalBackend`
- [ ] Implement `GpuBackend` trait for `MetalBackend`
- [ ] Implement buffer pool with unified memory
- [ ] Basic shader compilation and pipeline setup
- [ ] Unit test: device detection and initialization

### Phase 2: Score-Only Kernel (2 weeks)
- [ ] Write `sw_kernel.metal` for score computation
- [ ] Implement SoA buffer binding
- [ ] Command buffer encoding and dispatch
- [ ] Unit test: score parity with CPU SIMD

### Phase 3: Integration (1 week)
- [ ] Update `batch_extension/dispatch.rs` for Metal routing
- [ ] Implement batch size threshold logic
- [ ] Add `--gpu` CLI flag for explicit GPU mode
- [ ] Integration test: full pipeline with Metal backend

### Phase 4: Optimization (1-2 weeks)
- [ ] Profile with Metal System Trace
- [ ] Tune thread group size
- [ ] Optimize memory access patterns
- [ ] Benchmark against NEON baseline

### Phase 5: CIGAR Support (Future)
- [ ] Evaluate traceback on GPU vs CPU
- [ ] Implement if beneficial
- [ ] Otherwise, hybrid GPU-score + CPU-traceback

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| **Register pressure** | Medium | Use banded SW with limited band_width |
| **Dispatch overhead** | Medium | Batch threshold >= 1024 |
| **CIGAR complexity** | High | Score-only GPU, traceback on CPU |
| **metal-rs instability** | Low | Pin crate version, test on each macOS update |
| **M1 vs M4 differences** | Low | Test on multiple chips; use lowest common denominator |

## References

- [metal-rs crate](https://crates.io/crates/metal)
- [Using Metal and Rust to make FFT even faster](https://blog.lambdaclass.com/using-metal-and-rust-to-make-fft-even-faster/)
- [Unlocking the Power of Metal GPU on MacBooks](https://akashicmarga.github.io/2024/10/22/Metal-GPU-Kernels.html)
- [CUDASW++4.0: GPU-based Smith-Waterman](https://pubmed.ncbi.nlm.nih.gov/39488701/)
- [A Review of Parallel Implementations for Smith-Waterman](https://pmc.ncbi.nlm.nih.gov/articles/PMC8419822/)
- [Wavefront Parallelism on GPUs](https://synergy.cs.vt.edu/pubs/papers/hou-wavefront-ipdps18.pdf)

## Conclusion

Metal GPU acceleration for Smith-Waterman on Apple Silicon is viable and promising:

1. **Unified memory** eliminates transfer overhead (key advantage over discrete GPUs)
2. **Horizontal batching** (one thread per alignment) provides uniform workload
3. **Score-only GPU + CPU traceback** is the pragmatic initial approach
4. **Batch threshold >= 1024** ensures dispatch overhead is amortized

The existing SoA infrastructure in FerrousAlign aligns well with Metal's buffer model, making integration relatively straightforward. Expected speedup of 2-5x over NEON for large batches makes this worthwhile for high-throughput scenarios (e.g., full WGS runs).

Priority: Post-1.x, after core stability and SVE support.

---

## Appendix: Notes for Future CUDA/ROCm Contributors

This section provides guidance for contributors who wish to implement CUDA or ROCm backends. These implementations are **out of scope** for this document but the architecture is designed to accommodate them.

### What You Need to Implement

1. **Backend struct** implementing `GpuBackend` trait (see `src/core/compute/gpu/mod.rs`)
2. **Device/context management** (CUDA context, ROCm/HIP device)
3. **Buffer pool** with pinned memory for async transfers
4. **Kernel** in CUDA C or HIP C (equivalent to `sw_kernel.metal`)
5. **Async stream management** for overlapping compute and transfer

### Key Differences from Metal

| Aspect | Metal | CUDA/ROCm |
|--------|-------|-----------|
| **Memory** | Unified (zero-copy) | Discrete (explicit copy) |
| **Buffer allocation** | `StorageModeShared` | `cudaMallocHost` + `cudaMalloc` |
| **Data transfer** | Not needed | `cudaMemcpyAsync` |
| **Synchronization** | `waitUntilCompleted()` | `cudaStreamSynchronize` |
| **Kernel dispatch** | `dispatchThreadgroups` | `kernel<<<blocks, threads>>>` |

### Suggested Implementation Order

1. **Device detection**: `CudaBackend::is_available()` using `cudaGetDeviceCount`
2. **Context creation**: Initialize device, create stream
3. **Buffer pool**: Pre-allocate device buffers, pinned host buffers
4. **Kernel port**: Translate `sw_kernel.metal` to CUDA C (mostly straightforward)
5. **Async pipeline**: Overlap H2D transfer, kernel, D2H transfer
6. **Testing**: Verify score parity with CPU SIMD

### Performance Considerations for Discrete GPUs

Unlike Metal's unified memory, CUDA/ROCm must account for PCIe transfer overhead:

```
PCIe 4.0 x16: ~25 GB/s bidirectional
PCIe 5.0 x16: ~50 GB/s bidirectional

For 4096 alignments of 150bp:
- Upload: ~1.2 MB → ~50 μs (PCIe 4.0)
- Download: ~50 KB → ~2 μs
- Kernel: ~1-2 ms

Transfer is ~5% of total time; acceptable for large batches.
```

**Recommendation**: Use double-buffering with async streams to overlap transfer with compute.

### Rust Crate Options

| Backend | Crate | Notes |
|---------|-------|-------|
| CUDA | [`cudarc`](https://crates.io/crates/cudarc) | Modern, safe bindings |
| CUDA | [`cuda-sys`](https://crates.io/crates/cuda-sys) | Low-level FFI |
| ROCm | [`hip-sys`](https://crates.io/crates/hip-sys) | HIP FFI bindings |

### File Structure for CUDA Backend

```
src/core/compute/cuda/
├── mod.rs          # CudaBackend impl GpuBackend
├── context.rs      # Device, stream management
├── buffers.rs      # Pinned + device buffer pool
├── kernels/
│   └── sw_kernel.cu  # CUDA C kernel (compile with nvcc)
└── ffi.rs          # Optional: raw CUDA API bindings
```

### Getting Started

1. Fork FerrousAlign
2. Add `cuda = ["gpu", "cudarc"]` to `Cargo.toml` features
3. Create `src/core/compute/cuda/mod.rs` with `CudaBackend` struct
4. Implement `GpuBackend` trait methods
5. Port `sw_kernel.metal` to CUDA C
6. Test with `cargo test --features cuda`
7. Submit PR!

We welcome contributions from the community to expand GPU support beyond Apple Silicon.
