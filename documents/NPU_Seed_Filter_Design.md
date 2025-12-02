# NPU-Accelerated SMEM Seed Filtering

## Overview

This document outlines the design for using Neural Processing Units (NPUs) to pre-filter SMEM seeds before the expensive Smith-Waterman extension step. The goal is to train a lightweight CNN classifier that predicts seed viability, eliminating ~60-80% of seeds that would ultimately fail extension.

**Scope**: Apple Neural Engine via CoreML is the primary implementation. The architecture is designed to support emerging NPU platforms (Qualcomm Hexagon, Intel NPU, AMD XDNA, etc.) through a backend-agnostic trait.

**Training Data**: 1000 Genomes Project 30x coverage dataset ([PRJEB31736](https://www.ebi.ac.uk/ena/browser/view/PRJEB31736))

## Executive Summary

| Aspect | Specification |
|--------|---------------|
| **Target Hardware** | Apple ANE (primary), Qualcomm/Intel/AMD NPUs (future) |
| **Model Format** | ONNX (portable) → Platform-specific (CoreML, QNN, OpenVINO) |
| **Rust Integration** | `ort` crate with execution provider abstraction |
| **Input Encoding** | ONE-HOT (4 channels per base) |
| **Model Architecture** | 1D CNN with ~27K parameters |
| **Batch Size** | 512-2048 seeds per inference |
| **Target Latency** | < 1ms per batch |
| **Expected Filtering** | 60-80% of low-viability seeds eliminated |
| **Training Dataset** | 1000 Genomes PRJEB31736 (2504 samples, 30x WGS) |
| **Priority** | Post-1.x (after Metal GPU) |

## Multi-Backend NPU Architecture

### Design Goals

1. **CoreML first**: Apple Silicon is the primary development platform
2. **Backend-agnostic trait**: Define `NpuBackend` trait that any NPU can implement
3. **ONNX as interchange**: Train once, deploy everywhere
4. **Future-proof**: Support emerging AI accelerators as they mature

### NPU Landscape (2024-2025)

| Vendor | NPU | SDK | ONNX Runtime EP | Status |
|--------|-----|-----|-----------------|--------|
| **Apple** | Neural Engine (ANE) | CoreML | `CoreMLExecutionProvider` | **In Scope** |
| **Qualcomm** | Hexagon NPU | QNN SDK | `QNNExecutionProvider` | Out of Scope |
| **Intel** | NPU (Meteor Lake+) | OpenVINO | `OpenVINOExecutionProvider` | Out of Scope |
| **AMD** | XDNA (Ryzen AI) | ROCm/Vitis | `VitisAIExecutionProvider` | Out of Scope |
| **NVIDIA** | TensorRT | TensorRT | `TensorrtExecutionProvider` | Out of Scope |
| **MediaTek** | APU | NeuroPilot | Custom | Out of Scope |

### NpuBackend Trait

The key abstraction enabling multi-platform NPU support:

```rust
// src/core/compute/npu/mod.rs

/// Backend-agnostic NPU inference trait.
///
/// Implementations:
/// - CoreMLBackend (Apple ANE) - IN SCOPE
/// - QnnBackend (Qualcomm Hexagon) - OUT OF SCOPE
/// - OpenVinoBackend (Intel NPU) - OUT OF SCOPE
/// - VitisBackend (AMD XDNA) - OUT OF SCOPE
/// - TensorRtBackend (NVIDIA) - OUT OF SCOPE
/// - TractBackend (Pure Rust CPU fallback) - IN SCOPE
pub trait NpuBackend: Send + Sync {
    /// Backend identifier for logging/debugging
    fn name(&self) -> &'static str;

    /// Check if this backend is available on the current system
    fn is_available() -> bool where Self: Sized;

    /// Backend capabilities and constraints
    fn capabilities(&self) -> NpuCapabilities;

    /// Run seed viability inference on a batch.
    ///
    /// # Arguments
    /// * `batch` - ONE-HOT encoded seed contexts
    ///
    /// # Returns
    /// * Viability scores in [0, 1] for each seed
    fn predict_seed_viability(&self, batch: &SeedFilterBatch) -> NpuResult<Vec<f32>>;

    /// Optimal batch size for this backend
    fn optimal_batch_size(&self) -> usize;

    /// Supported batch sizes (NPUs often require fixed sizes)
    fn supported_batch_sizes(&self) -> &[usize];

    /// Warmup the model (first inference is often slow)
    fn warmup(&self) -> NpuResult<()>;
}

/// NPU backend capabilities and constraints.
#[derive(Debug, Clone)]
pub struct NpuCapabilities {
    /// Human-readable backend name
    pub name: &'static str,
    /// Supports INT8 quantized models
    pub supports_int8: bool,
    /// Supports FP16 models
    pub supports_fp16: bool,
    /// Requires fixed batch sizes
    pub requires_fixed_batch: bool,
    /// Maximum supported batch size
    pub max_batch_size: usize,
    /// Estimated TOPS (tera operations per second)
    pub estimated_tops: f32,
    /// Supports dynamic shapes (rare for NPUs)
    pub supports_dynamic_shapes: bool,
}

/// Result type for NPU operations.
pub type NpuResult<T> = Result<T, NpuError>;

/// NPU-specific errors.
#[derive(Debug, Clone)]
pub enum NpuError {
    /// Backend not available on this system
    NotAvailable,
    /// Model failed to load
    ModelLoadFailed(String),
    /// Inference failed
    InferenceFailed(String),
    /// Unsupported batch size
    UnsupportedBatchSize(usize),
    /// Unsupported model format
    UnsupportedFormat(String),
}

/// Input batch for seed filter model.
#[derive(Debug)]
pub struct SeedFilterBatch {
    /// ONE-HOT encoded contexts: shape (batch, 2, window, 4)
    /// Layout: batch-major, then channel, then position, then one-hot
    pub contexts: Vec<f32>,
    /// Actual number of seeds (may be less than allocated batch)
    pub batch_size: usize,
    /// Window size (typically 128)
    pub window_size: usize,
}

/// Configuration for seed filtering.
#[derive(Debug, Clone)]
pub struct SeedFilterConfig {
    /// Viability threshold (seeds below this are filtered)
    pub threshold: f32,  // Default: 0.3
    /// Minimum batch size to use NPU (below this, skip filtering)
    pub min_batch_size: usize,  // Default: 64
    /// Window size for context extraction
    pub window_size: usize,  // Default: 128
    /// Upstream context bases
    pub upstream_context: usize,  // Default: 32
    /// Downstream context bases
    pub downstream_context: usize,  // Default: 32
    /// Use INT8 quantized model if available
    pub prefer_quantized: bool,  // Default: true
}
```

### Module Structure

```
src/core/compute/
├── mod.rs                    # ComputeBackend enum
├── npu/                      # NPU backend abstraction
│   ├── mod.rs                # NpuBackend trait, types, errors
│   ├── types.rs              # SeedFilterBatch, SeedFilterConfig
│   ├── dispatch.rs           # Backend detection and selection
│   ├── coreml.rs             # Apple ANE (IN SCOPE)
│   ├── tract.rs              # Pure Rust CPU fallback (IN SCOPE)
│   ├── qnn.rs                # Qualcomm Hexagon (OUT OF SCOPE - placeholder)
│   ├── openvino.rs           # Intel NPU (OUT OF SCOPE - placeholder)
│   ├── vitis.rs              # AMD XDNA (OUT OF SCOPE - placeholder)
│   ├── tensorrt.rs           # NVIDIA TensorRT (OUT OF SCOPE - placeholder)
│   └── models/
│       ├── seed_filter_v1.onnx       # Portable ONNX model
│       └── seed_filter_v1_int8.onnx  # Quantized variant
├── gpu/                      # GPU backend (existing)
├── encoding.rs               # ONE-HOT encoding (existing)
└── simd_abstraction/         # SIMD backend (existing)
```

### Backend Detection and Selection

```rust
// src/core/compute/npu/dispatch.rs

/// Detect the optimal NPU backend for the current system.
pub fn detect_npu_backend() -> Option<Arc<dyn NpuBackend>> {
    // Priority order based on typical performance

    // 1. Apple Neural Engine (macOS/iOS)
    #[cfg(all(target_os = "macos", feature = "coreml"))]
    if coreml::CoreMLBackend::is_available() {
        if let Ok(backend) = coreml::CoreMLBackend::new() {
            log::info!("NPU: Using Apple Neural Engine via CoreML");
            return Some(Arc::new(backend));
        }
    }

    // 2. Qualcomm Hexagon NPU (Android, Windows on ARM)
    #[cfg(feature = "qnn")]
    if qnn::QnnBackend::is_available() {
        if let Ok(backend) = qnn::QnnBackend::new() {
            log::info!("NPU: Using Qualcomm Hexagon via QNN");
            return Some(Arc::new(backend));
        }
    }

    // 3. Intel NPU (Windows, Linux with Meteor Lake+)
    #[cfg(feature = "openvino")]
    if openvino::OpenVinoBackend::is_available() {
        if let Ok(backend) = openvino::OpenVinoBackend::new() {
            log::info!("NPU: Using Intel NPU via OpenVINO");
            return Some(Arc::new(backend));
        }
    }

    // 4. AMD XDNA (Windows, Linux with Ryzen AI)
    #[cfg(feature = "vitis")]
    if vitis::VitisBackend::is_available() {
        if let Ok(backend) = vitis::VitisBackend::new() {
            log::info!("NPU: Using AMD XDNA via Vitis AI");
            return Some(Arc::new(backend));
        }
    }

    // 5. NVIDIA TensorRT (Linux, Windows with NVIDIA GPU)
    #[cfg(feature = "tensorrt")]
    if tensorrt::TensorRtBackend::is_available() {
        if let Ok(backend) = tensorrt::TensorRtBackend::new() {
            log::info!("NPU: Using NVIDIA TensorRT");
            return Some(Arc::new(backend));
        }
    }

    // 6. Pure Rust CPU fallback (always available)
    #[cfg(feature = "tract")]
    {
        if let Ok(backend) = tract::TractBackend::new() {
            log::info!("NPU: Using tract (CPU fallback)");
            return Some(Arc::new(backend));
        }
    }

    log::debug!("NPU: No backend available");
    None
}

/// Get capabilities of all available backends (for diagnostics)
pub fn list_available_backends() -> Vec<NpuCapabilities> {
    let mut backends = Vec::new();

    #[cfg(all(target_os = "macos", feature = "coreml"))]
    if coreml::CoreMLBackend::is_available() {
        backends.push(coreml::CoreMLBackend::capabilities_static());
    }

    #[cfg(feature = "qnn")]
    if qnn::QnnBackend::is_available() {
        backends.push(qnn::QnnBackend::capabilities_static());
    }

    // ... other backends ...

    backends
}
```

### Feature Flags

```toml
# Cargo.toml

[features]
# NPU umbrella feature
npu = []

# Backend-specific features (IN SCOPE)
coreml = ["npu", "ort/coreml"]      # Apple Neural Engine
tract = ["npu", "tract-onnx"]       # Pure Rust CPU fallback

# Backend-specific features (OUT OF SCOPE - placeholders)
qnn = ["npu"]                        # Qualcomm Hexagon
openvino = ["npu"]                   # Intel NPU
vitis = ["npu"]                      # AMD XDNA
tensorrt = ["npu"]                   # NVIDIA TensorRT

# Convenience feature for all available backends
npu-all = ["coreml", "tract"]

[dependencies]
# ONNX Runtime (primary inference engine)
ort = { version = "2.0", optional = true, features = ["load-dynamic"] }

# Pure Rust fallback
tract-onnx = { version = "0.21", optional = true }

# Future (out of scope):
# openvino = { version = "0.6", optional = true }
# qnn-sys = { version = "...", optional = true }
```

### ComputeContext Integration

```rust
// src/core/compute/mod.rs

pub struct ComputeContext {
    /// Primary compute backend (SIMD or GPU)
    pub backend: ComputeBackend,

    /// GPU batch threshold
    pub gpu_batch_threshold: usize,

    /// NPU backend for seed filtering (optional)
    #[cfg(feature = "npu")]
    pub npu_backend: Option<Arc<dyn NpuBackend>>,

    /// Seed filter configuration
    #[cfg(feature = "npu")]
    pub seed_filter_config: SeedFilterConfig,

    /// Enable NPU seed filtering
    #[cfg(feature = "npu")]
    pub npu_filter_enabled: bool,
}

impl ComputeContext {
    pub fn new() -> Self {
        let backend = ComputeBackend::detect_optimal();

        #[cfg(feature = "npu")]
        let npu_backend = npu::dispatch::detect_npu_backend();

        Self {
            backend,
            gpu_batch_threshold: 1024,
            #[cfg(feature = "npu")]
            npu_backend,
            #[cfg(feature = "npu")]
            seed_filter_config: SeedFilterConfig::default(),
            #[cfg(feature = "npu")]
            npu_filter_enabled: true,
        }
    }
}

## Problem Statement

### Current Pipeline Bottleneck

In the current alignment pipeline:

```
Seeds → Extension (SW) → Chaining → Finalization
  │           │
  │      Expensive!
  │      (~80% of runtime)
  │
  └── Many seeds fail extension (low score, poor alignment)
```

Seed extension via Smith-Waterman is the most expensive step. However, many seeds:
- Have poor sequence context (repetitive regions)
- Span reference boundaries
- Have high mismatch density in flanking regions
- Would produce sub-threshold alignment scores

**Hypothesis**: A lightweight ML model can predict seed viability from sequence context, filtering 60-80% of low-quality seeds before the expensive SW step.

### Target Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| **True Positive Rate** | > 95% | Must not filter viable seeds |
| **False Positive Rate** | < 40% | Filter most bad seeds |
| **Inference Latency** | < 1ms / 512 seeds | Must not become bottleneck |
| **Model Size** | < 1 MB | Fit in ANE cache |

## Model Architecture

### Input Representation

Seeds are represented as ONE-HOT encoded sequence windows:

```
Seed Context Window (128 bp total):
┌──────────────────────────────────────────────────────────────┐
│  32 bp upstream  │  seed (variable)  │  32 bp downstream     │
│  query context   │    query match    │  query context        │
├──────────────────┼───────────────────┼───────────────────────┤
│  32 bp upstream  │  seed (variable)  │  32 bp downstream     │
│  ref context     │    ref match      │  ref context          │
└──────────────────┴───────────────────┴───────────────────────┘

Input tensor shape: (batch_size, 2, 128, 4)
  - 2 channels: query and reference
  - 128 positions: fixed window with padding
  - 4 features: ONE-HOT encoding (A, C, G, T)
```

### Network Architecture

Lightweight 1D CNN optimized for ANE:

```
Layer                    Output Shape       Parameters
─────────────────────────────────────────────────────────
Input                    (B, 2, 128, 4)     -
Reshape                  (B, 8, 128)        -         # Flatten channels
Conv1D(8→32, k=7)       (B, 32, 122)       1,824
BatchNorm + ReLU         (B, 32, 122)       64
MaxPool1D(2)            (B, 32, 61)        -
Conv1D(32→64, k=5)      (B, 64, 57)        10,304
BatchNorm + ReLU         (B, 64, 57)        128
MaxPool1D(2)            (B, 64, 28)        -
Conv1D(64→64, k=3)      (B, 64, 26)        12,352
BatchNorm + ReLU         (B, 64, 26)        128
GlobalAvgPool1D         (B, 64)            -
Dense(64→32)            (B, 32)            2,080
ReLU                     (B, 32)            -
Dense(32→1)             (B, 1)             33
Sigmoid                  (B, 1)             -
─────────────────────────────────────────────────────────
Total Parameters:        ~27K
Model Size:             ~110 KB (FP32), ~28 KB (INT8)
```

### Why This Architecture?

1. **1D Convolutions**: DNA is a 1D sequence; 2D convs waste compute
2. **Small Filters (3-7)**: Capture local motifs (k-mers)
3. **BatchNorm**: Enables INT8 quantization for ANE
4. **GlobalAvgPool**: Position-invariant aggregation
5. **< 50K params**: Fits entirely in ANE's 16 MB cache

### ANE Compatibility Requirements

Apple Neural Engine has specific requirements for optimal execution:

| Requirement | Our Design |
|-------------|------------|
| Fixed input shapes | 128 bp window (padded) |
| BatchNorm after Conv | Yes |
| No dynamic shapes | Fixed batch sizes (512, 1024, 2048) |
| Supported ops only | Conv1D, Pool, Dense, ReLU, Sigmoid |
| Quantization-friendly | BatchNorm enables INT8 |

## Training Pipeline

### Dataset: 1000 Genomes PRJEB31736

The [1000 Genomes 30x dataset](https://www.ebi.ac.uk/ena/browser/view/PRJEB31736) provides:

- **2504 unrelated samples** from diverse populations
- **30x mean coverage** per sample (NovaSeq 6000, 2x150bp)
- **GRCh38 aligned BAMs** publicly available
- **Unrestricted access** - no data access committee approval needed

### Training Data Generation

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Training Data Pipeline                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Sample Selection                                                │
│     ├── Select 100 samples from PRJEB31736 (diverse populations)   │
│     ├── Download FASTQ from ENA (or use existing BAMs)             │
│     └── ~10M read pairs per sample → 1B read pairs total           │
│                                                                      │
│  2. Seed Extraction (FerrousAlign)                                  │
│     ├── Run seeding stage only (--stage seeding)                   │
│     ├── Extract all SMEMs with context windows                      │
│     └── Record seed positions, lengths, interval sizes             │
│                                                                      │
│  3. Ground Truth Labeling                                           │
│     ├── Run full alignment pipeline                                 │
│     ├── Label seeds by extension outcome:                           │
│     │   ├── VIABLE: Seed contributed to alignment with score ≥ T   │
│     │   └── NON-VIABLE: Seed filtered or produced low score        │
│     └── Store labels with seed features                             │
│                                                                      │
│  4. Feature Extraction                                              │
│     ├── Query context: 32bp upstream + seed + 32bp downstream      │
│     ├── Reference context: Same window from reference               │
│     ├── ONE-HOT encode both sequences                               │
│     └── Additional features (optional):                             │
│         ├── Seed length                                             │
│         ├── BWT interval size                                       │
│         └── Position in read                                        │
│                                                                      │
│  5. Dataset Balancing                                               │
│     ├── ~70% of seeds are non-viable (class imbalance)             │
│     ├── Undersample non-viable OR use weighted loss                │
│     └── Target: 50-50 or 60-40 split                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Training Configuration

```python
# PyTorch training script (train_seed_filter.py)

config = {
    # Data
    "train_samples": 80,           # From PRJEB31736
    "val_samples": 10,
    "test_samples": 10,
    "seeds_per_sample": 1_000_000,  # ~100M total seeds

    # Model
    "window_size": 128,
    "channels": [8, 32, 64, 64],
    "kernel_sizes": [7, 5, 3],

    # Training
    "batch_size": 2048,
    "epochs": 20,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "pos_weight": 2.0,            # Handle class imbalance

    # Optimization
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingLR",
    "early_stopping_patience": 5,
}
```

### Export Pipeline

```
PyTorch Model (.pt)
       │
       ▼
ONNX Export (torch.onnx.export)
       │
       ▼
ONNX Model (.onnx)
       │
       ├──────────────────────┐
       ▼                      ▼
CoreML (coremltools)    TensorRT (future)
       │                      │
       ▼                      ▼
MLModel (.mlpackage)    TRT Engine (.plan)
       │
       ▼
Quantized INT8 (coremltools.optimize)
```

## Rust Integration

### Crate Selection

| Option | Pros | Cons | Recommendation |
|--------|------|------|----------------|
| [`ort`](https://github.com/pykeio/ort) | Mature, CoreML EP support | Build complexity | **Primary** |
| [`candle`](https://github.com/huggingface/candle) | Pure Rust, growing ecosystem | No ANE support | Fallback |
| [`tract`](https://github.com/sonos/tract) | Pure Rust, no deps | No ANE support | Fallback |

**Decision**: Use `ort` with CoreML execution provider for ANE acceleration, with `tract` as pure-Rust CPU fallback.

### Module Structure

```
src/core/compute/
├── mod.rs                    # ComputeBackend enum
├── npu/                      # NEW: NPU backend
│   ├── mod.rs                # NpuBackend trait, SeedFilterModel
│   ├── ort_backend.rs        # ONNX Runtime + CoreML implementation
│   ├── tract_backend.rs      # Pure Rust fallback (no ANE)
│   └── models/               # Embedded or loaded models
│       └── seed_filter_v1.onnx
├── encoding.rs               # ONE-HOT encoding (existing)
├── gpu/                      # GPU backend (existing)
└── simd_abstraction/         # SIMD backend (existing)
```

### NpuBackend Trait

```rust
// src/core/compute/npu/mod.rs

/// Backend-agnostic NPU inference trait.
///
/// Implementations:
/// - OrtCoreMLBackend (Apple ANE via ONNX Runtime)
/// - TractBackend (Pure Rust CPU fallback)
/// - TensorRTBackend (NVIDIA, out of scope)
pub trait NpuBackend: Send + Sync {
    /// Backend identifier
    fn name(&self) -> &'static str;

    /// Check hardware availability
    fn is_available() -> bool where Self: Sized;

    /// Run seed viability inference on a batch.
    ///
    /// # Arguments
    /// * `batch` - ONE-HOT encoded seed contexts
    ///
    /// # Returns
    /// * Viability scores in [0, 1] for each seed
    fn predict_seed_viability(&self, batch: &SeedFilterBatch) -> Vec<f32>;

    /// Optimal batch size for this backend
    fn optimal_batch_size(&self) -> usize;
}

/// Input batch for seed filter model.
#[derive(Debug)]
pub struct SeedFilterBatch {
    /// ONE-HOT encoded contexts: shape (batch, 2, window, 4)
    pub contexts: Vec<f32>,
    /// Batch size
    pub batch_size: usize,
    /// Window size (128)
    pub window_size: usize,
}

/// Configuration for seed filtering.
#[derive(Debug, Clone)]
pub struct SeedFilterConfig {
    /// Viability threshold (seeds below this are filtered)
    pub threshold: f32,  // Default: 0.3
    /// Minimum batch size to use NPU (below this, skip filtering)
    pub min_batch_size: usize,  // Default: 64
    /// Window size for context extraction
    pub window_size: usize,  // Default: 128
    /// Upstream context bases
    pub upstream_context: usize,  // Default: 32
    /// Downstream context bases
    pub downstream_context: usize,  // Default: 32
}

impl Default for SeedFilterConfig {
    fn default() -> Self {
        Self {
            threshold: 0.3,
            min_batch_size: 64,
            window_size: 128,
            upstream_context: 32,
            downstream_context: 32,
        }
    }
}
```

### ONNX Runtime + CoreML Backend

```rust
// src/core/compute/npu/ort_backend.rs

use ort::{Environment, Session, SessionBuilder, Value};
use super::{NpuBackend, SeedFilterBatch};

pub struct OrtCoreMLBackend {
    session: Session,
    environment: Environment,
}

impl OrtCoreMLBackend {
    pub fn new() -> Option<Self> {
        // Initialize ONNX Runtime with CoreML EP
        let environment = Environment::builder()
            .with_name("ferrous_align_npu")
            .build()
            .ok()?;

        // Load model (embedded or from file)
        let model_bytes = include_bytes!("models/seed_filter_v1.onnx");

        let session = SessionBuilder::new(&environment)
            .ok()?
            .with_execution_providers([
                // CoreML with ANE preference
                ort::ExecutionProvider::CoreML(
                    ort::CoreMLExecutionProviderOptions::default()
                        .with_ane_only(true)  // Prefer ANE
                ),
                // CPU fallback
                ort::ExecutionProvider::CPU(Default::default()),
            ])
            .ok()?
            .with_model_from_memory(model_bytes)
            .ok()?;

        Some(Self { session, environment })
    }
}

impl NpuBackend for OrtCoreMLBackend {
    fn name(&self) -> &'static str { "CoreML (ANE)" }

    fn is_available() -> bool {
        #[cfg(target_os = "macos")]
        { true }  // CoreML always available on macOS
        #[cfg(not(target_os = "macos"))]
        { false }
    }

    fn predict_seed_viability(&self, batch: &SeedFilterBatch) -> Vec<f32> {
        // Prepare input tensor
        let input_shape = [
            batch.batch_size as i64,
            2,  // query + reference channels
            batch.window_size as i64,
            4,  // ONE-HOT
        ];

        let input = Value::from_array(
            self.session.allocator(),
            &input_shape,
            &batch.contexts,
        ).unwrap();

        // Run inference
        let outputs = self.session.run(vec![input]).unwrap();

        // Extract predictions
        let predictions: Vec<f32> = outputs[0]
            .try_extract::<f32>()
            .unwrap()
            .view()
            .iter()
            .copied()
            .collect();

        predictions
    }

    fn optimal_batch_size(&self) -> usize {
        1024  // Optimized for ANE
    }
}
```

### Integration with Seeding Pipeline

```rust
// src/pipelines/linear/seeding.rs (modified)

pub fn find_seeds_with_filtering(
    query: &[u8],
    index: &BwaIndex,
    opts: &MemOptions,
    npu: Option<&dyn NpuBackend>,
    filter_config: &SeedFilterConfig,
) -> Vec<Seed> {
    // 1. Extract all SMEMs (existing code)
    let raw_seeds = find_smems(query, index, opts);

    // 2. Skip filtering if NPU unavailable or batch too small
    let npu = match npu {
        Some(n) if raw_seeds.len() >= filter_config.min_batch_size => n,
        _ => return raw_seeds,  // No filtering
    };

    // 3. Extract context windows for each seed
    let batch = prepare_seed_filter_batch(
        &raw_seeds,
        query,
        &index.reference,
        filter_config,
    );

    // 4. Run NPU inference
    let viability_scores = npu.predict_seed_viability(&batch);

    // 5. Filter seeds by threshold
    raw_seeds
        .into_iter()
        .zip(viability_scores)
        .filter(|(_, score)| *score >= filter_config.threshold)
        .map(|(seed, _)| seed)
        .collect()
}

fn prepare_seed_filter_batch(
    seeds: &[Seed],
    query: &[u8],
    reference: &BntSeq,
    config: &SeedFilterConfig,
) -> SeedFilterBatch {
    let mut contexts = Vec::with_capacity(
        seeds.len() * 2 * config.window_size * 4
    );

    for seed in seeds {
        // Extract query context window
        let query_window = extract_context_window(
            query,
            seed.query_pos as usize,
            seed.len as usize,
            config,
        );

        // Extract reference context window
        let ref_window = extract_ref_context_window(
            reference,
            seed.ref_pos,
            seed.len as usize,
            config,
        );

        // ONE-HOT encode both
        contexts.extend(encode_onehot_window(&query_window));
        contexts.extend(encode_onehot_window(&ref_window));
    }

    SeedFilterBatch {
        contexts,
        batch_size: seeds.len(),
        window_size: config.window_size,
    }
}
```

## Feature Gating

### Cargo.toml

```toml
[features]
# NPU umbrella feature
npu = []

# Backend-specific features
coreml = ["npu", "ort/coreml"]
tensorrt = ["npu"]  # Future: NVIDIA

[dependencies]
# ONNX Runtime (optional, for NPU)
ort = { version = "2.0", optional = true, features = ["load-dynamic"] }

# Pure Rust fallback (optional)
tract-onnx = { version = "0.21", optional = true }
```

### ComputeBackend Extension

```rust
// src/core/compute/mod.rs

pub enum ComputeBackend {
    CpuSimd(SimdEngineType),

    #[cfg(feature = "gpu")]
    Gpu(Arc<dyn GpuBackend>),

    #[cfg(feature = "npu")]
    Npu(Arc<dyn NpuBackend>),  // NEW
}

pub struct ComputeContext {
    pub backend: ComputeBackend,
    pub gpu_batch_threshold: usize,

    #[cfg(feature = "npu")]
    pub npu_backend: Option<Arc<dyn NpuBackend>>,  // NEW

    #[cfg(feature = "npu")]
    pub seed_filter_config: SeedFilterConfig,  // NEW
}
```

## Performance Analysis

### Latency Budget

```
Current Pipeline (without NPU):
──────────────────────────────────────────────────────────────
Seeding:     ~5 ms    (FM-Index search, SMEM extraction)
Extension:   ~40 ms   (Smith-Waterman on all seeds)
Chaining:    ~3 ms    (DP chaining)
Finalize:    ~2 ms    (CIGAR, tags)
──────────────────────────────────────────────────────────────
Total:       ~50 ms per read pair

With NPU Seed Filtering:
──────────────────────────────────────────────────────────────
Seeding:     ~5 ms    (unchanged)
NPU Filter:  ~1 ms    (batch inference on ANE)
Extension:   ~16 ms   (60% fewer seeds → 60% faster)
Chaining:    ~2 ms    (fewer chains)
Finalize:    ~2 ms    (unchanged)
──────────────────────────────────────────────────────────────
Total:       ~26 ms per read pair (1.9x speedup)
```

### Memory Overhead

| Component | Size | Notes |
|-----------|------|-------|
| Model weights | ~110 KB (FP32) | Embedded in binary |
| Model weights | ~28 KB (INT8) | Quantized for ANE |
| Input buffer | ~4 MB | 2048 seeds × 128 × 4 × 2 × f32 |
| Output buffer | ~8 KB | 2048 seeds × f32 |
| **Total** | ~4.2 MB | Per-thread overhead |

### ANE Throughput

Based on Apple's ANE specifications:

| Chip | ANE TOPS | Est. Seeds/sec | Batch Latency |
|------|----------|----------------|---------------|
| M1 | 11 | ~500K | ~1.0 ms |
| M2 | 15.8 | ~700K | ~0.7 ms |
| M3 | 18 | ~800K | ~0.6 ms |
| M4 | ~20 | ~900K | ~0.5 ms |

With 512-seed batches, ANE latency is well under 1ms.

## Implementation Phases

### Phase 0: Training Infrastructure (2 weeks)
- [ ] Create training data extraction tool (`ferrous-align --extract-seeds`)
- [ ] Download 100 samples from PRJEB31736
- [ ] Generate labeled seed dataset (~100M seeds)
- [ ] Implement PyTorch training script
- [ ] Train and validate model (target: >95% TPR, <40% FPR)
- [ ] Export to ONNX, convert to CoreML

### Phase 1: NPU Abstraction (1 week)
- [ ] Create `src/core/compute/npu/mod.rs` with `NpuBackend` trait
- [ ] Define `SeedFilterBatch`, `SeedFilterConfig` types
- [ ] Add `npu` umbrella feature to Cargo.toml
- [ ] Implement `OrtCoreMLBackend`
- [ ] Unit test: model loading and inference

### Phase 2: Pipeline Integration (1 week)
- [ ] Modify `seeding.rs` to accept optional NPU backend
- [ ] Implement `prepare_seed_filter_batch()`
- [ ] Implement context window extraction
- [ ] Activate ONE-HOT encoding in `encoding.rs`
- [ ] Integration test: seeding with filtering

### Phase 3: Optimization (1 week)
- [ ] Profile ANE utilization (Instruments)
- [ ] Tune batch sizes for optimal throughput
- [ ] Implement INT8 quantization
- [ ] Benchmark filtering accuracy and speedup
- [ ] A/B test: filtered vs unfiltered alignment quality

### Phase 4: CLI and Documentation (1 week)
- [ ] Add `--npu-filter` CLI flag
- [ ] Add `--npu-threshold` for tuning
- [ ] Document training data generation
- [ ] Document model retraining procedure
- [ ] Update CLAUDE.md with NPU architecture

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Model accuracy too low** | Medium | High | Iterate on architecture; use ensemble |
| **ANE not available** | Low | Medium | CPU fallback via tract |
| **ort crate breaking changes** | Medium | Medium | Pin version; consider pure-Rust |
| **Training data bias** | Medium | Medium | Use diverse 1KG populations |
| **Latency overhead** | Low | High | Skip filtering for small batches |

## References

- [1000 Genomes 30x Dataset (PRJEB31736)](https://www.ebi.ac.uk/ena/browser/view/PRJEB31736)
- [ort - ONNX Runtime Rust bindings](https://github.com/pykeio/ort)
- [CoreML Execution Provider](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)
- [Apple Neural Engine Overview](https://developer.apple.com/machine-learning/core-ml/)
- [DNA Sequence Classification with CNN](https://pmc.ncbi.nlm.nih.gov/articles/PMC8285202/)
- [Global Seed Filtering Methods](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-022-04745-4)
- [Deploying Transformers on ANE](https://machinelearning.apple.com/research/neural-engine-transformers)

## Conclusion

NPU-accelerated seed filtering is a promising optimization for Apple Silicon:

1. **Lightweight CNN** (~27K params) fits ANE constraints
2. **ONE-HOT encoding** (existing infrastructure) feeds the model
3. **60-80% seed reduction** translates to ~1.9x pipeline speedup
4. **1000 Genomes dataset** provides diverse, high-quality training data
5. **ort crate** provides mature CoreML integration

The architecture follows the established pattern of backend-agnostic traits (`NpuBackend`) with platform-specific implementations, enabling future TensorRT support for NVIDIA users.

Priority: Post-1.x, after Metal GPU acceleration.
