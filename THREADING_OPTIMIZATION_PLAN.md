# Multi-Threading Optimization Plan
**Date**: 2025-11-16
**Platform**: AMD Ryzen 9 7900X (12 cores, 24 threads)
**Current State**: Running single-threaded in benchmarks (100% CPU instead of 2400%)

---

## Phase 1: Immediate Fixes (Benchmark Accuracy)

### 1.1 Fix Benchmark Thread Configuration
**Problem**: Benchmark doesn't pass `-t` flag and silences stderr logging
**Impact**: Can't verify if threading is actually working
**Fix**:
```rust
// benches/bwa_mem2_comparison.rs
fn run_bwa_mem2(bwa_path: &str, ref_path: &str, read1_path: &str, read2_path: &str) {
    let num_threads = num_cpus::get().to_string();
    let output = Command::new(bwa_path)
        .arg("mem")
        .arg("-t").arg(&num_threads)  // ADD THIS
        .arg(ref_path)
        .arg(read1_path)
        .arg(read2_path)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())  // CHANGE: capture stderr to verify threading
        .output()
        .expect("Failed to execute bwa-mem2");

    // Optional: print thread info from stderr for first iteration
    if !output.stderr.is_empty() {
        eprintln!("bwa-mem2 stderr: {}", String::from_utf8_lossy(&output.stderr));
    }
}

fn run_ferrous_align(ref_path: &str, read1_path: &str, read2_path: &str) {
    let num_threads = num_cpus::get().to_string();
    let output = Command::new("cargo")
        .arg("run")
        .arg("--release")
        .arg("--")
        .arg("mem")
        .arg("-p")
        .arg("-t").arg(&num_threads)  // ADD THIS
        .arg(ref_path)
        .arg(read1_path)
        .arg(read2_path)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())  // CHANGE: capture stderr to verify threading
        .output()
        .expect("Failed to execute FerrousAlign");
}
```

**Validation**:
- Check stderr output shows "Using 24 threads"
- Monitor with `htop` during benchmark - should see 2000-2400% CPU

---

## Phase 2: Profiling & Bottleneck Analysis

### 2.1 Profile Current Threading Efficiency
**Tools**: `perf`, `cargo flamegraph`, `rayon::ThreadPoolBuilder::spawn_handler`

**Key Metrics to Measure**:
1. **Thread utilization**: % time threads are busy vs waiting
2. **Lock contention**: Time spent in mutex/sync operations
3. **Load balancing**: Work distribution across threads
4. **Memory bandwidth**: Are we memory-bound?

**Commands**:
```bash
# CPU profiling
perf record -g ./target/release/ferrous-align mem -t 24 ref.fna reads.fq > /dev/null
perf report

# Flamegraph
cargo flamegraph --bin ferrous-align -- mem -t 24 ref.fna reads.fq > /dev/null

# Thread-specific profiling
perf record -e cycles,cache-misses -ag ./target/release/ferrous-align mem -t 24 ref.fna reads.fq
```

**Expected Bottlenecks**:
- Gzip decompression (likely single-threaded)
- FM-Index backward search (shared read-only, should scale)
- Suffix array reconstruction (sequential cache lookups)
- SAM output (sequential write)

### 2.2 Measure Per-Stage Parallelism
Add timing instrumentation:
```rust
// src/mem.rs
let stage1_start = Instant::now();
let alignments: Vec<_> = batch.names
    .par_iter()  // Parallel
    .zip(batch.seqs.par_iter())
    .zip(batch.quals.par_iter())
    .map(|((name, seq), qual)| {
        // Alignment work
    })
    .collect();
log::debug!("Stage 1 (parallel alignment): {:?}", stage1_start.elapsed());
```

---

## Phase 3: Optimize Parallelizable Stages

### 3.1 Increase Batch Size (Quick Win)
**Current**: 512 reads per batch
**Issue**: Too small for 24 threads (21 reads/thread average)
**Recommendation**: Scale with thread count

```rust
// src/mem.rs or mem_opt.rs
const BASE_BATCH_SIZE: usize = 512;
let optimal_batch_size = BASE_BATCH_SIZE * (num_threads / 4).max(1);
// 24 threads → 512 * 6 = 3072 reads/batch
// 128 reads/thread (better amortization)
```

**Expected Impact**: 2-3x better thread utilization

### 3.2 Parallelize FASTQ Decompression
**Problem**: `bio::io::fastq::Reader` is single-threaded for gzip
**Solution**: Use `pgzip` or chunked parallel decompression

**Option A - Chunked Reading**:
```rust
use flate2::read::GzDecoder;
use std::io::BufReader;

// Read large chunks, decompress in parallel
let chunk_size = 10_000_000; // 10MB chunks
let chunks: Vec<_> = (0..num_chunks)
    .into_par_iter()
    .map(|i| decompress_chunk(i, chunk_size))
    .collect();
```

**Option B - Use `pgzip` crate** (if available):
```rust
use pgzip::Decoder;
let reader = Decoder::new(file)?;
```

**Expected Impact**: 3-5x faster I/O (gzip decompression is CPU-intensive)

### 3.3 Optimize Rayon Configuration
**Current**: Default Rayon settings
**Tuning Options**:

```rust
rayon::ThreadPoolBuilder::new()
    .num_threads(num_threads)
    .stack_size(4 * 1024 * 1024)  // 4MB (BWA uses large stacks)
    .thread_name(|i| format!("ferrous-{}", i))
    .build_global()?;
```

**Advanced - Work Stealing Tuning**:
```rust
// For compute-heavy tasks with varying complexity
.spawn_handler(|thread| {
    std::thread::Builder::new()
        .stack_size(4 * 1024 * 1024)
        .spawn(|| {
            // Set thread affinity for NUMA systems
            core_affinity::set_for_current(core_affinity::CoreId { id: thread.index() });
            thread.run()
        })
})
```

---

## Phase 4: Pipeline Parallelism

### 4.1 Implement 3-Stage Pipeline (Like C++ bwa-mem2)
**Current**: Sequential batch processing
**Target**: Overlapping I/O, compute, and output

```rust
use crossbeam::channel::{bounded, Sender, Receiver};

// Stage 0: Read FASTQ (1-2 threads)
let (read_tx, read_rx) = bounded(2);  // Buffer 2 batches
thread::spawn(move || {
    while let Some(batch) = read_fastq_batch() {
        read_tx.send(batch).unwrap();
    }
});

// Stage 1: Align reads (N-2 threads via Rayon)
let (align_tx, align_rx) = bounded(2);
let align_pool = rayon::ThreadPoolBuilder::new()
    .num_threads(num_threads - 2)
    .build()?;

align_pool.install(|| {
    for batch in read_rx {
        let alignments = batch.par_iter().map(align).collect();
        align_tx.send(alignments).unwrap();
    }
});

// Stage 2: Write SAM (1 thread, sequential for deterministic output)
thread::spawn(move || {
    for alignments in align_rx {
        write_sam_batch(alignments);
    }
});
```

**Expected Impact**:
- Overlap I/O with compute: ~1.5-2x throughput
- Eliminates read/write stalls

### 4.2 NUMA-Aware Memory Allocation
For large reference genomes on multi-socket systems:
```rust
// Replicate read-only index structures per NUMA node
let num_numa_nodes = hwloc::Topology::new().num_numa_nodes();
let indices: Vec<Arc<BwaIndex>> = (0..num_numa_nodes)
    .map(|node| {
        bind_to_numa_node(node);
        Arc::new(load_index_local())  // Allocates on local NUMA node
    })
    .collect();

// Workers use local copy
rayon::scope(|s| {
    for (thread_id, read) in reads.par_iter().enumerate() {
        let node = thread_id % num_numa_nodes;
        let local_idx = &indices[node];
        s.spawn(move |_| align_with_index(read, local_idx));
    }
});
```

---

## Phase 5: Low-Level SIMD & Cache Optimizations

### 5.1 Vectorize FM-Index Backward Search
**Current**: Scalar occurrence counting
**Target**: SIMD-accelerated `bwt_occ()`

```rust
// src/bwt.rs - Vectorize popcount64
#[cfg(target_arch = "x86_64")]
unsafe fn bwt_occ_simd(bwt: &[u64], k: u64, c: u8) -> u64 {
    let mut count = 0u64;
    let chunks = bwt.chunks_exact(4);

    for chunk in chunks {
        let v0 = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
        let popcounts = _mm256_popcnt_epi64(v0);
        // Sum and accumulate
    }
    count
}
```

### 5.2 Prefetch Suffix Array Lookups
**Problem**: SA lookups have random access patterns (cache misses)
**Solution**: Software prefetching

```rust
// src/align.rs - get_sa_entry()
#[inline(always)]
fn prefetch_sa(bwt: &Bwt, k: u64) {
    unsafe {
        let cp_idx = (k / 64) as usize;
        let sa_ptr = &bwt.sa_samples[cp_idx] as *const u64;
        std::arch::x86_64::_mm_prefetch::<3>(sa_ptr as *const i8);
    }
}

// In batch processing
for i in 0..batch.len() {
    if i + 4 < batch.len() {
        prefetch_sa(&bwt, batch[i+4].k);  // Prefetch ahead
    }
    let pos = get_sa_entry(&bwt, batch[i].k);
}
```

### 5.3 Cache-Aware Work Distribution
**Problem**: False sharing, poor cache locality
**Solution**: Thread-local buffers, aligned allocations

```rust
// Ensure thread-local buffers don't share cache lines
#[repr(align(64))]  // CPU cache line size
struct AlignedBuffer {
    data: Vec<Alignment>,
    _padding: [u8; 64],
}

thread_local! {
    static LOCAL_BUFFER: RefCell<AlignedBuffer> = RefCell::new(AlignedBuffer::new());
}
```

---

## Phase 6: Advanced Optimizations

### 6.1 GPU Acceleration (Long-term)
- Smith-Waterman on GPU (CUDA/OpenCL/Metal)
- Seed extension batching: 1000+ sequences in parallel
- Expected: 10-50x speedup for extension phase

### 6.2 Lock-Free Data Structures
Replace `Arc<Mutex<T>>` with lock-free alternatives:
- `crossbeam::queue::ArrayQueue` for work queues
- `atomic::Ordering::Relaxed` for statistics counters

### 6.3 Speculative Execution
```rust
// Start aligning reverse-complement speculatively while forward aligns
let (fwd, rev): (Vec<_>, Vec<_>) = rayon::join(
    || align_forward(seq),
    || align_reverse_complement(seq)
);
// Pick best result
```

---

## Validation & Benchmarking

### Success Metrics
1. **CPU Utilization**: 2000-2400% on 24-thread system (target: >90%)
2. **Throughput**:
   - Current: ~60-70 reads/second (estimated, single-thread)
   - Target: 1000+ reads/second (24 threads, linear scaling assumed)
   - bwa-mem2 baseline: ~1500 reads/second on this system
3. **Scaling Efficiency**:
   - 1 thread → 4 threads: >3.5x speedup (>87% efficiency)
   - 4 threads → 12 threads: >2.5x speedup (>83% efficiency)
   - 12 threads → 24 threads: >1.6x speedup (>80% efficiency, SMT overhead)

### Benchmark Suite
```bash
# Single-threaded baseline
./target/release/ferrous-align mem -t 1 ref.fna reads.fq > /dev/null

# Scaling test
for threads in 1 2 4 8 12 16 24; do
    echo "Testing $threads threads..."
    time ./target/release/ferrous-align mem -t $threads ref.fna reads.fq > /dev/null
done

# Compare to bwa-mem2
time /tmp/bwa-mem2-diag/bwa-mem2 mem -t 24 ref.fna reads.fq > /dev/null
```

---

## Implementation Priority

**Week 1 - Quick Wins**:
1. ✅ Fix benchmark thread configuration (Phase 1.1)
2. Profile current bottlenecks (Phase 2.1)
3. Increase batch size (Phase 3.1)

**Week 2-3 - Medium Impact**:
4. Parallelize gzip decompression (Phase 3.2)
5. Implement pipeline parallelism (Phase 4.1)
6. Optimize Rayon configuration (Phase 3.3)

**Month 2 - Advanced**:
7. NUMA awareness (Phase 4.2)
8. Vectorize FM-Index (Phase 5.1)
9. SA prefetching (Phase 5.2)

**Long-term**:
10. GPU acceleration (Phase 6.1)
11. Speculative execution (Phase 6.3)

---

## Risk Assessment

**Low Risk** (safe to implement immediately):
- Benchmark thread flag fixes
- Batch size tuning
- Profiling

**Medium Risk** (test thoroughly):
- Pipeline parallelism (complexity, correctness)
- Rayon tuning (stability)
- Gzip parallel decompression (library compatibility)

**High Risk** (prototype separately):
- NUMA allocation (hardware-specific)
- Lock-free data structures (subtle bugs)
- GPU acceleration (major architecture change)

---

## References

- C++ bwa-mem2 threading: `src/fastmap.cpp:674-810`
- Rayon documentation: https://docs.rs/rayon
- NUMA programming: `man 3 numa`
- Intel optimization guide: Software.intel.com/optimization-manual
