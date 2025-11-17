use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use ferrous_align::banded_swa::{BandedPairWiseSW, bwa_fill_scmat};

fn generate_random_sequence(len: usize, seed: u64) -> Vec<u8> {
    // Simple LCG random number generator for reproducible sequences
    let mut rng = seed;
    (0..len)
        .map(|_| {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            (rng / 65536) as u8 % 4 // Generate bases 0-3 (A, C, G, T)
        })
        .collect()
}

fn generate_sequence_with_mutations(seq: &[u8], mutation_rate: f64, seed: u64) -> Vec<u8> {
    let mut rng = seed;
    seq.iter()
        .map(|&base| {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let rand_val = (rng % 1000) as f64 / 1000.0;
            if rand_val < mutation_rate {
                // Mutate to a different base
                ((base + 1 + ((rng / 1000) % 3) as u8) % 4)
            } else {
                base
            }
        })
        .collect()
}

/// Benchmark scalar vs batched SIMD for varying sequence lengths
fn bench_scalar_vs_batched(c: &mut Criterion) {
    use ferrous_align::simd_abstraction::{SimdEngineType, detect_optimal_simd_engine};

    let mat = bwa_fill_scmat(1, 4, -1);
    let bsw = BandedPairWiseSW::new(6, 1, 6, 1, 100, 100, mat, 1, 1);

    let mut group = c.benchmark_group("smith_waterman");

    // Determine optimal batch size based on SIMD engine
    let engine = detect_optimal_simd_engine();
    let batch_size = match engine {
        #[cfg(target_arch = "x86_64")]
        SimdEngineType::Engine256 => 32,
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        SimdEngineType::Engine512 => 64,
        SimdEngineType::Engine128 => 16,
    };

    // Test different sequence lengths: 50bp, 100bp, 150bp
    for seq_len in [50, 100, 150].iter() {
        let query = generate_random_sequence(*seq_len, 42);
        let target = generate_sequence_with_mutations(&query, 0.05, 123); // 5% mutation rate

        // Benchmark scalar version
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::new("scalar", seq_len), seq_len, |b, &_size| {
            b.iter(|| {
                bsw.scalar_banded_swa(
                    black_box(query.len() as i32),
                    black_box(&query),
                    black_box(target.len() as i32),
                    black_box(&target),
                    black_box(100),
                    black_box(0),
                )
            })
        });

        // Benchmark batched SIMD version with optimal batch size
        let batch: Vec<_> = (0..batch_size)
            .map(|_| {
                (
                    query.len() as i32,
                    query.as_slice(),
                    target.len() as i32,
                    target.as_slice(),
                    100i32,
                    0i32,
                )
            })
            .collect();

        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("batched_simd", seq_len),
            seq_len,
            |b, &_size| b.iter(|| bsw.simd_banded_swa_dispatch(black_box(&batch))),
        );
    }

    group.finish();
}

/// Benchmark different batch sizes to find optimal batching
fn bench_batch_sizes(c: &mut Criterion) {
    let mat = bwa_fill_scmat(1, 4, -1);
    let bsw = BandedPairWiseSW::new(6, 1, 6, 1, 100, 100, mat, 1, 1);

    let mut group = c.benchmark_group("batch_sizes");

    let seq_len = 100;
    let num_alignments: usize = 128; // Total alignments to process

    // Generate diverse set of alignments
    let alignments: Vec<(Vec<u8>, Vec<u8>)> = (0..num_alignments)
        .map(|i| {
            let query = generate_random_sequence(seq_len, 42 + i as u64);
            let target = generate_sequence_with_mutations(&query, 0.05, 123 + i as u64);
            (query, target)
        })
        .collect();

    // Benchmark scalar: process all alignments one by one
    group.throughput(Throughput::Elements(num_alignments as u64));
    group.bench_function("scalar_128x", |b| {
        b.iter(|| {
            for (query, target) in &alignments {
                black_box(bsw.scalar_banded_swa(
                    query.len() as i32,
                    query,
                    target.len() as i32,
                    target,
                    100,
                    0,
                ));
            }
        })
    });

    // Benchmark auto-dispatch: uses optimal batch size for detected SIMD engine
    // SSE2 → 16, AVX2 → 32, AVX-512 → 64
    group.throughput(Throughput::Elements(num_alignments as u64));
    group.bench_function("auto_dispatch_128x", |b| {
        // Determine optimal batch size based on SIMD engine
        use ferrous_align::simd_abstraction::{SimdEngineType, detect_optimal_simd_engine};
        let engine = detect_optimal_simd_engine();
        let batch_size = match engine {
            #[cfg(target_arch = "x86_64")]
            SimdEngineType::Engine256 => 32,
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            SimdEngineType::Engine512 => 64,
            SimdEngineType::Engine128 => 16,
        };

        b.iter(|| {
            for chunk_start in (0..num_alignments).step_by(batch_size) {
                let batch: Vec<_> = (0..batch_size.min(num_alignments - chunk_start))
                    .map(|i| {
                        let idx = chunk_start + i;
                        let (query, target) = &alignments[idx];
                        (
                            query.len() as i32,
                            query.as_slice(),
                            target.len() as i32,
                            target.as_slice(),
                            100i32,
                            0i32,
                        )
                    })
                    .collect();

                // Pad to batch_size if needed
                let mut padded_batch = batch;
                while padded_batch.len() < batch_size {
                    let (query, target) = &alignments[0];
                    padded_batch.push((
                        query.len() as i32,
                        query.as_slice(),
                        target.len() as i32,
                        target.as_slice(),
                        100i32,
                        0i32,
                    ));
                }

                black_box(bsw.simd_banded_swa_dispatch(&padded_batch));
            }
        })
    });

    // Keep the old batch16 benchmark for comparison
    group.throughput(Throughput::Elements(num_alignments as u64));
    group.bench_function("batch16_128x_legacy", |b| {
        b.iter(|| {
            for chunk_start in (0..num_alignments).step_by(16) {
                let batch: Vec<_> = (0..16.min(num_alignments - chunk_start))
                    .map(|i| {
                        let idx = chunk_start + i;
                        let (query, target) = &alignments[idx];
                        (
                            query.len() as i32,
                            query.as_slice(),
                            target.len() as i32,
                            target.as_slice(),
                            100i32,
                            0i32,
                        )
                    })
                    .collect();

                // Pad to 16 if needed
                let mut padded_batch = batch;
                while padded_batch.len() < 16 {
                    let (query, target) = &alignments[0];
                    padded_batch.push((
                        query.len() as i32,
                        query.as_slice(),
                        target.len() as i32,
                        target.as_slice(),
                        100i32,
                        0i32,
                    ));
                }

                black_box(bsw.simd_banded_swa_batch16(&padded_batch));
            }
        })
    });

    group.finish();
}

/// Benchmark different mutation rates (affects DP complexity)
fn bench_mutation_rates(c: &mut Criterion) {
    use ferrous_align::simd_abstraction::{SimdEngineType, detect_optimal_simd_engine};

    let mat = bwa_fill_scmat(1, 4, -1);
    let bsw = BandedPairWiseSW::new(6, 1, 6, 1, 100, 100, mat, 1, 1);

    let mut group = c.benchmark_group("mutation_rates");
    let seq_len = 100;

    // Determine optimal batch size based on SIMD engine
    let engine = detect_optimal_simd_engine();
    let batch_size = match engine {
        #[cfg(target_arch = "x86_64")]
        SimdEngineType::Engine256 => 32,
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        SimdEngineType::Engine512 => 64,
        SimdEngineType::Engine128 => 16,
    };

    for mutation_rate in [0.0, 0.05, 0.10, 0.20].iter() {
        let query = generate_random_sequence(seq_len, 42);
        let target = generate_sequence_with_mutations(&query, *mutation_rate, 123);

        // Scalar
        group.bench_with_input(
            BenchmarkId::new("scalar", format!("{:.0}%", mutation_rate * 100.0)),
            mutation_rate,
            |b, &_rate| {
                b.iter(|| {
                    bsw.scalar_banded_swa(
                        black_box(query.len() as i32),
                        black_box(&query),
                        black_box(target.len() as i32),
                        black_box(&target),
                        black_box(100),
                        black_box(0),
                    )
                })
            },
        );

        // Batched SIMD with optimal batch size
        let batch: Vec<_> = (0..batch_size)
            .map(|_| {
                (
                    query.len() as i32,
                    query.as_slice(),
                    target.len() as i32,
                    target.as_slice(),
                    100i32,
                    0i32,
                )
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::new("batched_simd", format!("{:.0}%", mutation_rate * 100.0)),
            mutation_rate,
            |b, &_rate| b.iter(|| bsw.simd_banded_swa_dispatch(black_box(&batch))),
        );
    }

    group.finish();
}

/// Benchmark hybrid batched approach with CIGAR generation
fn bench_hybrid_with_cigar(c: &mut Criterion) {
    let mat = bwa_fill_scmat(1, 4, -1);
    let bsw = BandedPairWiseSW::new(6, 1, 6, 1, 100, 100, mat, 1, 1);

    let mut group = c.benchmark_group("hybrid_cigar");

    // Test with 100bp sequences (typical read length)
    let seq_len = 100;

    // Generate 16 different alignments
    let mut batch_data = Vec::new();
    let mut queries = Vec::new();
    let mut targets = Vec::new();

    for i in 0..16 {
        let query = generate_random_sequence(seq_len, 42 + i as u64);
        let target = generate_sequence_with_mutations(&query, 0.05, 123 + i as u64);
        queries.push(query);
        targets.push(target);
    }

    for i in 0..16 {
        batch_data.push((
            queries[i].len() as i32,
            queries[i].as_slice(),
            targets[i].len() as i32,
            targets[i].as_slice(),
            100,
            0,
        ));
    }

    // Benchmark: 16x scalar with CIGAR (baseline)
    group.throughput(Throughput::Elements(16));
    group.bench_function("scalar_16x_with_cigar", |b| {
        b.iter(|| {
            for i in 0..16 {
                let _ = bsw.scalar_banded_swa(
                    black_box(queries[i].len() as i32),
                    black_box(&queries[i]),
                    black_box(targets[i].len() as i32),
                    black_box(&targets[i]),
                    black_box(100),
                    black_box(0),
                );
            }
        });
    });

    // Benchmark: Hybrid batched with CIGAR
    group.bench_function("hybrid_batch16_with_cigar", |b| {
        b.iter(|| bsw.simd_banded_swa_batch16_with_cigar(black_box(&batch_data)));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_scalar_vs_batched,
    bench_batch_sizes,
    bench_mutation_rates,
    bench_hybrid_with_cigar
);
criterion_main!(benches);
