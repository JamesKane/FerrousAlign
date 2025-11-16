use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use ferrous_align::banded_swa::{BandedPairWiseSW, bwa_fill_scmat};

fn generate_random_sequence(len: usize, seed: u64) -> Vec<u8> {
    let mut rng = seed;
    (0..len)
        .map(|_| {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            (rng / 65536) as u8 % 4
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
                ((base + 1 + ((rng / 1000) % 3) as u8) % 4)
            } else {
                base
            }
        })
        .collect()
}

/// Detect and report available SIMD features
fn detect_simd_features() {
    eprintln!("\n=== SIMD Feature Detection ===");

    #[cfg(target_arch = "x86_64")]
    {
        eprintln!("Platform: x86_64");
        eprintln!("SSE2:     {}", is_x86_feature_detected!("sse2"));
        eprintln!("AVX2:     {}", is_x86_feature_detected!("avx2"));
        eprintln!("AVX-512F: {}", is_x86_feature_detected!("avx512f"));
        eprintln!("AVX-512BW:{}", is_x86_feature_detected!("avx512bw"));

        if is_x86_feature_detected!("avx512bw") {
            eprintln!("Expected SIMD engine: AVX-512 (512-bit, 64-way parallelism)");
        } else if is_x86_feature_detected!("avx2") {
            eprintln!("Expected SIMD engine: AVX2 (256-bit, 32-way parallelism)");
        } else {
            eprintln!("Expected SIMD engine: SSE2 (128-bit, 16-way parallelism)");
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        eprintln!("Platform: aarch64 (ARM)");
        eprintln!("NEON: Available (128-bit, 16-way parallelism)");
    }

    eprintln!("==============================\n");
}

/// Benchmark comprehensive SIMD engine comparison
fn bench_simd_engine_comparison(c: &mut Criterion) {
    detect_simd_features();

    let mat = bwa_fill_scmat(1, 4, -1);
    let bsw = BandedPairWiseSW::new(6, 1, 6, 1, 100, 100, mat, 1, 1);

    let mut group = c.benchmark_group("engine_comparison");

    // Test with realistic 100bp sequences (typical short read length)
    let seq_len = 100;
    let num_alignments = 128;

    // Generate diverse set of alignments
    let alignments: Vec<(Vec<u8>, Vec<u8>)> = (0..num_alignments)
        .map(|i| {
            let query = generate_random_sequence(seq_len, 42 + i as u64);
            let target = generate_sequence_with_mutations(&query, 0.05, 123 + i as u64);
            (query, target)
        })
        .collect();

    // Benchmark 1: Scalar baseline (no SIMD)
    group.throughput(Throughput::Elements(num_alignments));
    group.bench_function("scalar_128x_100bp", |b| {
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

    // Benchmark 2: Auto-detected SIMD (batch of 16)
    group.bench_function("auto_simd_128x_100bp", |b| {
        b.iter(|| {
            for chunk_start in (0..num_alignments).step_by(16) {
                let batch: Vec<_> = (0..16.min(num_alignments - chunk_start))
                    .map(|i| {
                        let idx = chunk_start + i as u64;
                        let (query, target) = &alignments[idx as usize];
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

/// Benchmark different sequence lengths with SIMD
fn bench_sequence_lengths(c: &mut Criterion) {
    let mat = bwa_fill_scmat(1, 4, -1);
    let bsw = BandedPairWiseSW::new(6, 1, 6, 1, 100, 100, mat, 1, 1);

    let mut group = c.benchmark_group("sequence_lengths");

    // Test typical read lengths: 50bp, 100bp, 150bp, 250bp
    for seq_len in [50, 100, 150, 250].iter() {
        let num_alignments = 64;

        let alignments: Vec<(Vec<u8>, Vec<u8>)> = (0..num_alignments)
            .map(|i| {
                let query = generate_random_sequence(*seq_len, 42 + i as u64);
                let target = generate_sequence_with_mutations(&query, 0.05, 123 + i as u64);
                (query, target)
            })
            .collect();

        // Scalar
        group.throughput(Throughput::Elements(num_alignments));
        group.bench_with_input(BenchmarkId::new("scalar", seq_len), seq_len, |b, &_size| {
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

        // Auto-detected SIMD
        group.bench_with_input(
            BenchmarkId::new("auto_simd", seq_len),
            seq_len,
            |b, &_size| {
                b.iter(|| {
                    for chunk_start in (0..num_alignments).step_by(16) {
                        let batch: Vec<_> = (0..16.min(num_alignments - chunk_start))
                            .map(|i| {
                                let idx = chunk_start + i as u64;
                                let (query, target) = &alignments[idx as usize];
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
            },
        );
    }

    group.finish();
}

/// Benchmark with varying mutation rates (affects alignment complexity)
fn bench_alignment_complexity(c: &mut Criterion) {
    let mat = bwa_fill_scmat(1, 4, -1);
    let bsw = BandedPairWiseSW::new(6, 1, 6, 1, 100, 100, mat, 1, 1);

    let mut group = c.benchmark_group("alignment_complexity");

    let seq_len = 100;
    let num_alignments = 64;

    // Test different mutation rates: 0% (perfect match), 5%, 10%, 20% (high divergence)
    for mutation_rate in [0.0, 0.05, 0.10, 0.20].iter() {
        let alignments: Vec<(Vec<u8>, Vec<u8>)> = (0..num_alignments)
            .map(|i| {
                let query = generate_random_sequence(seq_len, 42 + i as u64);
                let target =
                    generate_sequence_with_mutations(&query, *mutation_rate, 123 + i as u64);
                (query, target)
            })
            .collect();

        let label = format!("{:.0}%", mutation_rate * 100.0);

        // Scalar
        group.throughput(Throughput::Elements(num_alignments));
        group.bench_with_input(
            BenchmarkId::new("scalar", &label),
            mutation_rate,
            |b, &_rate| {
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
            },
        );

        // Auto-detected SIMD
        group.bench_with_input(
            BenchmarkId::new("auto_simd", &label),
            mutation_rate,
            |b, &_rate| {
                b.iter(|| {
                    for chunk_start in (0..num_alignments).step_by(16) {
                        let batch: Vec<_> = (0..16.min(num_alignments - chunk_start))
                            .map(|i| {
                                let idx = chunk_start + i as u64;
                                let (query, target) = &alignments[idx as usize];
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
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_simd_engine_comparison,
    bench_sequence_lengths,
    bench_alignment_complexity
);
criterion_main!(benches);
