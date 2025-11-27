use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use std::collections::BTreeMap;

// Import KBTree from the library
use ferrous_align::core::kbtree::KBTree;

/// Simple chain data for benchmarking (matches chaining use case)
#[derive(Clone, Copy, Default)]
struct ChainData {
    score: i32,
    query_start: i32,
    query_end: i32,
    is_rev: bool,
}

/// Generate pseudo-random positions like seed chaining would produce
fn generate_positions(n: usize, seed: u64) -> Vec<u64> {
    let mut rng = seed;
    let mut positions = Vec::with_capacity(n);
    let mut pos = 0u64;

    for _ in 0..n {
        // Seeds are roughly sorted with some variation
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let delta = 50 + (rng % 500); // 50-550 bp gaps
        pos += delta;
        positions.push(pos);
    }
    positions
}

/// Benchmark BTreeMap interval query simulation
fn bench_btreemap_interval(positions: &[u64]) -> usize {
    let mut tree: BTreeMap<u64, usize> = BTreeMap::new();
    let mut merges = 0;

    for (i, &pos) in positions.iter().enumerate() {
        // Simulate interval query: find closest position <= current
        if let Some((&_lower_key, &_lower_idx)) = tree.range(..=pos).next_back() {
            // Would do test_and_merge here
            merges += 1;
        }
        tree.insert(pos, i);
    }
    merges
}

/// Benchmark KBTree interval query
fn bench_kbtree_interval(positions: &[u64]) -> usize {
    let mut tree: KBTree<u64, usize> = KBTree::new();
    let mut merges = 0;

    for (i, &pos) in positions.iter().enumerate() {
        // Simulate interval query: find closest position <= current
        if let Some(_lower) = tree.get_lower_mut(&pos) {
            // Would do test_and_merge here
            merges += 1;
        }
        tree.insert(pos, i);
    }
    merges
}

fn bench_chaining_data_structures(c: &mut Criterion) {
    let mut group = c.benchmark_group("chaining_data_structure");

    // Test various sizes typical of chaining
    for n_seeds in [50, 100, 200, 500, 1000].iter() {
        let positions = generate_positions(*n_seeds, 42);

        group.throughput(Throughput::Elements(*n_seeds as u64));

        group.bench_with_input(
            BenchmarkId::new("BTreeMap", n_seeds),
            &positions,
            |b, pos| b.iter(|| black_box(bench_btreemap_interval(pos))),
        );

        group.bench_with_input(BenchmarkId::new("KBTree", n_seeds), &positions, |b, pos| {
            b.iter(|| black_box(bench_kbtree_interval(pos)))
        });
    }

    group.finish();
}

/// Benchmark with actual chain-like data
fn bench_chaining_realistic(c: &mut Criterion) {
    let mut group = c.benchmark_group("chaining_realistic");

    // Simulate multiple reads being chained (typical batch)
    let n_reads = 100;
    let seeds_per_read = 50;

    // Generate data for all reads
    let mut all_positions: Vec<Vec<u64>> = Vec::with_capacity(n_reads);
    for i in 0..n_reads {
        all_positions.push(generate_positions(seeds_per_read, 42 + i as u64));
    }

    group.throughput(Throughput::Elements((n_reads * seeds_per_read) as u64));

    group.bench_function("BTreeMap_batch", |b| {
        b.iter(|| {
            let mut total_merges = 0;
            for positions in &all_positions {
                total_merges += bench_btreemap_interval(positions);
            }
            black_box(total_merges)
        })
    });

    group.bench_function("KBTree_batch", |b| {
        b.iter(|| {
            let mut total_merges = 0;
            for positions in &all_positions {
                total_merges += bench_kbtree_interval(positions);
            }
            black_box(total_merges)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_chaining_data_structures,
    bench_chaining_realistic
);
criterion_main!(benches);
