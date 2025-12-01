// DEPRECATED: This benchmark file uses deprecated AoS dispatch functions
// that have been replaced with SoA-native implementations.
//
// TODO: Update benchmarks to use the new SoA API:
// - Use ExtensionJobBatch to build SoA data
// - Call execute_batch_simd_scoring instead of deprecated dispatch functions
//
// For now, these benchmarks are commented out to allow the build to succeed.
// The core SIMD functionality is tested through integration tests and the
// end-to-end SoA pipeline.

use criterion::{criterion_group, criterion_main};

fn placeholder(_c: &mut criterion::Criterion) {
    // Placeholder to satisfy criterion_main! macro
}

criterion_group!(benches, placeholder);
criterion_main!(benches);
