// benches/align_perf.rs
// Criterion benchmarks for banded_swa and kswv paths across ISAs.

#![allow(unsafe_op_in_unsafe_fn)]

use criterion::{BatchSize, Criterion, Throughput, black_box, criterion_group, criterion_main};
use rand::{Rng, SeedableRng, rngs::StdRng};

use ferrous_align::core::alignment::banded_swa::OutScore;
#[cfg(target_arch = "x86_64")]
use ferrous_align::core::alignment::banded_swa::isa_avx2::{
    simd_banded_swa_batch16_int16_soa, simd_banded_swa_batch32_soa,
};
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
use ferrous_align::core::alignment::banded_swa::isa_avx512_int8::simd_banded_swa_batch64_soa;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use ferrous_align::core::alignment::banded_swa::isa_sse_neon::simd_banded_swa_batch16_soa;
use ferrous_align::core::alignment::banded_swa::shared::{
    SoAInputs, SoAInputs16, pad_batch, soa_transform,
};

#[cfg(target_arch = "x86_64")]
use ferrous_align::core::alignment::kswv_avx2::kswv_batch32_soa as kswv32_soa;
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
use ferrous_align::core::alignment::kswv_avx512::kswv_batch64_soa as kswv64_soa;
use ferrous_align::core::alignment::kswv_batch::KswResult;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use ferrous_align::core::alignment::kswv_sse_neon::kswv_batch16_soa as kswv16_soa;
use ferrous_align::core::alignment::shared_types::KswSoA;
use ferrous_align::core::alignment::workspace::AlignmentWorkspace;

fn make_scoring_matrix() -> [i8; 25] {
    // Simple 5x5 matrix: match=1 on diagonal, mismatch=0 elsewhere
    let mut mat = [0i8; 25];
    for i in 0..4 {
        mat[i * 5 + i] = 1;
    }
    mat
}

fn make_batch(len: usize, lanes: usize) -> Vec<(i32, Vec<u8>, i32, Vec<u8>, i32, i32)> {
    let mut rng = StdRng::seed_from_u64(0xDEADBEEFCAFEBABE);
    let mut batch = Vec::with_capacity(lanes);
    for _ in 0..lanes {
        let mut q = vec![0u8; len];
        let mut t = vec![0u8; len];
        for i in 0..len {
            q[i] = rng.gen_range(0..4);
            t[i] = rng.gen_range(0..4);
        }
        // (qlen, &q, tlen, &t, band, h0)
        batch.push((len as i32, q, len as i32, t, 20, 0));
    }
    batch
}

fn bench_banded_swa(c: &mut Criterion) {
    let mut group = c.benchmark_group("banded_swa_soa");
    let mat = make_scoring_matrix();
    let params = [(64usize, 10i32), (128, 20), (151, 20), (256, 50), (400, 50)];

    // SSE/NEON width 16 (i8)
    #[allow(unused_variables)]
    {
        for (len, band) in params.iter().copied() {
            // Skip >128 for i8 benches to mirror dispatch policy (but allow measuring behavior if desired)
            if len > 128 {
                continue;
            }
            group.throughput(Throughput::Bytes((len as u64) * 16 * 2));
            group.bench_function(format!("i8_w16_len{len}_band{band}"), |b| {
                // Prepare inputs per iteration to avoid measuring allocs repeatedly; use batch to clone refs
                let batch_spec = make_batch(len, 16);
                // AoS view for pad/soa helper expects borrowed slices
                let batch_view: Vec<(i32, &[u8], i32, &[u8], i32, i32)> = batch_spec
                    .iter()
                    .map(|(ql, q, tl, t, w, h0)| (*ql, q.as_slice(), *tl, t.as_slice(), *w, *h0))
                    .collect();
                let (qlen, tlen, h0, w_arr, max_q, max_t, padded) = pad_batch::<16>(&batch_view);
                // MAX can be len rounded up to a small bound
                let (qsoa, tsoa) = soa_transform::<16, 512>(&padded);
                let inputs = SoAInputs {
                    query_soa: &qsoa,
                    target_soa: &tsoa,
                    qlen: &qlen,
                    tlen: &tlen,
                    w: &w_arr,
                    h0: &h0,
                    lanes: 16,
                    max_qlen: max_q,
                    max_tlen: max_t,
                };
                b.iter_batched(
                    || (),
                    |_| unsafe {
                        let _out: Vec<OutScore> =
                            simd_banded_swa_batch16_soa(&inputs, 16, 6, 1, 6, 1, 100, &mat, 5);
                        black_box(_out)
                    },
                    BatchSize::SmallInput,
                );
            });
        }
    }

    // AVX2 width 32 (i8) and AVX2 width 16 (i16)
    #[cfg(target_arch = "x86_64")]
    {
        for (len, band) in params.iter().copied() {
            if len > 128 {
                continue;
            }
            group.throughput(Throughput::Bytes((len as u64) * 32 * 2));
            group.bench_function(format!("i8_w32_len{len}_band{band}"), |b| {
                let batch_spec = make_batch(len, 32);
                let batch_view: Vec<(i32, &[u8], i32, &[u8], i32, i32)> = batch_spec
                    .iter()
                    .map(|(ql, q, tl, t, w, h0)| (*ql, q.as_slice(), *tl, t.as_slice(), *w, *h0))
                    .collect();
                let (qlen, tlen, h0, w_arr, max_q, max_t, padded) = pad_batch::<32>(&batch_view);
                let (qsoa, tsoa) = soa_transform::<32, 512>(&padded);
                let inputs = SoAInputs {
                    query_soa: &qsoa,
                    target_soa: &tsoa,
                    qlen: &qlen,
                    tlen: &tlen,
                    w: &w_arr,
                    h0: &h0,
                    lanes: 32,
                    max_qlen: max_q,
                    max_tlen: max_t,
                };
                b.iter_batched(
                    || (),
                    |_| unsafe {
                        let _out =
                            simd_banded_swa_batch32_soa(&inputs, 32, 6, 1, 6, 1, 100, &mat, 5);
                        black_box(_out)
                    },
                    BatchSize::SmallInput,
                );
            });
        }

        // i16 W16 for longer reads up to 400bp
        for (len, band) in params.iter().copied() {
            group.throughput(Throughput::Bytes((len as u64) * 16 * 2));
            group.bench_function(format!("i16_w16_len{len}_band{band}"), |b| {
                let batch_spec = make_batch(len, 16);
                let batch_view: Vec<(i32, &[u8], i32, &[u8], i32, i32)> = batch_spec
                    .iter()
                    .map(|(ql, q, tl, t, w, h0)| (*ql, q.as_slice(), *tl, t.as_slice(), *w, *h0))
                    .collect();
                let (qlen_i8, tlen_i8, h0_i8, w_arr, max_q, max_t, padded) =
                    pad_batch::<16>(&batch_view);
                let (qsoa_u8, tsoa_u8) = soa_transform::<16, 512>(&padded);
                // convert to i16 SoA
                let qsoa_i16: Vec<i16> = qsoa_u8.iter().map(|&v| v as i16).collect();
                let tsoa_i16: Vec<i16> = tsoa_u8.iter().map(|&v| v as i16).collect();
                let mut h0_i16 = [0i16; 16];
                let mut qlen_i16 = [0i16; 16];
                let mut tlen_i16 = [0i16; 16];
                for i in 0..16 {
                    h0_i16[i] = h0_i8[i] as i16;
                    qlen_i16[i] = qlen_i8[i] as i16;
                    tlen_i16[i] = tlen_i8[i] as i16;
                }
                let inputs16 = SoAInputs16 {
                    query_soa: &qsoa_i16,
                    target_soa: &tsoa_i16,
                    qlen: &qlen_i16,
                    tlen: &tlen_i16,
                    h0: &h0_i16,
                    w: &w_arr,
                    max_qlen: max_q,
                    max_tlen: max_t,
                };
                b.iter_batched(
                    || (),
                    |_| unsafe {
                        let _out = simd_banded_swa_batch16_int16_soa(
                            &inputs16, 16, 6, 1, 6, 1, 100, &mat, 5,
                        );
                        black_box(_out)
                    },
                    BatchSize::SmallInput,
                );
            });
        }
    }

    // AVX-512 width 64 (i8)
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    {
        if std::is_x86_feature_detected!("avx512bw") {
            for (len, band) in params.iter().copied() {
                if len > 128 {
                    continue;
                }
                group.throughput(Throughput::Bytes((len as u64) * 64 * 2));
                group.bench_function(format!("i8_w64_len{}_band{}", len, band), |b| {
                    let batch_spec = make_batch(len, 64);
                    let batch_view: Vec<(i32, &[u8], i32, &[u8], i32, i32)> = batch_spec
                        .iter()
                        .map(|(ql, q, tl, t, w, h0)| {
                            (*ql, q.as_slice(), *tl, t.as_slice(), *w, *h0)
                        })
                        .collect();
                    let (qlen, tlen, h0, w_arr, max_q, max_t, padded) =
                        pad_batch::<64>(&batch_view);
                    let (qsoa, tsoa) = soa_transform::<64, 512>(&padded);
                    let inputs = SoAInputs {
                        query_soa: &qsoa,
                        target_soa: &tsoa,
                        qlen: &qlen,
                        tlen: &tlen,
                        w: &w_arr,
                        h0: &h0,
                        lanes: 64,
                        max_qlen: max_q,
                        max_tlen: max_t,
                    };
                    b.iter_batched(
                        || (),
                        |_| unsafe {
                            let _out =
                                simd_banded_swa_batch64_soa(&inputs, 64, 6, 1, 6, 1, 100, &mat, 5);
                            black_box(_out)
                        },
                        BatchSize::SmallInput,
                    );
                });
            }
        } else {
            eprintln!("Skipping AVX-512 banded_swa benches: CPU lacks avx512bw");
        }
    }

    group.finish();
}

fn bench_kswv(c: &mut Criterion) {
    let mut group = c.benchmark_group("kswv_soa");
    let params = [(64usize, 10i32), (128, 20), (151, 20), (256, 50)];

    // SSE/NEON width 16 (i8)
    for (len, _band) in params.iter().copied() {
        if len > 128 {
            continue;
        }
        group.throughput(Throughput::Bytes((len as u64) * 16 * 2));
        group.bench_function(format!("i8_w16_len{len}"), |b| {
            let batch_spec = make_batch(len, 16);
            // Build KswSoA directly from the AoS buffers (already generated)
            let mut qsoa = vec![0xFFu8; len * 16];
            let mut rsoa = vec![0xFFu8; len * 16];
            for lane in 0..16 {
                let (_, q, _, t, _, _) = &batch_spec[lane];
                for pos in 0..len {
                    qsoa[pos * 16 + lane] = q[pos];
                    rsoa[pos * 16 + lane] = t[pos];
                }
            }
            let qlen = vec![len as i8; 16];
            let tlen = vec![len as i8; 16];
            let zeros = vec![0i8; 16];
            let inputs = KswSoA {
                ref_soa: &rsoa,
                query_soa: &qsoa,
                qlen: &qlen,
                tlen: &tlen,
                band: &zeros,
                h0: &zeros,
                lanes: 16,
                max_qlen: len as i32,
                max_tlen: len as i32,
            };
            b.iter_batched(
                || AlignmentWorkspace::new(),
                |mut ws| unsafe {
                    let _r: Vec<KswResult> =
                        kswv16_soa(&inputs, &mut ws, 16, 1, 0, 6, 1, 6, 1, -1, false);
                    black_box(_r)
                },
                BatchSize::SmallInput,
            );
        });
    }

    // AVX2 width 32
    #[cfg(target_arch = "x86_64")]
    for (len, _band) in params.iter().copied() {
        if len > 128 {
            continue;
        }
        group.throughput(Throughput::Bytes((len as u64) * 32 * 2));
        group.bench_function(format!("i8_w32_len{len}"), |b| {
            let batch_spec = make_batch(len, 32);
            let mut qsoa = vec![0xFFu8; len * 32];
            let mut rsoa = vec![0xFFu8; len * 32];
            for lane in 0..32 {
                let (_, q, _, t, _, _) = &batch_spec[lane];
                for pos in 0..len {
                    qsoa[pos * 32 + lane] = q[pos];
                    rsoa[pos * 32 + lane] = t[pos];
                }
            }
            let qlen = vec![len as i8; 32];
            let tlen = vec![len as i8; 32];
            let zeros = vec![0i8; 32];
            let inputs = KswSoA {
                ref_soa: &rsoa,
                query_soa: &qsoa,
                qlen: &qlen,
                tlen: &tlen,
                band: &zeros,
                h0: &zeros,
                lanes: 32,
                max_qlen: len as i32,
                max_tlen: len as i32,
            };
            b.iter_batched(
                || AlignmentWorkspace::new(),
                |mut ws| unsafe {
                    let _r: Vec<KswResult> =
                        kswv32_soa(&inputs, &mut ws, 32, 1, 0, 6, 1, 6, 1, -1, false);
                    black_box(_r)
                },
                BatchSize::SmallInput,
            );
        });
    }

    // AVX-512 width 64
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if std::is_x86_feature_detected!("avx512bw") {
        // Allow temporarily skipping unstable W64 kswv benches via env var
        let skip_w64 = std::env::var("SKIP_KSWV_W64").is_ok();
        if skip_w64 {
            eprintln!("Skipping AVX-512 kswv W64 benches due to SKIP_KSWV_W64 env var");
        } else {
            for (len, _band) in params.iter().copied() {
                if len > 128 {
                    continue;
                }
                group.throughput(Throughput::Bytes((len as u64) * 64 * 2));
                group.bench_function(format!("i8_w64_len{}", len), |b| {
                    let batch_spec = make_batch(len, 64);
                    let mut qsoa = vec![0xFFu8; len * 64];
                    let mut rsoa = vec![0xFFu8; len * 64];
                    for lane in 0..64 {
                        let (_, q, _, t, _, _) = &batch_spec[lane];
                        for pos in 0..len {
                            qsoa[pos * 64 + lane] = q[pos];
                            rsoa[pos * 64 + lane] = t[pos];
                        }
                    }
                    let qlen = vec![len as i8; 64];
                    let tlen = vec![len as i8; 64];
                    let zeros = vec![0i8; 64];
                    let inputs = KswSoA {
                        ref_soa: &rsoa,
                        query_soa: &qsoa,
                        qlen: &qlen,
                        tlen: &tlen,
                        band: &zeros,
                        h0: &zeros,
                        lanes: 64,
                        max_qlen: len as i32,
                        max_tlen: len as i32,
                    };
                    b.iter_batched(
                        || AlignmentWorkspace::new(),
                        |mut ws| unsafe {
                            let _r: Vec<KswResult> =
                                kswv64_soa(&inputs, &mut ws, 64, 1, 0, 6, 1, 6, 1, -1, false);
                            black_box(_r)
                        },
                        BatchSize::SmallInput,
                    );
                });
            }
        }
    } else {
        eprintln!("Skipping AVX-512 kswv benches: CPU lacks avx512bw");
    }

    group.finish();
}

fn configure() -> Criterion {
    Criterion::default().sample_size(20)
}

criterion_group! {
    name = benches;
    config = configure();
    targets = bench_banded_swa, bench_kswv
}
criterion_main!(benches);
