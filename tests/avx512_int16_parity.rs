#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
use ferrous_align::alignment::banded_swa::isa_avx512_int16::simd_banded_swa_batch32_int16;

fn mk_mat_match1() -> [i8; 25] {
    let mut m = [0i8; 25];
    for i in 0..4 {
        m[i * 5 + i] = 1;
    }
    m
}

#[test]
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
fn avx512_i16_deterministic_cases() {
    let mat = mk_mat_match1();

    // Cases exercise long reads (>128), narrow bands, non-zero h0, and Ns/padding
    let cases: Vec<(Vec<u8>, Vec<u8>, i32, i32, i32)> = vec![
        (vec![0; 256], vec![0; 256], 256, 256, 0), // long perfect match
        (
            vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1],
            vec![3, 2, 1, 0, 3, 2, 1, 0, 3, 2],
            10,
            10,
            1,
        ), // w=1 edge
        (
            vec![0, 4, 1, 4, 2, 4, 3, 4, 0, 4],
            vec![0, 4, 1, 4, 2, 4, 3, 4, 0, 4],
            10,
            10,
            5,
        ), // Ns and h0
    ];

    for (q, t, qlen, tlen, w) in cases.into_iter() {
        let batch = vec![(qlen, q.as_slice(), tlen, t.as_slice(), w, 0)];
        let avx2 = unsafe { simd_banded_swa_batch16_int16(&batch, 6, 1, 6, 1, 200, &mat, 5) };
        let avx512 = unsafe { simd_banded_swa_batch32_int16(&batch, 6, 1, 6, 1, 200, &mat, 5) };
        assert_eq!(avx2.len(), avx512.len());
        for (a, b) in avx2.iter().zip(avx512.iter()) {
            assert_eq!(
                a.score, b.score,
                "score: avx2={} avx512={}",
                a.score, b.score
            );
            assert_eq!(a.target_end_pos, b.target_end_pos);
            assert_eq!(a.query_end_pos, b.query_end_pos);
        }
    }
}

#[test]
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
fn avx512_i16_randomized_small_batches() {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xA5A5_5A5A_DEAD_BEEF);
    let mat = mk_mat_match1();

    let jobs = 8usize;
    let mut batch: Vec<(i32, Vec<u8>, i32, Vec<u8>, i32, i32)> = Vec::with_capacity(jobs);
    for _ in 0..jobs {
        let qlen = rng.gen_range(129..=300) as i32; // force i16 path
        let tlen = rng.gen_range(129..=300) as i32;
        let w = rng.gen_range(1..=20) as i32;
        let h0 = rng.gen_range(0..=10) as i32;
        let mut q = vec![0u8; qlen as usize];
        let mut t = vec![0u8; tlen as usize];
        for x in &mut q {
            *x = rng.gen_range(0..=4) as u8;
        }
        for x in &mut t {
            *x = rng.gen_range(0..=4) as u8;
        }
        batch.push((qlen, q, tlen, t, w, h0));
    }
    let batch_s: Vec<(i32, &[u8], i32, &[u8], i32, i32)> = batch
        .iter()
        .map(|(ql, q, tl, t, w, h)| (*ql, q.as_slice(), *tl, t.as_slice(), *w, *h))
        .collect();

    let avx2 = unsafe { simd_banded_swa_batch16_int16(&batch_s, 6, 1, 6, 1, 200, &mat, 5) };
    let avx512 = unsafe { simd_banded_swa_batch32_int16(&batch_s, 6, 1, 6, 1, 200, &mat, 5) };

    assert_eq!(avx2.len(), avx512.len());
    for (a, b) in avx2.iter().zip(avx512.iter()).enumerate() {
        let (i, (a, b)) = (a, b);
        assert_eq!(
            a.score, b.score,
            "lane {i} score avx2={} avx512={}",
            a.score, b.score
        );
        assert_eq!(a.target_end_pos, b.target_end_pos, "lane {i} tend");
        assert_eq!(a.query_end_pos, b.query_end_pos, "lane {i} qend");
    }
}
