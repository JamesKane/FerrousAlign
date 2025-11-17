use criterion::{criterion_group, criterion_main, Criterion};
use std::env;
use std::process::{Command, Stdio};
use std::path::Path;

fn run_bwa_mem2(bwa_path: &str, ref_path: &str, read1_path: &str, read2_path: &str) {
    let output = Command::new(bwa_path)
        .arg("mem")
        .arg(ref_path)
        .arg(read1_path)
        .arg(read2_path)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .expect("Failed to execute bwa-mem2");

    if !output.status.success() {
        panic!("bwa-mem2 failed with status: {}", output.status);
    }
}

fn run_ferrous_align(ref_path: &str, read1_path: &str, read2_path: &str) {
    let output = Command::new("cargo")
        .arg("run")
        .arg("--release")
        .arg("--")
        .arg("mem")
        .arg("-p") // paired-end reads
        .arg(ref_path)
        .arg(read1_path)
        .arg(read2_path)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .expect("Failed to execute FerrousAlign");

    if !output.status.success() {
        panic!("FerrousAlign failed with status: {}", output.status);
    }
}

fn benchmark_comparison(c: &mut Criterion) {
    let bwa_path = env::var("BWA_MEM2_PATH").unwrap_or_else(|_| "/tmp/bwa-mem2-diag/bwa-mem2".to_string());
    // Use chrM for faster benchmarking (or set via environment variable)
    let ref_path = env::var("REF_PATH").unwrap_or_else(|_| "test_data/chrM.fna".to_string());
    let read1_path = env::var("READ1_PATH").unwrap_or_else(|_| "/home/jkane/Genomics/HG002/test_20pairs_R1.fq".to_string());
    let read2_path = env::var("READ2_PATH").unwrap_or_else(|_| "/home/jkane/Genomics/HG002/test_20pairs_R2.fq".to_string());

    // Check if files exist
    if !Path::new(&bwa_path).exists() {
        eprintln!("bwa-mem2 executable not found at {}", bwa_path);
        return;
    }
    if !Path::new(ref_path).exists() {
        eprintln!("Reference file not found at {}", ref_path);
        return;
    }
    if !Path::new(&read1_path).exists() || !Path::new(&read2_path).exists() {
        eprintln!("Read files not found");
        return;
    }

    // Index reference for bwa-mem2 if needed
    let bwa_index_files = ["amb", "ann", "bwt.2bit.64", "pac"].iter().all(|ext| Path::new(&format!("{}.{}", ref_path, ext)).exists());
    if !bwa_index_files {
        Command::new(&bwa_path)
            .arg("index")
            .arg(ref_path)
            .output()
            .expect("Failed to index reference for bwa-mem2");
    }
    
    // Index reference for FerrousAlign if needed
    let fa_index_file = format!("{}.fai", ref_path);
    if !Path::new(&fa_index_file).exists() {
         Command::new("samtools")
            .arg("faidx")
            .arg(ref_path)
            .output()
            .expect("Failed to index reference for FerrousAlign with samtools");
    }


    let mut group = c.benchmark_group("FerrousAlign vs bwa-mem2");

    group.bench_function("bwa-mem2", |b| {
        b.iter(|| run_bwa_mem2(&bwa_path, ref_path, &read1_path, &read2_path))
    });

    group.bench_function("FerrousAlign", |b| {
        b.iter(|| run_ferrous_align(ref_path, &read1_path, &read2_path))
    });

    group.finish();
}

criterion_group!(benches, benchmark_comparison);
criterion_main!(benches);
