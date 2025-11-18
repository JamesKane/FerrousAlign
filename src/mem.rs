// bwa-mem2-rust/src/mem.rs

use crate::bntseq::BntSeq;
use crate::bwt::Bwt;
use crate::fastq_reader::FastqReader;
use crate::mem_opt::MemOpt;
use crossbeam_channel::{Receiver, Sender, bounded};
use rayon::prelude::*;
use std::fs::File;
use std::io::{self, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;

use crate::align;
use crate::align::{CP_SHIFT, CpOcc};

// Batch processing constants (matching C++ bwa-mem2)
// Chunk size in base pairs (from C++ bwamem.cpp mem_opt_init: o->chunk_size = 10000000)
const CHUNK_SIZE_BASES: usize = 10_000_000;
// Assumed average read length for batch size calculation (typical Illumina)
const AVG_READ_LEN: usize = 101;
// Minimum batch size (BATCH_SIZE from C++ macro.h)
const MIN_BATCH_SIZE: usize = 512;

// Paired-end insert size constants (from C++ bwamem_pair.cpp)
#[allow(dead_code)] // Reserved for future use in alignment scoring
const MIN_RATIO: f64 = 0.8; // Minimum ratio for unique alignment
const MIN_DIR_CNT: usize = 10; // Minimum pairs for orientation
const MIN_DIR_RATIO: f64 = 0.05; // Minimum ratio for orientation
const OUTLIER_BOUND: f64 = 2.0; // IQR multiplier for outliers
const MAPPING_BOUND: f64 = 3.0; // IQR multiplier for mapping
const MAX_STDDEV: f64 = 4.0; // Max standard deviations for boundaries

// Read data structure for batching
#[derive(Clone)]
#[allow(dead_code)] // Reserved for future batching optimization
struct ReadBatch {
    names: Vec<String>,
    seqs: Vec<Vec<u8>>,
    quals: Vec<String>,
}

// Message type for pipeline communication
type PairedBatchMessage = Option<(
    crate::fastq_reader::ReadBatch,
    crate::fastq_reader::ReadBatch,
)>;

// Insert size statistics for one orientation
#[derive(Debug, Clone)]
pub struct InsertSizeStats {
    pub avg: f64,     // Mean insert size
    pub std: f64,     // Standard deviation
    pub low: i32,     // Lower bound for proper pairs
    pub high: i32,    // Upper bound for proper pairs
    pub failed: bool, // Whether this orientation has enough data
}

pub struct BwaIndex {
    pub bwt: Bwt,
    pub bns: BntSeq,
    pub cp_occ: Vec<CpOcc>,
    pub sentinel_index: i64,
    pub min_seed_len: i32, // Kept for backwards compatibility, but will use MemOpt
}

impl BwaIndex {
    pub fn bwa_idx_load(prefix: &Path) -> io::Result<Self> {
        let mut bwt = Bwt::new();
        let bns = BntSeq::bns_restore(prefix)?;

        let cp_file_name = PathBuf::from(prefix.to_string_lossy().to_string() + ".bwt.2bit.64");
        let mut cp_file = BufReader::new(File::open(&cp_file_name)?);

        let mut buf_i64 = [0u8; 8];
        let mut buf_u64 = [0u8; 8];
        let mut buf_u8 = [0u8; 1];
        let mut buf_u32 = [0u8; 4];

        // 1. Read seq_len
        cp_file.read_exact(&mut buf_i64)?;
        bwt.seq_len = i64::from_le_bytes(buf_i64) as u64;

        // 2. Read count array (l2)
        for i in 0..5 {
            cp_file.read_exact(&mut buf_i64)?;
            bwt.l2[i] = i64::from_le_bytes(buf_i64) as u64;
        }

        // CRITICAL: Match C++ bwa-mem2 behavior - add 1 to all count values
        // See FMI_search.cpp:435 - this is required for correct SMEM generation
        for i in 0..5 {
            bwt.l2[i] += 1;
        }

        // 3. Read cp_occ array
        let cp_occ_size = (bwt.seq_len >> CP_SHIFT) + 1;
        let mut cp_occ: Vec<CpOcc> = Vec::with_capacity(cp_occ_size as usize);
        for _ in 0..cp_occ_size {
            let mut cp_count = [0i64; 4];
            for i in 0..4 {
                cp_file.read_exact(&mut buf_i64)?;
                cp_count[i] = i64::from_le_bytes(buf_i64);
            }
            let mut one_hot_bwt_str = [0u64; 4];
            for i in 0..4 {
                cp_file.read_exact(&mut buf_u64)?;
                one_hot_bwt_str[i] = u64::from_le_bytes(buf_u64);
            }
            cp_occ.push(CpOcc {
                cp_count,
                one_hot_bwt_str,
            });
        }

        // In C++, SA_COMPX is 3 (defined in macro.h), so sa_intv is 8
        let sa_compx = 3;
        let sa_intv = 1 << sa_compx; // sa_intv = 8
        // C++ uses: ((ref_seq_len >> SA_COMPX) + 1)
        // which equals: (ref_seq_len / 8) + 1
        let sa_len = (bwt.seq_len >> sa_compx) + 1;

        // 4. Read sa_ms_byte array
        bwt.sa_ms_byte.reserve_exact(sa_len as usize);
        for _ in 0..sa_len {
            cp_file.read_exact(&mut buf_u8)?;
            let val = u8::from_le_bytes(buf_u8) as i8;
            bwt.sa_ms_byte.push(val);
        }

        // 5. Read sa_ls_word array
        bwt.sa_ls_word.reserve_exact(sa_len as usize);
        for _ in 0..sa_len {
            cp_file.read_exact(&mut buf_u32)?;
            let val = u32::from_le_bytes(buf_u32);
            bwt.sa_ls_word.push(val);
        }

        // 6. Read sentinel_index
        cp_file.read_exact(&mut buf_i64)?;
        let sentinel_index = i64::from_le_bytes(buf_i64);
        bwt.primary = sentinel_index as u64;

        // Set other bwt fields that were not in the file
        bwt.sa_intv = 1 << sa_compx;
        bwt.n_sa = sa_len;

        // Debug: verify SA values look reasonable
        if bwt.sa_ms_byte.len() > 10 {
            log::debug!(
                "Loaded SA samples: n_sa={}, sa_intv={}",
                bwt.n_sa,
                bwt.sa_intv
            );
            log::debug!("First 5 SA values:");
            for i in 0..5.min(bwt.sa_ms_byte.len()) {
                let sa_val = ((bwt.sa_ms_byte[i] as i64) << 32) | (bwt.sa_ls_word[i] as i64);
                log::debug!("  SA[{}] = {}", i * bwt.sa_intv as usize, sa_val);
            }
        }

        Ok(BwaIndex {
            bwt,
            bns,
            cp_occ,
            sentinel_index,
            min_seed_len: 1, // Initialize with default value
        })
    }

    pub fn dump(&self, prefix: &Path) -> io::Result<()> {
        let bwt_file_path = prefix.with_extension("bwt.2bit.64");
        let mut file = File::create(&bwt_file_path)?;

        // Match C++ FMI_search::build_fm_index format
        // 1. ref_seq_len (i64)
        file.write_all(&(self.bwt.seq_len as i64).to_le_bytes())?;

        // 2. count array (l2) (5 * i64)
        for i in 0..5 {
            file.write_all(&(self.bwt.l2[i] as i64).to_le_bytes())?;
        }

        // 3. cp_occ array
        // eprintln!("Dumping cp_occ: cp_occ.len()={}", self.cp_occ.len());
        for (_idx, cp_occ_entry) in self.cp_occ.iter().enumerate() {
            // eprintln!("  cp_occ[{}]: cp_count=[{}, {}, {}, {}]", idx,
            //          cp_occ_entry.cp_count[0], cp_occ_entry.cp_count[1],
            //          cp_occ_entry.cp_count[2], cp_occ_entry.cp_count[3]);
            // eprintln!("    one_hot_bwt_str=[{:#018x}, {:#018x}, {:#018x}, {:#018x}]",
            //          cp_occ_entry.one_hot_bwt_str[0], cp_occ_entry.one_hot_bwt_str[1],
            //          cp_occ_entry.one_hot_bwt_str[2], cp_occ_entry.one_hot_bwt_str[3]);
            for i in 0..4 {
                file.write_all(&cp_occ_entry.cp_count[i].to_le_bytes())?;
            }
            for i in 0..4 {
                file.write_all(&cp_occ_entry.one_hot_bwt_str[i].to_le_bytes())?;
            }
        }

        // 4. sa_ms_byte array
        // eprintln!("Dumping SA: n_sa={}, sa_ms_byte.len()={}, sa_ls_word.len()={}",
        //           self.bwt.n_sa, self.bwt.sa_ms_byte.len(), self.bwt.sa_ls_word.len());
        for (_i, val) in self.bwt.sa_ms_byte.iter().enumerate() {
            // eprintln!("  Write sa_ms_byte[{}] = {}", i, val);
            file.write_all(&val.to_le_bytes())?;
        }

        // 5. sa_ls_word array
        for (_i, val) in self.bwt.sa_ls_word.iter().enumerate() {
            // eprintln!("  Write sa_ls_word[{}] = {}", i, val);
            file.write_all(&val.to_le_bytes())?;
        }

        // 6. sentinel_index (i64)
        file.write_all(&self.sentinel_index.to_le_bytes())?;

        Ok(())
    }
}

pub fn main_mem(
    idx_prefix: &Path,
    query_files: &Vec<String>,
    output: Option<&String>,
    opt: &MemOpt,
) {
    // Detect and display SIMD capabilities
    use crate::simd_abstraction::{detect_optimal_simd_engine, simd_engine_description};
    let simd_engine = detect_optimal_simd_engine();
    let simd_desc = simd_engine_description(simd_engine);
    log::info!("Using SIMD engine: {}", simd_desc);

    // Load the BWA index
    let bwa_idx = match BwaIndex::bwa_idx_load(idx_prefix) {
        Ok(idx) => idx,
        Err(e) => {
            log::error!("Error loading BWA index: {}", e);
            return; // Or handle error appropriately
        }
    };

    // Determine output writer
    let mut writer: Box<dyn Write> = match output {
        Some(file_name) => match File::create(file_name) {
            Ok(file) => Box::new(file),
            Err(e) => {
                log::error!("Error creating output file {}: {}", file_name, e);
                return;
            }
        },
        None => Box::new(io::stdout()),
    };

    // Write SAM header
    if let Err(e) = writeln!(writer, "@HD\tVN:1.0\tSO:unsorted") {
        log::error!("Error writing SAM header: {}", e);
        return;
    }

    // Write @SQ lines for reference sequences
    for ann in &bwa_idx.bns.anns {
        if let Err(e) = writeln!(writer, "@SQ\tSN:{}\tLN:{}", ann.name, ann.len) {
            log::error!("Error writing SAM header: {}", e);
            return;
        }
    }

    // Write @PG (program) line
    // Get program name and version from Cargo.toml at compile time
    // This allows the program name to automatically update if we rename the project
    // (e.g., to "FerrousAlign" or another name) without code changes
    const PKG_NAME: &str = env!("CARGO_PKG_NAME");
    const PKG_VERSION: &str = env!("CARGO_PKG_VERSION");

    if let Err(e) = writeln!(
        writer,
        "@PG\tID:{}\tPN:{}\tVN:{}\tCL:{}",
        PKG_NAME,
        PKG_NAME,
        PKG_VERSION,
        std::env::args().collect::<Vec<_>>().join(" ")
    ) {
        log::error!("Error writing @PG header: {}", e);
        return;
    }

    // Write read group header if provided (-R option)
    if let Some(ref rg_line) = opt.read_group {
        // Ensure line starts with @RG
        let formatted_rg = if rg_line.starts_with("@RG") {
            rg_line.clone()
        } else {
            format!("@RG\t{}", rg_line)
        };

        if let Err(e) = writeln!(writer, "{}", formatted_rg) {
            log::error!("Error writing read group header: {}", e);
            return;
        }
    }

    // Write custom header lines if provided (-H option)
    for header_line in &opt.header_lines {
        if let Err(e) = writeln!(writer, "{}", header_line) {
            log::error!("Error writing custom header: {}", e);
            return;
        }
    }

    // Detect paired-end mode: if exactly 2 query files provided
    if query_files.len() == 2 {
        // Paired-end mode
        process_paired_end(&bwa_idx, &query_files[0], &query_files[1], &mut writer, opt);
    } else {
        // Single-end mode (original behavior)
        process_single_end(&bwa_idx, query_files, &mut writer, opt);
    }
}

// Process single-end reads with parallel batching
fn process_single_end(
    bwa_idx: &BwaIndex,
    query_files: &Vec<String>,
    writer: &mut Box<dyn Write>,
    opt: &MemOpt,
) {
    use std::time::Instant;

    // Wrap index in Arc for thread-safe sharing
    let bwa_idx = Arc::new(bwa_idx);
    let opt = Arc::new(opt);

    // Track overall statistics
    let start_time = Instant::now();
    let mut total_reads = 0usize;
    let mut total_bases = 0usize;

    // Calculate optimal batch size based on thread count (matching C++ bwa-mem2)
    // Formula: (CHUNK_SIZE_BASES * n_threads) / AVG_READ_LEN
    // C++ fastmap.cpp:947: aux.task_size = opt->chunk_size * opt->n_threads
    // C++ fastmap.cpp:459: nreads = aux->actual_chunk_size / READ_LEN + 10
    let num_threads = opt.n_threads as usize;
    let batch_total_bases = CHUNK_SIZE_BASES * num_threads;
    let reads_per_batch = (batch_total_bases / AVG_READ_LEN).max(MIN_BATCH_SIZE);
    log::debug!(
        "Using batch size: {} reads ({} MB total, {} threads × {} MB/thread)",
        reads_per_batch,
        batch_total_bases / 1_000_000,
        num_threads,
        CHUNK_SIZE_BASES / 1_000_000
    );

    for query_file_name in query_files {
        let mut reader = match FastqReader::new(query_file_name) {
            Ok(r) => r,
            Err(e) => {
                log::error!("Error opening query file {}: {}", query_file_name, e);
                continue;
            }
        };

        loop {
            // Stage 0: Read batch of reads (matching C++ kt_pipeline step 0)
            let batch = match reader.read_batch(reads_per_batch) {
                Ok(b) => b,
                Err(e) => {
                    log::error!("Error reading batch from {}: {}", query_file_name, e);
                    break;
                }
            };

            if batch.names.is_empty() {
                break; // EOF
            }

            let batch_size = batch.names.len();

            // Calculate batch base pairs
            let batch_bp: usize = batch.seqs.iter().map(|s| s.len()).sum();
            total_reads += batch_size;
            total_bases += batch_bp;

            log::info!("Read {} sequences ({} bp)", batch_size, batch_bp);
            log::debug!("Processing batch of {} reads in parallel", batch_size);

            // Stage 1: Process batch in parallel (matching C++ kt_pipeline step 1)
            let bwa_idx_clone = Arc::clone(&bwa_idx);
            let opt_clone = Arc::clone(&opt);
            let alignments: Vec<Vec<align::Alignment>> = batch
                .names
                .par_iter()
                .zip(batch.seqs.par_iter())
                .zip(batch.quals.par_iter())
                .map(|((name, seq), qual)| {
                    align::generate_seeds(&bwa_idx_clone, name, seq, qual, &opt_clone)
                })
                .collect();

            // Stage 2: Write output sequentially (matching C++ kt_pipeline step 2)
            // Filter by minimum score threshold (-T)
            // Extract read group ID if specified
            let rg_id = opt
                .read_group
                .as_ref()
                .and_then(|rg| crate::mem_opt::MemOpt::extract_rg_id(rg));

            for alignment_vec in alignments {
                for mut alignment in alignment_vec {
                    // Filter alignments below score threshold (opt.t)
                    if alignment.score >= opt.t {
                        // Add RG tag if read group is specified
                        if let Some(ref rg) = rg_id {
                            alignment.tags.push(("RG".to_string(), format!("Z:{}", rg)));
                        }

                        let sam_record = alignment.to_sam_string();
                        if let Err(e) = writeln!(writer, "{}", sam_record) {
                            log::error!("Error writing SAM record: {}", e);
                        }
                    }
                }
            }

            if batch_size < reads_per_batch {
                break; // Last incomplete batch
            }
        }
    }

    // Print summary statistics
    let elapsed = start_time.elapsed();
    log::info!(
        "Processed {} reads ({} bp) in {:.2} sec",
        total_reads,
        total_bases,
        elapsed.as_secs_f64()
    );
}

// Reader thread function for pipeline parallelism
// Continuously reads batches from FASTQ files and sends them through channel
fn reader_thread(
    read1_file: String,
    read2_file: String,
    reads_per_batch: usize,
    sender: Sender<PairedBatchMessage>,
) {
    // Open both FASTQ files
    let mut reader1 = match FastqReader::new(&read1_file) {
        Ok(r) => r,
        Err(e) => {
            log::error!("Error opening read1 file {}: {}", read1_file, e);
            let _ = sender.send(None); // Signal error
            return;
        }
    };
    let mut reader2 = match FastqReader::new(&read2_file) {
        Ok(r) => r,
        Err(e) => {
            log::error!("Error opening read2 file {}: {}", read2_file, e);
            let _ = sender.send(None); // Signal error
            return;
        }
    };

    log::debug!(
        "[Reader thread] Started, reading batches of {} read pairs",
        reads_per_batch
    );

    loop {
        // Read batch from both files
        let batch1 = match reader1.read_batch(reads_per_batch) {
            Ok(b) => b,
            Err(e) => {
                log::error!("Error reading batch from read1 file: {}", e);
                let _ = sender.send(None); // Signal error
                break;
            }
        };
        let batch2 = match reader2.read_batch(reads_per_batch) {
            Ok(b) => b,
            Err(e) => {
                log::error!("Error reading batch from read2 file: {}", e);
                let _ = sender.send(None); // Signal error
                break;
            }
        };

        // Check for EOF
        if batch1.names.is_empty() && batch2.names.is_empty() {
            log::debug!("[Reader thread] EOF reached, shutting down");
            let _ = sender.send(None); // Signal EOF
            break;
        }

        // Check for mismatched batch sizes
        if batch1.names.len() != batch2.names.len() {
            log::warn!("Warning: Paired-end files have different number of reads");
            let _ = sender.send(None); // Signal error
            break;
        }

        let batch_size = batch1.names.len();
        log::debug!("[Reader thread] Read {} read pairs", batch_size);

        // Send batch through channel
        if sender.send(Some((batch1, batch2))).is_err() {
            log::error!("[Reader thread] Channel closed, shutting down");
            break;
        }

        // If this was a partial batch, it's the last one
        if batch_size < reads_per_batch {
            log::debug!("[Reader thread] Final partial batch, shutting down");
            let _ = sender.send(None); // Signal EOF
            break;
        }
    }

    log::debug!("[Reader thread] Exiting");
}

// Process paired-end reads with parallel batching
fn process_paired_end(
    bwa_idx: &BwaIndex,
    read1_file: &str,
    read2_file: &str,
    writer: &mut Box<dyn Write>,
    opt: &MemOpt,
) {
    use std::time::Instant;

    // Wrap index in Arc for thread-safe sharing
    let bwa_idx = Arc::new(bwa_idx);
    let opt = Arc::new(opt);

    // Track overall statistics
    let start_time = Instant::now();
    let mut total_reads = 0usize;
    let mut total_bases = 0usize;

    // Calculate optimal batch size based on thread count (matching C++ bwa-mem2)
    // Formula: (CHUNK_SIZE_BASES * n_threads) / AVG_READ_LEN
    // C++ fastmap.cpp:947: aux.task_size = opt->chunk_size * opt->n_threads
    // C++ fastmap.cpp:459: nreads = aux->actual_chunk_size / READ_LEN + 10
    let num_threads = opt.n_threads as usize;
    let batch_total_bases = CHUNK_SIZE_BASES * num_threads;
    let reads_per_batch = (batch_total_bases / AVG_READ_LEN).max(MIN_BATCH_SIZE);
    log::debug!(
        "Using batch size: {} reads ({} MB total, {} threads × {} MB/thread)",
        reads_per_batch,
        batch_total_bases / 1_000_000,
        num_threads,
        CHUNK_SIZE_BASES / 1_000_000
    );

    // Create channel for pipeline communication
    // Buffer size of 2 allows reader to stay 1 batch ahead
    let (sender, receiver): (Sender<PairedBatchMessage>, Receiver<PairedBatchMessage>) = bounded(2);

    // Spawn reader thread for pipeline parallelism
    let read1_file_owned = read1_file.to_string();
    let read2_file_owned = read2_file.to_string();
    let reader_handle = thread::spawn(move || {
        reader_thread(read1_file_owned, read2_file_owned, reads_per_batch, sender);
    });

    log::debug!("[Main] Pipeline started: reader thread spawned");

    // Load PAC file ONCE for all mate rescue operations
    // This eliminates catastrophic I/O that was occurring in the old design
    let pac = if let Some(pac_path) = &bwa_idx.bns.pac_file_path {
        std::fs::read(pac_path).unwrap_or_else(|e| {
            log::warn!(
                "Could not load PAC file: {}. Mate rescue will be disabled.",
                e
            );
            Vec::new()
        })
    } else {
        log::warn!("PAC file path not set. Mate rescue will be disabled.");
        Vec::new()
    };
    log::debug!("Loaded PAC file: {} bytes", pac.len());

    // === PHASE 1: Bootstrap insert size statistics from first batch ===
    log::info!("Phase 1: Bootstrapping insert size statistics from first batch");

    // Receive first batch
    let first_batch_msg = match receiver.recv() {
        Ok(msg) => msg,
        Err(_) => {
            log::error!("Failed to receive first batch from reader");
            return;
        }
    };

    let (first_batch1, first_batch2) = match first_batch_msg {
        Some((b1, b2)) => (b1, b2),
        None => {
            log::warn!("No data to process (empty input files)");
            return;
        }
    };

    let first_batch_size = first_batch1.names.len();
    let first_batch_bp: usize = first_batch1.seqs.iter().map(|s| s.len()).sum::<usize>()
        + first_batch2.seqs.iter().map(|s| s.len()).sum::<usize>();
    total_reads += first_batch_size * 2;
    total_bases += first_batch_bp;

    log::info!(
        "Read {} sequences ({} bp) [first batch]",
        first_batch_size * 2,
        first_batch_bp
    );

    // Process first batch in parallel
    let num_pairs = first_batch1.names.len();
    let bwa_idx_clone = Arc::clone(&bwa_idx);
    let opt_clone = Arc::clone(&opt);

    let mut first_batch_alignments: Vec<(Vec<align::Alignment>, Vec<align::Alignment>)> = (0
        ..num_pairs)
        .into_par_iter()
        .map(|i| {
            let aln1 = align::generate_seeds(
                &bwa_idx_clone,
                &first_batch1.names[i],
                &first_batch1.seqs[i],
                &first_batch1.quals[i],
                &opt_clone,
            );
            let aln2 = align::generate_seeds(
                &bwa_idx_clone,
                &first_batch2.names[i],
                &first_batch2.seqs[i],
                &first_batch2.quals[i],
                &opt_clone,
            );
            (aln1, aln2)
        })
        .collect();

    // Bootstrap insert size stats from first batch
    let stats = if let Some(ref is_override) = opt.insert_size_override {
        // Use manual insert size specification (-I option)
        log::info!(
            "[Paired-end] Using manual insert size: mean={:.1}, std={:.1}, max={}, min={}",
            is_override.mean,
            is_override.stddev,
            is_override.max,
            is_override.min
        );

        [
            InsertSizeStats {
                avg: 0.0,
                std: 0.0,
                low: 0,
                high: 0,
                failed: true,
            },
            InsertSizeStats {
                avg: is_override.mean,
                std: is_override.stddev,
                low: is_override.min,
                high: is_override.max,
                failed: false,
            },
            InsertSizeStats {
                avg: 0.0,
                std: 0.0,
                low: 0,
                high: 0,
                failed: true,
            },
            InsertSizeStats {
                avg: 0.0,
                std: 0.0,
                low: 0,
                high: 0,
                failed: true,
            },
        ]
    } else {
        bootstrap_insert_size_stats(&first_batch_alignments, bwa_idx.bns.l_pac as i64)
    };

    // Prepare sequences for mate rescue and output
    let first_batch_seqs1: Vec<_> = first_batch1
        .names
        .iter()
        .zip(&first_batch1.seqs)
        .zip(&first_batch1.quals)
        .map(|((n, s), q)| (n.as_str(), s.as_slice(), q.as_str()))
        .collect();
    let first_batch_seqs2: Vec<_> = first_batch2
        .names
        .iter()
        .zip(&first_batch2.seqs)
        .zip(&first_batch2.quals)
        .map(|((n, s), q)| (n.as_str(), s.as_slice(), q.as_str()))
        .collect();

    // Mate rescue on first batch
    let rescued_first = mate_rescue_batch(
        &mut first_batch_alignments,
        &first_batch_seqs1,
        &first_batch_seqs2,
        &pac,
        &stats,
        &bwa_idx,
    );
    log::info!("First batch: {} pairs rescued", rescued_first);

    // Prepare sequences for output (need owned versions)
    let first_batch_seqs1_owned: Vec<_> = first_batch1
        .names
        .into_iter()
        .zip(first_batch1.seqs)
        .zip(first_batch1.quals)
        .map(|((n, s), q)| (n, s, q))
        .collect();
    let first_batch_seqs2_owned: Vec<_> = first_batch2
        .names
        .into_iter()
        .zip(first_batch2.seqs)
        .zip(first_batch2.quals)
        .map(|((n, s), q)| (n, s, q))
        .collect();

    // Output first batch immediately
    let first_batch_records = output_batch_paired(
        first_batch_alignments,
        &first_batch_seqs1_owned,
        &first_batch_seqs2_owned,
        &stats,
        writer,
        &opt,
        bwa_idx.bns.l_pac as i64,
        0,
    )
    .unwrap_or_else(|e| {
        log::error!("Error writing first batch: {}", e);
        0
    });
    log::info!("First batch: {} records written", first_batch_records);

    // === PHASE 2: Stream remaining batches ===
    log::info!("Phase 2: Streaming remaining batches");

    let mut batch_num = 1u64;
    let mut total_rescued = rescued_first;
    let mut total_records = first_batch_records;

    loop {
        // Receive batch from reader thread (blocks until batch available)
        let batch_msg = match receiver.recv() {
            Ok(msg) => msg,
            Err(_) => {
                log::debug!("[Main] Reader channel closed");
                break;
            }
        };

        // Check for EOF or error signal
        let (batch1, batch2) = match batch_msg {
            Some((b1, b2)) => (b1, b2),
            None => {
                log::debug!("[Main] Received EOF/error signal from reader");
                break;
            }
        };

        batch_num += 1;

        let batch_size = batch1.names.len();
        let batch_bp: usize = batch1.seqs.iter().map(|s| s.len()).sum::<usize>()
            + batch2.seqs.iter().map(|s| s.len()).sum::<usize>();
        total_reads += batch_size * 2;
        total_bases += batch_bp;

        log::info!("Read {} sequences ({} bp)", batch_size * 2, batch_bp);

        // Process batch in parallel
        let num_pairs = batch1.names.len();
        let bwa_idx_clone = Arc::clone(&bwa_idx);
        let opt_clone = Arc::clone(&opt);

        let mut batch_alignments: Vec<(Vec<align::Alignment>, Vec<align::Alignment>)> = (0
            ..num_pairs)
            .into_par_iter()
            .map(|i| {
                let aln1 = align::generate_seeds(
                    &bwa_idx_clone,
                    &batch1.names[i],
                    &batch1.seqs[i],
                    &batch1.quals[i],
                    &opt_clone,
                );
                let aln2 = align::generate_seeds(
                    &bwa_idx_clone,
                    &batch2.names[i],
                    &batch2.seqs[i],
                    &batch2.quals[i],
                    &opt_clone,
                );
                (aln1, aln2)
            })
            .collect();

        // Prepare sequences for mate rescue
        let batch_seqs1: Vec<_> = batch1
            .names
            .iter()
            .zip(&batch1.seqs)
            .zip(&batch1.quals)
            .map(|((n, s), q)| (n.as_str(), s.as_slice(), q.as_str()))
            .collect();
        let batch_seqs2: Vec<_> = batch2
            .names
            .iter()
            .zip(&batch2.seqs)
            .zip(&batch2.quals)
            .map(|((n, s), q)| (n.as_str(), s.as_slice(), q.as_str()))
            .collect();

        // Mate rescue on this batch
        let rescued = mate_rescue_batch(
            &mut batch_alignments,
            &batch_seqs1,
            &batch_seqs2,
            &pac,
            &stats,
            &bwa_idx,
        );
        total_rescued += rescued;

        // Prepare sequences for output (need owned versions)
        let batch_seqs1_owned: Vec<_> = batch1
            .names
            .into_iter()
            .zip(batch1.seqs)
            .zip(batch1.quals)
            .map(|((n, s), q)| (n, s, q))
            .collect();
        let batch_seqs2_owned: Vec<_> = batch2
            .names
            .into_iter()
            .zip(batch2.seqs)
            .zip(batch2.quals)
            .map(|((n, s), q)| (n, s, q))
            .collect();

        // Output batch immediately
        let records = output_batch_paired(
            batch_alignments,
            &batch_seqs1_owned,
            &batch_seqs2_owned,
            &stats,
            writer,
            &opt,
            bwa_idx.bns.l_pac as i64,
            batch_num * reads_per_batch as u64,
        )
        .unwrap_or_else(|e| {
            log::error!("Error writing batch: {}", e);
            0
        });
        total_records += records;

        // Log progress every 10 batches
        if batch_num % 10 == 0 {
            log::info!(
                "Processed {} batches, {} records written, {} rescued",
                batch_num,
                total_records,
                total_rescued
            );
        }

        // batch_alignments and batch dropped here, memory freed
    }

    // Wait for reader thread to finish
    log::debug!("[Main] Waiting for reader thread to finish");
    if let Err(e) = reader_handle.join() {
        log::error!("[Main] Reader thread panicked: {:?}", e);
    }
    log::debug!("[Main] Reader thread finished, pipeline complete");

    // Print summary statistics
    let elapsed = start_time.elapsed();
    log::info!(
        "Complete: {} batches, {} reads ({} bp), {} records, {} pairs rescued in {:.2} sec",
        batch_num,
        total_reads,
        total_bases,
        total_records,
        total_rescued,
        elapsed.as_secs_f64()
    );
}

// Infer orientation of paired reads (from C++ mem_infer_dir)
// Returns: (orientation, insert_size)
// Orientation: 0=FF, 1=FR, 2=RF, 3=RR
fn infer_orientation(l_pac: i64, pos1: i64, pos2: i64) -> (usize, i64) {
    let r1 = if pos1 >= l_pac { 1 } else { 0 };
    let r2 = if pos2 >= l_pac { 1 } else { 0 };

    // p2 is the coordinate of read2 on the read1 strand
    let p2 = if r1 == r2 {
        pos2
    } else {
        (l_pac << 1) - 1 - pos2
    };

    let dist = if p2 > pos1 { p2 - pos1 } else { pos1 - p2 };

    // Calculate orientation
    // (r1 == r2 ? 0 : 1) ^ (p2 > pos1 ? 0 : 3)
    let orientation = if r1 == r2 { 0 } else { 1 } ^ if p2 > pos1 { 0 } else { 3 };

    (orientation, dist)
}

// Calculate insert size statistics for all 4 orientations
// Returns: [InsertSizeStats; 4] for orientations FF, FR, RF, RR
// Pair information for mem_pair scoring (equivalent to C++ pair64_t)
#[derive(Debug, Clone, Copy)]
struct AlignmentInfo {
    pos_key: u64, // (ref_id << 32) | forward_position
    info: u64,    // (score << 32) | (index << 2) | (is_rev << 1) | read_number
}

// Paired alignment scoring result
#[derive(Debug, Clone, Copy)]
struct PairScore {
    idx1: usize, // Index in read1 alignments
    idx2: usize, // Index in read2 alignments
    score: i32,  // Paired alignment score
    hash: u32,   // Hash for tie-breaking
}

// Score paired-end alignments based on insert size distribution (C++ mem_pair equivalent)
// Returns: Option<(best_idx1, best_idx2, pair_score, sub_score)>
fn mem_pair(
    stats: &[InsertSizeStats; 4],
    alns1: &[align::Alignment],
    alns2: &[align::Alignment],
    l_pac: i64,
    match_score: i32, // opt->a (match score for log-likelihood calculation)
    pair_id: u64,     // Read pair ID for hash
) -> Option<(usize, usize, i32, i32)> {
    if alns1.is_empty() || alns2.is_empty() {
        return None;
    }

    // Build sorted array of alignment positions (like C++ v array)
    let mut v: Vec<AlignmentInfo> = Vec::new();

    // Add alignments from read1
    for (i, aln) in alns1.iter().enumerate() {
        // Use forward-strand position directly (aln.pos is always on forward strand)
        let is_rev = (aln.flag & 0x10) != 0;
        let pos = aln.pos as i64;

        let pos_key = ((aln.ref_id as u64) << 32) | (pos as u64);
        let info = ((aln.score as u64) << 32) | ((i as u64) << 2) | ((is_rev as u64) << 1) | 0; // 0 = read1

        // DEBUG: Log positions for first few pairs
        if pair_id < 3 {
            log::debug!(
                "mem_pair: R1[{}]: aln.pos={}, is_rev={}, ref_id={}",
                i,
                aln.pos,
                is_rev,
                aln.ref_id
            );
        }

        v.push(AlignmentInfo { pos_key, info });
    }

    // Add alignments from read2
    for (i, aln) in alns2.iter().enumerate() {
        let is_rev = (aln.flag & 0x10) != 0;
        let pos = aln.pos as i64;

        let pos_key = ((aln.ref_id as u64) << 32) | (pos as u64);
        let info = ((aln.score as u64) << 32) | ((i as u64) << 2) | ((is_rev as u64) << 1) | 1; // 1 = read2

        // DEBUG: Log positions for first few pairs
        if pair_id < 3 {
            log::debug!(
                "mem_pair: R2[{}]: aln.pos={}, is_rev={}, ref_id={}",
                i,
                aln.pos,
                is_rev,
                aln.ref_id
            );
        }

        v.push(AlignmentInfo { pos_key, info });
    }

    // Sort by position (like C++ ks_introsort_128)
    v.sort_by_key(|a| a.pos_key);

    // Track last hit for each orientation combination [read][strand]
    let mut y = [-1i32; 4];

    // Array to store valid pairs (like C++ u array)
    let mut u: Vec<PairScore> = Vec::new();

    // For each alignment, look backward for compatible mates
    for i in 0..v.len() {
        for r in 0..2 {
            // Try both orientations
            let dir = ((r << 1) | ((v[i].info >> 1) & 1)) as usize; // orientation index

            if stats[dir].failed {
                continue; // Invalid orientation
            }

            let which = ((r << 1) | ((v[i].info & 1) ^ 1)) as usize; // Look for mate from other read

            if y[which] < 0 {
                continue; // No previous hits from mate
            }

            // Search backward for compatible pairs
            let mut k = y[which] as usize;
            loop {
                if k >= v.len() {
                    break;
                }

                if (v[k].info & 3) != which as u64 {
                    if k == 0 {
                        break;
                    }
                    k -= 1;
                    continue;
                }

                // Calculate distance
                let dist = (v[i].pos_key - v[k].pos_key) as i64;

                // DEBUG: Log distance checks for first few pairs
                if pair_id < 3 {
                    log::debug!(
                        "mem_pair: Checking pair i={}, k={}, dir={}, dist={}, bounds=[{}, {}]",
                        i,
                        k,
                        dir,
                        dist,
                        stats[dir].low,
                        stats[dir].high
                    );
                }

                if dist > stats[dir].high as i64 {
                    if pair_id < 3 {
                        log::debug!("mem_pair: Distance too far, breaking");
                    }
                    break; // Too far
                }

                if dist < stats[dir].low as i64 {
                    if pair_id < 3 {
                        log::debug!("mem_pair: Distance too close, continuing");
                    }
                    if k == 0 {
                        break;
                    }
                    k -= 1;
                    continue; // Too close
                }

                // Compute pairing score using normal distribution
                // q = score1 + score2 + log_prob(insert_size)
                let ns = (dist as f64 - stats[dir].avg) / stats[dir].std;

                // Log-likelihood penalty: .721 * log(2 * erfc(|ns| / sqrt(2))) * match_score
                // .721 = 1/log(4) converts to base-4 log
                let log_prob = 0.721
                    * ((2.0 * erfc(ns.abs() / std::f64::consts::SQRT_2)).ln())
                    * (match_score as f64);

                let score1 = (v[i].info >> 32) as i32;
                let score2 = (v[k].info >> 32) as i32;
                let mut q = score1 + score2 + (log_prob + 0.499) as i32;

                if q < 0 {
                    q = 0;
                }

                // Hash for tie-breaking
                let hash_input = (k as u64) << 32 | i as u64;
                let hash = (hash_64(hash_input ^ (pair_id << 8)) & 0xffffffff) as u32;

                u.push(PairScore {
                    idx1: if (v[k].info & 1) == 0 {
                        ((v[k].info >> 2) & 0x3fffffff) as usize
                    } else {
                        ((v[i].info >> 2) & 0x3fffffff) as usize
                    },
                    idx2: if (v[k].info & 1) == 1 {
                        ((v[k].info >> 2) & 0x3fffffff) as usize
                    } else {
                        ((v[i].info >> 2) & 0x3fffffff) as usize
                    },
                    score: q,
                    hash,
                });

                // DEBUG: Log when we find a valid pair
                if pair_id < 10 {
                    log::debug!(
                        "mem_pair: Found valid pair! dir={}, dist={}, score={}",
                        dir, dist, q
                    );
                }

                if k == 0 {
                    break;
                }
                k -= 1;
            }
        }

        y[(v[i].info & 3) as usize] = i as i32;
    }

    if u.is_empty() {
        // DEBUG: Log why no pairs were found for first few pairs
        if pair_id < 10 {
            log::debug!(
                "mem_pair: No valid pairs in u array. v.len()={}, y={:?}",
                v.len(),
                y
            );
        }
        return None; // No valid pairs found
    }

    // Sort by score (descending), then by hash
    u.sort_by(|a, b| match b.score.cmp(&a.score) {
        std::cmp::Ordering::Equal => b.hash.cmp(&a.hash),
        other => other,
    });

    // Best pair is first
    let best = &u[0];
    let sub_score = if u.len() > 1 { u[1].score } else { 0 };

    Some((best.idx1, best.idx2, best.score, sub_score))
}

/// Mate rescue using Smith-Waterman alignment
/// Equivalent to C++ mem_matesw
/// Returns number of rescued alignments added
fn mem_matesw(
    bwa_idx: &BwaIndex,
    pac: &[u8], // Pre-loaded PAC data (passed once, not loaded per call)
    stats: &[InsertSizeStats; 4],
    anchor: &align::Alignment,
    mate_seq: &[u8],
    mate_qual: &str,
    mate_name: &str,
    rescued_alignments: &mut Vec<align::Alignment>,
) -> usize {
    use crate::banded_swa::BandedPairWiseSW;

    let l_pac = bwa_idx.bns.l_pac as i64;
    let l_ms = mate_seq.len() as i32;
    let min_seed_len = bwa_idx.min_seed_len;

    // Check which orientations to skip (already have good pairs)
    let mut skip = [false; 4];
    for r in 0..4 {
        skip[r] = stats[r].failed;
    }

    // Check existing mate alignments to see if we already have pairs in each orientation
    for aln in rescued_alignments.iter() {
        if aln.ref_name == anchor.ref_name {
            let (dir, dist) = infer_orientation(l_pac, anchor.pos as i64, aln.pos as i64);
            if dist >= stats[dir].low as i64 && dist <= stats[dir].high as i64 {
                skip[dir] = true;
            }
        }
    }

    // If all orientations already have consistent pairs, no need for rescue
    if skip.iter().all(|&x| x) {
        return 0;
    }

    // PAC data is now passed as parameter (loaded once per batch, not per call)
    // This eliminates catastrophic I/O: was reading 740MB × 5000+ times per batch!

    // Setup Smith-Waterman aligner (same parameters as generate_seeds)
    let sw_params = BandedPairWiseSW::new(
        4,   // o_del
        2,   // e_del
        4,   // o_ins
        2,   // e_ins
        100, // zdrop
        0,   // end_bonus
        align::DEFAULT_SCORING_MATRIX,
        2,  // w_match
        -4, // w_mismatch
    );

    let mut n_rescued = 0;

    // Try each orientation
    for r in 0..4 {
        if skip[r] {
            continue;
        }

        let is_rev = (r >> 1) != (r & 1); // Whether to reverse complement the mate
        let is_larger = (r >> 1) == 0; // Whether the mate has larger coordinate

        // Prepare mate sequence (reverse complement if needed)
        let seq: Vec<u8>;
        if is_rev {
            seq = mate_seq
                .iter()
                .rev()
                .map(|&b| if b < 4 { 3 - b } else { 4 })
                .collect();
        } else {
            seq = mate_seq.to_vec();
        }

        // Calculate search region
        let (rb, re) = if !is_rev {
            let rb = if is_larger {
                anchor.pos as i64 + stats[r].low as i64
            } else {
                anchor.pos as i64 - stats[r].high as i64
            };
            let re = if is_larger {
                anchor.pos as i64 + stats[r].high as i64
            } else {
                anchor.pos as i64 - stats[r].low as i64
            } + l_ms as i64;
            (rb.max(0), re.min(l_pac << 1))
        } else {
            let rb = if is_larger {
                anchor.pos as i64 + stats[r].low as i64
            } else {
                anchor.pos as i64 - stats[r].high as i64
            } - l_ms as i64;
            let re = if is_larger {
                anchor.pos as i64 + stats[r].high as i64
            } else {
                anchor.pos as i64 - stats[r].low as i64
            };
            (rb.max(0), re.min(l_pac << 1))
        };

        if rb >= re {
            continue;
        }

        // Fetch reference sequence
        let (ref_seq, adj_rb, adj_re, rid) =
            bwa_idx.bns.bns_fetch_seq(&pac, rb, (rb + re) >> 1, re);

        // Check if on same reference and region is large enough
        if rid as usize != anchor.ref_id || (adj_re - adj_rb) < min_seed_len as i64 {
            continue;
        }

        // Perform Smith-Waterman alignment
        let ref_len = ref_seq.len() as i32;
        let (out_score, cigar) = sw_params.scalar_banded_swa(
            l_ms, &seq, ref_len, &ref_seq, 100, // w (bandwidth)
            0,   // h0 (initial score)
        );

        // Check if alignment is good enough
        if out_score.score < min_seed_len || cigar.is_empty() {
            continue;
        }

        // Calculate alignment start position from CIGAR
        // The end positions are qle and tle
        // We need to calculate start positions by walking back through CIGAR
        let mut _query_consumed = 0i32;
        let mut ref_consumed = 0i32;

        for &(op, len) in &cigar {
            match op {
                0 => {
                    // Match/Mismatch
                    _query_consumed += len;
                    ref_consumed += len;
                }
                1 => {
                    // Insertion (consumes query)
                    _query_consumed += len;
                }
                2 => {
                    // Deletion (consumes reference)
                    ref_consumed += len;
                }
                _ => {}
            }
        }

        // Calculate alignment position on reference
        let tb = (out_score.tle - ref_consumed).max(0);

        // Adjust for reverse complement and reference position
        let pos = if is_rev {
            ((l_pac << 1) - (adj_rb + out_score.tle as i64)).max(0) as u64
        } else {
            (adj_rb + tb as i64).max(0) as u64
        };

        // Create alignment structure
        let mut flag = 0u16;
        if is_rev {
            flag |= 0x10; // Reverse complement
        }

        let rescued_aln = align::Alignment {
            query_name: mate_name.to_string(),
            flag,
            ref_name: anchor.ref_name.clone(),
            ref_id: anchor.ref_id,
            pos,
            mapq: 0, // Will be calculated later
            score: out_score.score,
            cigar,
            rnext: String::from("*"),
            pnext: 0,
            tlen: 0,
            seq: String::from_utf8(
                mate_seq
                    .iter()
                    .map(|&b| match b {
                        0 => b'A',
                        1 => b'C',
                        2 => b'G',
                        3 => b'T',
                        _ => b'N',
                    })
                    .collect(),
            )
            .unwrap(),
            qual: mate_qual.to_string(),
            tags: Vec::new(),
            // Internal fields for alignment selection
            query_start: out_score.qle - out_score.qle, // Full query alignment
            query_end: out_score.qle,
            seed_coverage: l_ms, // Mate sequence length as coverage
            hash: 0, // Will be set later if needed
        };

        rescued_alignments.push(rescued_aln);
        n_rescued += 1;
    }

    n_rescued
}

// Simple hash function (C++ hash_64 equivalent)
fn hash_64(key: u64) -> u64 {
    let mut key = key;
    key = (!key).wrapping_add(key << 21);
    key = key ^ (key >> 24);
    key = key.wrapping_add(key << 3).wrapping_add(key << 8);
    key = key ^ (key >> 14);
    key = key.wrapping_add(key << 2).wrapping_add(key << 4);
    key = key ^ (key >> 28);
    key = key.wrapping_add(key << 31);
    key
}

// Complementary error function (approximation)
fn erfc(x: f64) -> f64 {
    // Use standard library if available, otherwise approximation
    // For now, using a simple approximation
    let t = 1.0 / (1.0 + 0.5 * x.abs());
    let tau = t
        * (-x * x - 1.26551223
            + t * (1.00002368
                + t * (0.37409196
                    + t * (0.09678418
                        + t * (-0.18628806
                            + t * (0.27886807
                                + t * (-1.13520398
                                    + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    if x >= 0.0 { tau } else { 2.0 - tau }
}

fn calculate_insert_size_stats(
    l_pac: i64,
    pairs: &[(i64, i64)], // (pos1, pos2) for each pair
) -> [InsertSizeStats; 4] {
    // Collect insert sizes for each orientation
    let mut insert_sizes: [Vec<i64>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];

    for &(pos1, pos2) in pairs {
        let (orientation, dist) = infer_orientation(l_pac, pos1, pos2);
        if dist > 0 {
            insert_sizes[orientation].push(dist);
        }
    }

    log::info!(
        "Paired-end: {} candidate pairs (FF={}, FR={}, RF={}, RR={})",
        insert_sizes.iter().map(|v| v.len()).sum::<usize>(),
        insert_sizes[0].len(),
        insert_sizes[1].len(),
        insert_sizes[2].len(),
        insert_sizes[3].len()
    );

    let mut stats = [
        InsertSizeStats {
            avg: 0.0,
            std: 0.0,
            low: 0,
            high: 0,
            failed: true,
        },
        InsertSizeStats {
            avg: 0.0,
            std: 0.0,
            low: 0,
            high: 0,
            failed: true,
        },
        InsertSizeStats {
            avg: 0.0,
            std: 0.0,
            low: 0,
            high: 0,
            failed: true,
        },
        InsertSizeStats {
            avg: 0.0,
            std: 0.0,
            low: 0,
            high: 0,
            failed: true,
        },
    ];

    let orientation_names = ["FF", "FR", "RF", "RR"];

    // Calculate statistics for each orientation
    for d in 0..4 {
        let sizes = &mut insert_sizes[d];

        if sizes.len() < MIN_DIR_CNT {
            log::debug!(
                "Skipping orientation {} (insufficient pairs: {} < {})",
                orientation_names[d],
                sizes.len(),
                MIN_DIR_CNT
            );
            continue;
        }

        log::info!(
            "Analyzing insert size for orientation {} ({} pairs)",
            orientation_names[d],
            sizes.len()
        );

        // Sort insert sizes
        sizes.sort_unstable();

        // Calculate percentiles
        let p25 = sizes[(0.25 * sizes.len() as f64 + 0.499) as usize];
        let p50 = sizes[(0.50 * sizes.len() as f64 + 0.499) as usize];
        let p75 = sizes[(0.75 * sizes.len() as f64 + 0.499) as usize];

        let iqr = p75 - p25;

        // Calculate initial bounds for mean/std calculation (outlier removal)
        let mut low = ((p25 as f64 - OUTLIER_BOUND * iqr as f64) + 0.499) as i32;
        if low < 1 {
            low = 1;
        }
        let mut high = ((p75 as f64 + OUTLIER_BOUND * iqr as f64) + 0.499) as i32;

        log::debug!("  Percentiles (25/50/75): {}/{}/{}", p25, p50, p75);
        log::debug!("  Outlier bounds: {} - {}", low, high);

        // Calculate mean (excluding outliers)
        let mut sum = 0i64;
        let mut count = 0usize;
        for &size in sizes.iter() {
            if size >= low as i64 && size <= high as i64 {
                sum += size;
                count += 1;
            }
        }

        if count == 0 {
            log::warn!(
                "No valid samples for orientation {} within bounds",
                orientation_names[d]
            );
            continue;
        }

        let avg = sum as f64 / count as f64;

        // Calculate standard deviation (excluding outliers)
        let mut sum_sq = 0.0;
        for &size in sizes.iter() {
            if size >= low as i64 && size <= high as i64 {
                let diff = size as f64 - avg;
                sum_sq += diff * diff;
            }
        }
        let std = (sum_sq / count as f64).sqrt();

        log::info!("  Insert size: mean={:.1}, std={:.1}", avg, std);

        // Calculate final bounds for proper pairs (mapping bounds)
        low = ((p25 as f64 - MAPPING_BOUND * iqr as f64) + 0.499) as i32;
        high = ((p75 as f64 + MAPPING_BOUND * iqr as f64) + 0.499) as i32;

        // Adjust using standard deviation
        let low_stddev = (avg - MAX_STDDEV * std + 0.499) as i32;
        let high_stddev = (avg + MAX_STDDEV * std + 0.499) as i32;

        if low > low_stddev {
            low = low_stddev;
        }
        if high < high_stddev {
            high = high_stddev;
        }
        if low < 1 {
            low = 1;
        }

        log::info!("  Proper pair bounds: {} - {}", low, high);

        stats[d] = InsertSizeStats {
            avg,
            std,
            low,
            high,
            failed: false,
        };
    }

    // Find max count across all orientations
    let max_count = insert_sizes.iter().map(|v| v.len()).max().unwrap_or(0);

    // Mark orientations with too few pairs as failed
    for d in 0..4 {
        if !stats[d].failed && insert_sizes[d].len() < (max_count as f64 * MIN_DIR_RATIO) as usize {
            stats[d].failed = true;
            log::debug!(
                "Skipping orientation {} (insufficient ratio)",
                orientation_names[d]
            );
        }
    }

    stats
}

// Bootstrap insert size statistics from first batch only
// This allows streaming subsequent batches without buffering all alignments
fn bootstrap_insert_size_stats(
    first_batch_alignments: &[(Vec<align::Alignment>, Vec<align::Alignment>)],
    l_pac: i64,
) -> [InsertSizeStats; 4] {
    log::info!(
        "Bootstrapping insert size statistics from first batch ({} pairs)",
        first_batch_alignments.len()
    );

    // Extract position pairs from first batch
    let mut position_pairs: Vec<(i64, i64)> = Vec::new();

    for (alns1, alns2) in first_batch_alignments {
        // Use best alignment from each read for statistics
        if let (Some(aln1), Some(aln2)) = (alns1.first(), alns2.first()) {
            // Only use pairs on same reference
            if aln1.ref_name == aln2.ref_name {
                // Convert positions to bidirectional coordinate space [0, 2*l_pac)
                // Forward strand: [0, l_pac), Reverse strand: [l_pac, 2*l_pac)
                let is_rev1 = (aln1.flag & 0x10) != 0;
                let is_rev2 = (aln2.flag & 0x10) != 0;

                let pos1 = if is_rev1 {
                    (l_pac << 1) - 1 - (aln1.pos as i64)
                } else {
                    aln1.pos as i64
                };

                let pos2 = if is_rev2 {
                    (l_pac << 1) - 1 - (aln2.pos as i64)
                } else {
                    aln2.pos as i64
                };

                position_pairs.push((pos1, pos2));
            }
        }
    }

    log::debug!(
        "Extracted {} concordant position pairs from first batch",
        position_pairs.len()
    );

    // Use existing calculation logic
    if !position_pairs.is_empty() {
        calculate_insert_size_stats(l_pac, &position_pairs)
    } else {
        // Return failed stats if no pairs found
        [
            InsertSizeStats {
                avg: 0.0,
                std: 0.0,
                low: 0,
                high: 0,
                failed: true,
            },
            InsertSizeStats {
                avg: 0.0,
                std: 0.0,
                low: 0,
                high: 0,
                failed: true,
            },
            InsertSizeStats {
                avg: 0.0,
                std: 0.0,
                low: 0,
                high: 0,
                failed: true,
            },
            InsertSizeStats {
                avg: 0.0,
                std: 0.0,
                low: 0,
                high: 0,
                failed: true,
            },
        ]
    }
}

// Perform mate rescue on a single batch
// Returns number of pairs rescued
fn mate_rescue_batch(
    batch_pairs: &mut [(Vec<align::Alignment>, Vec<align::Alignment>)],
    batch_seqs1: &[(&str, &[u8], &str)], // (name, seq, qual) for read1
    batch_seqs2: &[(&str, &[u8], &str)], // (name, seq, qual) for read2
    pac: &[u8],
    stats: &[InsertSizeStats; 4],
    bwa_idx: &BwaIndex,
) -> usize {
    let mut rescued = 0;

    // Encode sequence helper
    let encode_seq = |seq: &[u8]| -> Vec<u8> {
        seq.iter()
            .map(|&b| match b {
                b'A' | b'a' => 0,
                b'C' | b'c' => 1,
                b'G' | b'g' => 2,
                b'T' | b't' => 3,
                _ => 4,
            })
            .collect()
    };

    for (i, (alns1, alns2)) in batch_pairs.iter_mut().enumerate() {
        let (name1, seq1, qual1) = batch_seqs1[i];
        let (name2, seq2, qual2) = batch_seqs2[i];

        // DEBUG: Log alignment counts for diagnostic purposes
        log::debug!(
            "Pair {}: {} has {} alignments, {} has {} alignments",
            i,
            name1,
            alns1.len(),
            name2,
            alns2.len()
        );

        // Try to rescue read2 if read1 has good alignments but read2 doesn't
        if !alns1.is_empty() && alns2.is_empty() && !pac.is_empty() {
            let mate_seq = encode_seq(seq2);
            let n = mem_matesw(
                bwa_idx, pac, stats, &alns1[0], // Use best alignment as anchor
                &mate_seq, qual2, name2, alns2,
            );
            if n > 0 {
                rescued += 1;
            }
        }

        // Try to rescue read1 if read2 has good alignments but read1 doesn't
        if !alns2.is_empty() && alns1.is_empty() && !pac.is_empty() {
            let mate_seq = encode_seq(seq1);
            let n = mem_matesw(
                bwa_idx, pac, stats, &alns2[0], // Use best alignment as anchor
                &mate_seq, qual1, name1, alns1,
            );
            if n > 0 {
                rescued += 1;
            }
        }
    }

    rescued
}

// Output a batch of paired-end alignments with proper flags and flushing
// Returns number of records written
fn output_batch_paired(
    batch_pairs: Vec<(Vec<align::Alignment>, Vec<align::Alignment>)>,
    batch_seqs1: &[(String, Vec<u8>, String)], // (name, seq, qual) for read1
    batch_seqs2: &[(String, Vec<u8>, String)], // (name, seq, qual) for read2
    stats: &[InsertSizeStats; 4],
    writer: &mut Box<dyn Write>,
    opt: &MemOpt,
    l_pac: i64,
    starting_pair_id: u64,
) -> std::io::Result<usize> {
    let mut records_written = 0;

    // Extract RG ID once
    let rg_id = opt
        .read_group
        .as_ref()
        .and_then(|rg| crate::mem_opt::MemOpt::extract_rg_id(rg));

    for (pair_idx, (mut alignments1, mut alignments2)) in batch_pairs.into_iter().enumerate() {
        let (name1, seq1, qual1) = &batch_seqs1[pair_idx];
        let (name2, seq2, qual2) = &batch_seqs2[pair_idx];
        let pair_id = starting_pair_id + pair_idx as u64;

        // DEBUG: Log pair processing for diagnostic purposes
        log::debug!(
            "Processing pair {}: {} ({} alns), {} ({} alns)",
            pair_id,
            name1,
            alignments1.len(),
            name2,
            alignments2.len()
        );

        // ===  SCORE THRESHOLD FILTERING (matching C++ bwa-mem2 behavior) ===
        // Filter alignments by score threshold. If ALL alignments filtered,
        // create one unmapped alignment to ensure read is output.
        // This matches C++ bwa-mem2: if (aa.n == 0) { create unmapped record }
        alignments1.retain(|a| a.score >= opt.t);
        alignments2.retain(|a| a.score >= opt.t);

        // If all alignments filtered, create unmapped alignment
        if alignments1.is_empty() {
            log::debug!("{}: All alignments filtered by score threshold, creating unmapped", name1);
            alignments1.push(align::Alignment {
                query_name: name1.to_string(),
                flag: 0x4, // Unmapped (paired flags will be set later)
                ref_name: "*".to_string(),
                ref_id: 0,
                pos: 0,
                mapq: 0,
                score: 0,
                cigar: Vec::new(),
                rnext: "*".to_string(),
                pnext: 0,
                tlen: 0,
                seq: String::from_utf8_lossy(seq1).to_string(),
                qual: qual1.to_string(),
                tags: Vec::new(),
                query_start: 0,
                query_end: 0,
                seed_coverage: 0,
                hash: 0,
            });
        }

        if alignments2.is_empty() {
            log::debug!("{}: All alignments filtered by score threshold, creating unmapped", name2);
            alignments2.push(align::Alignment {
                query_name: name2.to_string(),
                flag: 0x4, // Unmapped (paired flags will be set later)
                ref_name: "*".to_string(),
                ref_id: 0,
                pos: 0,
                mapq: 0,
                score: 0,
                cigar: Vec::new(),
                rnext: "*".to_string(),
                pnext: 0,
                tlen: 0,
                seq: String::from_utf8_lossy(seq2).to_string(),
                qual: qual2.to_string(),
                tags: Vec::new(),
                query_start: 0,
                query_end: 0,
                seed_coverage: 0,
                hash: 0,
            });
        }

        // Use mem_pair to score paired alignments
        let pair_result = if !alignments1.is_empty() && !alignments2.is_empty() {
            let result = mem_pair(stats, &alignments1, &alignments2, l_pac, 2, pair_id);

            // DEBUG: Log mem_pair results
            if pair_id < 10 {
                if result.is_some() {
                    log::debug!("mem_pair SUCCESS for pair {}", pair_id);
                } else {
                    log::debug!(
                        "mem_pair FAIL for pair {}. alns1={}, alns2={}, stats[FR]: low={}, high={}, failed={}",
                        pair_id, alignments1.len(), alignments2.len(), stats[1].low, stats[1].high, stats[1].failed
                    );
                }
            }

            result
        } else {
            None
        };

        // Determine best paired alignment indices and check if properly paired
        let (best_idx1, best_idx2, is_properly_paired) =
            if let Some((idx1, idx2, _pair_score, _sub_score)) = pair_result {
                // If mem_pair returned a valid pair, it's properly paired
                // mem_pair already verified: same chromosome, insert size within bounds, valid orientation
                (idx1, idx2, true)
            } else if !alignments1.is_empty() && !alignments2.is_empty() {
                // mem_pair returned None - no valid pair found
                // Use first alignments but NOT properly paired
                // (insert size out of bounds, or wrong orientation)
                (0, 0, false)
            } else {
                // One or both reads unmapped
                (0, 0, false)
            };

        // Extract mate info from best alignments BEFORE creating dummies
        // (needed to populate dummy alignments with mate's position)
        let (mate2_ref_initial, mate2_pos_initial, mate2_flag_initial) =
            if let Some(aln2) = alignments2.get(best_idx2) {
                (aln2.ref_name.clone(), aln2.pos, aln2.flag)
            } else {
                ("*".to_string(), 0, 0)
            };

        let (mate1_ref_initial, mate1_pos_initial, mate1_flag_initial) =
            if let Some(aln1) = alignments1.get(best_idx1) {
                (aln1.ref_name.clone(), aln1.pos, aln1.flag)
            } else {
                ("*".to_string(), 0, 0)
            };

        // Handle unmapped reads - create dummy alignments
        if alignments1.is_empty() {
            let mut unmapped = align::Alignment {
                query_name: name1.clone(),
                flag: 0x1 | 0x4 | 0x40,
                ref_name: mate2_ref_initial.clone(),
                ref_id: 0,
                pos: mate2_pos_initial,
                mapq: 0,
                score: 0,
                cigar: Vec::new(),
                rnext: "*".to_string(),
                pnext: 0,
                tlen: 0,
                seq: String::from_utf8_lossy(seq1).to_string(),
                qual: qual1.clone(),
                tags: Vec::new(),
                // Internal fields for alignment selection (unmapped = defaults)
                query_start: 0,
                query_end: seq1.len() as i32,
                seed_coverage: 0,
                hash: 0,
            };

            if mate2_ref_initial != "*" {
                unmapped.rnext = "=".to_string();
                unmapped.pnext = mate2_pos_initial + 1;
                if mate2_flag_initial & 0x10 != 0 {
                    unmapped.flag |= 0x20;
                }
            }

            alignments1.push(unmapped);
        }

        if alignments2.is_empty() {
            let mut unmapped = align::Alignment {
                query_name: name2.clone(),
                flag: 0x1 | 0x4 | 0x80,
                ref_name: mate1_ref_initial.clone(),
                ref_id: 0,
                pos: mate1_pos_initial,
                mapq: 0,
                score: 0,
                cigar: Vec::new(),
                rnext: "*".to_string(),
                pnext: 0,
                tlen: 0,
                seq: String::from_utf8_lossy(seq2).to_string(),
                qual: qual2.clone(),
                tags: Vec::new(),
                // Internal fields for alignment selection (unmapped = defaults)
                query_start: 0,
                query_end: seq2.len() as i32,
                seed_coverage: 0,
                hash: 0,
            };

            if mate1_ref_initial != "*" {
                unmapped.rnext = "=".to_string();
                unmapped.pnext = mate1_pos_initial + 1;
                if mate1_flag_initial & 0x10 != 0 {
                    unmapped.flag |= 0x20;
                }
            }

            alignments2.push(unmapped);
        }

        // Re-extract mate info AFTER creating dummy alignments
        // This ensures mapped reads get correct mate information even when mate is unmapped
        let (mate2_ref, mate2_pos, mate2_flag) = if let Some(aln2) = alignments2.get(best_idx2) {
            (aln2.ref_name.clone(), aln2.pos, aln2.flag)
        } else {
            ("*".to_string(), 0, 0)
        };

        let (mate1_ref, mate1_pos, mate1_flag) = if let Some(aln1) = alignments1.get(best_idx1) {
            (aln1.ref_name.clone(), aln1.pos, aln1.flag)
        } else {
            ("*".to_string(), 0, 0)
        };

        // Set flags and mate information for read1
        for (idx, alignment) in alignments1.iter_mut().enumerate() {
            let is_unmapped = alignment.flag & 0x4 != 0;

            // ALWAYS set paired flag (0x1) - even for unmapped reads
            alignment.flag |= 0x1;

            // ALWAYS set first in pair flag (0x40) - even for unmapped reads
            alignment.flag |= 0x40;

            // Only set proper pair flag for mapped reads in proper pairs
            if !is_unmapped && is_properly_paired && idx == best_idx1 {
                alignment.flag |= 0x2;
            }

            // Set mate unmapped flag if mate is unmapped
            if mate2_ref == "*" {
                alignment.flag |= 0x8;
            }

            // Only set mate position and TLEN for mapped reads
            if !is_unmapped && mate2_ref != "*" {
                alignment.rnext = if alignment.ref_name == mate2_ref {
                    "=".to_string()
                } else {
                    mate2_ref.clone()
                };
                alignment.pnext = mate2_pos + 1;

                if mate2_flag & 0x10 != 0 {
                    alignment.flag |= 0x20;
                }

                if alignment.ref_name == mate2_ref {
                    let pos1 = alignment.pos as i64;
                    let pos2 = mate2_pos as i64;

                    // Calculate TLEN: outer_end - outer_start
                    // For FR orientation: leftmost_start to rightmost_end
                    if pos1 <= pos2 {
                        // R1 is leftmost: TLEN = (R2_start - R1_start) + R2_aligned_length
                        let mate2_len = if !alignments2.is_empty() && best_idx2 < alignments2.len() {
                            alignments2[best_idx2].reference_length()
                        } else {
                            0
                        };
                        alignment.tlen = ((pos2 - pos1) + mate2_len as i64) as i32;
                    } else {
                        // R1 is rightmost: TLEN = -((R1_start - R2_start) + R1_aligned_length)
                        let r1_len = alignment.reference_length();
                        alignment.tlen = -(((pos1 - pos2) + r1_len as i64) as i32);
                    }
                }
            }
        }

        // Set flags and mate information for read2
        for (idx, alignment) in alignments2.iter_mut().enumerate() {
            let is_unmapped = alignment.flag & 0x4 != 0;

            // ALWAYS set paired flag (0x1) - even for unmapped reads
            alignment.flag |= 0x1;

            // ALWAYS set second in pair flag (0x80) - even for unmapped reads
            alignment.flag |= 0x80;

            // Only set proper pair flag for mapped reads in proper pairs
            if !is_unmapped && is_properly_paired && idx == best_idx2 {
                alignment.flag |= 0x2;
            }

            // Set mate unmapped flag if mate is unmapped
            if mate1_ref == "*" {
                alignment.flag |= 0x8;
            }

            // Only set mate position and TLEN for mapped reads
            if !is_unmapped && mate1_ref != "*" {
                alignment.rnext = if alignment.ref_name == mate1_ref {
                    "=".to_string()
                } else {
                    mate1_ref.clone()
                };
                alignment.pnext = mate1_pos + 1;

                if mate1_flag & 0x10 != 0 {
                    alignment.flag |= 0x20;
                }

                if alignment.ref_name == mate1_ref {
                    let pos1 = mate1_pos as i64;
                    let pos2 = alignment.pos as i64;

                    // Calculate TLEN: outer_end - outer_start
                    // For FR orientation: leftmost_start to rightmost_end
                    if pos2 <= pos1 {
                        // R2 is leftmost: TLEN = (R1_start - R2_start) + R1_aligned_length
                        let mate1_len = if !alignments1.is_empty() && best_idx1 < alignments1.len() {
                            alignments1[best_idx1].reference_length()
                        } else {
                            0
                        };
                        alignment.tlen = ((pos1 - pos2) + mate1_len as i64) as i32;
                    } else {
                        // R2 is rightmost: TLEN = -((R2_start - R1_start) + R2_aligned_length)
                        let r2_len = alignment.reference_length();
                        alignment.tlen = -(((pos2 - pos1) + r2_len as i64) as i32);
                    }
                }
            }
        }

        // Get mate CIGARs for MC tag
        let mate1_cigar = if !alignments1.is_empty() && best_idx1 < alignments1.len() {
            alignments1[best_idx1].cigar_string()
        } else {
            "*".to_string()
        };

        let mate2_cigar = if !alignments2.is_empty() && best_idx2 < alignments2.len() {
            alignments2[best_idx2].cigar_string()
        } else {
            "*".to_string()
        };

        // Write alignments for read1
        for mut alignment in alignments1 {
            let is_unmapped = alignment.flag & 0x4 != 0;
            // Output unmapped reads always, or reads meeting score threshold
            let should_output = is_unmapped || alignment.score >= opt.t;

            if !should_output {
                continue;  // Skip low-scoring alignments
            }
            if let Some(ref rg) = rg_id {
                alignment.tags.push(("RG".to_string(), format!("Z:{}", rg)));
            }

            if !mate2_cigar.is_empty() {
                alignment
                    .tags
                    .push(("MC".to_string(), format!("Z:{}", mate2_cigar)));
            }

            let sam_record = alignment.to_sam_string();
            writeln!(writer, "{}", sam_record)?;
            records_written += 1;
        }

        // Write alignments for read2
        for mut alignment in alignments2 {
            let is_unmapped = alignment.flag & 0x4 != 0;
            // Output unmapped reads always, or reads meeting score threshold
            let should_output = is_unmapped || alignment.score >= opt.t;

            if !should_output {
                continue;  // Skip low-scoring alignments
            }
            if let Some(ref rg) = rg_id {
                alignment.tags.push(("RG".to_string(), format!("Z:{}", rg)));
            }

            if !mate1_cigar.is_empty() {
                alignment
                    .tags
                    .push(("MC".to_string(), format!("Z:{}", mate1_cigar)));
            }

            let sam_record = alignment.to_sam_string();
            writeln!(writer, "{}", sam_record)?;
            records_written += 1;
        }
    }

    // CRITICAL: Flush after each batch for incremental output
    writer.flush()?;

    Ok(records_written)
}
