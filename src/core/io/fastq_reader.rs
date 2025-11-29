// FASTQ reader module using bio::io::fastq
//
// This module provides a wrapper around bio::io::fastq with:
// - Automatic gzip/bgzip detection by file extension and magic bytes
// - Parallel BGZIP decompression for .gz files (if BGZIP format detected)
// - Batch reading to match our processing pattern (512 reads at a time)
// - Compatible API with our previous kseq implementation
//
// BGZIP format (used in bioinformatics) enables parallel decompression
// via independent compressed blocks. Standard gzip uses single-threaded fallback.

use bio::io::fastq;
use flate2::read::GzDecoder;
use noodles_bgzf as bgzf;
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread::{self, JoinHandle};

/// Batch of FASTQ reads
pub struct ReadBatch {
    pub names: Vec<String>,
    pub seqs: Vec<Vec<u8>>,
    pub quals: Vec<String>, // Store as String for compatibility with existing code
}

impl Default for ReadBatch {
    fn default() -> Self {
        Self::new()
    }
}

impl ReadBatch {
    /// Create an empty batch
    pub fn new() -> Self {
        Self {
            names: Vec::new(),
            seqs: Vec::new(),
            quals: Vec::new(),
        }
    }

    /// Number of reads in this batch
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// Check if batch is empty
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// Convert batch to vector of tuples with references
    /// Returns Vec<(&str, &[u8], &str)> for efficient passing to alignment functions
    /// Eliminates duplicate zip chains in mem.rs
    ///
    /// # Example
    /// ```
    /// use ferrous_align::io::fastq_reader::ReadBatch;
    ///
    /// let batch = ReadBatch {
    ///     names: vec!["read1".to_string()],
    ///     seqs: vec![b"ACGT".to_vec()],
    ///     quals: vec!["IIII".to_string()],
    /// };
    /// let tuples = batch.as_tuple_refs();
    /// assert_eq!(tuples.len(), 1);
    /// assert_eq!(tuples[0].0, "read1");
    /// ```
    pub fn as_tuple_refs(&self) -> Vec<(&str, &[u8], &str)> {
        self.names
            .iter()
            .zip(&self.seqs)
            .zip(&self.quals)
            .map(|((n, s), q)| (n.as_str(), s.as_slice(), q.as_str()))
            .collect()
    }
}

/// FASTQ reader with automatic gzip/bgzip detection
pub struct FastqReader {
    records: fastq::Records<BufReader<Box<dyn Read + Send>>>,
}

/// Detect if a gzipped file is BGZIP format by checking for BGZIP-specific header
fn is_bgzip_format(path: &str) -> io::Result<bool> {
    let mut file = File::open(path)?;
    let mut header = [0u8; 18]; // BGZIP header is at least 18 bytes

    // Try to read header
    if file.read(&mut header).unwrap_or(0) < 18 {
        return Ok(false); // Not enough bytes for BGZIP header
    }

    // Check for gzip magic bytes
    if header[0] != 0x1f || header[1] != 0x8b {
        return Ok(false); // Not gzip at all
    }

    // BGZIP uses extra field (FEXTRA flag = 0x04)
    if header[3] & 0x04 == 0 {
        return Ok(false); // No extra field, likely standard gzip
    }

    // BGZIP has specific extra field with 'BC' subfield ID
    // Extra field starts at byte 10, check for 'BC' signature at expected offset
    if header[12] == b'B' && header[13] == b'C' {
        return Ok(true); // BGZIP format detected
    }

    Ok(false)
}

impl FastqReader {
    /// Open a FASTQ file (auto-detects gzip/bgzip by .gz extension and magic bytes)
    ///
    /// Uses parallel BGZIP decompression for .gz files when BGZIP format is detected,
    /// utilizing multiple CPU cores for significant I/O speedup (3-10x faster).
    /// Falls back to single-threaded decompression for standard gzip.
    ///
    /// # Arguments
    /// * `path` - Path to FASTQ file (.fq, .fastq, .fq.gz, .fastq.gz)
    ///
    /// # Returns
    /// * `Ok(FastqReader)` on success
    /// * `Err(io::Error)` if file cannot be opened
    pub fn new(path: &str) -> io::Result<Self> {
        const BUFFER_SIZE: usize = 4 * 1024 * 1024; // 4MB buffer

        let reader: Box<dyn Read + Send> = if path.ends_with(".gz") {
            // Check if file is BGZIP format
            if is_bgzip_format(path)? {
                log::debug!("Detected BGZIP format, using parallel decompression");
                // Use noodles-bgzf for parallel decompression
                let file = File::open(path)?;
                let bgzf_reader = bgzf::MultithreadedReader::new(file);
                Box::new(BufReader::with_capacity(BUFFER_SIZE, bgzf_reader))
            } else {
                log::debug!("Detected standard gzip format, using single-threaded decompression");
                // Fall back to flate2 for standard gzip
                let file = File::open(path)?;
                Box::new(BufReader::with_capacity(BUFFER_SIZE, GzDecoder::new(file)))
            }
        } else {
            // Uncompressed file
            let file = File::open(path)?;
            Box::new(BufReader::with_capacity(BUFFER_SIZE, file))
        };

        let fastq_reader = fastq::Reader::new(reader);

        Ok(Self {
            records: fastq_reader.records(),
        })
    }

    /// Read a batch of reads (up to batch_size)
    ///
    /// Returns an empty batch when EOF is reached.
    ///
    /// # Arguments
    /// * `batch_size` - Maximum number of reads to return
    ///
    /// # Returns
    /// * `Ok(ReadBatch)` containing up to batch_size reads
    /// * `Err(io::Error)` on parse error
    pub fn read_batch(&mut self, batch_size: usize) -> io::Result<ReadBatch> {
        let mut batch = ReadBatch::new();

        for _ in 0..batch_size {
            match self.records.next() {
                Some(Ok(record)) => {
                    batch.names.push(record.id().to_string());
                    batch.seqs.push(record.seq().to_vec());
                    // Convert quality bytes to String (ASCII)
                    batch
                        .quals
                        .push(String::from_utf8_lossy(record.qual()).into_owned());
                }
                Some(Err(e)) => {
                    return Err(io::Error::other(e));
                }
                None => {
                    // EOF
                    break;
                }
            }
        }

        Ok(batch)
    }
}

// =============================================================================
// DOUBLE-BUFFERED PAIRED READER
// =============================================================================
//
// This reader overlaps I/O with computation by reading the next batch in a
// background thread while the main thread processes the current batch.
//
// Architecture:
// - Background thread: Reads paired batches and sends through bounded channel
// - Main thread: Receives pre-read batches (should be ready immediately)
// - Channel capacity of 1 provides double-buffering (one being processed,
//   one being read)
//
// Performance impact:
// - Eliminates serial I/O wait between batches
// - I/O happens in parallel with alignment computation
// - Improves CPU utilization from ~469% to ~800%+ (goal)

/// Result of reading a paired batch
pub type PairedBatchResult = Result<(ReadBatch, ReadBatch), io::Error>;

/// Double-buffered reader for paired FASTQ files
///
/// Reads batches in a background thread to overlap I/O with computation.
/// This eliminates the serial I/O bottleneck between parallel processing batches.
pub struct DoubleBufferedPairedReader {
    receiver: Receiver<PairedBatchResult>,
    reader_thread: Option<JoinHandle<()>>,
}

impl DoubleBufferedPairedReader {
    /// Create a new double-buffered paired reader
    ///
    /// Immediately starts reading the first batch in a background thread.
    ///
    /// # Arguments
    /// * `path1` - Path to read 1 FASTQ file
    /// * `path2` - Path to read 2 FASTQ file
    /// * `batch_size` - Number of read pairs per batch
    ///
    /// # Returns
    /// * `Ok(DoubleBufferedPairedReader)` on success
    /// * `Err(io::Error)` if files cannot be opened
    pub fn new(path1: &str, path2: &str, batch_size: usize) -> io::Result<Self> {
        // Open readers to validate paths before spawning thread
        let reader1 = FastqReader::new(path1)?;
        let reader2 = FastqReader::new(path2)?;

        // Bounded channel with capacity 1 = double buffering
        // One batch being processed, one batch being read
        let (sender, receiver): (SyncSender<PairedBatchResult>, Receiver<PairedBatchResult>) =
            mpsc::sync_channel(1);

        // Spawn background reader thread
        let reader_thread = thread::spawn(move || {
            Self::reader_loop(reader1, reader2, batch_size, sender);
        });

        Ok(Self {
            receiver,
            reader_thread: Some(reader_thread),
        })
    }

    /// Background reader loop - reads batches and sends through channel
    fn reader_loop(
        mut reader1: FastqReader,
        mut reader2: FastqReader,
        batch_size: usize,
        sender: SyncSender<PairedBatchResult>,
    ) {
        loop {
            // Read paired batch
            let batch1 = match reader1.read_batch(batch_size) {
                Ok(b) => b,
                Err(e) => {
                    let _ = sender.send(Err(e));
                    return;
                }
            };

            let batch2 = match reader2.read_batch(batch_size) {
                Ok(b) => b,
                Err(e) => {
                    let _ = sender.send(Err(e));
                    return;
                }
            };

            // Check for EOF (empty batch)
            if batch1.is_empty() {
                // Send empty batch to signal EOF, then exit
                let _ = sender.send(Ok((batch1, batch2)));
                return;
            }

            // Send batch through channel (blocks if channel full = backpressure)
            if sender.send(Ok((batch1, batch2))).is_err() {
                // Receiver dropped, exit thread
                return;
            }
        }
    }

    /// Receive the next pre-read batch pair
    ///
    /// This should return almost immediately if the reader thread has been
    /// reading while the previous batch was being processed.
    ///
    /// # Returns
    /// * `Ok(Some((batch1, batch2)))` - Next batch pair
    /// * `Ok(None)` - EOF reached
    /// * `Err(io::Error)` - Read error occurred
    pub fn next_batch(&self) -> io::Result<Option<(ReadBatch, ReadBatch)>> {
        match self.receiver.recv() {
            Ok(Ok((batch1, batch2))) => {
                if batch1.is_empty() {
                    Ok(None) // EOF
                } else {
                    Ok(Some((batch1, batch2)))
                }
            }
            Ok(Err(e)) => Err(e),
            Err(_) => {
                // Channel disconnected - reader thread exited unexpectedly
                Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "Reader thread disconnected",
                ))
            }
        }
    }
}

impl Drop for DoubleBufferedPairedReader {
    fn drop(&mut self) {
        // Wait for reader thread to finish
        if let Some(handle) = self.reader_thread.take() {
            let _ = handle.join();
        }
    }
}
