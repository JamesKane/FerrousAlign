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

/// Batch of FASTQ reads
pub struct ReadBatch {
    pub names: Vec<String>,
    pub seqs: Vec<Vec<u8>>,
    pub quals: Vec<String>, // Store as String for compatibility with existing code
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
    records: fastq::Records<BufReader<Box<dyn Read>>>,
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

        let reader: Box<dyn Read> = if path.ends_with(".gz") {
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
                    return Err(io::Error::new(io::ErrorKind::Other, e));
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
