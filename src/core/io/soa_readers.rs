//! This module contains the implementation for Structure-of-Arrays (SoA) based readers.
#![allow(dead_code)]

use bio::io::fastq;
use flate2::read::GzDecoder;
use noodles_bgzf as bgzf;
use std::fs::File;
use std::io::{self, BufReader, Read};

/// `SoAReadBatch` holds a batch of reads in a Structure-of-Arrays (SoA) format.
/// This layout is designed for efficient processing in later stages of the pipeline,
/// particularly for SIMD-heavy operations.
pub struct SoAReadBatch {
    /// Sequence data for all reads in the batch, stored contiguously.
    pub seqs: Vec<u8>,
    /// Quality scores for all reads.
    pub quals: Vec<u8>,
    /// Read names.
    pub names: Vec<String>,
    /// Offsets and lengths to delineate individual reads within the contiguous buffers.
    /// Each element is a tuple of (start_offset, length).
    pub read_boundaries: Vec<(usize, usize)>,
}

impl Default for SoAReadBatch {
    fn default() -> Self {
        Self::new()
    }
}

impl SoAReadBatch {
    /// Create an empty batch
    pub fn new() -> Self {
        Self {
            seqs: Vec::new(),
            quals: Vec::new(),
            names: Vec::new(),
            read_boundaries: Vec::new(),
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
}

/// FASTQ reader with automatic gzip/bgzip detection that reads into SoA buffers.
pub struct SoaFastqReader {
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

impl SoaFastqReader {
    /// Open a FASTQ file (auto-detects gzip/bgzip by .gz extension and magic bytes)
    pub fn new(path: &str) -> io::Result<Self> {
        const BUFFER_SIZE: usize = 4 * 1024 * 1024; // 4MB buffer

        let reader: Box<dyn Read + Send> = if path.ends_with(".gz") {
            if is_bgzip_format(path)? {
                log::debug!("Detected BGZIP format, using parallel decompression");
                let file = File::open(path)?;
                let bgzf_reader = bgzf::MultithreadedReader::new(file);
                Box::new(BufReader::with_capacity(BUFFER_SIZE, bgzf_reader))
            } else {
                log::debug!("Detected standard gzip format, using single-threaded decompression");
                let file = File::open(path)?;
                Box::new(BufReader::with_capacity(BUFFER_SIZE, GzDecoder::new(file)))
            }
        } else {
            let file = File::open(path)?;
            Box::new(BufReader::with_capacity(BUFFER_SIZE, file))
        };

        let fastq_reader = fastq::Reader::new(reader);

        Ok(Self {
            records: fastq_reader.records(),
        })
    }

    /// Read a batch of reads (up to batch_size) into an SoA buffer
    pub fn read_batch(&mut self, batch_size: usize) -> io::Result<SoAReadBatch> {
        let mut batch = SoAReadBatch::new();

        for _ in 0..batch_size {
            match self.records.next() {
                Some(Ok(record)) => {
                    batch.names.push(record.id().to_string());

                    let seq = record.seq();
                    let qual = record.qual();

                    // Handle mismatched seq/qual lengths (rare edge case in malformed FASTQ)
                    // Bio crate and BWA-MEM2 both accept mismatched lengths, so we do too
                    if seq.len() != qual.len() {
                        log::warn!(
                            "Read {} has mismatched seq/qual lengths ({} vs {}), using shorter length",
                            record.id(),
                            seq.len(),
                            qual.len()
                        );
                    }

                    let seq_start_offset = batch.seqs.len();
                    let seq_len = seq.len();

                    // Store sequence as-is (even if qual length differs)
                    // This matches bio crate and BWA-MEM2 behavior
                    batch.seqs.extend_from_slice(seq);

                    // Pad or truncate qual to match seq length
                    if qual.len() < seq.len() {
                        // Pad with 'I' (quality 40, common default)
                        batch.quals.extend_from_slice(qual);
                        batch
                            .quals
                            .resize(batch.quals.len() + (seq.len() - qual.len()), b'I');
                    } else {
                        // Truncate qual to seq length
                        batch.quals.extend_from_slice(&qual[..seq.len()]);
                    }

                    batch.read_boundaries.push((seq_start_offset, seq_len));
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_soa_fastq_reader() {
        // 1. Create a dummy FASTQ file
        let mut tmpfile = NamedTempFile::new().unwrap();
        let content = b"@read1\nACGT\n+\nIIII\n@read2\nTGCA\n+\nJJJJ\n";
        tmpfile.write_all(content).unwrap();

        // 2. Use SoaFastqReader to read a batch from it
        let mut reader = SoaFastqReader::new(tmpfile.path().to_str().unwrap()).unwrap();
        let batch = reader.read_batch(2).unwrap();

        // 3. Assert that the SoAReadBatch contains the correct data
        assert_eq!(batch.len(), 2);
        assert_eq!(batch.names, vec!["read1", "read2"]);

        assert_eq!(batch.seqs, b"ACGTTGCA");
        assert_eq!(batch.quals, b"IIIIJJJJ");

        assert_eq!(batch.read_boundaries, vec![(0, 4), (4, 4)]);

        // Test reconstruction of first read
        let (start1, len1) = batch.read_boundaries[0];
        assert_eq!(&batch.seqs[start1..start1 + len1], b"ACGT");
        assert_eq!(&batch.quals[start1..start1 + len1], b"IIII");

        // Test reconstruction of second read
        let (start2, len2) = batch.read_boundaries[1];
        assert_eq!(&batch.seqs[start2..start2 + len2], b"TGCA");
        assert_eq!(&batch.quals[start2..start2 + len2], b"JJJJ");
    }
}
