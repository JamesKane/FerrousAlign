// FASTA reader module using bio::io::fasta
//
// This module provides a wrapper around bio::io::fasta with:
// - Automatic gzip/bgzip detection by file extension and magic bytes
// - Parallel BGZIP decompression for .gz files (if BGZIP format detected)
//
// BGZIP format (used in bioinformatics) enables parallel decompression
// via independent compressed blocks. Standard gzip uses single-threaded fallback.

use bio::io::fasta;
use flate2::read::GzDecoder;
use noodles_bgzf as bgzf;
use std::fs::File;
use std::io::{self, BufReader, Read};

/// FASTA reader with automatic gzip/bgzip detection
pub struct FastaReader {
    records: fasta::Records<BufReader<Box<dyn Read>>>,
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

impl FastaReader {
    /// Open a FASTA file (auto-detects gzip/bgzip by .gz extension and magic bytes)
    ///
    /// Uses parallel BGZIP decompression for .gz files when BGZIP format is detected,
    /// utilizing multiple CPU cores for significant I/O speedup (3-10x faster).
    /// Falls back to single-threaded decompression for standard gzip.
    ///
    /// # Arguments
    /// * `path` - Path to FASTA file (.fa, .fasta, .fa.gz, .fasta.gz)
    ///
    /// # Returns
    /// * `Ok(FastaReader)` on success
    /// * `Err(io::Error)` if file cannot be opened
    pub fn new(path: &str) -> io::Result<Self> {
        const BUFFER_SIZE: usize = 4 * 1024 * 1024; // 4MB buffer

        let file = File::open(path)?;
        let reader: Box<dyn Read> = if path.ends_with(".gz") {
            // Check if file is BGZIP format
            if is_bgzip_format(path)? {
                log::debug!("Detected BGZIP format, using parallel decompression for FASTA");
                // Use noodles-bgzf for parallel decompression
                let bgzf_reader = bgzf::MultithreadedReader::new(File::open(path)?);
                Box::new(BufReader::with_capacity(BUFFER_SIZE, bgzf_reader))
            } else {
                log::debug!("Detected standard gzip format, using single-threaded decompression for FASTA");
                // Fall back to flate2 for standard gzip
                Box::new(BufReader::with_capacity(BUFFER_SIZE, GzDecoder::new(file)))
            }
        } else {
            // Uncompressed file
            Box::new(BufReader::with_capacity(BUFFER_SIZE, file))
        };

        let fasta_reader = fasta::Reader::new(reader);

        Ok(Self {
            records: fasta_reader.records(),
        })
    }

    /// Read the next FASTA record
    ///
    /// Returns `Ok(Some(record))` if a record is found, `Ok(None)` at EOF,
    /// and `Err(e)` on a parse error.
    pub fn read_record(&mut self) -> io::Result<Option<fasta::Record>> {
        match self.records.next() {
            Some(Ok(record)) => Ok(Some(record)),
            Some(Err(e)) => Err(io::Error::new(io::ErrorKind::Other, e)),
            None => Ok(None),
        }
    }
}
