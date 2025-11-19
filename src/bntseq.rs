use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

// From C's bntseq.h
pub const NST_NT4_TABLE: [u8; 256] = [
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 0, 4, 1, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 0, 4, 1, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
];

/// Per-sequence annotation
/// Corresponds to C++ bntann1_t (bntseq.h:41-48)
///
/// Stores metadata for a single reference sequence (e.g., chromosome).
#[derive(Debug)]
pub struct BntAnn1 {
    /// Offset in the concatenated packed sequence
    pub offset: u64,
    /// Length of this sequence
    pub sequence_length: i32,
    /// Number of ambiguous bases (N) in this sequence
    pub ambiguous_base_count: i32,
    /// GenInfo identifier (GI number from NCBI)
    pub geninfo_identifier: u32,
    /// Whether this is an alternate locus (0 = primary, non-zero = alternate)
    pub is_alternate_locus: i32,
    /// Sequence name (e.g., "chr1", "chrM")
    pub name: String,
    /// Optional annotation/comment from FASTA header
    pub anno: String,
}

/// Ambiguous base region
/// Corresponds to C++ bntamb1_t (bntseq.h:50-54)
///
/// Represents a contiguous region of ambiguous bases (typically 'N') in the
/// reference sequence. Multiple consecutive ambiguous bases are coalesced
/// into a single region.
#[derive(Debug)]
pub struct BntAmb1 {
    /// Offset of this ambiguous region in the packed sequence
    pub offset: u64,
    /// Length of this ambiguous region (number of consecutive N bases)
    pub region_length: i32,
    /// The ambiguous base character (typically 'N')
    pub ambiguous_base: char,
}

/// Reference sequence database (BNT format)
/// Corresponds to C++ bntseq_t (bntseq.h:56-64)
///
/// Stores reference genome sequences in 2-bit packed format (.pac file)
/// along with annotations for each sequence and ambiguous base regions.
#[derive(Debug)]
pub struct BntSeq {
    /// Total length of packed sequence (sum of all reference sequences)
    pub packed_sequence_length: u64,
    /// Number of reference sequences (e.g., chromosomes)
    pub sequence_count: i32,
    /// Random seed for ambiguous base replacement (fixed at 11)
    pub seed: u32,
    /// Per-sequence annotations (sequence_count elements)
    pub annotations: Vec<BntAnn1>,
    /// Number of ambiguous base regions (N-base "holes")
    pub ambiguous_region_count: i32,
    /// Ambiguous base regions (ambiguous_region_count elements)
    pub ambiguous_regions: Vec<BntAmb1>,
    /// Optional path to .pac file for on-demand loading
    pub pac_file_path: Option<PathBuf>,
}

#[path = "bntseq_test.rs"]
mod bntseq_test;

impl BntSeq {
    pub fn new() -> Self {
        BntSeq {
            packed_sequence_length: 0,
            sequence_count: 0,
            seed: 0,
            annotations: Vec::new(),
            ambiguous_region_count: 0,
            ambiguous_regions: Vec::new(),
            pac_file_path: None,
        }
    }

    pub fn bns_fasta2bntseq<R: Read + 'static>(
        reader: R,
        prefix: &Path,
        _for_only: bool, // for_only is not used in the C version's bns_fasta2bntseq
    ) -> io::Result<Self> {
        let mut bns = BntSeq::new();
        bns.seed = 11; // Fixed seed matching C++ bwa-mem2 (line 314 in bntseq.cpp)
        let mut rng = StdRng::seed_from_u64(bns.seed as u64);

        let mut kseq = crate::kseq::KSeq::new(Box::new(reader));
        let mut pac_data: Vec<u8> = Vec::new();

        let mut seq_id = 0;
        let mut pac_len = 0; // This will now track the total length including ambiguous bases
        let mut packed_base_count = 0; // This will track ALL bases (regular + ambiguous replaced with random)

        let mut current_ambs: Vec<BntAmb1> = Vec::new();

        loop {
            let len = kseq.read()?; // Reads one sequence
            if len < 0 {
                break;
            } // EOF or error

            let mut ann = BntAnn1 {
                // Create a new BntAnn1 for each sequence
                offset: packed_base_count,
                sequence_length: kseq.seq.len() as i32,
                ambiguous_base_count: 0, // Will be updated later
                geninfo_identifier: 0,
                is_alternate_locus: 0,
                name: kseq.name.clone(),
                anno: kseq.comment.clone(),
            };

            let mut n_ambs_in_seq = 0;
            for base in kseq.seq.bytes() {
                let mut nt4 = NST_NT4_TABLE[base as usize];

                // CRITICAL: Match C++ bwa-mem2 behavior (bntseq.cpp lines 264-292)
                // For ambiguous bases (N, etc.), record in ambs array AND replace with random base
                if nt4 >= 4 {
                    // Ambiguous base - record it
                    if current_ambs.is_empty()
                        || current_ambs.last().unwrap().offset
                            + current_ambs.last().unwrap().region_length as u64
                            != pac_len
                    {
                        current_ambs.push(BntAmb1 {
                            offset: pac_len,
                            region_length: 1,
                            ambiguous_base: base as char,
                        });
                    } else {
                        current_ambs.last_mut().unwrap().region_length += 1;
                    }
                    n_ambs_in_seq += 1;

                    // Replace with random base (C++ line 284: if (c >= 4) c = lrand48()&3;)
                    nt4 = rng.gen_range(0..4);
                }

                // Write ALL bases (regular + ambiguous-replaced-with-random) to pac_data
                let last_byte_idx = (packed_base_count / 4) as usize;
                if pac_data.len() <= last_byte_idx {
                    pac_data.resize(last_byte_idx + 1, 0);
                }
                // CRITICAL: Match C++ bwa-mem2 bit packing order
                // Same formula as extraction: ((~pos & 3) << 1)
                let shift = ((!(packed_base_count % 4)) & 3) << 1;
                pac_data[last_byte_idx] |= nt4 << shift;
                packed_base_count += 1;
                pac_len += 1;
            }
            ann.ambiguous_base_count = n_ambs_in_seq; // Update n_ambs for the current annotation

            bns.annotations.push(ann); // Push the completed annotation
            seq_id += 1;
        }

        bns.packed_sequence_length = packed_base_count; // ALL bases including ambiguous (replaced with random)
        bns.sequence_count = seq_id;
        bns.ambiguous_regions = current_ambs;
        bns.ambiguous_region_count = bns.ambiguous_regions.len() as i32;
        bns.pac_file_path = Some(PathBuf::from(prefix.to_string_lossy().to_string() + ".pac"));

        // Write .pac file with C++ bwa-mem2 format (bntseq.cpp lines 340-347)
        // Format: packed_bases + [optional_zero_byte] + count_byte
        let mut pac_file = File::create(prefix.with_extension("pac"))?;
        pac_file.write_all(&pac_data)?;

        // C++: if (bns->l_pac % 4 == 0) { ct = 0; err_fwrite(&ct, 1, 1, fp); }
        if bns.packed_sequence_length % 4 == 0 {
            pac_file.write_all(&[0u8])?;
        }

        // C++: ct = bns->l_pac % 4; err_fwrite(&ct, 1, 1, fp);
        let remainder = (bns.packed_sequence_length % 4) as u8;
        pac_file.write_all(&[remainder])?;

        log::info!(
            "Created .pac file: l_pac={}, file_bytes={}, metadata_bytes={}",
            bns.packed_sequence_length,
            pac_data.len(),
            if bns.packed_sequence_length % 4 == 0 {
                2
            } else {
                1
            }
        );

        Ok(bns)
    }

    pub fn bns_dump(&self, prefix: &Path) -> io::Result<()> {
        // Dump annotations
        let ann_file_path = prefix.with_extension("ann");
        let mut ann_file = BufWriter::new(File::create(&ann_file_path)?);
        writeln!(
            ann_file,
            "{} {} {}",
            self.packed_sequence_length, self.sequence_count, self.seed
        )?;
        for p in &self.annotations {
            if p.anno.is_empty() || p.anno == "(null)" {
                writeln!(ann_file, "{} {}", p.geninfo_identifier, p.name)?;
            } else {
                writeln!(ann_file, "{} {} {}", p.geninfo_identifier, p.name, p.anno)?;
            }
            writeln!(
                ann_file,
                "{} {} {}",
                p.offset, p.sequence_length, p.ambiguous_base_count
            )?;
        }
        ann_file.flush()?;

        // Dump ambiguous bases
        let amb_file_path = prefix.with_extension("amb");
        let mut amb_file = BufWriter::new(File::create(&amb_file_path)?);
        writeln!(
            amb_file,
            "{} {} {}",
            self.packed_sequence_length, self.sequence_count, self.ambiguous_region_count
        )?;
        for p in &self.ambiguous_regions {
            writeln!(amb_file, "{} {} {}", p.offset, p.region_length, p.ambiguous_base)?;
        }
        amb_file.flush()?;

        Ok(())
    }

    pub fn bns_restore(prefix: &Path) -> io::Result<Self> {
        let mut bns = BntSeq::new();

        // Restore annotations
        let ann_file_path = PathBuf::from(prefix.to_string_lossy().to_string() + ".ann");
        // eprintln!("Attempting to open .ann file: {:?}", ann_file_path);
        let ann_file = BufReader::new(File::open(&ann_file_path)?);
        // eprintln!("Successfully opened .ann file: {:?}", ann_file_path);
        let mut lines = ann_file.lines();

        // Read first line: l_pac n_seqs seed
        let first_line = lines.next().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "Missing first line in .ann file",
            )
        })??;
        let parts: Vec<&str> = first_line.split_whitespace().collect();
        bns.packed_sequence_length = parts[0]
            .parse()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid l_pac in .ann"))?;
        bns.sequence_count = parts[1]
            .parse()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid n_seqs in .ann"))?;
        bns.seed = parts[2]
            .parse()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid seed in .ann"))?;

        bns.annotations.reserve(bns.sequence_count as usize);
        for _ in 0..bns.sequence_count {
            // Read name and anno line: gi name [anno]
            let name_anno_line = lines.next().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Missing name/anno line in .ann file",
                )
            })??;
            let name_anno_parts: Vec<&str> = name_anno_line.splitn(3, ' ').collect();
            let gi = name_anno_parts[0]
                .parse()
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid gi in .ann"))?;
            let name = name_anno_parts[1].to_string();
            let anno = if name_anno_parts.len() > 2 {
                name_anno_parts[2].to_string()
            } else {
                String::new()
            };

            // Read offset, len, n_ambs line
            let offset_len_ambs_line = lines.next().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Missing offset/len/ambs line in .ann file",
                )
            })??;
            let offset_len_ambs_parts: Vec<&str> =
                offset_len_ambs_line.split_whitespace().collect();
            let offset = offset_len_ambs_parts[0].parse().map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidData, "Invalid offset in .ann")
            })?;
            let sequence_length = offset_len_ambs_parts[1]
                .parse()
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid len in .ann"))?;
            let ambiguous_base_count = offset_len_ambs_parts[2].parse().map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidData, "Invalid n_ambs in .ann")
            })?;

            bns.annotations.push(BntAnn1 {
                offset,
                sequence_length,
                ambiguous_base_count,
                geninfo_identifier: gi,
                is_alternate_locus: 0, // This is not stored in .ann, default to 0
                name,
                anno,
            });
        }

        // Restore ambiguous bases
        let amb_file_path = PathBuf::from(prefix.to_string_lossy().to_string() + ".amb");
        // eprintln!("Attempting to open .amb file: {:?}", amb_file_path);
        let amb_file = BufReader::new(File::open(&amb_file_path)?);
        // eprintln!("Successfully opened .amb file: {:?}", amb_file_path);
        let mut amb_lines = amb_file.lines();

        // Read first line: l_pac n_seqs n_holes
        let amb_first_line = amb_lines.next().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "Missing first line in .amb file",
            )
        })??;
        let amb_parts: Vec<&str> = amb_first_line.split_whitespace().collect();
        // We already have l_pac and n_seqs from .ann, so just read n_holes
        bns.ambiguous_region_count = amb_parts[2]
            .parse()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid n_holes in .amb"))?;

        bns.ambiguous_regions
            .reserve(bns.ambiguous_region_count as usize);
        for _ in 0..bns.ambiguous_region_count {
            let amb_data_line = amb_lines.next().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Missing amb data line in .amb file",
                )
            })??;
            let amb_data_parts: Vec<&str> = amb_data_line.split_whitespace().collect();
            let offset = amb_data_parts[0].parse().map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidData, "Invalid offset in .amb")
            })?;
            let len = amb_data_parts[1]
                .parse()
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid len in .amb"))?;
            let amb = amb_data_parts[2].chars().next().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "Missing amb char in .amb")
            })?;

            bns.ambiguous_regions.push(BntAmb1 { offset, region_length: len, ambiguous_base: amb });
        }

        bns.pac_file_path = Some(PathBuf::from(prefix.to_string_lossy().to_string() + ".pac"));
        // eprintln!("PAC file path set to: {:?}", bns.pac_file_path);

        Ok(bns)
    }

    // New method to get a segment of the reference sequence
    // CRITICAL: Handles both forward (start < l_pac) and reverse complement (start >= l_pac) strands
    // Matches C++ bwa-mem2 bns_get_seq() behavior
    pub fn get_reference_segment(&self, start: u64, len: u64) -> io::Result<Vec<u8>> {
        let pac_file_path = self
            .pac_file_path
            .as_ref()
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "PAC file path not set"))?;

        let end = start + len;

        // Check if we're bridging the forward-reverse boundary
        if start < self.packed_sequence_length && end > self.packed_sequence_length {
            log::warn!(
                "get_reference_segment: bridging forward-reverse boundary at start={}, end={}, l_pac={} - returning empty",
                start,
                end,
                self.packed_sequence_length
            );
            return Ok(Vec::new());
        }

        let mut segment = Vec::with_capacity(len as usize);

        if start >= self.packed_sequence_length {
            // Reverse complement strand: convert to forward coordinates
            // C++ formula: beg_f = (l_pac<<1) - 1 - end, end_f = (l_pac<<1) - 1 - beg
            // CRITICAL: Must match C++ exactly - was using (end - 1) which added +1 error
            let beg_f = ((self.packed_sequence_length << 1) - 1).saturating_sub(end);
            let end_f = ((self.packed_sequence_length << 1) - 1).saturating_sub(start);

            log::debug!(
                "get_reference_segment: RC strand start={}, len={}, l_pac={}, beg_f={}, end_f={}",
                start,
                len,
                self.packed_sequence_length,
                beg_f,
                end_f
            );

            // Read in reverse order with complementation
            let mut pac_file = BufReader::new(File::open(pac_file_path)?);
            for k in (beg_f + 1..=end_f).rev() {
                let byte_idx = k / 4;
                let mut byte_buf = [0u8; 1];
                pac_file.seek(SeekFrom::Start(byte_idx))?;
                pac_file.read_exact(&mut byte_buf)?;

                let base_in_byte_offset = ((!(k & 3)) & 3) * 2;
                let base = (byte_buf[0] >> base_in_byte_offset) & 0x3;
                let complement = 3 - base; // A<->T (0<->3), C<->G (1<->2)
                segment.push(complement);
            }
        } else {
            // Forward strand: read normally
            let start_byte_offset = start / 4;
            let end_byte_offset = (end - 1) / 4;
            let bytes_to_read = (end_byte_offset - start_byte_offset + 1) as usize;

            let mut pac_file = BufReader::new(File::open(pac_file_path)?);
            let mut pac_bytes = vec![0u8; bytes_to_read];
            pac_file.seek(SeekFrom::Start(start_byte_offset))?;
            pac_file.read_exact(&mut pac_bytes)?;

            log::debug!(
                "get_reference_segment: FWD strand start={}, len={}, l_pac={}, start_byte_offset={}, bytes_to_read={}",
                start,
                len,
                self.packed_sequence_length,
                start_byte_offset,
                bytes_to_read
            );

            for i in 0..len {
                let k = start + i;
                let byte_idx_in_segment = (k / 4 - start_byte_offset) as usize;
                // CRITICAL: Match C++ bwa-mem2 bit order
                // C++ uses: ((pac)[(l)>>2]>>((~(l)&3)<<1)&3)
                // Bit shifts are: l=0->6, l=1->4, l=2->2, l=3->0 (MSB to LSB)
                let base_in_byte_offset = ((!(k & 3)) & 3) * 2;
                let base = (pac_bytes[byte_idx_in_segment] >> base_in_byte_offset) & 0x3;
                segment.push(base);
            }
        }

        log::debug!(
            "get_reference_segment: extracted {} bases, first_10={:?}",
            segment.len(),
            &segment[..segment.len().min(10)]
        );

        Ok(segment)
    }

    /// Helper function: Convert position to forward strand position and determine if reverse
    /// Equivalent to C's bns_depos
    pub fn bns_depos(&self, pos: i64) -> (i64, bool) {
        let is_rev = pos >= self.packed_sequence_length as i64;
        let pos_f = if is_rev {
            ((self.packed_sequence_length as i64) << 1) - 1 - pos
        } else {
            pos
        };
        (pos_f, is_rev)
    }

    /// Helper function: Find which reference sequence contains the given forward position
    /// Equivalent to C's bns_pos2rid
    pub fn bns_pos2rid(&self, pos_f: i64) -> i32 {
        if pos_f >= self.packed_sequence_length as i64 {
            return -1;
        }

        let mut left = 0;
        let mut right = self.sequence_count as usize;
        let mut mid = 0;

        while left < right {
            mid = (left + right) >> 1;
            if pos_f >= self.annotations[mid].offset as i64 {
                if mid == self.sequence_count as usize - 1 {
                    break;
                }
                if pos_f < self.annotations[mid + 1].offset as i64 {
                    break;
                }
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        mid as i32
    }

    /// Extract a base from packed format
    /// Equivalent to C's _get_pac macro
    #[inline]
    fn get_pac(pac: &[u8], pos: i64) -> u8 {
        let byte_idx = (pos >> 2) as usize;
        let shift = ((!(pos as u8) & 3) << 1) as u32;
        (pac[byte_idx] >> shift) & 3
    }

    /// Get reference sequence from packed format (handles both strands)
    /// Equivalent to C's bns_get_seq
    pub fn bns_get_seq(&self, pac: &[u8], mut beg: i64, mut end: i64) -> Vec<u8> {
        // Swap if end < beg
        if end < beg {
            std::mem::swap(&mut beg, &mut end);
        }

        // Clamp to valid range
        if end > (self.packed_sequence_length as i64) << 1 {
            end = (self.packed_sequence_length as i64) << 1;
        }
        if beg < 0 {
            beg = 0;
        }

        // Check if bridging forward-reverse boundary
        if beg >= self.packed_sequence_length as i64 || end <= self.packed_sequence_length as i64 {
            let len = (end - beg) as usize;
            let mut seq = Vec::with_capacity(len);

            if beg >= self.packed_sequence_length as i64 {
                // Reverse strand
                let beg_f = ((self.packed_sequence_length as i64) << 1) - 1 - end;
                let end_f = ((self.packed_sequence_length as i64) << 1) - 1 - beg;
                for k in (beg_f + 1..=end_f).rev() {
                    seq.push(3 - Self::get_pac(pac, k));
                }
            } else {
                // Forward strand
                for k in beg..end {
                    seq.push(Self::get_pac(pac, k));
                }
            }
            seq
        } else {
            // Bridging forward-reverse boundary, return empty
            Vec::new()
        }
    }

    /// Fetch reference sequence for a region (with reference boundary checking)
    /// Equivalent to C's bns_fetch_seq
    /// Returns (sequence, adjusted_beg, adjusted_end, rid)
    pub fn bns_fetch_seq(
        &self,
        pac: &[u8],
        mut beg: i64,
        mid: i64,
        mut end: i64,
    ) -> (Vec<u8>, i64, i64, i32) {
        // Swap if end < beg
        if end < beg {
            std::mem::swap(&mut beg, &mut end);
        }

        // Get rid and strand info
        let (pos_f, is_rev) = self.bns_depos(mid);
        let rid = self.bns_pos2rid(pos_f);

        if rid < 0 {
            return (Vec::new(), beg, end, rid);
        }

        // Get reference boundaries
        let mut far_beg = self.annotations[rid as usize].offset as i64;
        let mut far_end = far_beg + self.annotations[rid as usize].sequence_length as i64;

        if is_rev {
            // Flip to reverse strand
            let tmp = far_beg;
            far_beg = ((self.packed_sequence_length as i64) << 1) - far_end;
            far_end = ((self.packed_sequence_length as i64) << 1) - tmp;
        }

        // Clamp to reference boundaries
        beg = beg.max(far_beg);
        end = end.min(far_end);

        // Get sequence
        let seq = self.bns_get_seq(pac, beg, end);

        (seq, beg, end, rid)
    }
}
