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

#[derive(Debug)]
pub struct BntAnn1 {
    pub offset: u64,
    pub len: i32,
    pub n_ambs: i32,
    pub gi: u32,
    pub is_alt: i32,
    pub name: String,
    pub anno: String,
}

#[derive(Debug)]
pub struct BntAmb1 {
    pub offset: u64,
    pub len: i32,
    pub amb: char,
}

#[derive(Debug)]
pub struct BntSeq {
    pub l_pac: u64,
    pub n_seqs: i32,
    pub seed: u32,
    pub anns: Vec<BntAnn1>,
    pub n_holes: i32,
    pub ambs: Vec<BntAmb1>,
    pub pac_file_path: Option<PathBuf>,
}

#[path = "bntseq_test.rs"]
mod bntseq_test;

impl BntSeq {
    pub fn new() -> Self {
        BntSeq {
            l_pac: 0,
            n_seqs: 0,
            seed: 0,
            anns: Vec::new(),
            n_holes: 0,
            ambs: Vec::new(),
            pac_file_path: None,
        }
    }

    pub fn bns_fasta2bntseq<R: Read + 'static>(
        reader: R,
        prefix: &Path,
        _for_only: bool, // for_only is not used in the C version's bns_fasta2bntseq
    ) -> io::Result<Self> {
        let mut bns = BntSeq::new();
        let mut kseq = crate::kseq::KSeq::new(Box::new(reader));
        let mut pac_data: Vec<u8> = Vec::new();

        let mut seq_id = 0;
        let mut pac_len = 0; // This will now track the total length including ambiguous bases
        let mut packed_base_count = 0; // This will track only non-ambiguous bases

        let mut current_ambs: Vec<BntAmb1> = Vec::new();

        loop {
            let len = kseq.read()?; // Reads one sequence
            if len < 0 {
                break;
            } // EOF or error

            let mut ann = BntAnn1 {
                // Create a new BntAnn1 for each sequence
                offset: packed_base_count,
                len: kseq.seq.len() as i32,
                n_ambs: 0, // Will be updated later
                gi: 0,
                is_alt: 0,
                name: kseq.name.clone(),
                anno: kseq.comment.clone(),
            };

            let mut n_ambs_in_seq = 0;
            for base in kseq.seq.bytes() {
                let nt4 = NST_NT4_TABLE[base as usize];
                if nt4 < 4 {
                    // Regular base
                    let last_byte_idx = (packed_base_count / 4) as usize;
                    if pac_data.len() <= last_byte_idx {
                        pac_data.resize(last_byte_idx + 1, 0);
                    }
                    pac_data[last_byte_idx] |= nt4 << ((packed_base_count % 4) << 1);
                    packed_base_count += 1;
                    pac_len += 1; // Still increment total pac_len for ambiguous base tracking
                } else {
                    // Ambiguous base
                    if current_ambs.is_empty()
                        || current_ambs.last().unwrap().offset
                            + current_ambs.last().unwrap().len as u64
                            != pac_len
                    {
                        current_ambs.push(BntAmb1 {
                            offset: pac_len,
                            len: 1,
                            amb: base as char,
                        });
                    } else {
                        current_ambs.last_mut().unwrap().len += 1;
                    }
                    pac_len += 1;
                    n_ambs_in_seq += 1;
                }
            }
            ann.n_ambs = n_ambs_in_seq; // Update n_ambs for the current annotation

            bns.anns.push(ann); // Push the completed annotation
            seq_id += 1;
        }

        bns.l_pac = packed_base_count; // Assign the count of non-ambiguous bases to l_pac
        bns.n_seqs = seq_id;
        bns.ambs = current_ambs;
        bns.n_holes = bns.ambs.len() as i32;
        bns.pac_file_path = Some(PathBuf::from(prefix.to_string_lossy().to_string() + ".pac"));

        let mut pac_file = File::create(prefix.with_extension("pac"))?;
        pac_file.write_all(&pac_data)?;

        Ok(bns)
    }

    pub fn bns_dump(&self, prefix: &Path) -> io::Result<()> {
        // Dump annotations
        let ann_file_path = prefix.with_extension("ann");
        let mut ann_file = BufWriter::new(File::create(&ann_file_path)?);
        writeln!(ann_file, "{} {} {}", self.l_pac, self.n_seqs, self.seed)?;
        for p in &self.anns {
            if p.anno.is_empty() || p.anno == "(null)" {
                writeln!(ann_file, "{} {}", p.gi, p.name)?;
            } else {
                writeln!(ann_file, "{} {} {}", p.gi, p.name, p.anno)?;
            }
            writeln!(ann_file, "{} {} {}", p.offset, p.len, p.n_ambs)?;
        }
        ann_file.flush()?;

        // Dump ambiguous bases
        let amb_file_path = prefix.with_extension("amb");
        let mut amb_file = BufWriter::new(File::create(&amb_file_path)?);
        writeln!(amb_file, "{} {} {}", self.l_pac, self.n_seqs, self.n_holes)?;
        for p in &self.ambs {
            writeln!(amb_file, "{} {} {}", p.offset, p.len, p.amb)?;
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
        bns.l_pac = parts[0]
            .parse()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid l_pac in .ann"))?;
        bns.n_seqs = parts[1]
            .parse()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid n_seqs in .ann"))?;
        bns.seed = parts[2]
            .parse()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid seed in .ann"))?;

        bns.anns.reserve(bns.n_seqs as usize);
        for _ in 0..bns.n_seqs {
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
            let len = offset_len_ambs_parts[1]
                .parse()
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid len in .ann"))?;
            let n_ambs = offset_len_ambs_parts[2].parse().map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidData, "Invalid n_ambs in .ann")
            })?;

            bns.anns.push(BntAnn1 {
                offset,
                len,
                n_ambs,
                gi,
                is_alt: 0, // This is not stored in .ann, default to 0
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
        bns.n_holes = amb_parts[2]
            .parse()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid n_holes in .amb"))?;

        bns.ambs.reserve(bns.n_holes as usize);
        for _ in 0..bns.n_holes {
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

            bns.ambs.push(BntAmb1 { offset, len, amb });
        }

        bns.pac_file_path = Some(PathBuf::from(prefix.to_string_lossy().to_string() + ".pac"));
        // eprintln!("PAC file path set to: {:?}", bns.pac_file_path);

        Ok(bns)
    }

    // New method to get a segment of the reference sequence
    pub fn get_reference_segment(&self, start: u64, len: u64) -> io::Result<Vec<u8>> {
        let pac_file_path = self
            .pac_file_path
            .as_ref()
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "PAC file path not set"))?;

        let mut pac_file = BufReader::new(File::open(pac_file_path)?);

        let mut segment = Vec::with_capacity(len as usize);

        let start_byte_offset = start / 4;
        let end_byte_offset = (start + len - 1) / 4;
        let bytes_to_read = (end_byte_offset - start_byte_offset + 1) as usize;

        let mut pac_bytes = vec![0u8; bytes_to_read];
        pac_file.seek(SeekFrom::Start(start_byte_offset as u64))?;
        pac_file.read_exact(&mut pac_bytes)?;

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

        Ok(segment)
    }

    /// Helper function: Convert position to forward strand position and determine if reverse
    /// Equivalent to C's bns_depos
    pub fn bns_depos(&self, pos: i64) -> (i64, bool) {
        let is_rev = pos >= self.l_pac as i64;
        let pos_f = if is_rev {
            ((self.l_pac as i64) << 1) - 1 - pos
        } else {
            pos
        };
        (pos_f, is_rev)
    }

    /// Helper function: Find which reference sequence contains the given forward position
    /// Equivalent to C's bns_pos2rid
    pub fn bns_pos2rid(&self, pos_f: i64) -> i32 {
        if pos_f >= self.l_pac as i64 {
            return -1;
        }

        let mut left = 0;
        let mut right = self.n_seqs as usize;
        let mut mid = 0;

        while left < right {
            mid = (left + right) >> 1;
            if pos_f >= self.anns[mid].offset as i64 {
                if mid == self.n_seqs as usize - 1 {
                    break;
                }
                if pos_f < self.anns[mid + 1].offset as i64 {
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
        if end > (self.l_pac as i64) << 1 {
            end = (self.l_pac as i64) << 1;
        }
        if beg < 0 {
            beg = 0;
        }

        // Check if bridging forward-reverse boundary
        if beg >= self.l_pac as i64 || end <= self.l_pac as i64 {
            let len = (end - beg) as usize;
            let mut seq = Vec::with_capacity(len);

            if beg >= self.l_pac as i64 {
                // Reverse strand
                let beg_f = ((self.l_pac as i64) << 1) - 1 - end;
                let end_f = ((self.l_pac as i64) << 1) - 1 - beg;
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
        let mut far_beg = self.anns[rid as usize].offset as i64;
        let mut far_end = far_beg + self.anns[rid as usize].len as i64;

        if is_rev {
            // Flip to reverse strand
            let tmp = far_beg;
            far_beg = ((self.l_pac as i64) << 1) - far_end;
            far_end = ((self.l_pac as i64) << 1) - tmp;
        }

        // Clamp to reference boundaries
        beg = beg.max(far_beg);
        end = end.min(far_end);

        // Get sequence
        let seq = self.bns_get_seq(pac, beg, end);

        (seq, beg, end, rid)
    }
}
