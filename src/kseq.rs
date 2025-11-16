use std::io::{self, Read};

const KSEQ_BUF_SIZE: usize = 16384;

pub struct KStream {
    reader: Box<dyn Read>,
    buf: Vec<u8>,
    begin: usize,
    end: usize,
    is_eof: bool,
}

#[path = "kseq_test.rs"]
mod kseq_test;

impl KStream {
    pub fn new(reader: Box<dyn Read>) -> Self {
        KStream {
            reader,
            buf: vec![0; KSEQ_BUF_SIZE],
            begin: 0,
            end: 0,
            is_eof: false,
        }
    }

    fn fill_buf(&mut self) -> io::Result<usize> {
        if self.is_eof {
            return Ok(0);
        }
        self.begin = 0;
        self.end = self.reader.read(&mut self.buf)?;
        if self.end == 0 {
            self.is_eof = true;
        }
        Ok(self.end)
    }

    pub fn getc(&mut self) -> io::Result<Option<u8>> {
        if self.begin >= self.end {
            if self.fill_buf()? == 0 {
                return Ok(None); // EOF
            }
        }
        let c = self.buf[self.begin];
        self.begin += 1;
        Ok(Some(c))
    }

    pub fn getuntil(
        &mut self,
        delimiter: i32,
        str: &mut String,
        append: bool,
    ) -> io::Result<Option<u8>> {
        let mut got_any = false;
        if !append {
            str.clear();
        }

        let mut delimiter_char: Option<u8> = None;

        loop {
            if self.begin >= self.end {
                if self.fill_buf()? == 0 {
                    break; // EOF
                }
            }

            let mut i = self.begin;
            let mut found_delimiter = false;

            if delimiter == 0 {
                // KS_SEP_SPACE
                while i < self.end && !self.buf[i].is_ascii_whitespace() {
                    i += 1;
                }
                if i < self.end {
                    found_delimiter = true;
                }
            } else if delimiter == 1 {
                // KS_SEP_TAB
                while i < self.end && !(self.buf[i].is_ascii_whitespace() && self.buf[i] != b' ') {
                    i += 1;
                }
                if i < self.end {
                    found_delimiter = true;
                }
            } else if delimiter == 2 {
                // KS_SEP_LINE
                while i < self.end && self.buf[i] != b'\n' {
                    i += 1;
                }
                if i < self.end {
                    found_delimiter = true;
                }
            } else if delimiter > 2 {
                // Custom delimiter
                while i < self.end && self.buf[i] != delimiter as u8 {
                    i += 1;
                }
                if i < self.end {
                    found_delimiter = true;
                }
            } else {
                // Should not happen based on C code's KS_SEP_MAX
                i = 0;
            }

            // Append to string
            if i > self.begin {
                str.push_str(
                    std::str::from_utf8(&self.buf[self.begin..i])
                        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?,
                );
                got_any = true;
            }

            if found_delimiter {
                delimiter_char = Some(self.buf[i]);
                self.begin = i + 1; // Move past the delimiter
                break; // Delimiter found, stop reading
            } else {
                self.begin = i; // Move to the end of buffer, do not consume if no delimiter
            }
        }

        if !got_any && self.is_eof && str.is_empty() {
            return Ok(None); // EOF and nothing read
        }

        // Handle Windows-style newlines if the delimiter was a line feed
        if delimiter == 2 && str.ends_with('\r') {
            str.pop();
        }

        Ok(delimiter_char)
    }
}

pub struct KSeq {
    pub name: String,
    pub comment: String,
    pub seq: String,
    pub qual: String,
    pub last_char: i32, // Using i32 to match C's int for EOF (-1)
    f: KStream,
}

impl KSeq {
    pub fn new(reader: Box<dyn Read>) -> Self {
        KSeq {
            name: String::new(),
            comment: String::new(),
            seq: String::new(),
            qual: String::new(),
            last_char: 0,
            f: KStream::new(reader),
        }
    }

    pub fn read(&mut self) -> io::Result<i64> {
        let mut c: Option<u8>;

        if self.last_char == 0 {
            // then jump to the next header line
            loop {
                c = self.f.getc()?;
                if c.is_none() {
                    return Ok(-1);
                } // EOF
                if c == Some(b'>') || c == Some(b'@') {
                    break;
                }
            }
            self.last_char = c.unwrap() as i32;
        }

        self.comment.clear();
        self.seq.clear();
        self.qual.clear();

        // Read name
        let name_delimiter = self.f.getuntil(0, &mut self.name, false)?; // 0 for KS_SEP_SPACE
        if name_delimiter.is_none() {
            return Ok(-1);
        } // EOF

        // Read comment if present
        if name_delimiter.unwrap() != b'\n' {
            // If name was not terminated by newline, there's a comment
            let comment_delimiter = self.f.getuntil(2, &mut self.comment, false)?; // 2 for KS_SEP_LINE
            if comment_delimiter.is_none() {
                return Ok(-1);
            } // EOF
        }

        // Read sequence
        loop {
            c = self.f.getc()?;
            if c.is_none() {
                break;
            } // EOF
            if c == Some(b'>') || c == Some(b'+') || c == Some(b'@') {
                break;
            }
            if c == Some(b'\n') {
                continue;
            } // skip empty lines
            self.seq.push(c.unwrap() as char);
            self.f.getuntil(2, &mut self.seq, true)?; // 2 for KS_SEP_LINE, append
        }

        if c.is_none() {
            // If EOF was reached during sequence reading
            self.last_char = 0; // Reset last_char to indicate EOF
        } else if c.is_some() && (c == Some(b'>') || c == Some(b'@')) {
            self.last_char = c.unwrap() as i32;
        }

        if c == Some(b'+') {
            // FASTQ format
            // Skip the rest of '+' line
            loop {
                c = self.f.getc()?;
                if c.is_none() || c == Some(b'\n') {
                    break;
                }
            }
            if c.is_none() {
                return Ok(-2);
            } // Error: no quality string

            // Read quality string
            loop {
                let res = self.f.getuntil(2, &mut self.qual, true)?; // 2 for KS_SEP_LINE, append
                if res.is_none() {
                    break;
                }
                if self.qual.len() >= self.seq.len() {
                    break;
                }
            }
            self.last_char = 0; // We have not come to the next header line
            if self.seq.len() != self.qual.len() {
                return Ok(-2);
            } // Error: qual string is of a different length
        }

        Ok(self.seq.len() as i64)
    }
}
