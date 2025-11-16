// bwa-mem2-rust/src/kseq_test.rs

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kseq::{KSEQ_BUF_SIZE, KSeq, KStream};
    use std::io::{self, Cursor};

    // --- KStream Tests ---

    #[test]
    fn test_kstream_new() {
        let reader = Cursor::new(b"");
        let ks = KStream::new(Box::new(reader));
        assert_eq!(ks.begin, 0);
        assert_eq!(ks.end, 0);
        assert!(!ks.is_eof);
        assert_eq!(ks.buf.len(), KSEQ_BUF_SIZE);
    }

    #[test]
    fn test_kstream_getc_empty() -> io::Result<()> {
        let reader = Cursor::new(b"");
        let mut ks = KStream::new(Box::new(reader));
        assert_eq!(ks.getc()?, None);
        assert!(ks.is_eof);
        Ok(())
    }

    #[test]
    fn test_kstream_getc_single_char() -> io::Result<()> {
        let reader = Cursor::new(b"A");
        let mut ks = KStream::new(Box::new(reader));
        assert_eq!(ks.getc()?, Some(b'A'));
        assert_eq!(ks.getc()?, None);
        Ok(())
    }

    #[test]
    fn test_kstream_getc_multiple_chars() -> io::Result<()> {
        let reader = Cursor::new(b"ABC");
        let mut ks = KStream::new(Box::new(reader));
        assert_eq!(ks.getc()?, Some(b'A'));
        assert_eq!(ks.getc()?, Some(b'B'));
        assert_eq!(ks.getc()?, Some(b'C'));
        assert_eq!(ks.getc()?, None);
        Ok(())
    }

    #[test]
    fn test_kstream_getuntil_empty() -> io::Result<()> {
        let reader = Cursor::new(b"");
        let mut ks = KStream::new(Box::new(reader));
        let mut s = String::new();
        assert_eq!(ks.getuntil(b'\n' as i32, &mut s, false)?, None);
        assert!(s.is_empty());
        Ok(())
    }

    #[test]
    fn test_kstream_getuntil_newline() -> io::Result<()> {
        let reader = Cursor::new(b"hello\nworld");
        let mut ks = KStream::new(Box::new(reader));
        let mut s = String::new();
        assert_eq!(ks.getuntil(b'\n' as i32, &mut s, false)?, Some(b'\n'));
        assert_eq!(s, "hello");
        assert_eq!(ks.getc()?, Some(b'w')); // Check if it moved past newline
        Ok(())
    }

    #[test]
    fn test_kstream_getuntil_space() -> io::Result<()> {
        let reader = Cursor::new(b"hello world\n");
        let mut ks = KStream::new(Box::new(reader));
        let mut s = String::new();
        assert_eq!(ks.getuntil(0, &mut s, false)?, Some(b' ')); // 0 for KS_SEP_SPACE
        assert_eq!(s, "hello");
        assert_eq!(ks.getc()?, Some(b'w')); // Check if it moved past space
        Ok(())
    }

    #[test]
    fn test_kstream_getuntil_append() -> io::Result<()> {
        let reader = Cursor::new(b"part1\npart2\n");
        let mut ks = KStream::new(Box::new(reader));
        let mut s = String::from("initial");
        assert_eq!(ks.getuntil(b'\n' as i32, &mut s, true)?, Some(b'\n'));
        assert_eq!(s, "initialpart1");
        assert_eq!(ks.getc()?, Some(b'p'));
        Ok(())
    }

    #[test]
    fn test_kstream_getuntil_no_delimiter() -> io::Result<()> {
        let reader = Cursor::new(b"no delimiter here");
        let mut ks = KStream::new(Box::new(reader));
        let mut s = String::new();
        assert_eq!(ks.getuntil(b'\n' as i32, &mut s, false)?, None);
        assert_eq!(s, "no delimiter here");
        assert!(ks.is_eof);
        Ok(())
    }

    // --- KSeq Tests ---

    #[test]
    fn test_kseq_new() {
        let reader = Cursor::new(b"");
        let ks = KSeq::new(Box::new(reader));
        assert!(ks.name.is_empty());
        assert!(ks.comment.is_empty());
        assert!(ks.seq.is_empty());
        assert!(ks.qual.is_empty());
        assert_eq!(ks.last_char, 0);
    }

    #[test]
    fn test_kseq_read_fasta_simple() -> io::Result<()> {
        let fasta_content = b">seq1\nAGCT\n";
        let reader = Cursor::new(fasta_content);
        let mut kseq = KSeq::new(Box::new(reader));

        let len = kseq.read()?;
        assert_eq!(len, 4);
        assert_eq!(kseq.name, "seq1");
        assert!(kseq.comment.is_empty());
        assert_eq!(kseq.seq, "AGCT");
        assert!(kseq.qual.is_empty());
        assert_eq!(kseq.last_char, 0); // Corrected assertion

        let len = kseq.read()?;
        assert_eq!(len, -1); // EOF
        Ok(())
    }

    #[test]
    fn test_kseq_read_fasta_with_comment() -> io::Result<()> {
        let fasta_content = b">seq2 comment_text\nTGCA\n";
        let reader = Cursor::new(fasta_content);
        let mut kseq = KSeq::new(Box::new(reader));

        let len = kseq.read()?;
        assert_eq!(len, 4);
        assert_eq!(kseq.name, "seq2");
        assert_eq!(kseq.comment, "comment_text");
        assert_eq!(kseq.seq, "TGCA");
        assert!(kseq.qual.is_empty());
        assert_eq!(kseq.last_char, 0); // Corrected assertion

        let len = kseq.read()?;
        assert_eq!(len, -1); // EOF
        Ok(())
    }

    #[test]
    fn test_kseq_read_fasta_multi_line_seq() -> io::Result<()> {
        let fasta_content = b">seq3\nAGC\nT\n";
        let reader = Cursor::new(fasta_content);
        let mut kseq = KSeq::new(Box::new(reader));

        let len = kseq.read()?;
        assert_eq!(len, 4);
        assert_eq!(kseq.name, "seq3");
        assert!(kseq.comment.is_empty());
        assert_eq!(kseq.seq, "AGCT");
        assert!(kseq.qual.is_empty());
        assert_eq!(kseq.last_char, 0); // Corrected assertion

        let len = kseq.read()?;
        assert_eq!(len, -1); // EOF
        Ok(())
    }

    #[test]
    fn test_kseq_read_fasta_multiple_entries() -> io::Result<()> {
        let fasta_content = b">seq1\nAGCT\n>seq2 comment\nTGCA\n";
        let reader = Cursor::new(fasta_content);
        let mut kseq = KSeq::new(Box::new(reader));

        let len = kseq.read()?;
        assert_eq!(len, 4);
        assert_eq!(kseq.name, "seq1");
        assert!(kseq.comment.is_empty());
        assert_eq!(kseq.seq, "AGCT");

        let len = kseq.read()?;
        assert_eq!(len, 4);
        assert_eq!(kseq.name, "seq2");
        assert_eq!(kseq.comment, "comment");
        assert_eq!(kseq.seq, "TGCA");
        assert_eq!(kseq.last_char, 0); // Corrected assertion, as it's EOF after seq2

        let len = kseq.read()?;
        assert_eq!(len, -1); // EOF
        Ok(())
    }

    #[test]
    fn test_kseq_read_fastq_simple() -> io::Result<()> {
        let fastq_content = b"@read1\nAGCT\n+\n!!!!\n";
        let reader = Cursor::new(fastq_content);
        let mut kseq = KSeq::new(Box::new(reader));

        let len = kseq.read()?;
        assert_eq!(len, 4);
        assert_eq!(kseq.name, "read1");
        assert!(kseq.comment.is_empty());
        assert_eq!(kseq.seq, "AGCT");
        assert_eq!(kseq.qual, "!!!!");
        assert_eq!(kseq.last_char, 0); // Reset after FASTQ entry

        let len = kseq.read()?;
        assert_eq!(len, -1); // EOF
        Ok(())
    }

    #[test]
    fn test_kseq_read_fastq_with_comment() -> io::Result<()> {
        let fastq_content = b"@read2 comment_text\nTGCA\n+comment_again\n####\n";
        let reader = Cursor::new(fastq_content);
        let mut kseq = KSeq::new(Box::new(reader));

        let len = kseq.read()?;
        assert_eq!(len, 4);
        assert_eq!(kseq.name, "read2");
        assert_eq!(kseq.comment, "comment_text");
        assert_eq!(kseq.seq, "TGCA");
        assert_eq!(kseq.qual, "####");
        assert_eq!(kseq.last_char, 0);

        let len = kseq.read()?;
        assert_eq!(len, -1); // EOF
        Ok(())
    }

    #[test]
    fn test_kseq_read_fastq_multi_line_seq() -> io::Result<()> {
        let fastq_content = b"@read3\nAGC\nT\n+\n!!!\n!\n";
        let reader = Cursor::new(fastq_content);
        let mut kseq = KSeq::new(Box::new(reader));

        let len = kseq.read()?;
        assert_eq!(len, 4);
        assert_eq!(kseq.name, "read3");
        assert!(kseq.comment.is_empty());
        assert_eq!(kseq.seq, "AGCT");
        assert_eq!(kseq.qual, "!!!!");
        assert_eq!(kseq.last_char, 0);

        let len = kseq.read()?;
        assert_eq!(len, -1); // EOF
        Ok(())
    }

    #[test]
    fn test_kseq_read_fastq_multiple_entries() -> io::Result<()> {
        let fastq_content = b"@read1\nAGCT\n+\n!!!!\n@read2 comment\nTGCA\n+comment_again\n####\n";
        let reader = Cursor::new(fastq_content);
        let mut kseq = KSeq::new(Box::new(reader));

        let len = kseq.read()?;
        assert_eq!(len, 4);
        assert_eq!(kseq.name, "read1");
        assert!(kseq.comment.is_empty());
        assert_eq!(kseq.seq, "AGCT");
        assert_eq!(kseq.qual, "!!!!");

        let len = kseq.read()?;
        assert_eq!(len, 4);
        assert_eq!(kseq.name, "read2");
        assert_eq!(kseq.comment, "comment");
        assert_eq!(kseq.seq, "TGCA");
        assert_eq!(kseq.qual, "####");
        assert_eq!(kseq.last_char, 0); // Corrected assertion, as it's EOF after read2

        let len = kseq.read()?;
        assert_eq!(len, -1); // EOF
        Ok(())
    }

    #[test]
    fn test_kseq_read_fastq_mismatched_qual_len() -> io::Result<()> {
        let fastq_content = b"@read1\nAGCT\n+\n!!!\n"; // Qual len 3, Seq len 4
        let reader = Cursor::new(fastq_content);
        let mut kseq = KSeq::new(Box::new(reader));

        let len = kseq.read()?;
        assert_eq!(len, -2); // Error: qual string is of a different length
        Ok(())
    }

    #[test]
    fn test_kseq_read_empty_file() -> io::Result<()> {
        let reader = Cursor::new(b"");
        let mut kseq = KSeq::new(Box::new(reader));

        let len = kseq.read()?;
        assert_eq!(len, -1); // EOF
        Ok(())
    }
}
