#[cfg(test)]
mod tests {
    use std::io::Write;
    use tempfile::NamedTempFile;
    use crate::io::fastq_reader::FastqReader;

    #[test]
    fn test_read_batch() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "@read1").unwrap();
        writeln!(file, "ACGT").unwrap();
        writeln!(file, "+").unwrap();
        writeln!(file, "IIII").unwrap();
        writeln!(file, "@read2").unwrap();
        writeln!(file, "TGCA").unwrap();
        writeln!(file, "+").unwrap();
        writeln!(file, "JJJJ").unwrap();
        file.flush().unwrap();

        let mut reader = FastqReader::new(file.path().to_str().unwrap()).unwrap();
        let batch = reader.read_batch(10).unwrap();

        assert_eq!(batch.len(), 2);
        assert_eq!(batch.names[0], "read1");
        assert_eq!(batch.names[1], "read2");
        assert_eq!(batch.seqs[0], b"ACGT");
        assert_eq!(batch.seqs[1], b"TGCA");
        assert_eq!(batch.quals[0], "IIII");
        assert_eq!(batch.quals[1], "JJJJ");
    }

    #[test]
    fn test_batch_size_limit() {
        let mut file = NamedTempFile::new().unwrap();
        for i in 0..100 {
            writeln!(file, "@read{}", i).unwrap();
            writeln!(file, "ACGT").unwrap();
            writeln!(file, "+").unwrap();
            writeln!(file, "IIII").unwrap();
        }
        file.flush().unwrap();

        let mut reader = FastqReader::new(file.path().to_str().unwrap()).unwrap();

        // Read in batches of 30
        let batch1 = reader.read_batch(30).unwrap();
        assert_eq!(batch1.len(), 30);

        let batch2 = reader.read_batch(30).unwrap();
        assert_eq!(batch2.len(), 30);

        let batch3 = reader.read_batch(30).unwrap();
        assert_eq!(batch3.len(), 30);

        // Last batch has remaining 10
        let batch4 = reader.read_batch(30).unwrap();
        assert_eq!(batch4.len(), 10);

        // No more reads
        let batch5 = reader.read_batch(30).unwrap();
        assert_eq!(batch5.len(), 0);
        assert!(batch5.is_empty());
    }

    #[test]
    fn test_empty_file() {
        let file = NamedTempFile::new().unwrap();

        let mut reader = FastqReader::new(file.path().to_str().unwrap()).unwrap();
        let batch = reader.read_batch(10).unwrap();

        assert_eq!(batch.len(), 0);
        assert!(batch.is_empty());
    }

    #[test]
    fn test_gzip_detection() {
        use flate2::Compression;
        use flate2::write::GzEncoder;
        use std::fs::File;

        // Create a gzipped FASTQ file
        let temp_dir = tempfile::tempdir().unwrap();
        let gz_path = temp_dir.path().join("test.fq.gz");

        {
            let file = File::create(&gz_path).unwrap();
            let mut encoder = GzEncoder::new(file, Compression::default());

            writeln!(encoder, "@read1").unwrap();
            writeln!(encoder, "ACGTACGT").unwrap();
            writeln!(encoder, "+").unwrap();
            writeln!(encoder, "IIIIIIII").unwrap();
            writeln!(encoder, "@read2").unwrap();
            writeln!(encoder, "TGCATGCA").unwrap();
            writeln!(encoder, "+").unwrap();
            writeln!(encoder, "JJJJJJJJ").unwrap();

            encoder.finish().unwrap();
        }

        // Read gzipped file
        let mut reader = FastqReader::new(gz_path.to_str().unwrap()).unwrap();
        let batch = reader.read_batch(10).unwrap();

        assert_eq!(batch.len(), 2);
        assert_eq!(batch.names[0], "read1");
        assert_eq!(batch.seqs[0], b"ACGTACGT");
        assert_eq!(batch.seqs[1], b"TGCATGCA");
    }

    #[test]
    fn test_quality_scores() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "@read1").unwrap();
        writeln!(file, "ACGT").unwrap();
        writeln!(file, "+").unwrap();
        writeln!(file, "!#$%").unwrap(); // Quality scores: 0, 2, 3, 4 (Phred+33)
        file.flush().unwrap();

        let mut reader = FastqReader::new(file.path().to_str().unwrap()).unwrap();
        let batch = reader.read_batch(10).unwrap();

        assert_eq!(batch.len(), 1);
        assert_eq!(batch.quals[0], "!#$%");
    }

    #[test]
    fn test_real_test_data() {
        // Test with actual test data if it exists
        let path = "test_data/paired_end/read1.fq";
        if !std::path::Path::new(path).exists() {
            println!("Skipping test - test data not found");
            return;
        }

        let mut reader = FastqReader::new(path).unwrap();
        let batch = reader.read_batch(10).unwrap();

        assert_eq!(batch.len(), 1);
        assert_eq!(batch.names[0], "read1/1");
        assert_eq!(batch.seqs[0].len(), 50);
        assert_eq!(batch.quals[0].len(), 50);
    }
}
