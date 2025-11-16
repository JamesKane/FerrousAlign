// Prototype test for bio::io::fastq migration
// This validates the API works for our batch reading pattern

use bio::io::fastq;
use flate2::read::GzDecoder;
use std::fs::File;
use std::io::Write;
use tempfile::NamedTempFile;

#[test]
fn test_basic_fastq_reading() {
    // Test with real test data
    let reader = fastq::Reader::from_file("test_data/paired_end/read1.fq");

    if reader.is_err() {
        println!("Test data not found, skipping test");
        return;
    }

    let reader = reader.unwrap();
    let records: Vec<_> = reader.records().collect::<Result<Vec<_>, _>>().unwrap();

    assert_eq!(records.len(), 1);

    // Check Record API
    let record = &records[0];
    let id = record.id(); // Returns &str
    let seq = record.seq(); // Returns &[u8]
    let qual = record.qual(); // Returns &[u8]

    println!("Read ID: {}", id);
    println!("Sequence length: {}", seq.len());
    println!("Quality length: {}", qual.len());

    assert_eq!(id, "read1/1");
    assert_eq!(seq.len(), 50);
    assert_eq!(qual.len(), 50);
    assert_eq!(seq, b"ACGTTAGCGATCGATAGCTGCATGCTAGCGATCGATCGATAGCTGATCGA");
}

#[test]
fn test_batch_reading_pattern() {
    // Test our batch reading pattern (512 reads at a time)
    let mut file = NamedTempFile::new().unwrap();

    // Write 1000 reads
    for i in 0..1000 {
        writeln!(file, "@read{}", i).unwrap();
        writeln!(file, "ACGT").unwrap();
        writeln!(file, "+").unwrap();
        writeln!(file, "IIII").unwrap();
    }
    file.flush().unwrap();

    let reader = fastq::Reader::from_file(file.path()).unwrap();

    // Read in batches of 512
    const BATCH_SIZE: usize = 512;
    let mut total_reads = 0;
    let mut batch_count = 0;

    let mut records_iter = reader.records();

    loop {
        let batch: Vec<_> = records_iter
            .by_ref()
            .take(BATCH_SIZE)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        if batch.is_empty() {
            break;
        }

        total_reads += batch.len();
        batch_count += 1;

        println!("Batch {}: {} reads", batch_count, batch.len());

        // Verify records have correct structure
        for record in &batch {
            assert_eq!(record.seq(), b"ACGT");
            assert_eq!(record.qual(), b"IIII");
        }
    }

    assert_eq!(total_reads, 1000);
    assert_eq!(batch_count, 2); // 512 + 488
}

#[test]
fn test_gzip_support() {
    use flate2::Compression;
    use flate2::write::GzEncoder;
    use std::io::Write;

    // Create a gzipped FASTQ file
    let mut temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_str().unwrap().to_string() + ".gz";

    {
        let file = File::create(&temp_path).unwrap();
        let mut encoder = GzEncoder::new(file, Compression::default());

        for i in 0..10 {
            writeln!(encoder, "@read{}", i).unwrap();
            writeln!(encoder, "ACGTACGT").unwrap();
            writeln!(encoder, "+").unwrap();
            writeln!(encoder, "IIIIIIII").unwrap();
        }

        encoder.finish().unwrap();
    }

    // Read gzipped file
    let file = File::open(&temp_path).unwrap();
    let decoder = GzDecoder::new(file);
    let reader = fastq::Reader::new(decoder);

    let records: Vec<_> = reader.records().collect::<Result<Vec<_>, _>>().unwrap();

    assert_eq!(records.len(), 10);

    for (i, record) in records.iter().enumerate() {
        let expected_id = format!("read{}", i);
        assert_eq!(record.id(), expected_id.as_str());
        assert_eq!(record.seq(), b"ACGTACGT");
    }

    // Cleanup
    std::fs::remove_file(&temp_path).unwrap();

    println!("✓ Gzip support works!");
}

#[test]
fn test_empty_file_handling() {
    let file = NamedTempFile::new().unwrap();

    let reader = fastq::Reader::from_file(file.path()).unwrap();
    let records: Vec<_> = reader.records().collect::<Result<Vec<_>, _>>().unwrap();

    assert_eq!(records.len(), 0);
    println!("✓ Empty file handling works!");
}

#[test]
fn test_quality_score_handling() {
    let mut file = NamedTempFile::new().unwrap();

    // Write FASTQ with various quality scores
    writeln!(file, "@read1").unwrap();
    writeln!(file, "ACGT").unwrap();
    writeln!(file, "+").unwrap();
    writeln!(file, "!#$%").unwrap(); // Quality scores: 0, 2, 3, 4 (Phred+33)

    file.flush().unwrap();

    let reader = fastq::Reader::from_file(file.path()).unwrap();
    let records: Vec<_> = reader.records().collect::<Result<Vec<_>, _>>().unwrap();

    assert_eq!(records.len(), 1);

    let qual = records[0].qual();
    assert_eq!(qual, b"!#$%");

    // Verify quality is returned as bytes (same as our kseq implementation)
    println!("Quality bytes: {:?}", qual);
    println!("✓ Quality score handling works!");
}
