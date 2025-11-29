// bwa-mem2-rust/src/utils_test.rs

#[cfg(test)]
mod tests {
    use crate::utils::*;
    use flate2::write::GzEncoder;
    // Import all from utils
    use flate2::Compression;
    use std::fs;
    use std::io::{self, Write};
    // Add Write trait
    use std::path::{Path, PathBuf};

    // Helper for creating temporary files
    fn create_temp_file(dir: &Path, name: &str, content: &[u8]) -> io::Result<PathBuf> {
        let path = dir.join(name);
        fs::write(&path, content)?;
        Ok(path)
    }

    // --- Pair64 Tests ---

    #[test]
    fn test_pair64_default() {
        let p = Pair64::default();
        assert_eq!(p.x, 0);
        assert_eq!(p.y, 0);
    }

    #[test]
    fn test_pair64_equality() {
        let p1 = Pair64 { x: 1, y: 2 };
        let p2 = Pair64 { x: 1, y: 2 };
        let p3 = Pair64 { x: 2, y: 1 };
        assert_eq!(p1, p2);
        assert_ne!(p1, p3);
    }

    #[test]
    fn test_pair64_ordering() {
        let p1 = Pair64 { x: 1, y: 2 };
        let p2 = Pair64 { x: 1, y: 3 };
        let p3 = Pair64 { x: 2, y: 1 };
        assert!(p1 < p2);
        assert!(p1 < p3);
        assert!(p2 < p3);
    }

    // --- hash_64 Tests ---

    #[test]
    fn test_hash_64_consistency() {
        let key = 1234567890u64;
        let hash1 = hash_64(key);
        let hash2 = hash_64(key);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_hash_64_different_inputs() {
        let key1 = 123u64;
        let key2 = 456u64;
        assert_ne!(hash_64(key1), hash_64(key2));
    }

    #[test]
    fn test_hash_64_matches_cpp_implementation() {
        // These test vectors are computed from C++ bwa-mem2's hash_64 function
        // in src/utils.h to ensure we match the reference implementation exactly
        //
        // C++ implementation:
        // key += ~(key << 32);
        // key ^= (key >> 22);
        // key += ~(key << 13);
        // key ^= (key >> 8);
        // key += (key << 3);
        // key ^= (key >> 15);
        // key += ~(key << 27);
        // key ^= (key >> 31);

        // Test vector 1: Small value
        assert_eq!(hash_64(0), 7654268697807496793);
        assert_eq!(hash_64(1), 2320827452992767577);
        assert_eq!(hash_64(42), 13135115513944745397);

        // Test vector 2: Typical genomic position hash input (position << 1 | strand)
        assert_eq!(hash_64(1000), 7418194391579287136);
        assert_eq!(hash_64(1000000), 3067516704660993436);

        // Test vector 3: Large values
        assert_eq!(hash_64(1234567890), 17337010322514605492);
        assert_eq!(hash_64(u64::MAX), 11347797999152016406);

        // Test vector 4: Powers of 2 (edge cases)
        assert_eq!(hash_64(1 << 10), 8099353856718621187);
        assert_eq!(hash_64(1 << 20), 518649583398939743);
        assert_eq!(hash_64(1 << 30), 2687339549123861914);
    }

    #[test]
    fn test_hash_64_zero_and_max() {
        // Special edge cases
        let hash_zero = hash_64(0);
        let hash_max = hash_64(u64::MAX);

        // These should be different
        assert_ne!(hash_zero, hash_max);

        // Hashing the hash should give a different value (avalanche property)
        assert_ne!(hash_64(hash_zero), hash_zero);
    }

    #[test]
    fn test_hash_64_avalanche() {
        // Small changes in input should cause large changes in output (avalanche effect)
        let h1 = hash_64(0x0000000000000000);
        let h2 = hash_64(0x0000000000000001);

        // Count bit differences
        let diff = (h1 ^ h2).count_ones();

        // Should have significant bit differences (typically > 20 for a good hash)
        assert!(
            diff > 15,
            "Poor avalanche: only {diff} bits differ between hash_64(0) and hash_64(1)"
        );
    }

    // --- realtime() Tests ---

    #[test]
    fn test_realtime_increases() {
        let t1 = realtime();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let t2 = realtime();
        assert!(t2 > t1);
    }

    // --- cputime() Tests ---
    // Note: cputime() is hard to test precisely due to OS scheduling and small time scales.
    // We'll just check if it returns a non-negative value and increases.
    #[test]
    fn test_cputime_non_negative() {
        let t = cputime();
        assert!(t >= 0.0);
    }

    #[test]
    fn test_cputime_increases() {
        let t1 = cputime();
        // Perform some CPU-bound work
        let mut sum: u64 = 0;
        for i in 0..1_000_000 {
            sum += i;
        }
        let t2 = cputime();
        // It's possible t2 == t1 on very fast systems or if work is optimized away,
        // but it should generally increase.
        // For robustness, we assert it's not less than t1.
        assert!(t2 >= t1);
        // A more robust test would involve significant, unavoidable CPU work.
    }

    // --- xopen Tests ---

    #[test]
    fn test_xopen_file() -> io::Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let file_path = create_temp_file(temp_dir.path(), "test.txt", b"hello world")?;

        let mut reader = xopen(&file_path, "r")?;
        let mut content = String::new();
        reader.read_to_string(&mut content)?;
        assert_eq!(content, "hello world");

        Ok(())
    }

    #[test]
    fn test_xopen_non_existent_file() {
        let non_existent_path = PathBuf::from("non_existent_file.txt");
        let result = xopen(&non_existent_path, "r");
        assert!(result.is_err());
        if let Err(e) = result {
            assert_eq!(e.kind(), io::ErrorKind::NotFound);
        } else {
            panic!("Expected an error, but got Ok");
        }
    }

    // --- xzopen Tests ---

    #[test]
    fn test_xzopen_gz_file() -> io::Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let gz_file_path = temp_dir.path().join("test.txt.gz");

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(b"gzipped content")?;
        let compressed_bytes = encoder.finish()?;
        fs::write(&gz_file_path, compressed_bytes)?;

        let mut reader = xzopen(&gz_file_path, "r")?;
        let mut content = String::new();
        reader.read_to_string(&mut content)?;
        assert_eq!(content, "gzipped content");

        Ok(())
    }

    #[test]
    fn test_xzopen_non_gz_file() -> io::Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let file_path = create_temp_file(temp_dir.path(), "test.txt", b"plain content")?;

        let mut reader = xzopen(&file_path, "r")?;
        let mut content = String::new();
        reader.read_to_string(&mut content)?;
        assert_eq!(content, "plain content");

        Ok(())
    }

    // --- Binary I/O Tests ---

    #[test]
    fn test_binary_write_u64_le() {
        use crate::utils::BinaryWrite;
        use std::io::Cursor;

        let mut buffer = Cursor::new(Vec::new());
        buffer.write_u64_le(0x0102030405060708u64).unwrap();

        let bytes = buffer.into_inner();
        // Little-endian: least significant byte first
        assert_eq!(bytes, vec![0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01]);
    }

    #[test]
    fn test_binary_write_i64_le() {
        use crate::utils::BinaryWrite;
        use std::io::Cursor;

        let mut buffer = Cursor::new(Vec::new());
        buffer.write_i64_le(-1i64).unwrap();

        let bytes = buffer.into_inner();
        // -1 in two's complement is all 0xFF
        assert_eq!(bytes, vec![0xFF; 8]);
    }

    #[test]
    fn test_binary_write_u32_le() {
        use crate::utils::BinaryWrite;
        use std::io::Cursor;

        let mut buffer = Cursor::new(Vec::new());
        buffer.write_u32_le(0x12345678u32).unwrap();

        let bytes = buffer.into_inner();
        assert_eq!(bytes, vec![0x78, 0x56, 0x34, 0x12]);
    }

    #[test]
    fn test_binary_write_u64_array_le() {
        use crate::utils::BinaryWrite;
        use std::io::Cursor;

        let mut buffer = Cursor::new(Vec::new());
        let values = vec![1u64, 2u64, 3u64];
        buffer.write_u64_array_le(&values).unwrap();

        let bytes = buffer.into_inner();
        // Each u64 is 8 bytes, total 24 bytes
        assert_eq!(bytes.len(), 24);

        // First u64 (1): [1, 0, 0, 0, 0, 0, 0, 0]
        assert_eq!(&bytes[0..8], &[1, 0, 0, 0, 0, 0, 0, 0]);
        // Second u64 (2): [2, 0, 0, 0, 0, 0, 0, 0]
        assert_eq!(&bytes[8..16], &[2, 0, 0, 0, 0, 0, 0, 0]);
        // Third u64 (3): [3, 0, 0, 0, 0, 0, 0, 0]
        assert_eq!(&bytes[16..24], &[3, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_binary_write_i64_array_le() {
        use crate::utils::BinaryWrite;
        use std::io::Cursor;

        let mut buffer = Cursor::new(Vec::new());
        let values = vec![100i64, -100i64];
        buffer.write_i64_array_le(&values).unwrap();

        let bytes = buffer.into_inner();
        assert_eq!(bytes.len(), 16);
    }

    #[test]
    fn test_binary_write_chaining() {
        use crate::utils::BinaryWrite;
        use std::io::Cursor;

        let mut buffer = Cursor::new(Vec::new());

        // Test chaining multiple writes
        buffer.write_u32_le(0xDEADBEEF).unwrap();
        buffer.write_u64_le(0x123456789ABCDEF0).unwrap();
        buffer.write_u32_le(0xCAFEBABE).unwrap();

        let bytes = buffer.into_inner();
        assert_eq!(bytes.len(), 4 + 8 + 4); // 16 bytes total
    }

    // --- Sorting Tests ---

    #[test]
    fn test_ks_introsort_64() {
        let mut arr = [5, 2, 8, 1, 9, 4];
        ks_introsort_64(&mut arr);
        assert_eq!(arr, [1, 2, 4, 5, 8, 9]);
    }

    #[test]
    fn test_ks_introsort_128() {
        let mut arr = [
            Pair64 { x: 5, y: 0 },
            Pair64 { x: 2, y: 0 },
            Pair64 { x: 8, y: 0 },
            Pair64 { x: 1, y: 0 },
            Pair64 { x: 9, y: 0 },
            Pair64 { x: 4, y: 0 },
        ];
        ks_introsort_128(&mut arr);
        assert_eq!(
            arr,
            [
                Pair64 { x: 1, y: 0 },
                Pair64 { x: 2, y: 0 },
                Pair64 { x: 4, y: 0 },
                Pair64 { x: 5, y: 0 },
                Pair64 { x: 8, y: 0 },
                Pair64 { x: 9, y: 0 },
            ]
        );

        let mut arr_with_y = [
            Pair64 { x: 1, y: 5 },
            Pair64 { x: 1, y: 2 },
            Pair64 { x: 2, y: 1 },
        ];
        ks_introsort_128(&mut arr_with_y);
        assert_eq!(
            arr_with_y,
            [
                Pair64 { x: 1, y: 2 },
                Pair64 { x: 1, y: 5 },
                Pair64 { x: 2, y: 1 },
            ]
        );
    }
}
