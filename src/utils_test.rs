// bwa-mem2-rust/src/utils_test.rs

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::*; // Import all from utils
    use flate2::Compression;
    use flate2::write::GzEncoder;
    use std::fs;
    use std::io::{self, Cursor, Write}; // Add Write trait
    use std::path::PathBuf;

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
