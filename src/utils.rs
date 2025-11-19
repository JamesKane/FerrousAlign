use flate2::read::GzDecoder;
use libc;
use std::fs::OpenOptions;
use std::io::{self, BufReader, Read, Write, stdin};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

#[path = "utils_test.rs"]
mod utils_test;

#[derive(Copy, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Debug)]
#[repr(C)]
pub struct Pair64 {
    pub x: u64,
    pub y: u64,
}

/// Binary I/O trait for writing little-endian values
/// Reduces boilerplate in serialization code (BWT, index files, etc.)
///
/// This trait provides convenience methods for writing binary data in
/// little-endian format, which is used by bwa-mem2 index files.
///
/// # Example
/// ```
/// use ferrous_align::utils::BinaryWrite;
/// use std::io::Cursor;
///
/// let mut buffer = Cursor::new(Vec::new());
/// buffer.write_u64_le(0x123456789ABCDEF0).unwrap();
/// buffer.write_u32_le(0xDEADBEEF).unwrap();
/// ```
pub trait BinaryWrite: Write {
    /// Write a u64 in little-endian format
    #[inline]
    fn write_u64_le(&mut self, val: u64) -> io::Result<()> {
        self.write_all(&val.to_le_bytes())
    }

    /// Write an i64 in little-endian format
    #[inline]
    fn write_i64_le(&mut self, val: i64) -> io::Result<()> {
        self.write_all(&val.to_le_bytes())
    }

    /// Write a u32 in little-endian format
    #[inline]
    fn write_u32_le(&mut self, val: u32) -> io::Result<()> {
        self.write_all(&val.to_le_bytes())
    }

    /// Write an i32 in little-endian format
    #[inline]
    fn write_i32_le(&mut self, val: i32) -> io::Result<()> {
        self.write_all(&val.to_le_bytes())
    }

    /// Write an i8 (single byte)
    #[inline]
    fn write_i8_le(&mut self, val: i8) -> io::Result<()> {
        self.write_all(&[val as u8])
    }

    /// Write a u8 (single byte)
    #[inline]
    fn write_u8_le(&mut self, val: u8) -> io::Result<()> {
        self.write_all(&[val])
    }

    /// Write an array of u64 values in little-endian format
    #[inline]
    fn write_u64_array_le(&mut self, vals: &[u64]) -> io::Result<()> {
        for &val in vals {
            self.write_u64_le(val)?;
        }
        Ok(())
    }

    /// Write an array of i64 values in little-endian format
    #[inline]
    fn write_i64_array_le(&mut self, vals: &[i64]) -> io::Result<()> {
        for &val in vals {
            self.write_i64_le(val)?;
        }
        Ok(())
    }
}

/// Blanket implementation for all types that implement Write
impl<W: Write + ?Sized> BinaryWrite for W {}

pub fn hash_64(key: u64) -> u64 {
    let mut key = key;
    key = key.wrapping_add(!key.wrapping_shl(32));
    key ^= key.wrapping_shr(22);
    key = key.wrapping_add(!key.wrapping_shl(13));
    key ^= key.wrapping_shr(8);
    key = key.wrapping_add(key.wrapping_shl(3));
    key ^= key.wrapping_shr(15);
    key = key.wrapping_add(!key.wrapping_shl(27));
    key ^= key.wrapping_shr(31);
    key
}

pub fn realtime() -> f64 {
    let now = SystemTime::now();
    let since_epoch = now.duration_since(UNIX_EPOCH).expect("Time went backwards");
    since_epoch.as_secs_f64()
}

pub fn cputime() -> f64 {
    let rusage = unsafe {
        let mut rusage = std::mem::MaybeUninit::uninit();
        libc::getrusage(libc::RUSAGE_SELF, rusage.as_mut_ptr());
        rusage.assume_init()
    };
    let user_time = rusage.ru_utime;
    let sys_time = rusage.ru_stime;
    (user_time.tv_sec as f64 + user_time.tv_usec as f64 * 1e-6)
        + (sys_time.tv_sec as f64 + sys_time.tv_usec as f64 * 1e-6)
}

pub fn err_fatal<S: AsRef<str>>(header: S, msg: &str) -> ! {
    log::error!("[{}] {}", header.as_ref(), msg);
    std::process::exit(1);
}

pub fn _err_fatal_simple<S: AsRef<str>>(func: S, msg: &str) -> ! {
    log::error!("[{}] {}", func.as_ref(), msg);
    std::process::exit(1);
}

pub fn xopen<'a>(path: &'a Path, _mode: &str) -> Result<Box<dyn Read + 'a>, io::Error> {
    if path.to_str() == Some("-") {
        return Ok(Box::new(BufReader::new(stdin())));
    }

    let file = OpenOptions::new().read(true).open(path)?;
    Ok(Box::new(BufReader::new(file)))
}

pub fn xzopen<'a>(path: &'a Path, mode: &str) -> Result<Box<dyn Read + 'a>, io::Error> {
    let input = xopen(path, mode)?;
    if path.extension().and_then(|s| s.to_str()) == Some("gz") {
        Ok(Box::new(GzDecoder::new(input)))
    } else {
        Ok(input)
    }
}

pub fn err_fwrite(ptr: &[u8], stream: &mut impl Write) {
    if let Err(e) = stream.write_all(ptr) {
        _err_fatal_simple("fwrite", &e.to_string());
    }
}

pub fn err_fread_noeof(ptr: &mut [u8], stream: &mut impl Read) {
    if let Err(e) = stream.read_exact(ptr) {
        _err_fatal_simple("fread", &e.to_string());
    }
}

pub fn ks_introsort_64(a: &mut [u64]) {
    a.sort();
}

pub fn ks_introsort_128(a: &mut [Pair64]) {
    a.sort();
}
