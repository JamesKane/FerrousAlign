//! Unit test to validate AVX-512 alignment hypothesis
//!
//! This test demonstrates that Vec<u8> is not 64-byte aligned,
//! causing _mm512_store_si512 to crash on misaligned buffers.

#![cfg(all(target_arch = "x86_64", feature = "avx512"))]
#![feature(stdarch_x86_avx512)]

use std::arch::x86_64::*;

#[test]
fn test_vec_u8_alignment() {
    // Allocate a Vec<u8> similar to the DP matrices in the kernel
    let buffer = vec![0u8; 9536]; // Same size as in the crash scenario
    let ptr = buffer.as_ptr();

    println!("Vec<u8> pointer: {:p}", ptr);
    println!("Alignment: {} bytes", ptr as usize % 64);

    // Check if the pointer is 64-byte aligned
    let is_aligned = (ptr as usize) % 64 == 0;

    println!("Is 64-byte aligned: {}", is_aligned);

    // This will almost certainly fail - Vec<u8> is typically 8 or 16-byte aligned
    assert!(
        !is_aligned,
        "Vec<u8> happened to be 64-byte aligned (rare!). Alignment: {} bytes",
        ptr as usize % 64
    );
}

#[test]
fn test_avx512_aligned_store_on_vec() {
    // This test should crash if run, demonstrating the bug
    // We skip it by default to avoid test failures
    if std::env::var("RUN_CRASH_TEST").is_ok() {
        unsafe {
            let mut buffer = vec![0u8; 128];
            let ptr = buffer.as_mut_ptr();

            println!("Attempting aligned store to Vec<u8> at {:p} (alignment: {})",
                ptr, ptr as usize % 64);

            let zero = _mm512_setzero_si512();

            // This will likely crash with SIGILL or SIGSEGV if not aligned
            _mm512_store_si512(ptr as *mut _, zero);

            panic!("Store succeeded - buffer happened to be aligned!");
        }
    } else {
        println!("Skipping crash test (set RUN_CRASH_TEST=1 to run)");
    }
}

#[test]
fn test_avx512_unaligned_store_works() {
    // Demonstrate that unaligned store works regardless of alignment
    unsafe {
        let mut buffer = vec![0u8; 128];
        let ptr = buffer.as_mut_ptr();

        println!("Testing unaligned store to Vec<u8> at {:p} (alignment: {})",
            ptr, ptr as usize % 64);

        let test_value = _mm512_set1_epi8(0x42);

        // This should always work, even on misaligned buffers
        _mm512_storeu_si512(ptr as *mut _, test_value);

        // Verify the write succeeded
        assert_eq!(buffer[0], 0x42);
        assert_eq!(buffer[63], 0x42);

        println!("✓ Unaligned store succeeded");
    }
}

#[test]
fn test_aligned_allocation() {
    // Demonstrate how to properly allocate aligned memory
    use std::alloc::{alloc, dealloc, Layout};

    unsafe {
        // Allocate 64-byte aligned memory
        let layout = Layout::from_size_align(9536, 64).unwrap();
        let ptr = alloc(layout);

        println!("Aligned allocation at {:p}", ptr);
        println!("Alignment: {} bytes", ptr as usize % 64);

        // This should be properly aligned
        assert_eq!(ptr as usize % 64, 0, "Aligned allocation failed!");

        // Test that aligned store works
        let zero = _mm512_setzero_si512();
        _mm512_store_si512(ptr as *mut _, zero);

        println!("✓ Aligned store succeeded on properly aligned buffer");

        // Clean up
        dealloc(ptr, layout);
    }
}

#[test]
fn test_workspace_buffer_alignment() {
    // Test if workspace buffers are properly aligned
    // This demonstrates what the fix should look like

    use std::alloc::{alloc, dealloc, Layout};

    unsafe {
        let size = 9536;
        let layout = Layout::from_size_align(size, 64).unwrap();
        let ptr = alloc(layout) as *mut u8;

        // Create a slice from the aligned allocation
        let buffer = std::slice::from_raw_parts_mut(ptr, size);

        println!("Workspace buffer at {:p} (alignment: {})",
            buffer.as_ptr(), buffer.as_ptr() as usize % 64);

        assert_eq!(buffer.as_ptr() as usize % 64, 0);

        // Simulate the initialization loop from the kernel
        let zero = _mm512_setzero_si512();
        let ncol = 148;

        for i in 0..=ncol {
            let offset = i * 64;
            let store_ptr = buffer.as_mut_ptr().add(offset);
            _mm512_store_si512(store_ptr as *mut _, zero);
        }

        println!("✓ Initialization loop completed successfully with aligned buffer");

        // Clean up
        dealloc(ptr, layout);
    }
}

#[test]
fn test_crash_scenario_reproduction() {
    // Reproduce the exact crash scenario from the logs
    // ncol=148, requires h_size=9536 bytes

    unsafe {
        // This is what the kernel does (WRONG - misaligned)
        let mut h0_buf_misaligned = vec![0u8; 9536];
        let ptr_misaligned = h0_buf_misaligned.as_mut_ptr();

        println!("=== Crash Scenario Reproduction ===");
        println!("ncol=148, required_h_size=9536");
        println!("Misaligned Vec<u8> at {:p} (alignment: {} bytes)",
            ptr_misaligned, ptr_misaligned as usize % 64);

        // This is the correct way (aligned)
        use std::alloc::{alloc, dealloc, Layout};
        let layout = Layout::from_size_align(9536, 64).unwrap();
        let ptr_aligned = alloc(layout);

        println!("Aligned buffer at {:p} (alignment: {} bytes)",
            ptr_aligned, ptr_aligned as usize % 64);

        assert_ne!(ptr_misaligned as usize % 64, 0, "Vec<u8> shouldn't be 64-byte aligned");
        assert_eq!(ptr_aligned as usize % 64, 0, "Aligned alloc should be 64-byte aligned");

        // Demonstrate the fix: use unaligned store for Vec<u8>
        let zero = _mm512_setzero_si512();

        // This would crash with aligned store on misaligned buffer:
        // _mm512_store_si512(ptr_misaligned as *mut _, zero); // CRASH!

        // But unaligned store works:
        _mm512_storeu_si512(ptr_misaligned as *mut _, zero);
        println!("✓ Unaligned store to Vec<u8> succeeded");

        // And aligned store works on aligned buffer:
        _mm512_store_si512(ptr_aligned as *mut _, zero);
        println!("✓ Aligned store to aligned buffer succeeded");

        dealloc(ptr_aligned, layout);
    }
}
