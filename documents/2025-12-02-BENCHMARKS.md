========================================
Summary Tables
========================================

**NOTE**: AVX-512 crash was **FIXED** on 2025-12-02 (commit e763e4a)
- Root cause: Misaligned buffer allocation (Vec<u8> provides ~48-byte alignment, AVX-512 requires 64-byte)
- Solution: Added aligned allocation helper using std::alloc::alloc with 64-byte Layout
- Status: AVX-512 now runs successfully on 10K and 100K read datasets
- Benchmarks below show **pre-fix** crash results; need to re-run for updated metrics

### ⏱️ Alignment Performance

| Tool         | Reference |  SIMD   | Wall Time | Max Memory (GB) | Throughput (reads/s) | Status |
|:-------------|:----------|:-------:|:---------:|:---------------:|:--------------------:|:------:|
| **AVX2**     | GRCh38    | 256-bit |  2:10.91  |      38.00      |        61,897        | ✅ |
| **AVX2**     | CHM13v2.0 | 256-bit |  2:27.10  |      41.16      |        54,713        | ✅ |
| **AVX512**   | GRCh38    | 512-bit |  1:21.82  |      19.52      |     ** CRASH **      | ✅ FIXED |
| **AVX512**   | CHM13v2.0 | 512-bit |  1:27.68  |      19.64      |     ** CRASH **      | ✅ FIXED |
| **BWA-MEM2** | GRCh38    |   N/A   |  1:46.27  |      20.31      |        75,770        | ✅ |
| **BWA-MEM2** | CHM13v2.0 |   N/A   |  2:24.42  |      21.94      |        55,554        | ✅ |

### 1. 1. AVX2 on GRCh38 - Detailed Metrics

| Metric          | Value                       |
|-----------------|-----------------------------|
| **SIMD Width**  | **256-bit (AVX2)**          |
| **Parallelism** | **32-way**                  |
| Total reads     | 8,000,000 (4,000,000 pairs) |
| Total records   | 8,102,683                   |
| Supplementary   | 102,683                     |
| Mapped          | 98.59%                      |
| Properly paired | 94.36%                      |
| Pairs rescued   | 0                           |
| **Wall time**   | **2:10.91 (130.91s)**       |
| CPU time        | 1042.31s                    |
| **Throughput**  | **61,897 reads/sec**        |
| CPU efficiency  | 796% (8.0x parallel)        |
| Max memory      | 38.0 GB                     |

### 2. 2. AVX2 on CHM13v2.0 - Detailed Metrics

| Metric          | Value                       |
|-----------------|-----------------------------|
| **SIMD Width**  | **256-bit (AVX2)**          |
| **Parallelism** | **32-way**                  |
| Total reads     | 8,000,000 (4,000,000 pairs) |
| Total records   | 8,048,253                   |
| Supplementary   | 48,253                      |
| Mapped          | 98.78%                      |
| Properly paired | 94.13%                      |
| Pairs rescued   | 0                           |
| **Wall time**   | **2:27.10 (147.10s)**       |
| CPU time        | 1206.37s                    |
| **Throughput**  | **54,713 reads/sec**        |
| CPU efficiency  | 820% (8.2x parallel)        |
| Max memory      | 41.2 GB                     |

### 3. 3. AVX-512 on GRCh38 - Detailed Metrics

| Metric          | Value                       |
|-----------------|-----------------------------|
| **SIMD Width**  | **512-bit (AVX-512)**       |
| **Parallelism** | **64-way**                  |
| Total reads     | 8,000,000 (4,000,000 pairs) |
| Total records   | 0                           |
| Supplementary   | -8,000,000                  |
| Mapped          | N/A                         |
| Properly paired | N/A                         |
| Pairs rescued   | N/A                         |
| **Wall time**   | **1:21.82 (81.82s)**        |
| CPU time        | 20.21s                      |
| **Throughput**  | ** CRASH **                 |
| CPU efficiency  | 25% (0.2x parallel)         |
| Max memory      | 19.5 GB                     |

### 4. 4. AVX-512 on CHM13v2.0 - Detailed Metrics

| Metric          | Value                       |
|-----------------|-----------------------------|
| **SIMD Width**  | **512-bit (AVX-512)**       |
| **Parallelism** | **64-way**                  |
| Total reads     | 8,000,000 (4,000,000 pairs) |
| Total records   | 0                           |
| Supplementary   | -8,000,000                  |
| Mapped          | N/A                         |
| Properly paired | N/A                         |
| Pairs rescued   | N/A                         |
| **Wall time**   | **1:27.68 (87.68s)**        |
| CPU time        | 20.59s                      |
| **Throughput**  | ** CRASH **                 |
| CPU efficiency  | 23% (0.2x parallel)         |
| Max memory      | 19.6 GB                     |

### 5. 5. BWA-MEM2 AVX2 on GRCh38 (reference) - Detailed Metrics

| Metric          | Value                       |
|-----------------|-----------------------------|
| **SIMD Width**  | **N/A**                     |
| **Parallelism** | **N/A**                     |
| Total reads     | 8,000,000 (4,000,000 pairs) |
| Total records   | 8,052,480                   |
| Supplementary   | 52,480                      |
| Mapped          | 99.47%                      |
| Properly paired | 97.10%                      |
| Pairs rescued   | N/A                         |
| **Wall time**   | **1:46.27 (106.27s)**       |
| CPU time        | 1553.17s                    |
| **Throughput**  | **75,770 reads/sec**        |
| CPU efficiency  | 1461% (14.6x parallel)      |
| Max memory      | 20.3 GB                     |

### 6. 6. BWA-MEM2 AVX2 on CHM13v2.0 (reference) - Detailed Metrics

| Metric          | Value                       |
|-----------------|-----------------------------|
| **SIMD Width**  | **N/A**                     |
| **Parallelism** | **N/A**                     |
| Total reads     | 8,000,000 (4,000,000 pairs) |
| Total records   | 8,023,095                   |
| Supplementary   | 23,095                      |
| Mapped          | 99.46%                      |
| Properly paired | 98.19%                      |
| Pairs rescued   | N/A                         |
| **Wall time**   | **2:24.42 (144.42s)**       |
| CPU time        | 2130.55s                    |
| **Throughput**  | **55,554 reads/sec**        |
| CPU efficiency  | 1475% (14.8x parallel)      |
| Max memory      | 21.9 GB                     |

========================================
Benchmark complete: 2025-12-02 07:14:28
========================================