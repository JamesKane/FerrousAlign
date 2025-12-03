========================================
Summary Tables
========================================

### ⏱️ Alignment Performance
| Tool | Reference | SIMD | Wall Time | Max Memory (GB) | Throughput (reads/s) |
|:---|:---|:---:|:---:|:---:|:---:|
| **AVX2** | GRCh38 | 256-bit | 2:22.43 | 38.42 | 57,663 |
| **AVX2** | CHM13v2.0 | 256-bit | 2:41.92 | 42.10 | 50,297 |
| **AVX512** | GRCh38 | 512-bit | 2:23.98 | 38.45 | 57,040 |
| **AVX512** | CHM13v2.0 | 512-bit | 2:40.83 | 42.12 | 50,637 |
| **BWA-MEM2** | GRCh38 | N/A | 1:50.12 | 20.34 | 73,127 |
| **BWA-MEM2** | CHM13v2.0 | N/A | 2:27.68 | 21.87 | 54,327 |


### 1. 1. AVX2 on GRCh38 - Detailed Metrics

| Metric | Value |
|--------|-------|
| **SIMD Width** | **256-bit (AVX2)** |
| **Parallelism** | **32-way** |
| Total reads | 8,000,000 (4,000,000 pairs) |
| Total records | 8,212,802 |
| Supplementary | 212,802 |
| Mapped | 98.53% |
| Properly paired | 96.01% |
| Pairs rescued | 448,175 |
| **Wall time** | **2:22.43 (142.43s)** |
| CPU time | 1067.17s |
| **Throughput** | **57,663 reads/sec** |
| CPU efficiency | 749% (7.5x parallel) |
| Max memory | 38.4 GB |


### 2. 2. AVX2 on CHM13v2.0 - Detailed Metrics

| Metric | Value |
|--------|-------|
| **SIMD Width** | **256-bit (AVX2)** |
| **Parallelism** | **32-way** |
| Total reads | 8,000,000 (4,000,000 pairs) |
| Total records | 8,144,050 |
| Supplementary | 144,050 |
| Mapped | 98.72% |
| Properly paired | 95.80% |
| Pairs rescued | 469,212 |
| **Wall time** | **2:41.92 (161.92s)** |
| CPU time | 1231.81s |
| **Throughput** | **50,297 reads/sec** |
| CPU efficiency | 761% (7.6x parallel) |
| Max memory | 42.1 GB |


### 3. 3. AVX-512 on GRCh38 - Detailed Metrics

| Metric | Value |
|--------|-------|
| **SIMD Width** | **512-bit (AVX-512)** |
| **Parallelism** | **64-way** |
| Total reads | 8,000,000 (4,000,000 pairs) |
| Total records | 8,212,800 |
| Supplementary | 212,800 |
| Mapped | 98.53% |
| Properly paired | 96.01% |
| Pairs rescued | 448,185 |
| **Wall time** | **2:23.98 (143.98s)** |
| CPU time | 1068.33s |
| **Throughput** | **57,040 reads/sec** |
| CPU efficiency | 742% (7.4x parallel) |
| Max memory | 38.4 GB |


### 4. 4. AVX-512 on CHM13v2.0 - Detailed Metrics

| Metric | Value |
|--------|-------|
| **SIMD Width** | **512-bit (AVX-512)** |
| **Parallelism** | **64-way** |
| Total reads | 8,000,000 (4,000,000 pairs) |
| Total records | 8,144,039 |
| Supplementary | 144,039 |
| Mapped | 98.72% |
| Properly paired | 95.80% |
| Pairs rescued | 469,212 |
| **Wall time** | **2:40.83 (160.83s)** |
| CPU time | 1244.55s |
| **Throughput** | **50,637 reads/sec** |
| CPU efficiency | 774% (7.7x parallel) |
| Max memory | 42.1 GB |


### 5. 5. BWA-MEM2 AVX2 on GRCh38 (reference) - Detailed Metrics

| Metric | Value |
|--------|-------|
| **SIMD Width** | **N/A** |
| **Parallelism** | **N/A** |
| Total reads | 8,000,000 (4,000,000 pairs) |
| Total records | 8,052,480 |
| Supplementary | 52,480 |
| Mapped | 99.47% |
| Properly paired | 97.10% |
| Pairs rescued | N/A |
| **Wall time** | **1:50.12 (110.12s)** |
| CPU time | 1580.03s |
| **Throughput** | **73,127 reads/sec** |
| CPU efficiency | 1435% (14.3x parallel) |
| Max memory | 20.3 GB |


### 6. 6. BWA-MEM2 AVX2 on CHM13v2.0 (reference) - Detailed Metrics

| Metric | Value |
|--------|-------|
| **SIMD Width** | **N/A** |
| **Parallelism** | **N/A** |
| Total reads | 8,000,000 (4,000,000 pairs) |
| Total records | 8,023,095 |
| Supplementary | 23,095 |
| Mapped | 99.46% |
| Properly paired | 98.19% |
| Pairs rescued | N/A |
| **Wall time** | **2:27.68 (147.68s)** |
| CPU time | 2175.46s |
| **Throughput** | **54,327 reads/sec** |
| CPU efficiency | 1473% (14.7x parallel) |
| Max memory | 21.9 GB |