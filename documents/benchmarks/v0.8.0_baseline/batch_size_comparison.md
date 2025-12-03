| Command | Mean [s] | Min [s] | Max [s] | Relative |
|:---|---:|---:|---:|---:|
| `./target/release/ferrous-align mem -t 16 --batch-size 100000 /home/jkane/Genomics/Reference/chm13v2.0/chm13v2.0.fa.gz /home/jkane/Genomics/HG002/test_100k_R1.fq /home/jkane/Genomics/HG002/test_100k_R2.fq > /dev/null` | 16.356 ± 0.065 | 16.286 | 16.414 | 1.00 |
| `./target/release/ferrous-align mem -t 16 --batch-size 200000 /home/jkane/Genomics/Reference/chm13v2.0/chm13v2.0.fa.gz /home/jkane/Genomics/HG002/test_100k_R1.fq /home/jkane/Genomics/HG002/test_100k_R2.fq > /dev/null` | 16.456 ± 0.157 | 16.283 | 16.588 | 1.01 ± 0.01 |
| `./target/release/ferrous-align mem -t 16 --batch-size 500000 /home/jkane/Genomics/Reference/chm13v2.0/chm13v2.0.fa.gz /home/jkane/Genomics/HG002/test_100k_R1.fq /home/jkane/Genomics/HG002/test_100k_R2.fq > /dev/null` | 16.458 ± 0.124 | 16.339 | 16.586 | 1.01 ± 0.01 |
