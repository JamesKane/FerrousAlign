# MIT License

## FerrousAlign

Copyright (c) 2025 James R. Kane

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Attributions

### BWA-MEM2

FerrousAlign is a Rust port based on the algorithms and architecture of BWA-MEM2.

**BWA-MEM2** - Copyright (c) 2019 Intel Corporation and Heng Li

Authors:
- Vasimuddin Md (Intel Corporation)
- Sanchit Misra (Intel Corporation)
- Heng Li (Harvard University)

BWA-MEM2 is licensed under the MIT License.

**Citation**: Vasimuddin Md, Sanchit Misra, Heng Li, Srinivas Aluru. "Efficient Architecture-Aware Acceleration of BWA-MEM for Multicore Systems." IEEE IPDPS 2019.

**Repository**: https://github.com/bwa-mem2/bwa-mem2

### BWA

BWA-MEM2 (and therefore FerrousAlign) builds upon the foundational algorithms developed in the original BWA (Burrows-Wheeler Aligner).

**BWA** - Copyright (c) 2008-2015 Heng Li

**Citation**: Li H. and Durbin R. "Fast and accurate short read alignment with Burrows-Wheeler transform." Bioinformatics, 25:1754-1760, 2009.

**Repository**: https://github.com/lh3/bwa

---

## License Compatibility Notes

- **FerrousAlign**: MIT License (this project)
- **BWA-MEM2**: MIT License (source of algorithms and architecture)
- **BWA**: GPL v3 (original algorithm)

While BWA is licensed under GPL v3, BWA-MEM2 is a clean rewrite under the MIT License. FerrousAlign is based on BWA-MEM2's implementation and continues under the MIT License, maintaining attribution to all original authors.
