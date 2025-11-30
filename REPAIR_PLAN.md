# Plan to Align SW_Kernel with Project's Smith-Waterman Algorithm

## 1. Analysis of Discrepancy

Following a review of all manual SIMD implementations (SSE/NEON, AVX2, and AVX512), it has been determined that they all consistently use a specific Smith-Waterman DP recurrence. My initial analysis incorrectly labeled this as a bug.

-   **Established Algorithm (SSE, AVX2, AVX512):** All existing manual implementations calculate the open-gap penalty for both E (deletions) and F (insertions) based on the `M` score, which is derived from the diagonal `H(i-1, j-1)` cell. This is the **correct and intended algorithm** for this project, likely inherited from the original BWA-MEM2 C implementation.

    ```rust
    // Established pattern in manual SIMD code
    let m_vec = H(i-1, j-1) + score;
    // ... m_vec is clamped to zero ...

    // Gap open calculations are based on m_vec
    let e_open = m_vec - (o_del + e_del);
    let f_open = m_vec - (o_ins + e_ins);
    ```

-   **`sw_kernel` (New Shared Kernel):** The new generic kernel is the actual outlier. It was implemented using a more common "textbook" Smith-Waterman recurrence, where gap penalties are based on the adjacent `H(i-1, j)` (for E) and `H(i, j-1)` (for F) cells.

    ```rust
    // Current logic in sw_kernel
    let e_from_open = H(i-1, j) - (o_del + e_del);
    let f_from_open = H(i, j-1) - (o_ins + e_ins);
    ```

The discrepancy in these two approaches is the root cause of the failing parity tests. To resolve this, the new `sw_kernel` must be modified to conform to the project's established algorithm.

## 2. Proposed Solution

The goal is to replace the manual `simd_banded_swa_batch32` with the new macro-based version that uses `sw_kernel`. To do this, the `sw_kernel` must behave identically to the code it is replacing.

The solution is to **modify `sw_kernel` to implement the same DP recurrence as the manual SSE, AVX2, and AVX512 implementations.**

This will:
1.  Fix the failing parity tests.
2.  Unify all Smith-Waterman implementations under a single, consistent algorithm.
3.  Allow the refactoring to proceed as planned.

## 3. Implementation Steps

I will modify the `sw_kernel` function in `src/core/alignment/banded_swa_kernel.rs`.

1.  **Locate the DP recurrence loop:** `for j in 0..qmax`.
2.  **Calculate `M` score:** Just after calculating `score_vec`, compute the `M` score (`h_from_diag`) and clamp it to zero, mirroring how `m_vec` is calculated in the manual implementations.
3.  **Modify Gap Calculations:** Change the `e_from_open` and `f_from_open` calculations to use this `M` score instead of `h_up` and `h_left`.
4.  **Update Final `H` Calculation:** Ensure the final `h_val` is a `max()` of the new `M`, `E`, and `F` scores.

**Code Snippet (Conceptual):**

```rust
// In src/core/alignment/banded_swa_kernel.rs -> sw_kernel()

// ... inside `for j in 0..qmax` loop

// Calculate M score from the diagonal, mirroring the manual implementations.
let h_from_diag = E::adds_epi8(h_diag, score_vec);
let m_vec = E::max_epi8(h_from_diag, zero); // Clamp to zero.

// Align gap penalty logic with the project's established algorithm.
// Gap opens are calculated from the M score, not H(i-1,j) or H(i,j-1).
let e_from_open = E::subs_epi8(m_vec, oe_del_vec);
let e_from_ext = E::subs_epi8(e_prev, e_del_vec);
let mut e_val = E::max_epi8(e_from_open, e_from_ext);
e_val = E::max_epi8(e_val, zero);


let f_from_open = E::subs_epi8(m_vec, oe_ins_vec);
let f_from_ext = E::subs_epi8(f_vec, e_ins_vec);
f_vec = E::max_epi8(f_from_open, f_from_ext);
f_vec = E::max_epi8(f_vec, zero);

// The final H score is the max of M, E, and F.
let mut h_val = E::max_epi8(m_vec, e_val);
h_val = E::max_epi8(h_val, f_vec);

// ... rest of the loop
```

This change will align the `sw_kernel`'s behavior with the `simd_banded_swa_batch32` implementation, resolving the test failures.
