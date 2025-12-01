// Helper function to create scoring matrix (similar to bwa_fill_scmat in main_banded.cpp)
pub fn bwa_fill_scmat(match_score: i8, mismatch_penalty: i8, ambig_penalty: i8) -> [i8; 25] {
    let mut mat = [0i8; 25];
    let mut k = 0;

    // Fill 5x5 matrix for A, C, G, T, N
    for i in 0..4 {
        for j in 0..4 {
            mat[k] = if i == j {
                match_score
            } else {
                -mismatch_penalty
            };
            k += 1;
        }
        mat[k] = ambig_penalty; // ambiguous base (N)
        k += 1;
    }

    // Last row for N
    for _ in 0..5 {
        mat[k] = ambig_penalty;
        k += 1;
    }

    mat
}
