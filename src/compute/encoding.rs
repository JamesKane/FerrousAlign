//! # Sequence Encoding Strategies
//!
//! This module provides the abstraction for different sequence encoding schemes
//! used by various compute backends. It is a key integration point for NPU-based
//! acceleration which typically requires ONE-HOT encoding.
//!
//! ## Architecture Overview
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                    ENCODING STRATEGY INTEGRATION POINT                  │
//! │                                                                         │
//! │  Different compute backends require different sequence encodings:       │
//! │                                                                         │
//! │  • Classic (2-bit): A=0, C=1, G=2, T=3 - Used by CPU SIMD & GPU        │
//! │  • ONE-HOT: A=[1,0,0,0], C=[0,1,0,0], etc. - Used by NPU/ML models     │
//! │                                                                         │
//! │  To add a new encoding:                                                 │
//! │  1. Add variant to EncodingStrategy enum                                │
//! │  2. Implement encode_base() and encode_sequence() for the variant       │
//! │  3. Update the compute backend to use the appropriate encoding          │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```

// ============================================================================
// ENCODING STRATEGY ENUM
// ============================================================================
//
// This enum defines the available sequence encoding schemes. When adding
// a new encoding (e.g., for a specific NPU architecture), add a variant here.
//
// CURRENT STATUS: ONE-HOT is a NO-OP that falls back to Classic encoding.
// It exists as a documented integration point for NPU implementation.
// ============================================================================

/// Sequence encoding strategy for compute backends.
///
/// # Heterogeneous Compute Integration Point
///
/// Different compute backends may require different encodings:
///
/// - **Classic (2-bit)**: Compact encoding used by FM-Index and SIMD alignment
///   - A=0, C=1, G=2, T=3, N=4
///   - 3 bits per base (fits in u8)
///
/// - **ONE-HOT**: Expanded encoding used by neural network models
///   - A=[1,0,0,0], C=[0,1,0,0], G=[0,0,1,0], T=[0,0,0,1]
///   - 4 values per base (typically f32 or u8 for quantized models)
///
/// ## Current Status
///
/// - `Classic`: Fully implemented, used by all current backends
/// - `OneHot`: NO-OP placeholder, falls back to Classic encoding
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EncodingStrategy {
    // ========================================================================
    // CLASSIC 2-BIT ENCODING (ACTIVE)
    // ========================================================================
    /// Classic 2-bit encoding used by BWA-MEM and FM-Index.
    ///
    /// Encoding: A=0, C=1, G=2, T=3, N=4
    ///
    /// This is the standard encoding for:
    /// - FM-Index backward search
    /// - CPU SIMD Smith-Waterman
    /// - GPU Smith-Waterman (future)
    #[default]
    Classic,

    // ========================================================================
    // ONE-HOT ENCODING (FUTURE - CURRENTLY NO-OP)
    // ========================================================================
    /// ONE-HOT encoding for neural network models.
    ///
    /// # CURRENT STATUS: NO-OP
    ///
    /// This variant exists as an integration point. When used, it currently
    /// produces Classic encoding. To implement:
    ///
    /// 1. Update `encode_base_onehot()` to return actual ONE-HOT vectors
    /// 2. Update `encode_sequence_onehot()` to produce tensor-compatible output
    /// 3. Integrate with NPU backend in compute module
    ///
    /// ## ONE-HOT Format
    ///
    /// Each base is encoded as a 4-element vector:
    /// - A = [1, 0, 0, 0]
    /// - C = [0, 1, 0, 0]
    /// - G = [0, 0, 1, 0]
    /// - T = [0, 0, 0, 1]
    /// - N = [0, 0, 0, 0] (or [0.25, 0.25, 0.25, 0.25] for probabilistic)
    ///
    /// ## Use Cases
    ///
    /// - NPU seed viability classifier
    /// - ML-based alignment scoring
    /// - Neural network quality prediction
    OneHot,
}

impl EncodingStrategy {
    /// Encode a single base using the selected strategy.
    ///
    /// # Heterogeneous Compute Integration Point
    ///
    /// Returns the encoded representation of a DNA base (A, C, G, T, N).
    ///
    /// ## Return Value
    ///
    /// - `Classic`: Returns a single u8 (0-4)
    /// - `OneHot`: Returns 4 u8 values (one-hot vector)
    ///   **Currently NO-OP: returns Classic encoding**
    pub fn encode_base(&self, base: u8) -> Vec<u8> {
        match self {
            EncodingStrategy::Classic => {
                vec![classic_base_to_code(base)]
            }
            // ================================================================
            // ONE-HOT: NO-OP - Falls back to Classic encoding
            // ================================================================
            // TODO: When NPU is implemented, return actual ONE-HOT vector:
            // EncodingStrategy::OneHot => encode_base_onehot(base),
            EncodingStrategy::OneHot => {
                // Placeholder: return classic encoding until NPU is implemented
                vec![classic_base_to_code(base)]
            }
        }
    }

    /// Encode a sequence using the selected strategy.
    ///
    /// # Heterogeneous Compute Integration Point
    ///
    /// Encodes an entire DNA sequence for the selected compute backend.
    ///
    /// ## Return Value
    ///
    /// - `Classic`: Vec<u8> with one value per base (length = sequence length)
    /// - `OneHot`: Vec<u8> with 4 values per base (length = 4 × sequence length)
    ///   **Currently NO-OP: returns Classic encoding**
    pub fn encode_sequence(&self, seq: &[u8]) -> Vec<u8> {
        match self {
            EncodingStrategy::Classic => seq.iter().map(|&b| classic_base_to_code(b)).collect(),
            // ================================================================
            // ONE-HOT: NO-OP - Falls back to Classic encoding
            // ================================================================
            // TODO: When NPU is implemented, return actual ONE-HOT matrix:
            // EncodingStrategy::OneHot => encode_sequence_onehot(seq),
            EncodingStrategy::OneHot => {
                // Placeholder: return classic encoding until NPU is implemented
                seq.iter().map(|&b| classic_base_to_code(b)).collect()
            }
        }
    }

    /// Get reverse complement of encoded sequence.
    ///
    /// For Classic encoding, this reverses and complements (A↔T, C↔G).
    /// For ONE-HOT encoding, this reverses and swaps channels.
    pub fn reverse_complement(&self, encoded: &[u8]) -> Vec<u8> {
        match self {
            EncodingStrategy::Classic => encoded
                .iter()
                .rev()
                .map(|&code| classic_complement_code(code))
                .collect(),
            // ONE-HOT: NO-OP - Falls back to Classic
            EncodingStrategy::OneHot => encoded
                .iter()
                .rev()
                .map(|&code| classic_complement_code(code))
                .collect(),
        }
    }

    /// Returns a human-readable name for this encoding strategy.
    pub fn name(&self) -> &'static str {
        match self {
            EncodingStrategy::Classic => "Classic (2-bit)",
            EncodingStrategy::OneHot => "ONE-HOT (4-channel) [NO-OP]",
        }
    }

    /// Returns the number of values produced per base.
    ///
    /// - Classic: 1 (single u8 per base)
    /// - OneHot: 4 (four u8 values per base)
    pub fn values_per_base(&self) -> usize {
        match self {
            EncodingStrategy::Classic => 1,
            EncodingStrategy::OneHot => 4, // Will be used when implemented
        }
    }
}

// ============================================================================
// CLASSIC ENCODING HELPERS
// ============================================================================

/// Convert ASCII base to 2-bit code (Classic encoding).
///
/// - A/a → 0
/// - C/c → 1
/// - G/g → 2
/// - T/t → 3
/// - N/n/other → 4
#[inline]
pub fn classic_base_to_code(base: u8) -> u8 {
    match base {
        b'A' | b'a' => 0,
        b'C' | b'c' => 1,
        b'G' | b'g' => 2,
        b'T' | b't' => 3,
        _ => 4, // N or unknown
    }
}

/// Get complement of 2-bit code (A↔T, C↔G).
#[inline]
pub fn classic_complement_code(code: u8) -> u8 {
    match code {
        0 => 3, // A → T
        1 => 2, // C → G
        2 => 1, // G → C
        3 => 0, // T → A
        _ => 4, // N → N
    }
}

// ============================================================================
// ONE-HOT ENCODING HELPERS (FUTURE IMPLEMENTATION)
// ============================================================================
//
// These functions are placeholders for ONE-HOT encoding implementation.
// When NPU support is added, uncomment and complete these functions.
// ============================================================================

/// Convert ASCII base to ONE-HOT vector (4 u8 values).
///
/// # Future Implementation
///
/// When NPU is implemented, this will return:
/// - A → [1, 0, 0, 0]
/// - C → [0, 1, 0, 0]
/// - G → [0, 0, 1, 0]
/// - T → [0, 0, 0, 1]
/// - N → [0, 0, 0, 0]
#[allow(dead_code)]
fn encode_base_onehot(base: u8) -> Vec<u8> {
    match base {
        b'A' | b'a' => vec![1, 0, 0, 0],
        b'C' | b'c' => vec![0, 1, 0, 0],
        b'G' | b'g' => vec![0, 0, 1, 0],
        b'T' | b't' => vec![0, 0, 0, 1],
        _ => vec![0, 0, 0, 0], // N or unknown
    }
}

/// Encode entire sequence to ONE-HOT matrix.
///
/// # Future Implementation
///
/// Returns a flat Vec<u8> with 4 values per base, suitable for
/// reshaping into a (seq_len, 4) tensor for NPU inference.
#[allow(dead_code)]
fn encode_sequence_onehot(seq: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(seq.len() * 4);
    for &base in seq {
        result.extend(encode_base_onehot(base));
    }
    result
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classic_base_encoding() {
        assert_eq!(classic_base_to_code(b'A'), 0);
        assert_eq!(classic_base_to_code(b'C'), 1);
        assert_eq!(classic_base_to_code(b'G'), 2);
        assert_eq!(classic_base_to_code(b'T'), 3);
        assert_eq!(classic_base_to_code(b'N'), 4);
        assert_eq!(classic_base_to_code(b'a'), 0); // lowercase
    }

    #[test]
    fn test_classic_complement() {
        assert_eq!(classic_complement_code(0), 3); // A → T
        assert_eq!(classic_complement_code(1), 2); // C → G
        assert_eq!(classic_complement_code(2), 1); // G → C
        assert_eq!(classic_complement_code(3), 0); // T → A
        assert_eq!(classic_complement_code(4), 4); // N → N
    }

    #[test]
    fn test_encoding_strategy_classic() {
        let strategy = EncodingStrategy::Classic;
        let encoded = strategy.encode_base(b'A');
        assert_eq!(encoded, vec![0]);

        let seq = b"ACGT";
        let encoded_seq = strategy.encode_sequence(seq);
        assert_eq!(encoded_seq, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_encoding_strategy_onehot_fallback() {
        // ONE-HOT currently falls back to Classic
        let strategy = EncodingStrategy::OneHot;
        let encoded = strategy.encode_base(b'A');
        // Currently returns Classic encoding (NO-OP)
        assert_eq!(encoded, vec![0]);

        let seq = b"ACGT";
        let encoded_seq = strategy.encode_sequence(seq);
        // Currently returns Classic encoding (NO-OP)
        assert_eq!(encoded_seq, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_reverse_complement() {
        let strategy = EncodingStrategy::Classic;
        let seq = b"ACGT";
        let encoded = strategy.encode_sequence(seq);
        let rc = strategy.reverse_complement(&encoded);
        // ACGT → TGCA → [3,2,1,0] reversed and complemented = [0,1,2,3]
        // Actually: ACGT encoded = [0,1,2,3]
        // Reverse: [3,2,1,0]
        // Complement: [0,1,2,3] (T→A, G→C, C→G, A→T)
        assert_eq!(rc, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_onehot_helper_functions() {
        // Test the helper functions directly (for future use)
        let a_onehot = encode_base_onehot(b'A');
        assert_eq!(a_onehot, vec![1, 0, 0, 0]);

        let seq_onehot = encode_sequence_onehot(b"AC");
        assert_eq!(seq_onehot, vec![1, 0, 0, 0, 0, 1, 0, 0]);
    }

    #[test]
    fn test_values_per_base() {
        assert_eq!(EncodingStrategy::Classic.values_per_base(), 1);
        assert_eq!(EncodingStrategy::OneHot.values_per_base(), 4);
    }
}
