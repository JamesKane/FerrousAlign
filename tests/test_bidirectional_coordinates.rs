/// Test to verify bidirectional coordinate system understanding
///
/// In bwa-mem2, reference positions use a bidirectional coordinate system:
/// - Forward strand: [0, l_pac)
/// - Reverse strand: [l_pac, 2*l_pac)
/// - Conversion: forward_pos = (2 * l_pac) - 1 - bidirectional_pos

#[test]
fn test_bns_depos_forward_strand() {
    let l_pac = 1000u64;

    // Forward strand position
    let bidirectional_pos = 500u64;

    // Check if reverse strand
    let is_rev = bidirectional_pos >= l_pac;
    assert!(!is_rev, "Position 500 should be on forward strand");

    // Convert to forward position
    let forward_pos = if is_rev {
        (l_pac << 1) - 1 - bidirectional_pos
    } else {
        bidirectional_pos
    };

    assert_eq!(forward_pos, 500, "Forward strand position should be unchanged");
}

#[test]
fn test_bns_depos_reverse_strand() {
    let l_pac = 1000u64;

    // Reverse strand position (1500 in bidirectional coordinates)
    let bidirectional_pos = 1500u64;

    // Check if reverse strand
    let is_rev = bidirectional_pos >= l_pac;
    assert!(is_rev, "Position 1500 should be on reverse strand");

    // Convert to forward position
    let forward_pos = if is_rev {
        (l_pac << 1) - 1 - bidirectional_pos
    } else {
        bidirectional_pos
    };

    // (2 * 1000) - 1 - 1500 = 2000 - 1 - 1500 = 499
    assert_eq!(forward_pos, 499, "Reverse position 1500 maps to forward position 499");
}

#[test]
fn test_bns_depos_symmetry() {
    let l_pac = 1000u64;

    // Position 0 on forward strand
    let fwd_0 = 0u64;
    let is_rev_0 = fwd_0 >= l_pac;
    assert!(!is_rev_0);

    // Should map to position (2*l_pac - 1) on reverse strand
    let rev_999 = (l_pac << 1) - 1 - fwd_0;
    assert_eq!(rev_999, 1999, "Forward 0 maps to reverse 1999");

    // Verify reverse: convert back
    let is_rev_999 = rev_999 >= l_pac;
    assert!(is_rev_999);
    let back_to_fwd = (l_pac << 1) - 1 - rev_999;
    assert_eq!(back_to_fwd, 0, "Reverse 1999 converts back to forward 0");
}

#[test]
fn test_actual_genome_coordinates() {
    // For human genome b38: l_pac ≈ 3.1 billion
    let l_pac = 3_099_441_038u64;

    // Test read seed.ref_pos = 1299604748
    let bidirectional_pos = 1299604748u64;

    // Is it on reverse strand?
    let is_rev = bidirectional_pos >= l_pac;
    assert!(!is_rev, "Position 1299604748 should be on forward strand (< l_pac)");

    // No conversion needed for forward strand
    let forward_pos = bidirectional_pos;
    assert_eq!(forward_pos, 1299604748);
}
