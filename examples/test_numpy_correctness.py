#!/usr/bin/env python3
"""
Test correctness of NumPy-optimized co-retweet approach.

This script verifies that CoRetweetsNumpyApproach produces identical
results to the standard CoRetweetsApproach implementation.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from approaches import CoRetweetsApproach, CoRetweetsNumpyApproach


def generate_test_data(num_users=10, num_tweets=5, num_rts=50, window_sec=60):
    """
    Generate synthetic retweet data for testing.

    Args:
        num_users: Number of users
        num_tweets: Number of tweets
        num_rts: Number of retweets to generate
        window_sec: Time window for co-retweets

    Returns:
        List of (user_id, tweet_id, timestamp) tuples
    """
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    RTs = []

    for _ in range(num_rts):
        user = random.randint(0, num_users - 1)
        tweet = random.randint(0, num_tweets - 1)
        # Random offset up to 2 * window_sec
        offset = random.randint(0, window_sec * 2)
        timestamp = base_time + timedelta(seconds=offset)
        RTs.append((user, tweet, timestamp))

    return RTs


def test_approaches_match(RTs, window_sec=60, min_coactions=1):
    """
    Test that both approaches produce identical results.

    Args:
        RTs: List of retweets
        window_sec: Time window
        min_coactions: Minimum co-actions

    Returns:
        True if results match, False otherwise
    """
    standard = CoRetweetsApproach(window_sec, min_coactions)
    numpy_opt = CoRetweetsNumpyApproach(window_sec, min_coactions)

    result_standard = standard.compute_pairs_scores(RTs)
    result_numpy = numpy_opt.compute_pairs_scores(RTs)

    return result_standard == result_numpy


def main():
    print("=" * 80)
    print("NUMPY APPROACH CORRECTNESS TESTS")
    print("=" * 80)

    all_passed = True

    # Test 1: Empty dataset
    print("\nTest 1: Empty dataset")
    RTs_empty = []
    if test_approaches_match(RTs_empty):
        print("  ✓ PASS: Both return empty dict")
    else:
        print("  ✗ FAIL: Results differ on empty dataset")
        all_passed = False

    # Test 2: Single retweet
    print("\nTest 2: Single retweet")
    RTs_single = [(0, 0, datetime(2024, 1, 1, 12, 0, 0))]
    if test_approaches_match(RTs_single):
        print("  ✓ PASS: Both return empty dict (no pairs possible)")
    else:
        print("  ✗ FAIL: Results differ on single retweet")
        all_passed = False

    # Test 3: Two users, same tweet, within window
    print("\nTest 3: Two users, same tweet, within window")
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    RTs_pair = [
        (0, 0, base_time),
        (1, 0, base_time + timedelta(seconds=30))
    ]
    if test_approaches_match(RTs_pair):
        print("  ✓ PASS: Both detect the pair")
    else:
        print("  ✗ FAIL: Results differ on simple pair")
        all_passed = False

    # Test 4: Two users, same tweet, outside window
    print("\nTest 4: Two users, same tweet, outside window")
    RTs_no_pair = [
        (0, 0, base_time),
        (1, 0, base_time + timedelta(seconds=120))
    ]
    if test_approaches_match(RTs_no_pair, window_sec=60):
        print("  ✓ PASS: Both return empty (outside window)")
    else:
        print("  ✗ FAIL: Results differ on out-of-window pair")
        all_passed = False

    # Test 5: Same user retweeting same tweet (should be excluded)
    print("\nTest 5: Same user, same tweet (should be excluded)")
    RTs_same_user = [
        (0, 0, base_time),
        (0, 0, base_time + timedelta(seconds=30))
    ]
    if test_approaches_match(RTs_same_user):
        print("  ✓ PASS: Both exclude same-user pairs")
    else:
        print("  ✗ FAIL: Results differ on same-user scenario")
        all_passed = False

    # Test 6: Multiple co-retweets from same pair
    print("\nTest 6: Multiple co-retweets from same pair")
    RTs_multiple = [
        (0, 0, base_time),
        (1, 0, base_time + timedelta(seconds=30)),
        (0, 1, base_time + timedelta(seconds=100)),
        (1, 1, base_time + timedelta(seconds=130)),
    ]
    standard = CoRetweetsApproach(60, 1)
    numpy_opt = CoRetweetsNumpyApproach(60, 1)
    result_std = standard.compute_pairs_scores(RTs_multiple)
    result_np = numpy_opt.compute_pairs_scores(RTs_multiple)

    if result_std == result_np:
        pair_count = result_std.get((0, 1), 0)
        print(f"  ✓ PASS: Both count {pair_count} co-retweets for pair (0,1)")
    else:
        print("  ✗ FAIL: Results differ on multiple co-retweets")
        print(f"    Standard: {result_std}")
        print(f"    NumPy:    {result_np}")
        all_passed = False

    # Test 7: min_coactions filtering
    print("\nTest 7: min_coactions filtering")
    RTs_filter = [
        (0, 0, base_time),
        (1, 0, base_time + timedelta(seconds=30)),
        (2, 1, base_time + timedelta(seconds=100)),
        (3, 1, base_time + timedelta(seconds=130)),
        (2, 2, base_time + timedelta(seconds=200)),
        (3, 2, base_time + timedelta(seconds=230)),
    ]
    if test_approaches_match(RTs_filter, min_coactions=2):
        print("  ✓ PASS: Both filter by min_coactions correctly")
    else:
        print("  ✗ FAIL: Results differ on min_coactions filtering")
        all_passed = False

    # Test 8: Random synthetic data
    print("\nTest 8: Random synthetic data")
    random.seed(42)  # Reproducible
    RTs_random = generate_test_data(num_users=10, num_tweets=5, num_rts=100)

    if test_approaches_match(RTs_random):
        standard = CoRetweetsApproach(60, 1)
        result = standard.compute_pairs_scores(RTs_random)
        print(f"  ✓ PASS: Both produce {len(result)} pairs on random data")
    else:
        print("  ✗ FAIL: Results differ on random synthetic data")
        all_passed = False

    # Test 9: Large random dataset
    print("\nTest 9: Large random dataset (stress test)")
    random.seed(123)
    RTs_large = generate_test_data(num_users=50, num_tweets=20, num_rts=1000)

    if test_approaches_match(RTs_large):
        standard = CoRetweetsApproach(60, 1)
        result = standard.compute_pairs_scores(RTs_large)
        print(f"  ✓ PASS: Both produce {len(result)} pairs on large dataset")
    else:
        print("  ✗ FAIL: Results differ on large dataset")
        all_passed = False

    # Test 10: Edge case - boundary time windows
    print("\nTest 10: Boundary time windows")
    RTs_boundary = [
        (0, 0, base_time),
        (1, 0, base_time + timedelta(seconds=59.9)),  # Just within
        (2, 0, base_time + timedelta(seconds=60.1)),  # Just outside
    ]
    standard = CoRetweetsApproach(60, 1)
    numpy_opt = CoRetweetsNumpyApproach(60, 1)
    result_std = standard.compute_pairs_scores(RTs_boundary)
    result_np = numpy_opt.compute_pairs_scores(RTs_boundary)

    if result_std == result_np:
        print(f"  ✓ PASS: Both handle boundary correctly ({len(result_std)} pairs)")
    else:
        print("  ✗ FAIL: Results differ on boundary cases")
        print(f"    Standard: {result_std}")
        print(f"    NumPy:    {result_np}")
        all_passed = False

    # Summary
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("CoRetweetsNumpyApproach produces identical results to standard approach.")
    else:
        print("✗ SOME TESTS FAILED")
        print("There may be correctness issues with the NumPy implementation.")
    print("=" * 80)

    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
