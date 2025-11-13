#!/usr/bin/env python3
"""
Benchmark comparison between standard and NumPy-optimized co-retweet detection.

This script compares the performance of CoRetweetsApproach vs CoRetweetsNumpyApproach
on real datasets to demonstrate the speedup from NumPy vectorization.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from approaches import CoRetweetsApproach, CoRetweetsNumpyApproach
from synchronous_repeated_detection import import_data


def benchmark_approach(approach, RTs, num_runs=3):
    """
    Benchmark an approach multiple times and return average time.

    Args:
        approach: Approach instance to benchmark
        RTs: List of retweets
        num_runs: Number of runs to average

    Returns:
        Tuple of (avg_time, result)
    """
    times = []
    result = None

    for i in range(num_runs):
        start_time = time.time()
        result = approach.compute_pairs_scores(RTs)
        elapsed = time.time() - start_time
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    return avg_time, result


def compare_approaches(processed_dir, window_sec=60, min_coactions=1, num_runs=3):
    """
    Compare standard vs NumPy approaches on a dataset.

    Args:
        processed_dir: Path to processed dataset directory
        window_sec: Time window in seconds
        min_coactions: Minimum co-actions threshold
        num_runs: Number of runs to average
    """
    processed_dir = Path(processed_dir)

    if not processed_dir.exists():
        print(f"Error: Directory not found: {processed_dir}")
        print("\nPlease provide a valid dataset path.")
        print("Example: uv run examples/benchmark_coretweets.py ../data/Armenia/Processed/")
        return

    print("=" * 80)
    print("CO-RETWEETS PERFORMANCE BENCHMARK")
    print("=" * 80)
    print(f"\nDataset: {processed_dir.parent.name}")
    print(f"Path: {processed_dir.absolute()}")
    print(f"Parameters: window_sec={window_sec}, min_coactions={min_coactions}")
    print(f"Number of runs: {num_runs}")

    # Load data
    print("\n" + "-" * 80)
    print("Loading dataset...")
    print("-" * 80)

    start_time = time.time()
    RTs, io_users = import_data(processed_dir)
    load_time = time.time() - start_time

    print(f"Loaded {len(RTs):,} retweets, {len(io_users)} IO users")
    print(f"Load time: {load_time:.3f} seconds")

    # Create approach instances
    standard_approach = CoRetweetsApproach(window_sec, min_coactions)
    numpy_approach = CoRetweetsNumpyApproach(window_sec, min_coactions)

    # Benchmark standard approach
    print("\n" + "-" * 80)
    print("Benchmark 1: Standard CoRetweetsApproach")
    print("-" * 80)

    standard_time, standard_result = benchmark_approach(standard_approach, RTs, num_runs)

    print(f"Found {len(standard_result):,} user pairs")
    print(f"Average time: {standard_time:.4f} seconds")

    # Benchmark NumPy approach
    print("\n" + "-" * 80)
    print("Benchmark 2: NumPy CoRetweetsNumpyApproach")
    print("-" * 80)

    numpy_time, numpy_result = benchmark_approach(numpy_approach, RTs, num_runs)

    print(f"Found {len(numpy_result):,} user pairs")
    print(f"Average time: {numpy_time:.4f} seconds")

    # Verify results match
    print("\n" + "-" * 80)
    print("Verification")
    print("-" * 80)

    if standard_result == numpy_result:
        print("✓ Results match perfectly!")
    else:
        print("✗ WARNING: Results differ!")
        print(f"  Standard pairs: {len(standard_result)}")
        print(f"  NumPy pairs: {len(numpy_result)}")

        # Check for differences
        only_in_standard = set(standard_result.keys()) - set(numpy_result.keys())
        only_in_numpy = set(numpy_result.keys()) - set(standard_result.keys())

        if only_in_standard:
            print(f"  Pairs only in standard: {len(only_in_standard)}")
        if only_in_numpy:
            print(f"  Pairs only in NumPy: {len(only_in_numpy)}")

    # Performance comparison
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)

    speedup = standard_time / numpy_time if numpy_time > 0 else 0
    time_saved = standard_time - numpy_time
    percent_faster = ((standard_time - numpy_time) / standard_time * 100) if standard_time > 0 else 0

    print(f"\nStandard approach:  {standard_time:.4f} seconds")
    print(f"NumPy approach:     {numpy_time:.4f} seconds")
    print(f"\nSpeedup:            {speedup:.2f}x")
    print(f"Time saved:         {time_saved:.4f} seconds ({percent_faster:.1f}% faster)")

    # Analyze dataset characteristics
    print("\n" + "-" * 80)
    print("Dataset Characteristics")
    print("-" * 80)

    # Group by tweet to analyze retweet distribution
    from collections import defaultdict
    tweet_data = defaultdict(list)
    for user, tid, ts in RTs:
        tweet_data[tid].append((user, ts))

    retweet_counts = [len(events) for events in tweet_data.values()]
    avg_retweets = sum(retweet_counts) / len(retweet_counts) if retweet_counts else 0
    max_retweets = max(retweet_counts) if retweet_counts else 0

    print(f"Unique tweets: {len(tweet_data):,}")
    print(f"Average retweets per tweet: {avg_retweets:.1f}")
    print(f"Max retweets per tweet: {max_retweets:,}")

    print("\n💡 Performance notes:")
    if speedup > 2.0:
        print("   Excellent speedup! NumPy vectorization is highly effective on this dataset.")
    elif speedup > 1.5:
        print("   Good speedup! NumPy provides noticeable performance improvement.")
    elif speedup > 1.0:
        print("   Modest speedup. Dataset may be small or have few viral tweets.")
    else:
        print("   Minimal or negative speedup. Standard approach may be better for this case.")

    if max_retweets > 1000:
        print("   Dataset has viral tweets - NumPy benefits from vectorization.")
    if len(RTs) > 100000:
        print("   Large dataset - NumPy's efficiency scales well.")

    print("\n" + "=" * 80)


def benchmark_multiple_datasets(data_dir, datasets, **kwargs):
    """
    Benchmark multiple datasets and show aggregate results.

    Args:
        data_dir: Base data directory
        datasets: List of dataset names
        **kwargs: Additional arguments for compare_approaches
    """
    results = []

    print("=" * 80)
    print("MULTI-DATASET BENCHMARK")
    print("=" * 80)
    print(f"\nBenchmarking {len(datasets)} datasets...")

    for dataset in datasets:
        processed_dir = Path(data_dir) / dataset / "Processed"
        if not processed_dir.exists():
            print(f"\n⊘ Skipping {dataset}: directory not found")
            continue

        print(f"\n{'=' * 80}")
        print(f"Dataset: {dataset}")
        print(f"{'=' * 80}")

        try:
            # Load data
            RTs, io_users = import_data(processed_dir)

            # Benchmark both approaches
            standard_approach = CoRetweetsApproach(**kwargs)
            numpy_approach = CoRetweetsNumpyApproach(**kwargs)

            standard_time, _ = benchmark_approach(standard_approach, RTs, num_runs=3)
            numpy_time, _ = benchmark_approach(numpy_approach, RTs, num_runs=3)

            speedup = standard_time / numpy_time if numpy_time > 0 else 0

            results.append({
                'dataset': dataset,
                'num_rts': len(RTs),
                'standard_time': standard_time,
                'numpy_time': numpy_time,
                'speedup': speedup
            })

            print(f"\n  Standard: {standard_time:.4f}s | NumPy: {numpy_time:.4f}s | Speedup: {speedup:.2f}x")

        except Exception as e:
            print(f"\n✗ Error processing {dataset}: {e}")

    # Summary
    if results:
        print("\n" + "=" * 80)
        print("AGGREGATE RESULTS")
        print("=" * 80)

        avg_speedup = sum(r['speedup'] for r in results) / len(results)
        total_standard = sum(r['standard_time'] for r in results)
        total_numpy = sum(r['numpy_time'] for r in results)

        print(f"\nDatasets processed: {len(results)}")
        print(f"Average speedup: {avg_speedup:.2f}x")
        print(f"Total time (standard): {total_standard:.2f}s")
        print(f"Total time (NumPy): {total_numpy:.2f}s")
        print(f"Total time saved: {total_standard - total_numpy:.2f}s")

        # Best and worst
        best = max(results, key=lambda r: r['speedup'])
        worst = min(results, key=lambda r: r['speedup'])

        print(f"\nBest speedup: {best['dataset']} ({best['speedup']:.2f}x)")
        print(f"Worst speedup: {worst['dataset']} ({worst['speedup']:.2f}x)")

        print("=" * 80)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]

        # Check if this is a multi-dataset benchmark
        if dataset_path == '--multi':
            data_dir = sys.argv[2] if len(sys.argv) > 2 else '../data'
            datasets = ['Armenia', 'Catalonia', 'Spain', 'Thailand']
            benchmark_multiple_datasets(data_dir, datasets, window_sec=60, min_coactions=1)
        else:
            compare_approaches(dataset_path, window_sec=60, min_coactions=1, num_runs=3)
    else:
        # Try a default path
        dataset_path = "../data/Armenia/Processed/"
        print("No dataset path provided. Using default:", dataset_path)
        print("Usage: uv run examples/benchmark_coretweets.py <dataset_path>")
        print("   or: uv run examples/benchmark_coretweets.py --multi [data_dir]\n")
        compare_approaches(dataset_path, window_sec=60, min_coactions=1, num_runs=3)
