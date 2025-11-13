#!/usr/bin/env python3
"""
Test pickle caching performance improvement.

This script demonstrates the speed improvement from using pickle caching.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from synchronous_repeated_detection import import_data


def test_loading_performance(processed_dir):
    """
    Test loading performance with and without pickle cache.

    Args:
        processed_dir: Path to a processed dataset directory
    """
    processed_dir = Path(processed_dir)

    if not processed_dir.exists():
        print(f"Error: Directory not found: {processed_dir}")
        print("\nPlease provide a valid dataset path.")
        print("Example: python examples/test_pickle_performance.py ../data/Armenia/Processed/")
        return

    print("=" * 80)
    print("PICKLE CACHING PERFORMANCE TEST")
    print("=" * 80)
    print(f"\nDataset: {processed_dir.parent.name}")
    print(f"Path: {processed_dir.absolute()}")

    # Test 1: Load from text files (no cache)
    print("\n" + "-" * 80)
    print("Test 1: Loading from text files (first time, no cache)")
    print("-" * 80)

    start_time = time.time()
    RTs, io_users = import_data(processed_dir, use_cache=False)
    text_load_time = time.time() - start_time

    print(f"Loaded {len(RTs):,} retweets, {len(io_users)} IO users")
    print(f"Time: {text_load_time:.3f} seconds")

    # Test 2: Create pickle cache
    print("\n" + "-" * 80)
    print("Test 2: Creating pickle cache")
    print("-" * 80)

    cache_file = processed_dir / "data_cache.pkl"
    if cache_file.exists():
        cache_file.unlink()  # Remove old cache

    start_time = time.time()
    import_data(processed_dir, use_cache=True, force_reload=True)
    cache_creation_time = time.time() - start_time

    cache_size_mb = cache_file.stat().st_size / (1024 * 1024)
    print(f"Cache created: {cache_file.name}")
    print(f"Cache size: {cache_size_mb:.2f} MB")
    print(f"Time: {cache_creation_time:.3f} seconds")

    # Test 3: Load from pickle cache
    print("\n" + "-" * 80)
    print("Test 3: Loading from pickle cache")
    print("-" * 80)

    # Run multiple times to get consistent timing
    times = []
    for i in range(3):
        start_time = time.time()
        RTs_cached, io_users_cached = import_data(processed_dir, use_cache=True)
        pickle_load_time = time.time() - start_time
        times.append(pickle_load_time)

    avg_pickle_time = sum(times) / len(times)
    print(f"Loaded {len(RTs_cached):,} retweets, {len(io_users_cached)} IO users")
    print(f"Average time (3 runs): {avg_pickle_time:.4f} seconds")
    print(f"Individual times: {', '.join(f'{t:.4f}s' for t in times)}")

    # Verify data integrity
    assert RTs == RTs_cached, "Data mismatch!"
    assert io_users == io_users_cached, "IO users mismatch!"

    # Results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    speedup = text_load_time / avg_pickle_time
    time_saved = text_load_time - avg_pickle_time

    print(f"\nText file loading:  {text_load_time:.3f} seconds")
    print(f"Pickle loading:     {avg_pickle_time:.4f} seconds")
    print(f"\nSpeedup:            {speedup:.1f}x faster")
    print(f"Time saved:         {time_saved:.3f} seconds per load")

    print(f"\n💡 Impact when processing multiple approaches:")
    for num_approaches in [4, 10, 20]:
        total_saved = time_saved * num_approaches
        print(f"   {num_approaches} approaches: ~{total_saved:.1f}s saved")

    print("\n" + "=" * 80)
    print("Cache will be automatically used for subsequent loads!")
    print("Cache is invalidated if source files are modified.")
    print("=" * 80)


if __name__ == '__main__':
    # Example usage
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        # Try a default path
        dataset_path = "../data/Armenia/Processed/"

    test_loading_performance(dataset_path)
