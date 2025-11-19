#!/usr/bin/env python3
"""
Convert text-based datasets to pickle format for faster loading.

This script scans all datasets in the data directory and creates pickle caches
for each one. Subsequent loads will be 10-100x faster.

Usage:
    uv run bin/convert_to_pickle.py [--data-dir DATA_DIR] [--force]
"""

import argparse
import sys
from pathlib import Path
import time

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from data_loader import import_data, is_cache_valid


# Default datasets (same as run_experiments.py)
DEFAULT_DATASETS = (
    'Armenia', 'Catalonia', 'Ghana_Nigeria', 'Iran_5', 'Russia_3', 'Spain',
    'Venezuela_2', 'Thailand', 'Ecuador', 'Iran_1', 'Iran_6', 'Russia_5',
    'Qatar', 'Russia_2', 'China_1', 'China_2', 'Russia_1', 'Russia_4',
    'Iran_2', 'Iran_3', 'Iran_4'
)


def convert_dataset(data_dir, dataset_name, force=False):
    """
    Convert a single dataset to pickle format.

    Args:
        data_dir: Base data directory
        dataset_name: Name of dataset to convert
        force: Force reconversion even if cache exists

    Returns:
        Tuple of (success: bool, time_saved: float, message: str)
    """
    processed_dir = Path(data_dir) / dataset_name / "Processed"

    if not processed_dir.exists():
        return False, 0.0, "Directory not found"

    # Check required files exist
    required_files = ["data.txt", "io_users.txt", "RTs.txt"]
    missing = [f for f in required_files if not (processed_dir / f).exists()]
    if missing:
        return False, 0.0, f"Missing files: {', '.join(missing)}"

    pickle_file = processed_dir / "data_cache.pkl"

    # Check if already converted
    if not force and pickle_file.exists():
        return True, 0.0, "Already cached (use --force to reconvert)"

    try:
        # Time the text loading
        start_time = time.time()
        RTs, io_users = import_data(processed_dir, use_cache=False)
        text_load_time = time.time() - start_time

        # Force create the pickle cache
        import_data(processed_dir, use_cache=True, force_reload=True)

        # Time the pickle loading
        start_time = time.time()
        import_data(processed_dir, use_cache=True)
        pickle_load_time = time.time() - start_time

        speedup = text_load_time / pickle_load_time if pickle_load_time > 0 else 0
        file_size_mb = pickle_file.stat().st_size / (1024 * 1024)

        message = (
            f"✓ Converted ({len(RTs):,} RTs, {len(io_users)} IO users)\n"
            f"      Text load: {text_load_time:.2f}s, Pickle load: {pickle_load_time:.3f}s\n"
            f"      Speedup: {speedup:.1f}x, Cache size: {file_size_mb:.2f} MB"
        )

        return True, text_load_time - pickle_load_time, message

    except Exception as e:
        return False, 0.0, f"Error: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Convert text datasets to pickle format for faster loading"
    )
    parser.add_argument(
        '--data-dir',
        default='data',
        help='Directory containing datasets (default: data)'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        help='Specific datasets to convert (default: all)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force reconversion even if cache exists'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List datasets and their cache status'
    )

    args = parser.parse_args()

    datasets = args.datasets if args.datasets else DEFAULT_DATASETS
    data_dir = Path(args.data_dir)

    print("=" * 80)
    print("DATASET PICKLE CONVERSION")
    print("=" * 80)
    print(f"\nData directory: {data_dir.absolute()}")
    print(f"Datasets to process: {len(datasets)}")
    print(f"Force reconvert: {args.force}")

    # List mode
    if args.list:
        print("\nDataset cache status:")
        print("-" * 80)

        for dataset in datasets:
            processed_dir = data_dir / dataset / "Processed"
            pickle_file = processed_dir / "data_cache.pkl"

            if not processed_dir.exists():
                status = "✗ NOT FOUND"
                size = ""
            elif pickle_file.exists():
                size_mb = pickle_file.stat().st_size / (1024 * 1024)
                status = "✓ CACHED"
                size = f" ({size_mb:.2f} MB)"
            else:
                status = "○ NOT CACHED"
                size = ""

            print(f"  {dataset:30s} {status}{size}")

        print()
        return

    # Convert datasets
    print("\nConverting datasets...")
    print("=" * 80)

    total_converted = 0
    total_skipped = 0
    total_failed = 0
    total_time_saved = 0.0

    for dataset in datasets:
        print(f"\n{dataset}:")

        success, time_saved, message = convert_dataset(
            data_dir, dataset, force=args.force
        )

        print(f"  {message}")

        if success:
            if "Already cached" in message:
                total_skipped += 1
            else:
                total_converted += 1
                total_time_saved += time_saved
        else:
            total_failed += 1

    # Summary
    print("\n" + "=" * 80)
    print("CONVERSION SUMMARY")
    print("=" * 80)
    print(f"  Converted: {total_converted}")
    print(f"  Skipped (already cached): {total_skipped}")
    print(f"  Failed: {total_failed}")
    print(f"  Total datasets: {len(datasets)}")

    if total_converted > 0:
        print(f"\n  Estimated time saved per run: {total_time_saved:.1f}s")
        print(f"  (This adds up quickly when processing multiple approaches!)")

    print("=" * 80)
    print("\nNext steps:")
    print("  - Run experiments with: uv run bin/run_experiments.py")
    print("  - Data will now load much faster from pickle cache")
    print("  - Caches are automatically invalidated if source files change")
    print("=" * 80)


if __name__ == '__main__':
    main()
