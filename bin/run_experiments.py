#!/usr/bin/env python3
"""
Run experiments for detecting coordinated inauthentic behavior.

This script processes datasets and computes detection scores using multiple approaches.
Results are stored in pickle files (one per dataset+approach) to avoid reprocessing.
"""

import os
import sys
import pickle
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from synchronous_repeated_detection import import_data, filter_RTs
from approaches import ApproachFactory, Approach, CoRetweetsApproach


# Default datasets to process
DEFAULT_DATASETS = (
    'Armenia', 'Catalonia', 'Ghana_Nigeria', 'Iran_5', 'Russia_3', 'Spain',
    'Venezuela_2', 'Thailand', 'Ecuador', 'Iran_1', 'Iran_6', 'Russia_5',
    'Qatar', 'Russia_2', 'China_1', 'China_2', 'Russia_1', 'Russia_4',
    'Iran_2', 'Iran_3', 'Iran_4'
)


def get_result_path(output_dir, dataset, approach_key):
    """Get the path for storing results for a dataset+approach."""
    dataset_dir = Path(output_dir) / dataset
    return dataset_dir / f"{approach_key}.pkl"


def is_approach_processed(output_dir, dataset, approach_key):
    """Check if a dataset+approach has already been processed."""
    result_path = get_result_path(output_dir, dataset, approach_key)
    return result_path.exists()


def save_approach_results(output_dir, dataset, approach_key, results):
    """Save results for a dataset+approach."""
    result_path = get_result_path(output_dir, dataset, approach_key)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    with open(result_path, 'wb') as f:
        pickle.dump(results, f)


def load_approach_results(output_dir, dataset, approach_key):
    """Load results for a dataset+approach."""
    result_path = get_result_path(output_dir, dataset, approach_key)

    with open(result_path, 'rb') as f:
        return pickle.load(f)


def process_dataset_approach(dataset: str, approach: Approach,
                            data_dir: str, output_dir: str, force: bool = False):
    """
    Process a single dataset with a specific approach.

    Args:
        dataset: Dataset name
        approach: Approach instance to use
        data_dir: Directory containing datasets
        output_dir: Directory to save results
        force: Force reprocessing even if results exist

    Returns:
        True if processed, False if skipped, None if error/missing
    """
    approach_key = approach.get_approach_key()

    # Check if already processed
    if not force and is_approach_processed(output_dir, dataset, approach_key):
        return False

    processed_dir = f"{data_dir}/{dataset}/Processed/"
    if not os.path.exists(processed_dir):
        return None

    # Import data
    RTs, io_users = import_data(processed_dir)

    # Prepare data based on approach needs
    if approach.needs_filtered_data():
        # Need to filter RTs based on coretweets first
        coretweets_approach = CoRetweetsApproach(approach.window_sec, approach.min_coactions)
        pairs_coretweets = coretweets_approach.compute_pairs_scores(RTs)

        if not pairs_coretweets:
            return None

        users_coretweeted = set().union(*pairs_coretweets)
        RTs_to_use = filter_RTs(RTs, users_coretweeted)
    else:
        RTs_to_use = RTs

    # Compute pairs scores using the approach
    pairs_scores = approach.compute_pairs_scores(RTs_to_use)

    if not pairs_scores:
        return None

    # Save results with metadata from approach
    results = {
        'dataset': dataset,
        'pairs_scores': pairs_scores,
        'io_users': io_users,
        'num_pairs': len(pairs_scores),
        **approach.get_metadata()  # Includes all approach configuration
    }

    save_approach_results(output_dir, dataset, approach_key, results)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run detection experiments on social media datasets"
    )
    parser.add_argument(
        '--data-dir',
        default='../data',
        help='Directory containing datasets (default: ../data)'
    )
    parser.add_argument(
        '--output-dir',
        default='results',
        help='Directory to store results (default: results)'
    )
    parser.add_argument(
        '--window-sec',
        type=int,
        default=60,
        help='Time window in seconds for synchronous actions (default: 60)'
    )
    parser.add_argument(
        '--min-coactions',
        type=int,
        default=1,
        help='Minimum co-retweets to consider a pair (default: 1)'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        help='Specific datasets to process (default: all)'
    )
    parser.add_argument(
        '--approaches',
        nargs='+',
        choices=ApproachFactory.get_all_keys(),
        help='Specific approaches to run (default: all)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Reprocess even if results exist'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all dataset+approach combinations and their status'
    )

    args = parser.parse_args()

    # Determine which datasets and approaches to process
    datasets = args.datasets if args.datasets else DEFAULT_DATASETS

    # Create approach instances
    if args.approaches:
        approaches = [ApproachFactory.create(key, args.window_sec, args.min_coactions)
                     for key in args.approaches]
    else:
        approaches = ApproachFactory.get_all_approaches(args.window_sec, args.min_coactions)

    # List mode
    if args.list:
        print(f"\nDataset+Approach status (output dir: {args.output_dir}):")
        print("=" * 80)

        for dataset in datasets:
            print(f"\n{dataset}:")
            for approach in approaches:
                processed = is_approach_processed(args.output_dir, dataset,
                                                 approach.get_approach_key())
                status = "✓ PROCESSED" if processed else "✗ NOT PROCESSED"
                print(f"  {approach.get_approach_name():40s} {status}")
        print()
        return

    # Process combinations
    print("=" * 80)
    print("RUNNING EXPERIMENTS")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Window: {args.window_sec} seconds")
    print(f"  Min co-actions: {args.min_coactions}")
    print(f"  Force reprocess: {args.force}")
    print(f"  Datasets: {len(datasets)}")
    print(f"  Approaches: {len(approaches)}")
    print(f"  Total combinations: {len(datasets) * len(approaches)}")

    processed_count = 0
    skipped_count = 0
    error_count = 0
    missing_count = 0

    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset}")
        print(f"{'='*80}")

        for approach in approaches:
            print(f"\n  Approach: {approach.get_approach_name()}")

            try:
                result = process_dataset_approach(
                    dataset, approach, args.data_dir, args.output_dir, args.force
                )

                if result is True:
                    print(f"    ✓ Processed and saved")
                    processed_count += 1
                elif result is False:
                    print(f"    ⊘ Skipped (already processed, use --force to reprocess)")
                    skipped_count += 1
                elif result is None:
                    print(f"    ✗ Skipped (missing data or no results)")
                    missing_count += 1

            except Exception as e:
                print(f"    ✗ ERROR: {e}")
                error_count += 1
                import traceback
                traceback.print_exc()

    # Summary
    total = len(datasets) * len(approaches)
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Processed: {processed_count}")
    print(f"  Skipped (already done): {skipped_count}")
    print(f"  Skipped (missing/no results): {missing_count}")
    print(f"  Errors: {error_count}")
    print(f"  Total combinations: {total}")
    print("=" * 80)


if __name__ == '__main__':
    main()
