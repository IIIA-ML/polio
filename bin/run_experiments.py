#!/usr/bin/env python3
"""
Run experiments for detecting coordinated inauthentic behavior.

This script processes datasets and computes detection scores using multiple approaches.
Results are stored in pickle files (one per dataset+approach) to avoid reprocessing.

The script accepts a JSON configuration file that specifies all experiment parameters.

Usage:
    ./bin/run_experiments.py <config.json>      # Run experiments from config

Configuration file format (JSON) example in experiments/
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from synchronous_repeated_detection import import_data
from approaches import ApproachFactory, Approach


# Default datasets to process
DEFAULT_DATASETS = (
    'Armenia', 'Catalonia', 'Ghana_Nigeria', 'Iran_5', 'Russia_3', 'Spain',
    'Venezuela_2', 'Thailand', 'Ecuador', 'Iran_1', 'Iran_6', 'Russia_5',
    'Qatar', 'Russia_2', 'China_1', 'China_2', 'Russia_1', 'Russia_4',
    'Iran_2', 'Iran_3', 'Iran_4', 'UAE', 'Cuba'#, 'Egypt_UAE', 'Bangladesh', 'Venezuela_1'
)


def process_dataset_approach(dataset: str, approach: Approach,
                            data_dir: str, output_dir: str, force: bool = False,
                            filter_min_coactions: int | None = None):
    """
    Process a single dataset with a specific approach.

    Args:
        dataset: Dataset name
        approach: Approach instance to use
        data_dir: Directory containing datasets
        output_dir: Directory to save results
        force: Force reprocessing even if results exist
        filter_min_coactions: Minimum coactions to filter suspicious users
    Returns:
        True if processed, False if skipped, None if error/missing
    """
    # Check if already processed (unless force is True)
    if not force and approach.is_result_cached(output_dir, dataset):
        return False

    processed_dir = f"{data_dir}/{dataset}/Processed/"
    if not os.path.exists(processed_dir):
        return None

    # Import data
    RTs, io_users = import_data(processed_dir)

    # Compute suspicious users (caching is handled transparently by get_suspicious)
    suspicious_users = approach.get_suspicious(
        RTs,
        output_dir=output_dir,
        dataset=dataset,
        io_users=io_users,
        filter_min_coactions=filter_min_coactions,
        data_dir=data_dir,
        force=force,
    )

    if not suspicious_users:
        return None

    return True


def load_experiment_config(config_path):
    """
    Load experiment configuration from JSON file.

    Args:
        config_path: Path to JSON configuration file

    Returns:
        Dictionary with experiment configuration

    Raises:
        ValueError: If configuration is invalid
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Validate required fields
    if 'name' not in config:
        raise ValueError("Configuration must include 'name' field")

    # Set defaults
    config.setdefault('data_dir', 'data')
    config.setdefault('output_dir', None)  # Will be set based on JSON location if not specified
    config.setdefault('window_sec', 60)
    config.setdefault('datasets', None)  # None means all datasets
    config.setdefault('approaches', None)  # None means all approaches
    config.setdefault('force', False)
    # Optional override for the min_coactions used by coretweets filter step
    config.setdefault('filter_min_coactions', None)

    # Convert null datasets to all datasets
    if config['datasets'] is None:
        config['datasets'] = list(DEFAULT_DATASETS)

    # Validate approaches if specified
    if config['approaches'] is not None:
        for approach in config['approaches']:
            ApproachFactory.validate_approach_key(approach)

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Run detection experiments on social media datasets using JSON configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Example configuration file (experiment.json) in experiments/."""
    )
    parser.add_argument(
        'config',
        help='Path to JSON configuration file'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all dataset+approach combinations and their status'
    )

    args = parser.parse_args()

    # Load configuration from JSON file
    try:
        config = load_experiment_config(args.config)
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: Invalid configuration: {e}")
        sys.exit(1)

    # Extract configuration
    datasets = config['datasets']
    data_dir = config['data_dir']
    window_sec = config['window_sec']
    force = config['force']

    # Set output directory based on JSON file location if not specified
    if config['output_dir'] is None:
        config_path = Path(args.config).resolve()
        json_dir = config_path.parent
        output_dir = str(json_dir / config['name'])
    else:
        output_dir = Path(config['output_dir']) / config['name']

    # Create experiment directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create approach instances
    # All parameters (min_coactions, ranking_mode) are now specified per-approach
    
    if config['approaches'] is not None:
        approaches = [ApproachFactory.create(key, window_sec)
                     for key in config['approaches']]
    else:
        approaches = ApproachFactory.get_all_approaches(window_sec, 1, 'L2')

    # List mode
    if args.list:
        print(f"\nExperiment: {config['name']}")
        print(f"Dataset+Approach status (output dir: {output_dir}):")
        print("=" * 80)

        for dataset in datasets:
            print(f"\n{dataset}:")
            for approach in approaches:
                processed = approach.is_result_cached(output_dir, dataset)
                status = "✓ PROCESSED" if processed else "✗ NOT PROCESSED"
                mode_info = f" ({approach.ranking_mode})" if approach.ranking_mode != 'max' else ""
                print(f"  {approach.get_approach_name():40s}{mode_info:10s} {status}")
        print()
        return

    # Process combinations
    print("=" * 80)
    print("RUNNING EXPERIMENTS")
    print("=" * 80)
    print(f"\nExperiment name: {config['name']}")
    print(f"\nConfiguration:")
    print(f"  Data directory: {data_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Window: {window_sec} seconds")
    print(f"  Force reprocess: {force}")
    print(f"  Datasets: {len(datasets)}")
    print(f"  Approaches: {len(approaches)}")
    for approach in approaches:
        print(f"    {approach.get_approach_name()}: ranking_mode={approach.ranking_mode}, min_coactions={approach.min_coactions}")
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
                    dataset, approach, data_dir, output_dir, force,
                    filter_min_coactions=config.get('filter_min_coactions')
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
