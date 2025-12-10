#!/usr/bin/env python3
"""
Run experiments for detecting coordinated inauthentic behavior.

This script processes datasets and computes detection scores using multiple approaches.
Results are stored in pickle files (one per dataset+approach) to avoid reprocessing.

The script accepts a JSON configuration file that specifies all experiment parameters.
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

from synchronous_repeated_detection import import_data, filter_RTs
from approaches import ApproachFactory, Approach
from approaches.coretweets import CoRetweetsApproach


# Default datasets to process
DEFAULT_DATASETS = (
    'Armenia', 'Catalonia', 'Ghana_Nigeria', 'Iran_5', 'Russia_3', 'Spain',
    'Venezuela_2', 'Thailand', 'Ecuador', 'Iran_1', 'Iran_6', 'Russia_5',
    'Qatar', 'Russia_2', 'China_1', 'China_2', 'Russia_1', 'Russia_4',
    'Iran_2', 'Iran_3', 'Iran_4', 'UAE', 'Cuba'#, 'Egypt_UAE', 'Bangladesh', 'Venezuela_1'
)


def get_result_path(output_dir, dataset, approach):
    """Get the path for storing results for a dataset+approach."""
    # Results are stored in: {output_dir}/results/{dataset}/{approach}.pkl
    # Use full key which includes ranking mode if not default
    if isinstance(approach, str):
        approach_key = approach
    else:
        approach_key = approach.get_full_approach_key()
    
    dataset_dir = Path(output_dir) / "results" / dataset
    return dataset_dir / f"{approach_key}.pkl"


def is_approach_processed(output_dir, dataset, approach):
    """Check if a dataset+approach has already been processed."""
    result_path = get_result_path(output_dir, dataset, approach)
    return result_path.exists()


def save_approach_results(output_dir, dataset, approach, results):
    """Save results for a dataset+approach."""
    result_path = get_result_path(output_dir, dataset, approach)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    with open(result_path, 'wb') as f:
        pickle.dump(results, f)


def load_approach_results(output_dir, dataset, approach):
    """Load results for a dataset+approach."""
    result_path = get_result_path(output_dir, dataset, approach)

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
    # Check if already processed
    # Pass the approach object so it can use get_full_approach_key() which includes ranking mode
    if not force and is_approach_processed(output_dir, dataset, approach):
        return False

    processed_dir = f"{data_dir}/{dataset}/Processed/"
    if not os.path.exists(processed_dir):
        return None

    # Import data
    RTs, io_users = import_data(processed_dir)

    # Prepare data based on approach needs
    if approach.needs_filtered_data():
        # Need to filter RTs based on coretweets first (with transparent caching)
        coretweets_approach = CoRetweetsApproach(approach.window_sec, approach.min_coactions)
        pairs_coretweets = coretweets_approach.compute_pairs_scores(
            RTs, processed_dir=processed_dir
        )

        if not pairs_coretweets:
            return None

        users_coretweeted = set().union(*pairs_coretweets)
        RTs_to_use = filter_RTs(RTs, users_coretweeted)
    else:
        RTs_to_use = RTs

    # Compute suspicious users using the approach
    suspicious_users = approach.get_suspicious(RTs_to_use)

    if not suspicious_users:
        return None

    # Save results with metadata from approach
    results = {
        'dataset': dataset,
        'suspicious_users': suspicious_users,
        'io_users': io_users,
        'num_suspicious_groups': len(suspicious_users),
        'num_suspicious_users': sum(len(group) for group in suspicious_users),
        **approach.get_metadata()  # Includes all approach configuration
    }

    save_approach_results(output_dir, dataset, approach, results)
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
    config.setdefault('min_coactions', 1)
    config.setdefault('ranking_mode', 'L2')  # Default to max (current behavior)
    config.setdefault('datasets', None)  # None means all datasets
    config.setdefault('approaches', None)  # None means all approaches
    config.setdefault('force', False)

    # Validate ranking_mode (should be string - per-approach modes are now in approach keys)
    ranking_mode = config['ranking_mode']
    if not isinstance(ranking_mode, str):
        raise ValueError(f"ranking_mode must be a string ('L1', 'L2', or 'Linf'), got {type(ranking_mode).__name__}")
    if ranking_mode not in ('L1', 'L2', 'Linf'):
        raise ValueError(f"Invalid ranking_mode: {ranking_mode}. Must be 'L1', 'L2', or 'Linf'")

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
        epilog="""
Example configuration file (experiment.json):
{
  "name": "baseline_experiment",
  "data_dir": "../data",
  "output_dir": null,
  "window_sec": 60,
  "min_coactions": 1,
  "ranking_mode": "Linf",
  "datasets": ["Armenia", "Thailand"],
  "approaches": ["coretweets", "ignoring_tweet_fast"],
  "force": false
}

Example with per-approach ranking modes (embedded in approach keys):
{
  "name": "compare_ranking_modes",
  "ranking_mode": "L1",
  "approaches": [
    "coretweets[L1]",
    "coretweets[L2]",
    "coretweets[Linf]",
    "lexicographic:coretweets[L1]+ignoring_tweet_fast[Linf]"
  ],
  "datasets": ["Armenia"]
}

Notes:
- If output_dir is null, an experiment folder will be created in the same directory
  as the JSON file with structure: {json_dir}/{name}/results/{dataset}/
- If datasets is null or omitted, all datasets will be processed
- If approaches is null or omitted, all approaches will be used
- ranking_mode is a default for approaches without explicit mode in their key
- Specify ranking mode in approach key: "approach[mode]" or 
  "lexicographic:approach1[mode1]+approach2[mode2]+..."
- Examples: "coretweets[L1]", "lexicographic:coretweets[L1]+ignoring_tweet_fast[Linf]"
        """
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
    min_coactions = config['min_coactions']
    force = config['force']

    # Set output directory based on JSON file location if not specified
    if config['output_dir'] is None:
        config_path = Path(args.config).resolve()
        json_dir = config_path.parent
        output_dir = str(json_dir / config['name'])
    else:
        output_dir = config['output_dir']

    # Create experiment directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create approach instances
    # Ranking modes can now be specified in approach keys like "coretweets[sum]"
    ranking_mode = config['ranking_mode']
    
    if config['approaches'] is not None:
        approaches = [ApproachFactory.create(key, window_sec, min_coactions, ranking_mode)
                     for key in config['approaches']]
    else:
        approaches = ApproachFactory.get_all_approaches(window_sec, min_coactions, ranking_mode)

    # List mode
    if args.list:
        print(f"\nExperiment: {config['name']}")
        print(f"Dataset+Approach status (output dir: {output_dir}):")
        print("=" * 80)

        for dataset in datasets:
            print(f"\n{dataset}:")
            for approach in approaches:
                processed = is_approach_processed(output_dir, dataset, approach)
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
    print(f"  Min co-actions: {min_coactions}")
    print(f"  Default ranking mode: {ranking_mode}")
    print(f"  Force reprocess: {force}")
    print(f"  Datasets: {len(datasets)}")
    print(f"  Approaches: {len(approaches)}")
    for approach in approaches:
        print(f"    {approach.get_approach_name()}: ranking_mode={approach.ranking_mode}")
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
                    dataset, approach, data_dir, output_dir, force
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
