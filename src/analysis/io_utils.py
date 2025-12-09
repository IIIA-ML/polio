"""
Input/Output utilities for experiment analysis.

This module provides functions for loading experiment configurations and results:
- Load experiment configuration from JSON
- Load result files from experiment directories
"""

import json
import pickle
from pathlib import Path
from collections import defaultdict


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

    return config


def load_all_results(results_dir, approach_keys, approach_storage_keys):
    """
    Load all result files from the results directory.

    Args:
        results_dir: Path to results directory
        approach_keys: List of approach keys from config (for result dict keys)
        approach_storage_keys: List of storage keys (for filenames)

    Returns:
        dict: {dataset_name: {approach_key: results_dict}}
    """
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"ERROR: Results directory {results_dir} does not exist.")
        return {}

    all_results = defaultdict(dict)

    # Scan for dataset directories
    for dataset_dir in results_path.iterdir():
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name

        # Load each approach file using storage key for filename
        for approach_key, storage_key in zip(approach_keys, approach_storage_keys):
            pkl_file = dataset_dir / f"{storage_key}.pkl"

            if pkl_file.exists():
                try:
                    with open(pkl_file, 'rb') as f:
                        all_results[dataset_name][approach_key] = pickle.load(f)
                except Exception as e:
                    print(f"WARNING: Failed to load {pkl_file}: {e}")

    return dict(all_results)
