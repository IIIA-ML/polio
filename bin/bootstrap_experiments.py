#!/usr/bin/env python3
"""
Subsampling evaluation for experiment configurations.

For each dataset in a configuration, this script:
1) Loads RTs and IO users
2) Randomly samples from all users with replacement
3) Computes approach rankings on each subsample
4) Computes metrics (AUC, NDCG, AP) directly (no caching)
5) Writes per-metric CSVs with approach columns
6) Computes percentile confidence intervals from CSVs

Outputs:
  experiments/{experiment_name}/analysis/bootstrap/
    {dataset}_auc.csv
    {dataset}_ndcg.csv
    {dataset}_ap.csv
    {dataset}_auc_ci.txt
    {dataset}_ndcg_ci.txt
    {dataset}_ap_ci.txt
"""

import sys
import json
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager

import numpy as np
from numpy.random import SeedSequence

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from data_loader import import_data
from approaches import ApproachFactory, Approach
from analysis import compute_score


DEFAULT_DATASETS = (
    'Armenia', 'Catalonia', 'Ghana_Nigeria', 'Iran_5', 'Russia_3', 'Spain',
    'Venezuela_2', 'Thailand', 'Ecuador', 'Iran_1', 'Iran_6', 'Russia_5',
    'Qatar', 'Russia_2', 'China_1', 'China_2', 'Russia_1', 'Russia_4',
    'Iran_2', 'Iran_3', 'Iran_4', 'UAE', 'Cuba'
)


def load_experiment_config(config_path: str) -> dict:
    """Load and validate experiment configuration (same defaults as run_experiments.py)."""
    with open(config_path, 'r') as f:
        config = json.load(f)

    if 'name' not in config:
        raise ValueError("Configuration must include 'name' field")

    config.setdefault('data_dir', 'data')
    config.setdefault('output_dir', None)
    config.setdefault('window_sec', 60)
    config.setdefault('datasets', None)
    config.setdefault('approaches', None)
    config.setdefault('force', False)
    config.setdefault('filter_min_coactions', None)

    if config['datasets'] is None:
        config['datasets'] = list(DEFAULT_DATASETS)

    if config['approaches'] is not None:
        for approach in config['approaches']:
            ApproachFactory.validate_approach_key(approach)

    return config

def build_user_map(RTs: List[Tuple]) -> Dict:
    user_map = {}
    for row in RTs:
        user = row[0]
        if user not in user_map:
            user_map[user] = []
        # Store the whole row so we don't have to re-create it later
        user_map[user].append(row) 
    return user_map

def bootstrap_resample_users(users: List, sample_size: int, user_map: Dict, io_users: set, rng: np.random.Generator) -> Tuple[List[Tuple], set]:
    """Sample from all users randomly with replacement and rebuild RTs and IO user set."""
    if not users:
        return [], set()

    sampled_users = rng.choice(users, size=sample_size, replace=True)

    # Delete all repeated users from the sample
    sampled_users = list(set(sampled_users))

    boot_RTs: List[Tuple] = []
    
    # Optimized version using set intersection and list extension
    boot_io_users = set(sampled_users).intersection(io_users)

    # Flattening the lists using a fast list comprehension
    boot_RTs = [
        row for u in sampled_users for row in user_map[u]
    ]

    return boot_RTs, boot_io_users

def unique_user_count(suspicious_users: List[List[int]]) -> int:
    """Count unique users in suspicious list of lists."""
    return len(set(u for group in suspicious_users for u in group))

def compute_metrics_for_sample(approaches: List[Approach], approach_names: List[str], RTs: List[Tuple], io_users: set, filter_min_coactions: int | None) -> Dict[str, Dict[str, float]]:
    """Compute AUC, NDCG, and AP for one bootstrap sample."""
    suspicious_by_approach = {}

    for approach, name in zip(approaches, approach_names):
        suspicious = approach.get_suspicious(
            RTs,
            io_users=io_users,
            filter_min_coactions=filter_min_coactions,
        )
        suspicious_by_approach[name] = suspicious

    u_counts = [len(set(u for group in s for u in group)) for s in list(suspicious_by_approach.values())]
    x_max = min(u_counts) if u_counts else 0

    metrics = {'auc': {}, 'ndcg': {}, 'ap': {}}
    for name in approach_names:
        suspicious = suspicious_by_approach[name]
        metrics['auc'][name] = compute_score(suspicious, io_users, x_max, metric='auc')
        metrics['ndcg'][name] = compute_score(suspicious, io_users, x_max, metric='ndcg')
        metrics['ap'][name] = compute_score(suspicious, io_users, x_max, metric='ap')

    return metrics

def bootstrap_worker(
    iteration_idx: int,
    seed_int: int,
    sample_size: int,
    users: List,
    user_map: Dict,
    io_users: set,
    approaches: List[Approach],
    approach_names: List[str],
    filter_min_coactions: int | None,
) -> Dict:
    """Worker function for one bootstrap iteration.
    
    Returns results as {metric: {approach_name: value}} for memory efficiency.
    """
    
    # Create independent RNG from seed
    rng = np.random.default_rng(seed_int)
    
    # Sample users with replacement and rebuild RTs and IO user set for this bootstrap sample
    boot_RTs, boot_io_users = bootstrap_resample_users(users, sample_size, user_map, io_users, rng)
    
    # Compute metrics
    metrics = compute_metrics_for_sample(
        approaches,
        approach_names,
        boot_RTs,
        boot_io_users,
        filter_min_coactions,
    )
    
    return {'iteration': iteration_idx, 'metrics': metrics}

def compute_ci_from_csv(csv_path: Path, ci_level: float) -> Dict[str, Tuple[float, float, float]]:
    """Compute percentile CI from CSV. Returns {approach: (mean, low, high)}."""
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        values = {name: [] for name in columns}
        for row in reader:
            for name in columns:
                try:
                    values[name].append(float(row[name]))
                except (ValueError, TypeError):
                    continue

    alpha = (100.0 - ci_level) / 2.0
    ci = {}
    for name, vals in values.items():
        if not vals:
            ci[name] = (float('nan'), float('nan'), float('nan'))
            continue
        arr = np.array(vals, dtype=float)
        mean = float(np.mean(arr))
        low = float(np.percentile(arr, alpha))
        high = float(np.percentile(arr, 100.0 - alpha))
        ci[name] = (mean, low, high)
    return ci

def write_ci_report(report_path: Path, metric: str, dataset: str, ci_level: float,
                    ci: Dict[str, Tuple[float, float, float]]) -> None:
    """Write CI report to text file."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Metric: {metric.upper()}\n")
        f.write(f"CI level: {ci_level:.1f}%\n\n")
        for name, (mean, low, high) in ci.items():
            f.write(f"{name}\n")
            f.write(f"  Mean: {mean:.6f}\n")
            f.write(f"  CI: [{low:.6f}, {high:.6f}]\n\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Subsampling evaluation (resample all users with replacement) for experiment configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Example usage:
            uv run bin/bootstrap_experiments.py experiments/my_experiment.json
            uv run bin/bootstrap_experiments.py experiments/my_experiment.json --n-bootstrap 200 --seed 123
            uv run bin/bootstrap_experiments.py experiments/my_experiment.json --n-bootstrap 200 --workers 8
        """,
    )
    parser.add_argument('config', help='Path to experiment JSON configuration file')
    parser.add_argument('--n-bootstrap', type=int, default=1000,
                        help='Number of subsamples (all users with replacement) per dataset (default: 1000)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--ci', type=float, default=95.0,
                        help='Confidence interval level in percent (default: 95)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes (default: auto-detect)')

    args = parser.parse_args()

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

    config_path = Path(args.config).resolve()
    json_dir = config_path.parent

    if config['output_dir'] is None:
        experiment_dir = json_dir / config['name']
    else:
        experiment_dir = Path(config['output_dir']) / config['name']

    output_dir = experiment_dir / "analysis" / "bootstrap"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build approaches from config
    window_sec = config['window_sec']
    if config['approaches'] is not None:
        approaches = [ApproachFactory.create(spec, window_sec)
                      for spec in config['approaches']]
    else:
        approaches = ApproachFactory.get_all_approaches(window_sec, 1, 'L2')

    approach_names = [approach.get_full_approach_name() for approach in approaches]

    # Determine number of workers with memory constraints in mind
    # Each worker holds large RTs lists and IO sets, so be conservative
    if args.workers is None:
        import os
        cpu_count = os.cpu_count() or 1
        num_workers = max(1, min(4, cpu_count // 2))  # Conservative: max 4 or half CPU count
    else:
        num_workers = args.workers

    print("=" * 80)
    print("SUBSAMPLING EVALUATION (100% of users with replacement)")
    print("=" * 80)
    print(f"Experiment: {config['name']}")
    print(f"Configuration file: {args.config}")
    print(f"Output directory: {output_dir}")
    print(f"Number of subsamples: {args.n_bootstrap}")
    print(f"Number of workers: {num_workers} (conservative for memory efficiency)")

    for dataset in config['datasets']:
        print("\n" + "-" * 80)
        print(f"Dataset: {dataset}")
        print("-" * 80)

        processed_dir = Path(config['data_dir']) / dataset / "Processed"
        if not processed_dir.exists():
            print(f"  WARNING: Missing data directory: {processed_dir}")
            continue

        RTs, io_users = import_data(processed_dir)
        user_map = build_user_map(RTs)
        print(f"  Total users: {len(user_map)}, IO users: {len(io_users)}")
        
        # Pre-compute values used in every bootstrap iteration
        users = list(user_map.keys())
        n_users = len(users)
        sample_size = n_users
        print(f"  Sampling {sample_size} users per bootstrap iteration")

        # Open CSV files and initialize writers with headers
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_paths = {}
        csv_files = {}
        csv_writers = {}
        
        for metric in ('auc', 'ndcg', 'ap'):
            csv_path = output_dir / f"{dataset}_{metric}_replacement.csv"
            csv_paths[metric] = csv_path
            csv_files[metric] = open(csv_path, 'w', newline='')
            csv_writers[metric] = csv.writer(csv_files[metric])
            csv_writers[metric].writerow(approach_names)
            csv_files[metric].flush()

        # Generate independent seeds for each bootstrap iteration using SeedSequence
        ss = SeedSequence(args.seed)
        seeds = ss.spawn(args.n_bootstrap)
        # Properly generate independent integer seeds from spawned sequences
        seed_ints = [s.generate_state(1)[0] for s in seeds]
        
        # Run bootstrap iterations in parallel
        results_by_iteration = {}
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    bootstrap_worker,
                    iteration_idx=i,
                    seed_int=seed_ints[i],
                    sample_size=sample_size,
                    users=users,
                    user_map=user_map,
                    io_users=io_users,
                    approaches=approaches,
                    approach_names=approach_names,
                    filter_min_coactions=config.get('filter_min_coactions'),
                )
                for i in range(args.n_bootstrap)
            ]
            
            try:
                # Process results as they complete (memory efficient streaming)
                for completed_future in enumerate(futures):
                    idx, future = completed_future
                    result = future.result()
                    iteration_idx = result['iteration']
                    metrics = result['metrics']
                    results_by_iteration[iteration_idx] = metrics
                    
                    # Write to CSV using already-open writers and flush immediately
                    for metric in ('auc', 'ndcg', 'ap'):
                        csv_writers[metric].writerow([metrics[metric][name] for name in approach_names])
                        csv_files[metric].flush()  # Force write to disk for real-time streaming
                    
                    # Progress update
                    if (idx + 1) % 100 == 0 or (idx + 1) == args.n_bootstrap:
                        print(f"  Completed {idx + 1}/{args.n_bootstrap} samples")
            finally:
                # Close all CSV files
                for f in csv_files.values():
                    f.close()

        # Compute confidence intervals from completed CSVs
        for metric in ('auc', 'ndcg', 'ap'):
            print(f"  Saved CSV: {csv_paths[metric]}")
            ci = compute_ci_from_csv(csv_paths[metric], args.ci)
            report_path = output_dir / f"{dataset}_{metric}_replacement_ci.txt"
            write_ci_report(report_path, metric, dataset, args.ci, ci)
            print(f"  Saved CI: {report_path}")

    print("\n" + "=" * 80)
    print("SUBSAMPLING COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
