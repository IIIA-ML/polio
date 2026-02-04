#!/usr/bin/env python3
"""
Bootstrap evaluation for experiment configurations.

For each dataset in a configuration, this script:
1) Loads RTs and IO users
2) Bootstraps users by resampling with replacement
3) Computes approach rankings on each bootstrap subdataset
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
import multiprocessing as mp

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
    """Map user -> list of (tweet_id, timestamp)."""
    user_map: Dict = {}
    for user, tid, ts in RTs:
        if user not in user_map:
            user_map[user] = []
        user_map[user].append((tid, ts))
    return user_map


def bootstrap_resample_users(user_map: Dict, io_users: set, rng: np.random.Generator) -> Tuple[List[Tuple], set]:
    """Resample users with replacement and rebuild RTs and IO user set."""
    users = list(user_map.keys())
    n_users = len(users)

    if n_users == 0:
        return [], set()

    sampled_users = rng.choice(users, size=n_users, replace=True)

    boot_RTs: List[Tuple] = []
    boot_io_users = set()

    for idx, orig_user in enumerate(sampled_users):
        if orig_user in io_users:
            boot_io_users.add(orig_user)
        for tid, ts in user_map[orig_user]:
            boot_RTs.append((orig_user, tid, ts))
    return boot_RTs, boot_io_users


def unique_user_count(suspicious_users: List[List[int]]) -> int:
    """Count unique users in suspicious list of lists."""
    seen = set()
    for group in suspicious_users:
        for u in group:
            seen.add(u)
    return len(seen)


def compute_metrics_for_sample(approaches: List[Approach], approach_names: List[str], RTs: List[Tuple], io_users: set, filter_min_coactions: int | None) -> Dict[str, Dict[str, float]]:
    """Compute AUC, NDCG, and AP for one bootstrap sample."""
    suspicious_by_approach = {}
    x_max_candidates = []

    for approach, name in zip(approaches, approach_names):
        suspicious = approach.get_suspicious(
            RTs,
            io_users=io_users,
            filter_min_coactions=filter_min_coactions,
        )
        suspicious_by_approach[name] = suspicious
        x_max_candidates.append(unique_user_count(suspicious))

    x_max = min(x_max_candidates) if x_max_candidates else 0

    metrics = {'auc': {}, 'ndcg': {}, 'ap': {}}
    for name in approach_names:
        suspicious = suspicious_by_approach[name]
        metrics['auc'][name] = compute_score(suspicious, io_users, x_max, metric='auc')
        metrics['ndcg'][name] = compute_score(suspicious, io_users, x_max, metric='ndcg')
        metrics['ap'][name] = compute_score(suspicious, io_users, x_max, metric='ap')

    return metrics


def initialize_metric_csv(csv_path: Path, approach_names: List[str]) -> None:
    """Initialize CSV with header row (columns = approaches)."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(approach_names)


def append_metric_row(csv_path: Path, approach_names: List[str], row: Dict[str, float]) -> None:
    """Append a single metric row to CSV."""
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([row[name] for name in approach_names])


def bootstrap_worker(bootstrap_idx: int, seed: int, user_map: Dict, io_users: set, approaches: List[Approach], approach_names: List[str], filter_min_coactions: int | None,) -> Tuple[int, Dict[str, Dict[str, float]]]:
    """Worker function for parallel bootstrap sampling.
    
    Returns: (bootstrap_idx, metrics_dict)
    """
    rng = np.random.default_rng(seed)
    boot_RTs, boot_io_users = bootstrap_resample_users(user_map, io_users, rng)
    metrics = compute_metrics_for_sample(
        approaches,
        approach_names,
        boot_RTs,
        boot_io_users,
        filter_min_coactions,
    )
    return bootstrap_idx, metrics


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
        description="Bootstrap evaluation for experiment configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Example usage:
            uv run bin/bootstrap_experiments.py experiments/my_experiment.json
            uv run bin/bootstrap_experiments.py experiments/my_experiment.json --n-bootstrap 200 --seed 123
        """,
    )
    parser.add_argument('config', help='Path to experiment JSON configuration file')
    parser.add_argument('--n-bootstrap', type=int, default=1000,
                        help='Number of bootstrap samples per dataset (default: 1000)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--ci', type=float, default=95.0,
                        help='Confidence interval level in percent (default: 95)')

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

    print("=" * 80)
    print("BOOTSTRAP EVALUATION")
    print("=" * 80)
    print(f"Experiment: {config['name']}")
    print(f"Configuration file: {args.config}")
    print(f"Output directory: {output_dir}")
    print(f"Bootstrap samples: {args.n_bootstrap}")

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

        # Initialize CSV files with headers
        csv_paths = {}
        for metric in ('auc', 'ndcg', 'ap'):
            csv_path = output_dir / f"{dataset}_{metric}.csv"
            initialize_metric_csv(csv_path, approach_names)
            csv_paths[metric] = csv_path

        # Generate unique seeds for each bootstrap sample using SeedSequence
        ss = SeedSequence(args.seed)
        child_seeds = ss.spawn(args.n_bootstrap)

        # Parallel bootstrap loop
        n_workers = mp.cpu_count() - 2 if mp.cpu_count() > 2 else 1
        print(f"  Using {n_workers} parallel workers for bootstrap sampling")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(
                    bootstrap_worker,
                    b,
                    int(child_seeds[b].generate_state(1)[0]),
                    user_map,
                    set(io_users),
                    approaches,
                    approach_names,
                    config.get('filter_min_coactions'),
                )
                for b in range(args.n_bootstrap)
            ]

            # Process results as they complete and append to CSVs immediately
            completed = 0
            for future in futures:
                bootstrap_idx, metrics = future.result()
                # Append metrics to each metric CSV immediately (low memory footprint)
                for metric in ('auc', 'ndcg', 'ap'):
                    append_metric_row(csv_paths[metric], approach_names, metrics[metric])
                completed += 1
                if completed % max(1, args.n_bootstrap // 10) == 0:
                    print(f"  Completed {completed}/{args.n_bootstrap} samples")

        # Compute confidence intervals from completed CSVs
        for metric in ('auc', 'ndcg', 'ap'):
            print(f"  Saved CSV: {csv_paths[metric]}")
            ci = compute_ci_from_csv(csv_paths[metric], args.ci)
            report_path = output_dir / f"{dataset}_{metric}_ci.txt"
            write_ci_report(report_path, metric, dataset, args.ci, ci)
            print(f"  Saved CI: {report_path}")

    print("\n" + "=" * 80)
    print("BOOTSTRAP COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
