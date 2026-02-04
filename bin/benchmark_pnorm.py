#!/usr/bin/env python3
"""
Benchmark different Lp-norm ranking modes for a given approach.

Compares performance across different ranking modes (L1, L2, ..., Linf) by:
1. Loading results for all ranking mode variants of the specified approach
2. Computing AUC, AP, and NDCG metrics for each variant
3. Plotting all three metrics in a single figure with error bars
4. Saving the plot to {experiment_dir}/analysis/pnorm_benchmark/

Expects the experiment config to contain multiple specs of the same approach
with different 'ranking_mode' parameters.
"""

import os
import sys
import json
import pickle
import argparse
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from approaches import ApproachFactory
from analysis import (
    load_experiment_config,
    load_all_results,
    compute_score,
)


def get_approach_name(approach_key):
    """
    Get the human-readable name for an approach key.
    
    Args:
        approach_key: Approach key (e.g., 'coretweets_fast')
    
    Returns:
        Approach name (e.g., 'CT')
    """
    try:
        approach = ApproachFactory.create(approach_key)
        return approach.get_approach_name()
    except Exception:
        return approach_key


def extract_norm_value(ranking_mode):
    """
    Extract numeric norm value from ranking mode string for sorting.
    
    Args:
        ranking_mode: Ranking mode string (e.g., 'L1', 'L2', 'L10', 'Linf')
    
    Returns:
        Numeric value for sorting (int for L<n>, float('inf') for Linf, None otherwise)
    """
    if ranking_mode == 'Linf':
        return float('inf')
    elif ranking_mode.startswith('L'):
        try:
            return int(ranking_mode[1:])
        except ValueError:
            return None
    return None


def load_results_by_ranking_mode(results_dir, base_approach_key, approach_specs):
    """
    Load results for a single approach with different ranking modes.
    
    Creates approach instances from specs, extracts their ranking_mode,
    and loads results using load_all_results().
    
    Args:
        results_dir: Path to results directory
        base_approach_key: Base approach key (unused, kept for compatibility)
        approach_specs: List of approach specs with different ranking modes
    
    Returns:
        dict: {ranking_mode: {dataset_name: {approach_key: results_dict}}}
    """
    results_by_mode = {}
    
    for spec in approach_specs:
        # Create approach from spec
        approach = ApproachFactory.create(spec)
        storage_key = approach.get_full_approach_key()
        
        # Extract ranking mode from the approach
        # The ranking_mode is stored in the approach config or can be extracted from storage_key
        if hasattr(approach, 'ranking_mode'):
            ranking_mode = approach.ranking_mode
        else:
            # Default to L2 if no ranking mode is specified
            ranking_mode = 'L2'
        
        # Load results for this specific approach
        all_results = load_all_results(results_dir, [storage_key], [storage_key])
        
        if all_results:
            results_by_mode[ranking_mode] = all_results
    
    return results_by_mode


def compute_metric_scores(all_results, metric='auc'):
    """
    Compute metric scores for all datasets and approaches.
    
    Args:
        all_results: {dataset_name: {approach_key: results_dict}}
        metric: Metric to compute ('auc', 'ndcg', 'ap')
    
    Returns:
        List of scores for each dataset
    """
    scores = []
    
    for dataset_name, dataset_results in all_results.items():
        if not dataset_results:
            continue
        
        # Get IO users from the available result
        approach_key = list(dataset_results.keys())[0]
        io_users = set(dataset_results[approach_key]['io_users'])
        
        # Compute x_max
        x_max = float('inf')
        for key in dataset_results:
            x_max_approach = len(set([u for group in dataset_results[key]['suspicious_users'] for u in group]))
            x_max = min(x_max, x_max_approach)
        
        # Compute score for this approach
        suspicious_users = dataset_results[approach_key]['suspicious_users']
        score = compute_score(suspicious_users, io_users, x_max, metric=metric)
        scores.append(score)
    
    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark different Lp-norm ranking modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Example usage:
            ./bin/benchmark_pnorm.py experiments/my_experiment.json coretweets_fast
            ./bin/benchmark_pnorm.py experiments/my_experiment.json ignoring_tweet_fast

            The script will:
            1. Load results from the experiment's results directory
            2. Find results for different ranking modes (L1, L2, ..., L10, Linf)
            3. Compute AUC, AP, and NDCG for each norm
            4. Plot all three metrics in a single figure
            5. Save plot to {experiment_dir}/analysis/pnorm_benchmark/
        """
    )
    parser.add_argument(
        'config',
        help='Path to experiment JSON configuration file'
    )
    parser.add_argument(
        'approach',
        help='Base approach key to benchmark (e.g., coretweets_fast)'
    )
    
    args = parser.parse_args()
    
    # Load experiment configuration
    try:
        config = load_experiment_config(args.config)
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Invalid configuration: {e}")
        sys.exit(1)
    
    # Determine experiment directory based on JSON location
    config_path = Path(args.config).resolve()
    json_dir = config_path.parent
    experiment_dir = json_dir / config['name']
    results_dir = experiment_dir / "results"
    
    # Get approach name
    approach_name = get_approach_name(args.approach)
    
    # Set output directory
    output_dir = experiment_dir / "analysis" / "pnorm_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.approach}_pnorm_benchmark.png"
    
    print("\n" + "="*70)
    print("BENCHMARK Lp-NORM RANKING MODES")
    print("="*70)
    print(f"\nExperiment: {config['name']}")
    print(f"Approach: {approach_name} ({args.approach})")
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    
    # Get approaches from config that match the base approach
    # Filter for different ranking modes of the same base approach
    if config.get('approaches') is None:
        print(f"ERROR: No approaches specified in config!")
        sys.exit(1)
    
    approach_specs = config['approaches']
    
    # Filter for specs that match the base approach key
    matching_specs = []
    for spec in approach_specs:
        if isinstance(spec, str):
            if spec == args.approach:
                matching_specs.append(spec)
        elif isinstance(spec, dict):
            if spec.get('name') == args.approach:
                matching_specs.append(spec)
    
    if not matching_specs:
        print(f"ERROR: No approach specs found for '{args.approach}' in config!")
        print(f"Available approaches: {', '.join(str(s) if isinstance(s, str) else s['name'] for s in approach_specs)}")
        sys.exit(1)
    
    # Load results grouped by ranking mode
    print(f"\nLoading results for different ranking modes...")
    results_by_mode = load_results_by_ranking_mode(results_dir, args.approach, matching_specs)
    
    if not results_by_mode:
        print(f"ERROR: No results found for approach '{args.approach}' in {results_dir}!")
        print(f"Have you run the experiment first using run_experiments.py?")
        sys.exit(1)
    
    # Sort modes by norm value
    sorted_modes = sorted(results_by_mode.keys(), key=extract_norm_value)
    print(f"Found {len(sorted_modes)} ranking modes: {', '.join(sorted_modes)}")
    
    # Compute scores for each metric and ranking mode
    metrics = ['auc', 'ndcg', 'ap']
    results_data = defaultdict(list)  # {metric: [scores]}
    x_positions = []  # Sequential positions for x-axis (0, 1, 2, ...)
    norm_labels = []
    
    for i, mode in enumerate(sorted_modes):
        x_positions.append(i)  # Sequential positions regardless of actual norm value
        norm_labels.append(mode)
        
        # Get all results for this ranking mode
        all_results = results_by_mode[mode]
        
        # Find the approach storage key (there should be only one per mode)
        if not all_results:
            print(f"  WARNING: No results found for mode {mode}")
            continue
        
        # Get first dataset to find the approach key
        first_dataset = next(iter(all_results.values()))
        if not first_dataset:
            print(f"  WARNING: Empty dataset results for mode {mode}")
            continue
        
        approach_key = list(first_dataset.keys())[0]
        
        # Reconstruct all_results with approach_key as the consistent key
        reformatted_results = {}
        for dataset_name, dataset_data in all_results.items():
            reformatted_results[dataset_name] = {approach_key: dataset_data[approach_key]}
        
        all_results = reformatted_results
        
        for metric in metrics:
            print(f"  Computing {metric.upper()} for {mode}...")
            scores = compute_metric_scores(all_results, metric=metric)
            
            if scores:
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                results_data[metric].append((mean_score, std_score))
                print(f"    {metric.upper()}: {mean_score:.4f} ± {std_score:.4f}")
            else:
                print(f"    WARNING: No scores computed for {metric}")
                results_data[metric].append((0.0, 0.0))
    
    # Create plot
    print(f"\nGenerating plot...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot each metric
    colors = {'auc': '#1f77b4', 'ndcg': '#ff7f0e', 'ap': '#2ca02c'}
    markers = {'auc': 'o', 'ndcg': 's', 'ap': '^'}
    
    for metric in metrics:
        means, stds = zip(*results_data[metric])
        ax.errorbar(x_positions, means, yerr=stds, 
                   label=metric.upper(), 
                   color=colors[metric],
                   marker=markers[metric],
                   markersize=8,
                   linewidth=2,
                   capsize=5,
                   capthick=2)
    
    # Customize plot
    ax.set_xlabel("Lp-norm (p value)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Mean Score", fontsize=12, fontweight='bold')
    ax.set_title(f"Benchmark: {approach_name} Performance by Lp-norm", 
                fontsize=14, fontweight='bold', pad=20)
    
    # Set x-axis with equal spacing
    ax.set_xticks(x_positions)
    ax.set_xticklabels(norm_labels, rotation=0)
    ax.set_xlim(-0.5, len(x_positions) - 0.5)
    
    # Set y-axis limits
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    
    # Add value labels on points with background for better readability
    for metric in metrics:
        means, _ = zip(*results_data[metric])
        for x, y in zip(x_positions, means):
            # Add white background box behind text for better visibility
            ax.text(x, y + 0.035, f'{y:.3f}', 
                   ha='center', va='bottom', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {output_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nApproach: {approach_name} ({args.approach})")
    print(f"\nResults by ranking mode:\n")
    
    print(f"{'Mode':<10} {'AUC':<15} {'AP':<15} {'NDCG':<15}")
    print("-" * 55)
    for i, mode in enumerate(sorted_modes):
        auc_mean, auc_std = results_data['auc'][i]
        ap_mean, ap_std = results_data['ap'][i]
        ndcg_mean, ndcg_std = results_data['ndcg'][i]
        print(f"{mode:<10} {auc_mean:.4f}±{auc_std:.4f}  {ap_mean:.4f}±{ap_std:.4f}  {ndcg_mean:.4f}±{ndcg_std:.4f}")
    
    # Find best mode for each metric
    print(f"\nBest modes:")
    for metric in metrics:
        means, _ = zip(*results_data[metric])
        best_idx = np.argmax(means)
        best_mode = sorted_modes[best_idx]
        best_score = means[best_idx]
        print(f"  {metric.upper()}: {best_mode} ({best_score:.4f})")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()