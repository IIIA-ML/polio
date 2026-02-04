#!/usr/bin/env python3
"""
Analyze experiment results and perform statistical comparisons.

This script:
1. Loads experiment configuration from JSON file
2. Loads results from the experiment's results directory
3. Generates comparison plots for each dataset
4. Computes rankings for each method
5. Performs Friedman test
6. Conducts post-hoc analysis (Nemenyi test)
7. Generates critical difference diagrams

Supports multiple evaluation modes: standard, multi-metric, no-truncation, and ideal.

Usage:
    ./bin/analyze_results.py <config.json> [--no-plots] [--metric <metric/all>] [--notruncation] [--ideal]
"""

import sys
import argparse
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from approaches import ApproachFactory
from analysis import (
    load_experiment_config,
    load_all_results,
    compute_score,
    compute_rankings_with_ties,
    count_io_until_first_nonio,
    users_until_reaching_io_fraction,
    wilcoxon_signed_rank_test,
    friedman_test,
    nemenyi_test,
    plot_method_comparison,
    plot_critical_difference_diagram,
    count_total_io_in_suspicious,
    count_total_accounts_in_suspicious,
)
from analysis.visualization import plot_method_comparison_no_truncation, plot_method_comparison_ideal
from analysis.reporting import write_metric_summary, write_general_summary


def process_dataset(dataset_name, dataset_results, approach_keys, approach_names,
                    plots_dir, current_metric, generate_plots, ideal_mode=False):
    """
    Process a single dataset: compute scores, rankings, and optionally generate plots.

    Args:
        dataset_name: Name of the dataset
        dataset_results: Dictionary of results for each approach
        approach_keys: List of approach keys
        approach_names: Dictionary mapping approach keys to display names
        plots_dir: Directory to save plots
        current_metric: Metric to use for scoring
        generate_plots: Whether to generate comparison plots
        ideal_mode: Whether to generate ideal comparison plots

    Returns:
        Tuple of (rankings, scores, first_nonio_counts, users_to_80pct, total_io_counts, total_accounts_counts) or (None, None, None, None, None, None)
    """
    print(f"\nProcessing: {dataset_name}")

    # Check if all approaches are available
    if not all(approach_key in dataset_results for approach_key in approach_keys):
        missing = [key for key in approach_keys if key not in dataset_results]
        print(f"  WARNING: Incomplete results, skipping (missing: {', '.join(missing)})")
        return None, None, None, None, None, None

    # Get method names
    method_names = [approach_names[key] for key in approach_keys]

    # Compute scores for all methods
    first_approach = approach_keys[0]
    io_users = set(dataset_results[first_approach]['io_users'])
    
    # Find minimum x_max across all approaches
    x_max = float('inf')
    for k in approach_keys:
        x_max_approach = len(set([u for group in dataset_results[k]['suspicious_users'] for u in group]))
        x_max = min(x_max, x_max_approach)
    
    scores = {
        approach_names[approach_key]: compute_score(
            dataset_results[approach_key]['suspicious_users'], io_users, x_max, metric=current_metric
        )
        for approach_key in approach_keys
    }

    # Compute additional evaluations for this dataset
    first_nonio_counts = {
        approach_names[approach_key]: count_io_until_first_nonio(
            dataset_results[approach_key]['suspicious_users'], io_users
        )
        for approach_key in approach_keys
    }

    users_to_80pct = {
        approach_names[approach_key]: users_until_reaching_io_fraction(
            dataset_results[approach_key]['suspicious_users'], io_users, fraction=0.9
        )
        for approach_key in approach_keys
    }

    total_io_counts = {
        approach_names[approach_key]: count_total_io_in_suspicious(
            dataset_results[approach_key]['suspicious_users'], io_users
        )
        for approach_key in approach_keys
    }

    total_accounts_counts = {
        approach_names[approach_key]: count_total_accounts_in_suspicious(
            dataset_results[approach_key]['suspicious_users']
        )
        for approach_key in approach_keys
    }

    # Generate plot or just compute rankings
    if generate_plots:
        if ideal_mode:
            rankings = plot_method_comparison_ideal(dataset_name, dataset_results, approach_keys, 
                                                   approach_names, plots_dir, metric=current_metric)
        else:
            rankings = plot_method_comparison(dataset_name, dataset_results, approach_keys, 
                                             approach_names, plots_dir, metric=current_metric)
    else:
        # Just compute rankings without plotting
        rankings = compute_rankings_with_ties(scores)

    if rankings is None:
        return None, None, None, None, None, None

    # Return data in consistent order
    return (
        [rankings[method] for method in method_names],
        [scores[method] for method in method_names],
        [first_nonio_counts[method] for method in method_names],
        [(users_to_80pct[method] if users_to_80pct[method] is not None else -1) for method in method_names],
        [total_io_counts[method] for method in method_names],
        [total_accounts_counts[method] for method in method_names]
    )


def display_results(dataset_names, method_names, rankings_matrix, scores_matrix, metric_display):
    """
    Display rankings and scores matrices to console.

    Args:
        dataset_names: List of dataset names
        method_names: List of method names
        rankings_matrix: 2D array of rankings
        scores_matrix: 2D array of scores
        metric_display: Display name of metric
    """
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS")
    print("="*70)

    # Display rankings matrix
    print("\nRankings Matrix:")
    print(f"{'Dataset':<20} " + " ".join([f"{m:>18}" for m in method_names]))
    print("-" * (20 + 19 * len(method_names)))
    for dataset, ranks in zip(dataset_names, rankings_matrix):
        print(f"{dataset:<20} " + " ".join([f"{r:>18}" for r in ranks]))

    # Compute average ranks
    avg_ranks = np.mean(rankings_matrix, axis=0)
    print(f"\n{'Average Rank':<20} " + " ".join([f"{r:>18.2f}" for r in avg_ranks]))

    # Display scores matrix
    print(f"\n\n{metric_display} Scores Matrix:")
    print(f"{'Dataset':<20} " + " ".join([f"{m:>18}" for m in method_names]))
    print("-" * (20 + 19 * len(method_names)))
    for dataset, scores in zip(dataset_names, scores_matrix):
        print(f"{dataset:<20} " + " ".join([f"{s:>18.4f}" for s in scores]))

    # Compute average scores
    avg_scores = np.mean(scores_matrix, axis=0)
    print(f"\n{'Average ' + metric_display:<20} " + " ".join([f"{s:>18.4f}" for s in avg_scores]))


def perform_statistical_tests(rankings_matrix, scores_matrix, method_names, metric_display):
    """
    Perform appropriate statistical tests based on number of methods.

    Args:
        rankings_matrix: 2D array of rankings
        scores_matrix: 2D array of scores
        method_names: List of method names
        metric_display: Display name of metric

    Returns:
        Tuple of (statistic, p_value, n_methods, median_diff, cd, pvalues_matrix, avg_ranks)
    """
    n_methods = len(method_names)
    avg_ranks = np.mean(rankings_matrix, axis=0)
    
    if n_methods == 2:
        # Use Wilcoxon signed-rank test for two methods
        print("\n" + "-"*70)
        print("Wilcoxon Signed-Rank Test (for two methods):")
        print("-"*70)

        statistic, p_value, median_diff = wilcoxon_signed_rank_test(
            scores_matrix[:, 0], scores_matrix[:, 1]
        )

        print(f"Test statistic: {statistic:.4f}")
        print(f"P-value: {p_value:.6f}")
        print(f"Median difference ({metric_display}): {median_diff:.4f}")
        print(f"\nMethod 1: {method_names[0]}")
        print(f"Method 2: {method_names[1]}")

        if p_value < 0.05:
            if median_diff > 0:
                print(f"\nResult: {method_names[0]} significantly outperforms {method_names[1]} (p < 0.05)")
            else:
                print(f"\nResult: {method_names[1]} significantly outperforms {method_names[0]} (p < 0.05)")
        else:
            print("\nResult: No significant difference detected (p >= 0.05)")

        return statistic, p_value, n_methods, median_diff, None, None, avg_ranks

    else:
        # Use Friedman test for more than two methods
        print("\n" + "-"*70)
        print("Friedman Test:")
        print("-"*70)

        statistic, p_value = friedman_test(rankings_matrix)
        print(f"Chi-square statistic: {statistic:.4f}")
        print(f"P-value: {p_value:.6f}")

        if p_value < 0.05:
            print("Result: Significant differences detected (p < 0.05)")
            print("Proceeding with post-hoc analysis...")
        else:
            print("Result: No significant differences detected (p >= 0.05)")

        # Nemenyi post-hoc test
        print("\n" + "-"*70)
        print("Nemenyi Post-hoc Test:")
        print("-"*70)

        avg_ranks, cd, pvalues_matrix = nemenyi_test(rankings_matrix, method_names)

        print(f"Critical Difference (CD): {cd:.4f}")
        print(f"\nAverage Ranks:")
        for name, rank in zip(method_names, avg_ranks):
            print(f"  {name:<25} {rank:.3f}")

        print(f"\nPairwise Comparisons (p-values):")
        print(f"(p-value < 0.05 indicates significant difference)")
        print(f"\n{'':>20} " + " ".join([f"{m[:10]:>12}" for m in method_names]))
        for i, name_i in enumerate(method_names):
            row = f"{name_i[:20]:<20} "
            for j in range(len(method_names)):
                if i == j:
                    row += f"{'1.000':>12} "
                else:
                    row += f"{pvalues_matrix[i, j]:>12.4f} "
            print(row)

        # Show significant pairs
        print(f"\nSignificant differences (p < 0.05):")
        sig_pairs = []
        for i in range(len(method_names)):
            for j in range(i + 1, len(method_names)):
                if pvalues_matrix[i, j] < 0.05:
                    sig_pairs.append((method_names[i], method_names[j], pvalues_matrix[i, j]))
        
        if sig_pairs:
            for name1, name2, pval in sig_pairs:
                print(f"  {name1} vs {name2}: p = {pval:.4f}")
        else:
            print("  No significant pairwise differences found")

        # Generate critical difference diagram
        print("\n" + "-"*70)
        print("Generating Critical Difference Diagram...")
        print("-"*70)

        return statistic, p_value, n_methods, None, cd, pvalues_matrix, avg_ranks


def main():
    parser = argparse.ArgumentParser(
        description="Analyze experiment results and perform statistical comparisons",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Usage:
                ./bin/analyze_results.py <config.json> [--no-plots] [--metric <metric/all>] [--notruncation] [--ideal]

            Output saved to {experiment_name}/analysis/ with metric-specific and general summaries.
            Statistical tests: Wilcoxon (2 methods) or Friedman+Nemenyi (>2 methods).
        """
    )
    parser.add_argument(
        'config',
        help='Path to experiment JSON configuration file'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating individual comparison plots'
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='auc',
        choices=['auc', 'ndcg', 'ap', 'all'],
        help='Metric to use for evaluation (default: auc). '
             'Options: auc (sklearn ROC-AUC), ndcg (sklearn NDCG), '
             'ap (sklearn Average Precision), all (compute all three metrics)'
    )
    parser.add_argument(
        '--notruncation',
        action='store_true',
        help='Generate plots without truncating by x_max and without computing areas. '
             'Outputs will be saved to experiments/{experiment_name}/analysis/no_truncation/'
    )
    parser.add_argument(
        '--ideal',
        action='store_true',
        help='Generate ideal comparison plots where each method is compared to the ideal detection curve '
             '(y=x until all IO users found, then flat). Outputs saved as *_comparison_ideal.png'
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

    # Set defaults for optional fields
    config.setdefault('output_dir', None)
    
    # Determine experiment directory based on output_dir setting or JSON location
    config_path = Path(args.config).resolve()
    json_dir = config_path.parent
    
    if config['output_dir'] is None:
        # Use JSON file location
        experiment_dir = json_dir / config['name']
    else:
        # Use specified output_dir
        experiment_dir = Path(config['output_dir']) / config['name']

    # Set results directory
    results_dir = experiment_dir / "results"
    
    # Handle no-truncation mode
    if args.notruncation:
        print("\n" + "="*70)
        print("NO-TRUNCATION MODE")
        print("="*70)
        print(f"\nExperiment: {config['name']}")
        print(f"Configuration file: {args.config}")
        print(f"Experiment directory: {experiment_dir}")
        print(f"Results directory: {results_dir}")
        
        # Create output directory for no-truncation plots
        no_trunc_dir = experiment_dir / "analysis" / "no_truncation"
        no_trunc_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {no_trunc_dir}")
        
        # Get approaches from config
        if config.get('approaches') is not None:
            approach_specs = config['approaches']
        else:
            approach_specs = ApproachFactory.get_all_keys()
        
        # Create approach instances and build lookup tables
        # Use specs directly to get correct storage keys with min_coactions, ranking_mode, and _nofilter
        approach_storage_keys = []
        approach_names = {}
        for spec in approach_specs:
            # Create approach from spec (not with hardcoded defaults)
            # This ensures storage keys match what run_experiments.py generates
            approach = ApproachFactory.create(spec)
            storage_key = approach.get_full_approach_key()
            approach_storage_keys.append(storage_key)
            
            # Use storage_key as the lookup key (it's always a string)
            approach_names[storage_key] = approach.get_full_approach_name()
        
        print(f"Approaches: {', '.join(str(s) if isinstance(s, str) else s['name'] for s in approach_specs)}")
        
        # Load all results
        print(f"\nLoading results...")
        all_results = load_all_results(results_dir, approach_storage_keys, approach_storage_keys)
        
        if not all_results:
            print(f"ERROR: No results found in {results_dir}!")
            print("Have you run the experiment first using run_experiments.py?")
            return
        
        print(f"Loaded {len(all_results)} datasets")
        
        # Generate plots without truncation
        print(f"\nGenerating plots without truncation...")
        for dataset_name, dataset_results in sorted(all_results.items()):
            print(f"\nProcessing: {dataset_name}")
            
            # Check if all approaches are available
            if not all(approach_key in dataset_results for approach_key in approach_storage_keys):
                missing = [key for key in approach_storage_keys if key not in dataset_results]
                print(f"  WARNING: Incomplete results, skipping (missing: {', '.join(missing)})")
                continue
            
            # Generate plot without truncation
            plot_method_comparison_no_truncation(
                dataset_name, dataset_results, approach_storage_keys, 
                approach_names, no_trunc_dir
            )
        
        print("\n" + "="*70)
        print("NO-TRUNCATION PLOTS COMPLETE")
        print("="*70)
        print(f"\nPlots saved to: {no_trunc_dir}/")
        return
    
    # Determine which metrics to process
    if args.metric == 'all':
        metrics_to_process = ['auc', 'ndcg', 'ap']
        print(f"\nProcessing all metrics: {', '.join(metrics_to_process)}")
    else:
        metrics_to_process = [args.metric]
    
    # Get approaches from config, or use all if not specified
    if config.get('approaches') is not None:
        approach_specs = config['approaches']
    else:
        # Get all available approaches
        approach_specs = ApproachFactory.get_all_keys()

    # Create approach instances to get storage keys and display names
    # Use specs directly to ensure storage keys match run_experiments.py output
    # (including min_coactions, ranking_mode, and _nofilter suffix)
    approach_storage_keys = []
    approach_names = {}

    for spec in approach_specs:
        # Create approach from spec without hardcoded defaults
        approach = ApproachFactory.create(spec)
        storage_key = approach.get_full_approach_key()  # Use full key to include ranking mode suffix and _nofilter
        approach_storage_keys.append(storage_key)
        approach_names[storage_key] = approach.get_full_approach_name()  # Use full name to include ranking mode

    # Create general plots directory (shared across all metrics)
    general_plots_dir = experiment_dir / "analysis" / "plots"
    # Only generate plots on the first metric to avoid redundant work
    generate_plots = not args.no_plots and args.metric != 'all'
    
    if generate_plots and args.metric == 'all':
        # If processing all metrics, only generate plots once (on first metric)
        generate_plots = True
        generate_plots_on_first_metric_only = True
    else:
        generate_plots_on_first_metric_only = False
    
    if generate_plots or (args.metric == 'all' and not args.no_plots):
        print(f"\nGenerating comparison plots...")
        general_plots_dir.mkdir(parents=True, exist_ok=True)

    # Process each metric
    for metric_idx, current_metric in enumerate(metrics_to_process):
        # Only generate plots on first metric if processing all metrics
        should_generate_plots = not args.no_plots and (args.metric != 'all' or metric_idx == 0)
        # Set output directory for this metric
        output_dir = experiment_dir / "analysis" / current_metric
        
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*70)
        print("EXPERIMENT RESULTS ANALYSIS")
        print("="*70)
        print(f"\nExperiment: {config['name']}")
        print(f"Configuration file: {args.config}")
        print(f"Experiment directory: {experiment_dir}")
        print(f"Results directory: {results_dir}")
        print(f"Analysis output: {output_dir}")
        print(f"Metric: {current_metric}")
        print(f"Approaches: {', '.join(str(s) if isinstance(s, str) else s['name'] for s in approach_specs)}")

        # Load all results (only once, reuse for all metrics)
        if current_metric == metrics_to_process[0]:
            print(f"\nLoading results...")
            all_results = load_all_results(results_dir, approach_storage_keys, approach_storage_keys)

            if not all_results:
                print(f"ERROR: No results found in {results_dir}!")
                print("Have you run the experiment first using run_experiments.py?")
                return

            print(f"Loaded {len(all_results)} datasets")

        # Method names (in consistent order)
        method_names = [approach_names[key] for key in approach_storage_keys]

        # Store rankings and scores for each dataset
        all_rankings = []
        all_scores = []
        dataset_names = []
        all_first_nonio_counts = []
        all_users_to_80pct = []
        all_total_io_counts = []
        all_total_accounts_counts = []

        # Use general plots directory only on first metric if processing all metrics
        plots_dir = general_plots_dir if should_generate_plots else None

        for dataset_name, dataset_results in sorted(all_results.items()):
            rankings, scores, first_nonio, users80, total_io, total_accounts = process_dataset(
                dataset_name, dataset_results, approach_storage_keys, approach_names,
                plots_dir, current_metric, should_generate_plots, ideal_mode=args.ideal
            )

            if rankings is None:
                continue

            all_rankings.append(rankings)
            all_scores.append(scores)
            all_first_nonio_counts.append(first_nonio)
            all_users_to_80pct.append(users80)
            all_total_io_counts.append(total_io)
            all_total_accounts_counts.append(total_accounts)
            dataset_names.append(dataset_name)

        if len(all_rankings) == 0:
            print("\nERROR: No complete datasets found for analysis!")
            continue

        # Convert to numpy arrays
        rankings_matrix = np.array(all_rankings)
        scores_matrix = np.array(all_scores)

        # Display results
        metric_display = current_metric.upper()
        display_results(dataset_names, method_names, rankings_matrix, scores_matrix, metric_display)

        # Perform statistical tests
        statistic, p_value, n_methods, median_diff, cd, pvalues_matrix, avg_ranks = \
            perform_statistical_tests(rankings_matrix, scores_matrix, method_names, metric_display)

        # Generate critical difference diagram for multiple methods
        if n_methods > 2:
            cd_path = Path(output_dir) / "critical_difference_diagram.png"
            plot_critical_difference_diagram(scores_matrix, method_names, cd_path, lower_better=False)

        # Save summary report (metric-specific)
        avg_scores = np.mean(scores_matrix, axis=0)
        summary_path = Path(output_dir) / "summary.txt"
        write_metric_summary(
            summary_path, metric_display, dataset_names, method_names,
            scores_matrix, rankings_matrix, avg_scores, avg_ranks,
            statistic, p_value, n_methods, median_diff, cd, pvalues_matrix
        )

        print(f"\nSummary report saved: {summary_path}")

    # After processing all metrics, create a general summary with metric-independent info
    general_summary_path = experiment_dir / "analysis" / "summary.txt"
    write_general_summary(
        general_summary_path, config['name'], dataset_names, method_names,
        approach_storage_keys, all_first_nonio_counts, all_users_to_80pct,
        all_total_io_counts, all_total_accounts_counts
    )

    print(f"\nGeneral summary saved: {general_summary_path}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nOutputs saved to: {experiment_dir / 'analysis'}/")
    print(f"  - General summary: {general_summary_path}")
    for metric in metrics_to_process:
        if args.metric == 'all':
            metric_dir = experiment_dir / "analysis" / metric
        else:
            metric_dir = experiment_dir / "analysis"
        print(f"  - {metric.upper()} summary: {metric_dir / 'summary.txt'}")
        if not args.no_plots:
            print(f"  - {metric.upper()} plots: {metric_dir / 'plots/'}")
        if n_methods > 2:
            print(f"  - {metric.upper()} CD diagram: {metric_dir / 'critical_difference_diagram.png'}")


if __name__ == '__main__':
    main()
