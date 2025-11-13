#!/usr/bin/env python3
"""
Analyze experiment results and perform statistical comparisons.

This script:
1. Loads results from all processed dataset+approach combinations
2. Generates comparison plots for each dataset
3. Computes rankings for each method
4. Performs Friedman test
5. Conducts post-hoc analysis (Nemenyi test)
6. Generates critical difference diagrams
"""

import os
import sys
import pickle
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from synchronous_repeated_detection import plot_comparison, _compute_curve


# Detection approaches (must match run_experiments.py)
APPROACHES = {
    'coretweets': 'Co-retweets',
    'coretweets_numpy': 'Co-retweets (NumPy)',
    'ignoring_tweet': 'Ignoring Tweet',
    'shared_tweets': 'Shared Tweets',
    'same_tweet_same_time': 'Same Tweet Same Time'
}


def load_all_results(results_dir):
    """
    Load all result files from the results directory.

    Returns:
        dict: {dataset_name: {approach: results_dict}}
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

        # Load each approach file
        for approach in APPROACHES.keys():
            pkl_file = dataset_dir / f"{approach}.pkl"

            if pkl_file.exists():
                try:
                    with open(pkl_file, 'rb') as f:
                        all_results[dataset_name][approach] = pickle.load(f)
                except Exception as e:
                    print(f"WARNING: Failed to load {pkl_file}: {e}")

    return dict(all_results)


def compute_area_under_curve(pairs_scores, io_users):
    """
    Compute normalized area under the detection curve.

    Returns area as a fraction of the ideal area (higher is better).
    """
    if not pairs_scores:
        return 0.0

    x_vals, y_vals = _compute_curve(pairs_scores, set(io_users))

    # Total users and IO users in the pairs
    users_in_pairs = set([u for p in pairs_scores.keys() for u in p])
    total_users = len(users_in_pairs)
    total_io = len(users_in_pairs & set(io_users))

    if total_io == 0:
        return 0.0

    # Compute areas
    area_obtained = float(np.sum(y_vals))
    area_ideal = float(np.sum([min(x, total_io) for x in range(total_users + 1)]))

    if area_ideal == 0:
        return 0.0

    # Return as fraction of ideal (0 to 1, higher is better)
    return area_obtained / area_ideal


def plot_method_comparison(dataset_name, dataset_results, output_dir):
    """
    Generate comparison plots for all methods on a dataset.
    """
    # Check if all approaches are available
    if not all(approach in dataset_results for approach in APPROACHES.keys()):
        print(f"  WARNING: Not all approaches available for {dataset_name}, skipping plot")
        return None

    io_users = set(dataset_results['coretweets']['io_users'])

    # Extract pairs_scores for each method
    methods = {
        APPROACHES[approach]: dataset_results[approach]['pairs_scores']
        for approach in APPROACHES.keys()
    }

    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Method Comparison - {dataset_name}', fontsize=16, fontweight='bold')

    baseline_scores = methods['Co-retweets']

    plot_configs = [
        ('Ignoring Tweet', methods['Ignoring Tweet'], axes[0, 0]),
        ('Shared Tweets', methods['Shared Tweets'], axes[0, 1]),
        ('Same Tweet Same Time', methods['Same Tweet Same Time'], axes[1, 0]),
    ]

    for method_name, pairs_scores, ax in plot_configs:
        # Compute curves
        x1, y1 = _compute_curve(pairs_scores, io_users)
        x2, y2 = _compute_curve(baseline_scores, io_users)

        # Calculate differences
        diff = y1 - y2
        total_area = np.sum(diff)
        negative_area = np.sum(np.minimum(0, diff))

        # Plot
        ax.plot(x1, y1, label=method_name, linewidth=2)
        ax.plot(x2, y2, label='Co-retweets (baseline)', linewidth=2, linestyle='--')

        ax.set_xlabel("Number of users studied")
        ax.set_ylabel("Number of IO users detected")
        ax.set_title(f'{method_name} vs Co-retweets')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add text box with area information
        textstr = f'Total area diff: {total_area:.2f}\n'
        textstr += f'Negative area: {negative_area:.2f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', bbox=props)

    # Use the last subplot for summary statistics
    ax = axes[1, 1]
    ax.axis('off')

    # Compute area scores for all methods
    scores = {}
    for name, pairs in methods.items():
        scores[name] = compute_area_under_curve(pairs, io_users)

    # Sort by score
    sorted_methods = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Display summary
    summary_text = "Performance Summary\n" + "="*30 + "\n\n"
    summary_text += "Method Rankings (by AUC):\n\n"

    for rank, (name, score) in enumerate(sorted_methods, 1):
        summary_text += f"{rank}. {name:20s} {score:.4f}\n"

    # Get dataset info from coretweets results
    coretweet_result = dataset_results['coretweets']
    users_in_pairs = set([u for p in coretweet_result['pairs_scores'].keys() for u in p])

    summary_text += f"\n\nDataset Info:\n"
    summary_text += f"Total users: {len(users_in_pairs)}\n"
    summary_text += f"IO users: {len(users_in_pairs & io_users)}\n"
    summary_text += f"Pairs: {coretweet_result['num_pairs']}\n"

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()

    # Save figure
    output_path = Path(output_dir) / f"{dataset_name}_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved plot: {output_path}")

    # Return rankings for this dataset
    return {name: rank for rank, (name, _) in enumerate(sorted_methods, 1)}


def friedman_test(rankings_matrix):
    """
    Perform Friedman test on rankings matrix.

    Args:
        rankings_matrix: 2D array where rows are datasets and columns are methods

    Returns:
        statistic, p_value
    """
    return stats.friedmanchisquare(*rankings_matrix.T)


def nemenyi_test(rankings_matrix, method_names):
    """
    Perform Nemenyi post-hoc test.

    Returns matrix of p-values for pairwise comparisons.
    """
    n_datasets, n_methods = rankings_matrix.shape

    # Compute average ranks
    avg_ranks = np.mean(rankings_matrix, axis=0)

    # Critical difference for Nemenyi test
    # CD = q_alpha * sqrt(k(k+1) / (6N))
    # where q_alpha is from studentized range distribution

    from scipy.stats import studentized_range

    # For alpha=0.05, k methods, infinity df
    q_alpha = studentized_range.ppf(0.95, n_methods, np.inf)

    cd = q_alpha * np.sqrt(n_methods * (n_methods + 1) / (6 * n_datasets))

    # Compute pairwise differences
    p_values = np.ones((n_methods, n_methods))

    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            rank_diff = abs(avg_ranks[i] - avg_ranks[j])
            # If difference exceeds CD, methods are significantly different
            p_values[i, j] = p_values[j, i] = 1.0 if rank_diff < cd else 0.0

    return avg_ranks, cd, p_values


def plot_critical_difference_diagram(avg_ranks, cd, method_names, output_path):
    """
    Generate critical difference diagram (CD diagram).

    Shows average ranks and connects methods that are NOT significantly different.
    """
    n_methods = len(method_names)

    # Sort methods by average rank
    sorted_indices = np.argsort(avg_ranks)
    sorted_names = [method_names[i] for i in sorted_indices]
    sorted_ranks = avg_ranks[sorted_indices]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot settings
    y_positions = np.arange(n_methods)
    rank_min, rank_max = 1, n_methods

    # Draw horizontal lines for each method
    for i, (name, rank) in enumerate(zip(sorted_names, sorted_ranks)):
        ax.plot([rank, rank], [i - 0.3, i + 0.3], 'k-', linewidth=2)
        ax.text(rank_max + 0.3, i, name, va='center', fontsize=11)

    # Draw connections for non-significant differences
    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            idx_i = sorted_indices[i]
            idx_j = sorted_indices[j]

            if abs(sorted_ranks[i] - sorted_ranks[j]) < cd:
                # Not significantly different - draw connection
                y_conn = max(i, j) + 0.5
                ax.plot([sorted_ranks[i], sorted_ranks[j]],
                       [y_conn, y_conn], 'b-', linewidth=2, alpha=0.6)

    # Add critical difference bar
    cd_y = -1
    cd_center = (rank_min + rank_max) / 2
    ax.plot([cd_center - cd/2, cd_center + cd/2], [cd_y, cd_y], 'r-', linewidth=3)
    ax.plot([cd_center - cd/2, cd_center - cd/2], [cd_y - 0.1, cd_y + 0.1], 'r-', linewidth=2)
    ax.plot([cd_center + cd/2, cd_center + cd/2], [cd_y - 0.1, cd_y + 0.1], 'r-', linewidth=2)
    ax.text(cd_center, cd_y - 0.5, f'CD = {cd:.3f}', ha='center', fontsize=10, color='red')

    # Styling
    ax.set_xlim(rank_min - 0.5, rank_max + 3)
    ax.set_ylim(-2, n_methods)
    ax.set_xlabel('Average Rank', fontsize=12)
    ax.set_title('Critical Difference Diagram\n(methods connected by blue lines are not significantly different)',
                 fontsize=14, fontweight='bold')
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nCritical difference diagram saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze experiment results and perform statistical comparisons"
    )
    parser.add_argument(
        '--results-dir',
        default='results',
        help='Directory containing result files (default: results)'
    )
    parser.add_argument(
        '--output-dir',
        default='analysis',
        help='Directory to save analysis outputs (default: analysis)'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating individual comparison plots'
    )

    args = parser.parse_args()

    print("="*70)
    print("EXPERIMENT RESULTS ANALYSIS")
    print("="*70)

    # Load all results
    print(f"\nLoading results from: {args.results_dir}")
    all_results = load_all_results(args.results_dir)

    if not all_results:
        print("ERROR: No results found!")
        return

    print(f"Loaded {len(all_results)} datasets")

    # Method names (in consistent order)
    method_names = list(APPROACHES.values())

    # Store rankings for each dataset
    all_rankings = []
    dataset_names = []

    # Generate plots and collect rankings
    if not args.no_plots:
        print(f"\nGenerating comparison plots...")
        plots_dir = Path(args.output_dir) / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name, dataset_results in sorted(all_results.items()):
        print(f"\nProcessing: {dataset_name}")

        # Check if all approaches are available
        if not all(approach in dataset_results for approach in APPROACHES.keys()):
            print(f"  WARNING: Incomplete results, skipping")
            continue

        # Generate plot or just compute rankings
        if not args.no_plots:
            rankings = plot_method_comparison(dataset_name, dataset_results, plots_dir)
        else:
            # Just compute rankings without plotting
            io_users = set(dataset_results['coretweets']['io_users'])
            scores = {
                APPROACHES[approach]: compute_area_under_curve(
                    dataset_results[approach]['pairs_scores'], io_users
                )
                for approach in APPROACHES.keys()
            }
            sorted_methods = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            rankings = {name: rank for rank, (name, _) in enumerate(sorted_methods, 1)}

        if rankings is None:
            continue

        # Store rankings in consistent order
        dataset_rankings = [rankings[method] for method in method_names]
        all_rankings.append(dataset_rankings)
        dataset_names.append(dataset_name)

    if len(all_rankings) == 0:
        print("\nERROR: No complete datasets found for analysis!")
        return

    # Convert to numpy array
    rankings_matrix = np.array(all_rankings)

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

    # Friedman test
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

    avg_ranks, cd, p_values = nemenyi_test(rankings_matrix, method_names)

    print(f"Critical Difference (CD): {cd:.4f}")
    print(f"\nAverage Ranks:")
    for name, rank in zip(method_names, avg_ranks):
        print(f"  {name:<25} {rank:.3f}")

    print(f"\nPairwise Comparisons:")
    print(f"(1.0 = significantly different, 0.0 = not significantly different)")
    print(f"\n{'':>20} " + " ".join([f"{m[:10]:>12}" for m in method_names]))
    for i, name_i in enumerate(method_names):
        row = f"{name_i[:20]:<20} "
        for j in range(len(method_names)):
            if i == j:
                row += f"{'--':>12} "
            else:
                row += f"{p_values[i, j]:>12.2f} "
        print(row)

    # Generate critical difference diagram
    print("\n" + "-"*70)
    print("Generating Critical Difference Diagram...")
    print("-"*70)

    cd_path = Path(args.output_dir) / "critical_difference_diagram.png"
    plot_critical_difference_diagram(avg_ranks, cd, method_names, cd_path)

    # Save summary report
    summary_path = Path(args.output_dir) / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("EXPERIMENT RESULTS ANALYSIS SUMMARY\n")
        f.write("="*70 + "\n\n")

        f.write(f"Number of datasets: {len(dataset_names)}\n")
        f.write(f"Methods compared: {len(method_names)}\n\n")

        f.write("Average Ranks:\n")
        for name, rank in zip(method_names, avg_ranks):
            f.write(f"  {name:<25} {rank:.3f}\n")

        f.write(f"\nFriedman Test:\n")
        f.write(f"  Chi-square: {statistic:.4f}\n")
        f.write(f"  P-value: {p_value:.6f}\n")
        f.write(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}\n")

        f.write(f"\nNemenyi Test:\n")
        f.write(f"  Critical Difference: {cd:.4f}\n")

    print(f"\nSummary report saved: {summary_path}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nOutputs saved to: {args.output_dir}/")
    if not args.no_plots:
        print(f"  - Individual plots: {args.output_dir}/plots/")
    print(f"  - CD diagram: {args.output_dir}/critical_difference_diagram.png")
    print(f"  - Summary: {args.output_dir}/summary.txt")


if __name__ == '__main__':
    main()
