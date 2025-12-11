"""
Visualization utilities for experiment analysis.

This module provides functions for creating plots and diagrams:
- Method comparison plots (showing detection curves)
- Critical difference diagrams (for statistical significance)
"""

import math
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from aeon.visualisation import plot_critical_difference

from .metrics import compute_score
from .statistics import compute_rankings_with_ties


def _clean_approach_name(name: str) -> str:
    """
    Remove (Linf) and [Linf] from approach names.
    
    Args:
        name: The approach name string
        
    Returns:
        The cleaned name with (Linf) and [Linf] removed
    """
    # Remove (Linf) and [Linf] patterns
    cleaned = re.sub(r'\s*\[Linf\]', '', name)
    cleaned = re.sub(r'\s*\(Linf\)', '', cleaned)
    return cleaned


def plot_method_comparison(dataset_name, dataset_results, approach_keys, approach_names, 
                          output_dir, metric='auc', compute_curve_func=None):
    """
    Generate comparison plots for all methods on a dataset.

    Args:
        dataset_name: Name of the dataset
        dataset_results: Dictionary of results for each approach
        approach_keys: List of approach keys in the experiment
        approach_names: Dictionary mapping approach keys to display names
        output_dir: Directory to save plots
        metric: Metric to use for scoring (default: 'auc')
        compute_curve_func: Function to compute detection curves (x, y) from (suspicious_users, io_users)

    Returns:
        Dictionary mapping method names to their ranks, or None if plot cannot be generated
    """
    # Check if all approaches are available
    if not all(approach_key in dataset_results for approach_key in approach_keys):
        print(f"  WARNING: Not all approaches available for {dataset_name}, skipping plot")
        return None

    # Get IO users from first available approach
    first_approach = approach_keys[0]
    io_users = set(dataset_results[first_approach]['io_users'])

    # Extract suspicious_users for each method
    methods = {
        approach_names[approach_key]: dataset_results[approach_key]['suspicious_users']
        for approach_key in approach_keys
    }

    # Determine grid size based on number of approaches
    n_methods = len(methods) - 1  # exclude baseline
    n_plots = max(1, n_methods)

    # Compute grid: as square as possible
    n_cols = math.ceil(math.sqrt(n_plots))
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = np.array(axes).flatten()  # make it iterable

    # Use first method as baseline
    baseline_name = list(methods.keys())[0]
    baseline_suspicious = methods[baseline_name]

    # Plot each method vs baseline (except baseline itself)
    plot_idx = 0
    for method_name, suspicious_users in methods.items():
        if method_name == baseline_name:
            continue

        if plot_idx >= len(axes):
            break

        ax = axes[plot_idx]

        # Compute curves
        if compute_curve_func is None:
            # Use default curve computation
            from synchronous_repeated_detection import _compute_curve
            compute_curve_func = _compute_curve
            
        x1, y1 = compute_curve_func(suspicious_users, io_users)
        x2, y2 = compute_curve_func(baseline_suspicious, io_users)

        if x1[-1] > x2[-1]:
            x1 = x1[:x2[-1]+1]
            y1 = y1[:x2[-1]+1]
        elif x2[-1] > x1[-1]:
            x2 = x2[:x1[-1]+1]
            y2 = y2[:x1[-1]+1]
        
        x_max = len(x2)

        # Calculate differences
        diff = y1 - y2
        total_area = np.sum(diff)
        negative_area = np.sum(np.minimum(0, diff))

        # Plot
        ax.plot(x1, y1, label=method_name, linewidth=2)
        ax.plot(x2, y2, label=f'{baseline_name} (baseline)', linewidth=2, linestyle='--')

        ax.set_xlabel("Number of users studied")
        ax.set_ylabel("Number of IO users detected")
        ax.set_title(f'{method_name} vs {baseline_name}')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add text box with area information
        textstr = f'Total area diff: {total_area:.2f}\n'
        textstr += f'Negative area: {negative_area:.2f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.65, 0.25, textstr, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', bbox=props)

        plot_idx += 1

    # Hide unused axes AFTER the summary
    for i in range(plot_idx, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    # Save figure
    output_path = Path(output_dir) / f"{dataset_name}_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved plot: {output_path}")

    # Compute scores for all methods using specified metric
    scores = {}
    for name, users in methods.items():
        scores[name] = compute_score(users, io_users, x_max, metric=metric)

    # Compute rankings with proper tie handling
    rankings = compute_rankings_with_ties(scores)

    # Return rankings for this dataset
    return rankings


def plot_critical_difference_diagram(scores_matrix, method_names, output_path, lower_better=False):
    """
    Generate critical difference diagram (CD diagram) using aeon library.

    This function uses the standard aeon.visualisation.plot_critical_difference
    which properly handles statistical testing and visualization.
    
    Args:
        scores_matrix: 2D array where rows are datasets and columns are methods (raw scores, not ranks)
        method_names: List of method names
        output_path: Path to save the diagram
        lower_better: If True, lower scores are better (e.g., error rates).
                     If False, higher scores are better (e.g., accuracy, AUC)
    """
    # Clean method names to remove (Linf) and [Linf]
    cleaned_method_names = [_clean_approach_name(name) for name in method_names]
    
    # Set figure size - using width parameter for better control
    # Adjust size based on number of methods for readability
    n_methods = len(cleaned_method_names)
    fig_width = max(6, n_methods * 0.8)
    fig_height = max(5, n_methods * 0.6)
    
    plt.figure(figsize=(fig_width, fig_height))
    
    # Generate the critical difference diagram using aeon
    # test='nemenyi' uses the Nemenyi post-hoc test
    # correction='holm' applies Holm correction for multiple comparisons
    plot_critical_difference(
        scores_matrix,
        cleaned_method_names,
        lower_better=lower_better,
        test='nemenyi',
        correction='holm',
    )
    
    # Get the current axes for customization
    ax = plt.gca()
    
    # Adjust margins to provide more space for labels
    plt.subplots_adjust(bottom=0.25, left=0.1, right=0.95)
    
    # Optionally adjust figure size if needed (workaround for size issues)
    fig = plt.gcf()
    fig.set_size_inches(fig_width, fig_height)
    
    # Save and close
    plt.savefig(output_path, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"\nCritical difference diagram saved: {output_path}")
