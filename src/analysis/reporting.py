"""
Report generation utilities for experiment analysis.

This module provides functions for generating text-based summary reports:
- Metric-specific summaries (with rankings and statistical tests)
- General summaries (with method-independent metrics)
"""

import numpy as np


def write_metric_summary(summary_path, metric_display, dataset_names, method_names,
                        scores_matrix, rankings_matrix, avg_scores, avg_ranks,
                        statistic, p_value, n_methods, median_diff=None, 
                        cd=None, pvalues_matrix=None):
    """
    Write metric-specific summary report to file.

    Args:
        summary_path: Path to output file
        metric_display: Display name of the metric (e.g., 'AUC', 'NDCG')
        dataset_names: List of dataset names
        method_names: List of method names
        scores_matrix: 2D array of metric scores
        rankings_matrix: 2D array of rankings
        avg_scores: Average scores for each method
        avg_ranks: Average ranks for each method
        statistic: Test statistic from statistical test
        p_value: P-value from statistical test
        n_methods: Number of methods being compared
        median_diff: Median difference (for Wilcoxon test, optional)
        cd: Critical difference value (for Nemenyi test, optional)
        pvalues_matrix: Matrix of pairwise p-values (for Nemenyi test, optional)
    """
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"EXPERIMENT RESULTS ANALYSIS SUMMARY ({metric_display})\n")
        f.write("="*70 + "\n\n")

        f.write(f"Metric: {metric_display}\n")
        f.write(f"Number of datasets: {len(dataset_names)}\n")
        f.write(f"Methods compared: {len(method_names)}\n\n")

        # Calculate dynamic column width based on longest method name
        col_width = max(len(m) for m in method_names) + 2
        col_width = max(col_width, 12)  # minimum width of 12

        f.write("=" * 70 + "\n")
        f.write(f"{metric_display} SCORES MATRIX\n")
        f.write("=" * 70 + "\n\n")
        
        # Create header with full method names
        header = f"{'Dataset':<20} " + " ".join([f"{m:>{col_width}}" for m in method_names])
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        
        # Write each dataset's scores
        for dataset, scores in zip(dataset_names, scores_matrix):
            f.write(f"{dataset:<20} " + " ".join([f"{s:>{col_width}.4f}" for s in scores]) + "\n")
        
        f.write("\n")
        f.write(f"{'Average ' + metric_display:<20} " + " ".join([f"{s:>{col_width}.4f}" for s in avg_scores]) + "\n")
        f.write("\n\n")

        f.write("=" * 70 + "\n")
        f.write("RANKINGS MATRIX\n")
        f.write("=" * 70 + "\n\n")
        
        # Create header with full method names
        header = f"{'Dataset':<20} " + " ".join([f"{m:>{col_width}}" for m in method_names])
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        
        # Write each dataset's rankings
        for dataset, ranks in zip(dataset_names, rankings_matrix):
            f.write(f"{dataset:<20} " + " ".join([f"{r:>{col_width}.0f}" for r in ranks]) + "\n")
        
        f.write("\n")
        f.write(f"{'Average Rank':<20} " + " ".join([f"{r:>{col_width}.2f}" for r in avg_ranks]) + "\n")
        f.write("\n\n")

        f.write("=" * 70 + "\n")
        f.write("STATISTICAL TESTS\n")
        f.write("=" * 70 + "\n\n")

        if n_methods == 2:
            f.write(f"Wilcoxon Signed-Rank Test:\n")
            f.write(f"  Test statistic: {statistic:.4f}\n")
            f.write(f"  P-value: {p_value:.6f}\n")
            f.write(f"  Median difference ({metric_display}): {median_diff:.4f}\n")
            f.write(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}\n")
            if p_value < 0.05:
                winner = method_names[0] if median_diff > 0 else method_names[1]
                f.write(f"  Winner: {winner}\n")
        else:
            f.write(f"Friedman Test:\n")
            f.write(f"  Chi-square: {statistic:.4f}\n")
            f.write(f"  P-value: {p_value:.6f}\n")
            f.write(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}\n\n")

            f.write(f"Nemenyi Post-hoc Test:\n")
            f.write(f"  Critical Difference: {cd:.4f}\n\n")

            f.write("  Pairwise P-values:\n")
            f.write("  (p-value < 0.05 indicates significant difference)\n\n")
            
            # Calculate dynamic column width for p-values table
            pval_col_width = max(len(m) for m in method_names) + 2
            pval_col_width = max(pval_col_width, 10)  # minimum width of 10
            
            # Write header for p-values matrix
            header = "  " + f"{'':>{pval_col_width + 8}} " + " ".join([f"{m:>{pval_col_width}}" for m in method_names])
            f.write(header + "\n")
            
            # Write each row of p-values
            for i, name_i in enumerate(method_names):
                row = "  " + f"{name_i[:pval_col_width + 8]:<{pval_col_width + 8}} "
                for j in range(len(method_names)):
                    if i == j:
                        row += f"{'1.000':>{pval_col_width}} "
                    else:
                        row += f"{pvalues_matrix[i, j]:>{pval_col_width}.4f} "
                f.write(row + "\n")
            
            # Add significant pairs
            f.write(f"\n  Significant differences (p < 0.05):\n")
            sig_pairs = []
            for i in range(len(method_names)):
                for j in range(i + 1, len(method_names)):
                    if pvalues_matrix[i, j] < 0.05:
                        sig_pairs.append((method_names[i], method_names[j], pvalues_matrix[i, j]))
            
            if sig_pairs:
                for name1, name2, pval in sig_pairs:
                    f.write(f"    {name1} vs {name2}: p = {pval:.4f}\n")
            else:
                f.write("    No significant pairwise differences found\n")


def write_general_summary(summary_path, experiment_name, dataset_names, method_names,
                         approach_keys, all_first_nonio_counts, all_users_to_80pct):
    """
    Write general summary report with metric-independent information.

    Args:
        summary_path: Path to output file
        experiment_name: Name of the experiment
        dataset_names: List of dataset names
        method_names: List of method names
        approach_keys: List of approach keys
        all_first_nonio_counts: 2D array of IO counts until first non-IO
        all_users_to_80pct: 2D array of users needed to reach 80% IO
    """
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("EXPERIMENT RESULTS ANALYSIS - GENERAL SUMMARY\n")
        f.write("="*70 + "\n\n")

        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Number of datasets: {len(dataset_names)}\n")
        f.write(f"Methods compared: {len(method_names)}\n\n")

        f.write("Methods:\n")
        for i, (key, name) in enumerate(zip(approach_keys, method_names), 1):
            f.write(f"  {i}. {name} ({key})\n")
        f.write("\n")

        f.write("Datasets:\n")
        for i, dataset in enumerate(sorted(dataset_names), 1):
            f.write(f"  {i}. {dataset}\n")
        f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("IO UNTIL FIRST NON-IO (Counts)\n")
        f.write("=" * 70 + "\n\n")

        # Calculate dynamic column width
        col_width = max(len(m) for m in method_names) + 2
        col_width = max(col_width, 12)  # minimum width of 12
        
        header = f"{'Dataset':<20} " + " ".join([f"{m:>{col_width}}" for m in method_names])
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for dataset, row in zip(dataset_names, all_first_nonio_counts):
            f.write(f"{dataset:<20} " + " ".join([f"{val:>{col_width}d}" for val in row]) + "\n")
        
        # Compute and write mean for IO until first non-IO
        first_nonio_matrix = np.array(all_first_nonio_counts)
        avg_first_nonio = np.mean(first_nonio_matrix, axis=0)
        f.write("\n")
        f.write(f"{'Mean':<20} " + " ".join([f"{val:>{col_width}.2f}" for val in avg_first_nonio]) + "\n")

        f.write("\n\n")

        f.write("=" * 70 + "\n")
        f.write("USERS UNTIL 80% IO FOUND\n")
        f.write("=" * 70 + "\n\n")

        header = f"{'Dataset':<20} " + " ".join([f"{m:>{col_width}}" for m in method_names])
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for dataset, row in zip(dataset_names, all_users_to_80pct):
            # Values are ints or -1 when not reached
            def fmt(v):
                return "NA" if v == -1 else str(v)
            f.write(f"{dataset:<20} " + " ".join([f"{fmt(v):>{col_width}}" for v in row]) + "\n")
        
        # Compute and write mean for users until 80% IO found (excluding NA values)
        users_80pct_matrix = np.array(all_users_to_80pct)
        avg_users_80pct = []
        for method_idx in range(len(method_names)):
            # Get values for this method, excluding -1 (NA) values
            valid_values = [val for val in users_80pct_matrix[:, method_idx] if val != -1]
            if valid_values:
                avg_users_80pct.append(np.mean(valid_values))
            else:
                avg_users_80pct.append(-1)  # All were NA
        
        f.write("\n")
        def fmt_mean(v):
            return "NA" if v == -1 else f"{v:.2f}"
        f.write(f"{'Mean':<20} " + " ".join([f"{fmt_mean(val):>{col_width}}" for val in avg_users_80pct]) + "\n")
