"""
Statistical testing and ranking utilities.

This module provides functions for statistical analysis of experimental results:
- Ranking computation with tie handling
- Wilcoxon signed-rank test (for two methods)
- Friedman test (for multiple methods)
- Nemenyi post-hoc test
"""

import numpy as np
from scipy import stats
from scipy.stats import studentized_range
import scikit_posthocs as sp


def compute_rankings_with_ties(scores):
    """
    Compute rankings from scores, assigning the same rank to tied scores.

    Uses standard competition ranking (1224 ranking): if two items are tied for
    first place, they both get rank 1, and the next item gets rank 3.

    Args:
        scores: Dictionary mapping names to scores (higher is better)

    Returns:
        Dictionary mapping names to ranks (1 = best)
    """
    # Sort by score (descending)
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    rankings = {}
    current_rank = 1

    for i, (name, score) in enumerate(sorted_items):
        # Check if this score is tied with the previous one
        if i > 0 and abs(score - sorted_items[i-1][1]) < 1e-10:
            # Same score as previous, use same rank
            rankings[name] = rankings[sorted_items[i-1][0]]
        else:
            # New score, use current rank (which accounts for ties)
            rankings[name] = current_rank

        # Increment rank for next iteration
        current_rank = i + 2

    return rankings


def wilcoxon_signed_rank_test(scores_method1, scores_method2):
    """
    Perform Wilcoxon signed-rank test for paired samples.

    This test is appropriate when comparing exactly two methods across multiple datasets.
    It tests whether the median difference between paired observations is zero.

    Args:
        scores_method1: Array of scores for method 1 across datasets
        scores_method2: Array of scores for method 2 across datasets

    Returns:
        statistic, p_value, median_diff
    """
    # Compute differences
    differences = scores_method1 - scores_method2
    median_diff = np.median(differences)

    # Perform Wilcoxon signed-rank test
    # alternative='two-sided' tests if the distributions differ
    statistic, p_value = stats.wilcoxon(scores_method1, scores_method2,
                                        alternative='two-sided', zero_method='wilcox')

    return statistic, p_value, median_diff


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
    Perform Nemenyi post-hoc test using scikit-posthocs.

    The Nemenyi test is used for pairwise comparisons after a significant Friedman test.
    It returns p-values for all pairwise comparisons.
    
    Args:
        rankings_matrix: 2D array where rows are datasets and columns are methods
        method_names: List of method names

    Returns:
        avg_ranks: Average rank for each method
        cd: Critical difference value
        pvalues_matrix: Matrix of p-values for pairwise comparisons
    """
    n_datasets, n_methods = rankings_matrix.shape

    # Compute average ranks
    avg_ranks = np.mean(rankings_matrix, axis=0)

    # Perform Nemenyi post-hoc test using scikit-posthocs
    # The function expects data in the format: rows=observations, columns=groups
    # This matches our rankings_matrix format
    pvalues_matrix = sp.posthoc_nemenyi_friedman(rankings_matrix)
    
    # Convert DataFrame to numpy array if needed
    if hasattr(pvalues_matrix, 'values'):
        pvalues_matrix = pvalues_matrix.values

    # Calculate critical difference for visualization
    # CD = q_alpha * sqrt(k(k+1) / (6N))
    # where q_alpha is from studentized range distribution
    q_alpha = studentized_range.ppf(0.95, n_methods, np.inf)
    cd = q_alpha * np.sqrt(n_methods * (n_methods + 1) / (6 * n_datasets))

    return avg_ranks, cd, pvalues_matrix
