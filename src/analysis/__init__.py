"""
Analysis utilities for experiment results.

This package provides tools for analyzing experimental results, including:
- Metrics computation (AUC, NDCG, mAP)
- Statistical testing (Friedman, Wilcoxon, Nemenyi)
- Visualization (comparison plots, critical difference diagrams)
- Report generation
"""

from .metrics import (
    compute_score,
    compute_ndcg,
    compute_average_precision,
    compute_roc_auc,
    count_io_until_first_nonio,
    users_until_reaching_io_fraction,
    precision_k,
    count_total_io_in_suspicious,
    count_total_accounts_in_suspicious,
)

from .statistics import (
    compute_rankings_with_ties,
    wilcoxon_signed_rank_test,
    friedman_test,
    nemenyi_test,
)

from .visualization import (
    plot_method_comparison,
    plot_critical_difference_diagram,
    plot_all_approaches_together,
)

from .io_utils import (
    load_experiment_config,
    load_all_results,
)

__all__ = [
    # Metrics
    'compute_score',
    'compute_ndcg',
    'compute_average_precision',
    'compute_roc_auc',
    'count_io_until_first_nonio',
    'users_until_reaching_io_fraction',
    'precision_k',
    'count_total_io_in_suspicious',
    'count_total_accounts_in_suspicious',
    # Statistics
    'compute_rankings_with_ties',
    'wilcoxon_signed_rank_test',
    'friedman_test',
    'nemenyi_test',
    # Visualization
    'plot_method_comparison',
    'plot_critical_difference_diagram',
    'plot_all_approaches_together',
    # IO
    'load_experiment_config',
    'load_all_results',
]
