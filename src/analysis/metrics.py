"""
Metrics computation for ranking evaluation.

This module provides functions to compute various metrics for evaluating
the quality of suspicious user rankings:
- ROC-AUC (Area Under the Receiver Operating Characteristic curve)
- NDCG (Normalized Discounted Cumulative Gain)
- AP (Average Precision)
- Custom metrics (IO users until first non-IO, users to reach IO fraction)
"""

import numpy as np
from sklearn.metrics import ndcg_score, average_precision_score, roc_auc_score


def _prepare_sklearn_data(suspicious_users, io_users, x_max):
    """
    Prepare data in sklearn format for metric computation.
    
    Args:
        suspicious_users: List of lists of user IDs grouped by score level
        io_users: Set/list of known inauthentic user IDs
        x_max: Maximum number of users to consider
    
    Returns:
        y_true: Binary array (1 for IO users, 0 for non-IO)
        y_score: Score array (higher scores for more suspicious users, same score for ties/unknown ordering)
    """
    if not suspicious_users:
        return np.array([]), np.array([])
    
    # Flatten suspicious_users while preserving order and score groups
    ordered_users = []
    user_scores = {}
    seen = set()
    
    # Assign scores based on position (earlier groups = higher scores)
    max_score = len(suspicious_users)
    for group_idx, group in enumerate(suspicious_users):
        score = max_score - group_idx  # Higher score for earlier groups
        for u in group:
            if u not in seen:
                ordered_users.append(u)
                user_scores[u] = score
                seen.add(u)
    
    # Truncate to x_max
    ordered_users = ordered_users[:x_max]
    
    # Create arrays
    y_true = np.array([1 if u in io_users else 0 for u in ordered_users])
    y_score = np.array([user_scores[u] for u in ordered_users])
    
    return y_true, y_score


def compute_ndcg(suspicious_users, io_users, x_max, k=None):
    """
    Compute Normalized Discounted Cumulative Gain (NDCG).
    
    NDCG measures ranking quality, giving higher weight to correctly ranked
    items at the top of the list. Returns a value between 0 and 1, where
    higher is better.
    
    Args:
        suspicious_users: List of lists of user IDs grouped by score level
        io_users: Set/list of known inauthentic user IDs
        x_max: Maximum number of users to consider
        k: If specified, compute NDCG@k (only top k positions)
    
    Returns:
        NDCG score (0 to 1, higher is better)
    """
    y_true, y_score = _prepare_sklearn_data(suspicious_users, io_users, x_max)
    
    if len(y_true) == 0 or np.sum(y_true) == 0:
        return 0.0
        
    try:
        return float(ndcg_score([y_true], [y_score]))
    except Exception as e:
        print(f"Warning: NDCG computation failed: {e}")
        return 0.0


def compute_average_precision(suspicious_users, io_users, x_max):
    """
    Compute Average Precision (AP).
    
    AP summarizes the precision-recall curve as the weighted mean of precisions
    at each threshold. For a single ranking, this is AP.
    Returns a value between 0 and 1, where higher is better.
    
    Args:
        suspicious_users: List of lists of user IDs grouped by score level
        io_users: Set/list of known inauthentic user IDs
        x_max: Maximum number of users to consider
    
    Returns:
        AP score (0 to 1, higher is better)
    """
    y_true, y_score = _prepare_sklearn_data(suspicious_users, io_users, x_max)
    
    if len(y_true) == 0 or np.sum(y_true) == 0:
        return 0.0
    
    try:
        return float(average_precision_score(y_true, y_score))
    except Exception as e:
        print(f"Warning: AP computation failed: {e}")
        return 0.0


def compute_roc_auc(suspicious_users, io_users, x_max):
    """
    Compute ROC-AUC (Area Under the Receiver Operating Characteristic curve).
    
    ROC-AUC measures the probability that a randomly chosen positive example
    is ranked higher than a randomly chosen negative example. Returns a value
    between 0 and 1, where higher is better (0.5 is random).
    
    Args:
        suspicious_users: List of lists of user IDs grouped by score level
        io_users: Set/list of known inauthentic user IDs
        x_max: Maximum number of users to consider
    
    Returns:
        ROC-AUC score (0 to 1, higher is better)
    """
    y_true, y_score = _prepare_sklearn_data(suspicious_users, io_users, x_max)
    
    if len(y_true) == 0 or np.sum(y_true) == 0:
        return 0.0
    
    # Need at least one positive and one negative sample
    if len(np.unique(y_true)) < 2:
        return 0.0
    
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception as e:
        print(f"Warning: ROC-AUC computation failed: {e}")
        return 0.0

def compute_score(suspicious_users, io_users, x_max, metric='auc'):
    """
    Compute performance score using the specified metric.
    
    This is the main scoring function that dispatches to the appropriate
    metric implementation.
    
    Args:
        suspicious_users: List of lists of user IDs grouped by score level
        io_users: Set/list of known inauthentic user IDs
        x_max: Maximum number of users to consider
        metric: Metric to use ('auc', 'ndcg', 'ap')
                'auc' - sklearn ROC-AUC
                'ndcg' - sklearn NDCG
                'ap' - sklearn Average Precision (AP)
    
    Returns:
        Score value (higher is better for all metrics)
    """
    metric = metric.lower()
    
    if metric == 'auc':
        return compute_roc_auc(suspicious_users, io_users, x_max)
    elif metric == 'ndcg':
        return compute_ndcg(suspicious_users, io_users, x_max)
    elif metric == 'ap':
        return compute_average_precision(suspicious_users, io_users, x_max)
    else:
        raise ValueError(f"Unknown metric: {metric}. Choose from: auc, ndcg, ap")


def count_io_until_first_nonio(suspicious_users, io_users):
    """
    Count how many IO users are found before the first non-IO appears.

    Traverses users in the given suspicious order (groups preserve ties),
    skipping duplicates, and stops at the first non-IO user encountered.

    Args:
        suspicious_users: List[List[user_id]] ordered by score groups
        io_users: Set of known IO users

    Returns:
        Integer count of IO users until the first non-IO; 0 if the first user is non-IO.
    """
    count = 0
    for group in suspicious_users:
        for u in group:
            if u not in io_users:
                return count
        count += len(group)
    return count


def users_until_reaching_io_fraction(suspicious_users, io_users, fraction=0.9):
    """
    Return how many users must be studied to reach a fraction of total IOs.

    Uses the ordered suspicious users and counts how many
    need to be examined to detect at least `fraction` of all IO users in the
    dataset. If the threshold is not reached, returns None.

    Args:
        suspicious_users: List[List[user_id]] ordered by score groups
        io_users: Set of known IO users
        fraction: Target fraction (e.g., 0.9 for 90%)

    Returns:
        Integer number of users studied to reach the target IO fraction, or None if not reached.
    """
    total_io = sum(1 for group in suspicious_users for u in group if u in io_users)
    if total_io == 0:
        return 0

    target = int(np.ceil(total_io * float(fraction)))
    studied = 0
    found_io = 0

    for group in suspicious_users:
        new_users_len = len(group)
        new_ios_len = sum(1 for u in group if u in io_users)

        if found_io + new_ios_len < target:
            # All new users are needed
            studied += new_users_len
            found_io += new_ios_len
        
        else:
            # Only part of new users are needed
            # Linear interpolation within score group
            for i in range(1, new_users_len + 1):
                studied += 1
                if found_io + (new_ios_len * (i / new_users_len)) >= target:
                    return studied

    return None


def count_total_io_in_suspicious(suspicious_users, io_users):
    """
    Count total number of IO accounts in the suspicious users list.
    
    Args:
        suspicious_users: List of lists of user IDs grouped by score level
        io_users: Set of known IO users
    
    Returns:
        Integer count of IO accounts found in suspicious_users
    """
    seen = set()
    count = 0
    for group in suspicious_users:
        for u in group:
            if u not in seen:
                seen.add(u)
                if u in io_users:
                    count += 1
    return count


def count_total_accounts_in_suspicious(suspicious_users):
    """
    Count total number of accounts (both IO and non-IO) in the suspicious users list.
    
    Args:
        suspicious_users: List of lists of user IDs grouped by score level
    
    Returns:
        Integer count of total unique accounts in suspicious_users
    """
    seen = set()
    for group in suspicious_users:
        for u in group:
            seen.add(u)
    return len(seen)


def precision_k(suspicious_users, io_users, k=100):
    """
    Compute precision@k for a suspicious users ranking.

    precision@k = (# IO accounts in the top k positions) / k

    Args:
        suspicious_users: List[List[user_id]] ordered by score groups
        io_users: Set of known IO users
        k: Number of top positions to consider

    Returns:
        Precision@k as a float in [0, 1]. Returns 0.0 if k <= 0.
    """
    if k <= 0:
        return 0.0

    seen = set()
    ordered_users = []

    for group in suspicious_users:
        for u in group:
            if u in seen:
                continue
            ordered_users.append(u)
            seen.add(u)
            if len(ordered_users) >= k:
                break
        if len(ordered_users) >= k:
            break

    io_count = sum(1 for u in ordered_users if u in io_users)
    return float(io_count) / float(k)