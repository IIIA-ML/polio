"""
Script for detecting coordinated behavior through synchronous repeated actions.
Analyzes retweet patterns to identify users who repeatedly interact with content
at similar times, which may indicate inauthentic coordinated behavior.
"""

import sys
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.path.abspath(".."))

# Import shared data loading utilities
from data_loader import import_data

def group_RTs_by_user(RTs):
    """Group retweets by user, sorted by timestamp."""
    user_data = defaultdict(list)
    for user, tid, ts in RTs:
        user_data[user].append((tid, ts))
    # Sort each user's retweets chronologically
    for u in user_data:
        user_data[u].sort(key=lambda x: x[1])
    return user_data

def group_RTs_by_tweet(RTs):
    """Group retweets by tweet, sorted by timestamp."""
    tweet_data = defaultdict(list)
    for user, tid, ts in RTs:
        tweet_data[tid].append((user, ts))
    # Sort each tweet's retweets chronologically
    for tid in tweet_data:
        tweet_data[tid].sort(key=lambda x: x[1])
    return tweet_data

def count_coretweets(RTs, window_sec=60, min_coactions=1):
    """
    Count co-retweets: pairs of users who retweeted the same tweet within a time window.
    Returns dict of user pairs and their co-retweet count (filtered by min_coactions).

    This function now delegates to CoRetweetsApproach for backward compatibility.
    """
    from approaches import CoRetweetsApproach
    approach = CoRetweetsApproach(window_sec=window_sec, min_coactions=min_coactions)
    return approach.compute_pairs_scores(RTs)

def filter_RTs(RTs, users):
    """Filter retweets to only include specified users."""
    users_set = set(users)
    filtered_RTs = [rt for rt in RTs if rt[0] in users_set]
    return filtered_RTs

def count_daily_coincidences(times1, times2, window_sec):
    """
    Two-pointer scan: counts distinct days where timestamps are within window_sec.
    Efficiently finds temporal coincidences between two sorted lists of timestamps.
    """
    i = j = 0
    days = set()

    while i < len(times1) and j < len(times2):
        t1 = times1[i]
        t2 = times2[j]
        diff = abs((t1 - t2).total_seconds())

        # If timestamps are within window, record the day
        if diff <= window_sec:
            days.add(max(t1, t2).date())
            i += 1
            j += 1
        # Advance the earlier timestamp
        elif t1 < t2:
            i += 1
        else:
            j += 1

    return len(days)

def coincide_ignoring_tweet(RTs, window_sec=60):
    """
    Count days where two users posted within the time window, regardless of tweet content.
    Measures general temporal synchronization between users.

    This function now delegates to IgnoringTweetApproach for backward compatibility.
    """
    from approaches import IgnoringTweetApproach
    approach = IgnoringTweetApproach(window_sec=window_sec)
    return approach.compute_pairs_scores(RTs)

def coincide_shared_tweets(RTs, window_sec=60):
    """
    Count days where two users retweeted shared content within the time window.
    Only considers tweets both users interacted with.

    This function now delegates to SharedTweetsApproach for backward compatibility.
    """
    from approaches import SharedTweetsApproach
    approach = SharedTweetsApproach(window_sec=window_sec)
    return approach.compute_pairs_scores(RTs)

def coincide_same_tweet_same_time(RTs, window_sec=60):
    """
    Count distinct days where users co-retweeted the same tweet within the time window.
    Strongest indicator of coordination: same content, same time.

    This function now delegates to SameTweetSameTimeApproach for backward compatibility.
    """
    from approaches import SameTweetSameTimeApproach
    approach = SameTweetSameTimeApproach(window_sec=window_sec)
    return approach.compute_pairs_scores(RTs)

def _pairs_scores_to_suspicious_users(pairs_scores):
    """
    Convert pairs_scores dict to suspicious_users list format.

    Helper function to maintain backward compatibility with functions that
    still use pairs_scores instead of the new suspicious_users format.

    Args:
        pairs_scores: Dictionary mapping user pairs (tuples) to their scores

    Returns:
        List of lists where each inner list contains users from pairs at the
        same score level, ordered from highest to lowest scores.
    """
    if not pairs_scores:
        return []

    # Sort pairs by score in descending order
    sorted_pairs = sorted(pairs_scores.items(), key=lambda x: x[1], reverse=True)

    # Track users already seen to avoid duplicates
    users_seen = set()
    suspicious_users = []

    idx = 0
    n = len(sorted_pairs)

    # Process pairs in score blocks (all pairs with same score)
    while idx < n:
        current_score = sorted_pairs[idx][1]
        block_users = set()

        # Collect all users from pairs with the same score
        while idx < n and sorted_pairs[idx][1] == current_score:
            pair, _ = sorted_pairs[idx]
            for user in pair:
                if user not in users_seen and user not in block_users:
                    block_users.add(user)
            idx += 1

        # Add this score block's users as a new list
        if block_users:
            suspicious_users.append(sorted(block_users))
            users_seen.update(block_users)

    return suspicious_users

def _compute_curve(suspicious_users, io_users):
    """
    Helper function to compute x and y values for a detection curve.
    Shows how many inauthentic users are detected as more users are examined.
    For users with equal score, uses linear interpolation within that score group.

    Args:
        suspicious_users: List of lists where each inner list contains users from
                         pairs at the same score level, ordered highest to lowest.
                         Each user appears only once (in their highest-scoring group).
        io_users: Set of known inauthentic user IDs

    Returns:
        (x_vals, y_vals): Numpy arrays showing cumulative users studied vs IO users detected
    """
    x_vals, y_vals = [0], [0.0]  # y_vals stores floats (interpolated counts)
    io_users_count = 0
    users_seen_count = 0

    # suspicious_users is already grouped by score, with no duplicates
    for user_group in suspicious_users:
        new_users_len = len(user_group)
        new_ios_len = sum(1 for u in user_group if u in io_users)

        # Linear interpolation within score group
        for i in range(1, new_users_len + 1):
            x_vals.append(users_seen_count + i)
            y_vals.append(io_users_count + new_ios_len * (i / new_users_len))

        # Update totals
        users_seen_count += new_users_len
        io_users_count += new_ios_len

    return np.array(x_vals), np.array(y_vals)

def plot_ios_vs_studied_approach(approach, RTs, io_users, figsize=(8,5), **kwargs):
    """
    Plot progressive detection of IO users using an Approach instance.

    This function uses the approach's get_suspicious method directly.

    Args:
        approach: An Approach instance (e.g., CoretweetsApproach)
        RTs: List of (user_id, tweet_id, timestamp) tuples
        io_users: Set of known inauthentic user IDs
        figsize: Figure size tuple
        **kwargs: Additional approach-specific parameters
    """
    suspicious_users = approach.get_suspicious(RTs, **kwargs)
    x_vals, y_vals = _compute_curve(suspicious_users, io_users)

    # ---- Plot ----
    plt.figure(figsize=figsize)
    plt.plot(x_vals, y_vals, linestyle='-')

    plt.xlabel("Number of users studied")
    plt.ylabel("Number of IO users detected")
    plt.title(f"Progressive detection of IO users: {approach.get_approach_name()}")
    plt.grid(True)
    plt.show()

def plot_ios_vs_studied(pairs_scores, io_users, figsize=(8,5)):
    """
    Plot progressive detection of IO users as edges are studied.

    Legacy function for backward compatibility. For new code, prefer
    plot_ios_vs_studied_approach() which works directly with Approach instances.
    """
    suspicious_users = _pairs_scores_to_suspicious_users(pairs_scores)
    x_vals, y_vals = _compute_curve(suspicious_users, io_users)

    # ---- Plot ----
    plt.figure(figsize=figsize)
    plt.plot(x_vals, y_vals, linestyle='-')

    plt.xlabel("Number of users studied")
    plt.ylabel("Number of IO users detected")
    plt.title("Progressive detection of IO users (score-group averaged)")
    plt.grid(True)
    plt.show()

def plot_comparison_approaches(approach1, approach2, RTs, io_users,
                              label1=None, label2=None, figsize=(10, 6), **kwargs):
    """
    Compare two Approach instances by plotting their detection curves.

    This function uses the approaches' get_suspicious methods directly.

    Args:
        approach1: First Approach instance
        approach2: Second Approach instance (baseline)
        RTs: List of (user_id, tweet_id, timestamp) tuples
        io_users: Set of known inauthentic user IDs
        label1: Label for first method (defaults to approach name)
        label2: Label for second method (defaults to approach name)
        figsize: Figure size tuple
        **kwargs: Additional approach-specific parameters

    Returns:
        dict with 'total_area' and 'negative_area' (where method 1 < baseline)
    """
    # Get labels from approaches if not provided
    if label1 is None:
        label1 = approach1.get_approach_name()
    if label2 is None:
        label2 = f"{approach2.get_approach_name()} (baseline)"

    # Get suspicious users using the approach's method
    suspicious_users1 = approach1.get_suspicious(RTs, **kwargs)
    suspicious_users2 = approach2.get_suspicious(RTs, **kwargs)

    # Compute curves
    x1, y1 = _compute_curve(suspicious_users1, io_users)
    x2, y2 = _compute_curve(suspicious_users2, io_users)

    # Calculate differences (positive when method 1 > baseline)
    diff = y1 - y2

    # Calculate areas
    total_area = np.sum(diff)
    negative_area = np.sum(np.minimum(0, diff))

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot both curves
    ax.plot(x1, y1, label=label1, linewidth=2)
    ax.plot(x2, y2, label=label2, linewidth=2, linestyle='--')

    ax.set_xlabel("Number of users studied")
    ax.set_ylabel("Number of IO users detected")
    ax.set_title("Comparison of Detection Methods")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add text box with area information
    textstr = f'Total area difference: {total_area:.2f}\n'
    textstr += f'Area where baseline better: {negative_area:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.show()

    return {'total_area': total_area, 'negative_area': negative_area}

def plot_comparison(pairs_scores1, pairs_scores2, io_users,
                  label1="Method 1", label2="Method 2 (baseline)",
                  figsize=(10, 6)):
    """
    Compare two detection methods by plotting their curves side by side.
    Calculates and displays the area between curves (total and negative only).

    Legacy function for backward compatibility. For new code, prefer
    plot_comparison_approaches() which works directly with Approach instances.

    Args:
        pairs_scores1: First method's user pair scores
        pairs_scores2: Second method's user pair scores (used as baseline)
        io_users: Set of known inauthentic user IDs
        label1: Label for first method
        label2: Label for second method (baseline)
        figsize: Figure size tuple

    Returns:
        dict with 'total_area' and 'negative_area' (where method 1 < baseline)
    """
    # Compute curves for both methods
    suspicious_users1 = _pairs_scores_to_suspicious_users(pairs_scores1)
    suspicious_users2 = _pairs_scores_to_suspicious_users(pairs_scores2)
    x1, y1 = _compute_curve(suspicious_users1, io_users)
    x2, y2 = _compute_curve(suspicious_users2, io_users)
            
    # Calculate differences (positive when method 1 > baseline)
    diff = y1 - y2
    
    # Calculate areas
    total_area = np.sum(diff)
    negative_area = np.sum(np.minimum(0, diff))
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot both curves
    ax.plot(x1, y1, label=label1, linewidth=2)
    ax.plot(x2, y2, label=label2, linewidth=2, linestyle='--')
        
    ax.set_xlabel("Number of users studied")
    ax.set_ylabel("Number of IO users detected")
    ax.set_title("Comparison of Detection Methods")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add text box with area information
    textstr = f'Total area difference: {total_area:.2f}\n'
    textstr += f'Area where baseline better: {negative_area:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.show()
