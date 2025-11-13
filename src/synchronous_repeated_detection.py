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

def _compute_curve(pairs_scores, io_users):
    """
    Helper function to compute x and y values for a detection curve.
    Shows how many inauthentic users are detected as more user pairs are examined.
    For edges with equal score, aggregate all new users introduced at that score
    Returns (x_vals, y_vals) arrays.
    """
    # Sort by decreasing score, but keep equal-score items grouped
    sorted_pairs = sorted(pairs_scores.items(), key=lambda x: x[1], reverse=True)

    users_seen = set()
    x_vals, y_vals = [0], [0]
    io_users_count = 0
    users_seen_count = 0

    idx = 0
    n = len(sorted_pairs)

    while idx < n:
        # Start of a score block (pairs with same score)
        score = sorted_pairs[idx][1]
        block_new_users = set()

        # Collect all edges with the same score
        while idx < n and sorted_pairs[idx][1] == score:
            pair, _ = sorted_pairs[idx]
            # Add users from this pair if not seen before
            for user in pair:
                if user not in users_seen and user not in block_new_users:
                    block_new_users.add(user)
            idx += 1

        # Count totals for this score block
        new_users = block_new_users
        new_users_len = len(new_users)
        new_ios_len = sum(1 for u in new_users if u in io_users)

        # Append the mean-line step (linear interpolation within score group)
        for i in range(1, new_users_len+1):
            x_vals.append(users_seen_count + i)
            y_vals.append(io_users_count + new_ios_len * (i / new_users_len))
        
        # Update totals
        users_seen.update(new_users)
        users_seen_count += new_users_len
        io_users_count += new_ios_len

    return np.array(x_vals), np.array(y_vals)

def plot_ios_vs_studied(pairs_scores, io_users, figsize=(8,5)):
    """
    Plot progressive detection of IO users as edges are studied.
    """
    x_vals, y_vals = _compute_curve(pairs_scores, io_users)

    # ---- Plot ----
    plt.figure(figsize=figsize)
    plt.plot(x_vals, y_vals, linestyle='-')

    plt.xlabel("Number of users studied")
    plt.ylabel("Number of IO users detected")
    plt.title("Progressive detection of IO users (score-group averaged)")
    plt.grid(True)
    plt.show()

def plot_comparison(pairs_scores1, pairs_scores2, io_users, 
                  label1="Method 1", label2="Method 2 (baseline)", 
                  figsize=(10, 6)):
    """
    Compare two detection methods by plotting their curves side by side.
    Calculates and displays the area between curves (total and negative only).
    
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
    x1, y1 = _compute_curve(pairs_scores1, io_users)
    x2, y2 = _compute_curve(pairs_scores2, io_users)
            
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
