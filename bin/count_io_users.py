#!/usr/bin/env python3
"""
Count IO (inauthentic operation) users in each dataset with different filtering criteria.

This script analyzes the number of IO users in each dataset:
1. Without filtering (all IO users)
2. Filtering by users with minimum 1 coretweet
3. Filtering by users with minimum 2 coretweets

A coretweet is when two users retweet the same tweet within a time window.
Users are ranked by the CoRetweetsApproach which identifies suspicious user pairs.
"""

import sys
from pathlib import Path
from typing import Tuple

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import import_data
from approaches import CoRetweetsApproach


def count_io_users_with_min_coactions(RTs, io_users_set: set, min_coactions: int) -> Tuple[int, int]:
    """
    Count IO users with minimum coactions threshold.
    
    Args:
        RTs: List of (user_id, tweet_id, timestamp) tuples
        io_users_set: Set of IO user IDs
        min_coactions: Minimum number of co-retweets required
    
    Returns:
        Tuple of (io_users_count, total_users_count)
    """
    approach = CoRetweetsApproach(min_coactions=min_coactions)
    pairs = approach.compute_pairs_scores(RTs)
    users = set()
    for (user1, user2) in pairs.keys():
        users.add(user1)
        users.add(user2)
    io_count = len(io_users_set & users)
    total_count = len(users)
    return io_count, total_count


def count_io_users(dataset_name: str) -> Tuple[int, int, int, int, int, int]:
    """
    Count IO users in a dataset with different filtering criteria.
    
    Uses the CoRetweetsApproach to identify users with coretweet pairs.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'Armenia', 'Bangladesh')
    
    Returns:
        Tuple of (total_io_users, total_all_users, io_users_min1, total_users_min1,
                  io_users_min2, total_users_min2)
    """
    data_dir = Path(__file__).parent.parent / "data" / dataset_name / "Processed"
    
    if not data_dir.exists():
        return None
    
    # Load data
    RTs, io_users = import_data(data_dir)
    io_users_set = set(io_users)
    
    # Count without filtering
    total_io_users = len(io_users_set)
    all_users = set()
    for user, _, _ in RTs:
        all_users.add(user)
    total_all_users = len(all_users)
    
    # Get users with minimum 1 coretweet
    io_users_min1, total_users_min1 = count_io_users_with_min_coactions(RTs, io_users_set, min_coactions=1)
    
    # Get users with minimum 2 coretweets
    io_users_min2, total_users_min2 = count_io_users_with_min_coactions(RTs, io_users_set, min_coactions=2)
    
    return total_io_users, total_all_users, io_users_min1, total_users_min1, io_users_min2, total_users_min2


def main():
    """Main function."""
    # List of all datasets
    datasets = [
        'Armenia', 'Bangladesh', 'Catalonia', 'China_1', 'China_2',
        'Cuba', 'Ecuador', 'Egypt_UAE', 'Ghana_Nigeria',
        'Iran_1', 'Iran_2', 'Iran_3', 'Iran_4', 'Iran_5', 'Iran_6',
        'Qatar', 'Russia_1', 'Russia_2', 'Russia_3', 'Russia_4', 'Russia_5',
        'Spain', 'Thailand', 'UAE', 'Venezuela_1', 'Venezuela_2'
    ]
    
    # Print header
    print(f"{'Dataset':<20} {'No Filter':<20} {'Min 1 Coretweet':<25} {'Min 2 Coretweets':<25}")
    print("-" * 90)
    
    # Process each dataset
    for dataset in datasets:
        result = count_io_users(dataset)
        if result is None:
            print(f"{dataset:<20} (data not found)")
        else:
            total_io, total_all, io_min1, total_min1, io_min2, total_min2 = result
            no_filter_str = f"{total_io} ({total_all})"
            min1_str = f"{io_min1} ({total_min1})"
            min2_str = f"{io_min2} ({total_min2})"
            print(f"{dataset:<20} {no_filter_str:<20} {min1_str:<25} {min2_str:<25}")
    
    print("\nNote: Coretweets are pairs of users retweeting the same tweet within a 1-minute window.")


if __name__ == "__main__":
    main()
