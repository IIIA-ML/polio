#!/usr/bin/env python3
"""
Count IO (inauthentic operation) users detected by different approaches.

Analyzes how many IO users appear in user pairs identified by:
1. No filtering (all users who retweeted)
2. CoRetweetsApproach with min 1 and 2 coactions

Usage: python count_io_users.py
(Edit 'datasets' list in main() to select datasets to analyze)
"""

import sys
from pathlib import Path
from typing import Tuple

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from approaches.same_tweet_same_time import SameTweetSameTimeApproach
from data_loader import import_data
from approaches.coretweets import CoRetweetsApproach
from approaches.ignoring_tweet_fast import IgnoringTweetFastApproach


def obtain_users_from_pairs(pairs: dict) -> set:
    """
    Extract unique users from user pairs.

    Args:
        pairs: Dictionary with user pairs as keys
    """
    users = set()
    for (user1, user2) in pairs.keys():
        users.add(user1)
        users.add(user2)
    return users

def count_users_and_io(users, io_users):
    """
    Count how many IO users are in the given user set.
    
    Args:
        users: Set of user IDs
        io_users: Set of known inauthentic user IDs
    """
    return len(users), len(io_users & users)

def get_users_and_io_users_from_pairs(pairs: dict, io_users_set: set) -> Tuple[int, int]:
    """
    Count total users and IO users from user pairs.
    
    Args:
        pairs: Dictionary with user pairs as keys
        io_users_set: Set of known inauthentic user IDs
    """
    users = obtain_users_from_pairs(pairs)
    return count_users_and_io(users, io_users_set)

def count_connections(pairs: dict) -> int:
    """Count the number of connections (non-zero pairs).
    
    Args:
        pairs: Dictionary with user pairs as keys
    
    Returns:
        Number of non-zero pairs
    """
    return len(pairs)
    
def load_RTs_and_io_users(dataset_name, base_data_dir):
    data_dir = base_data_dir / dataset_name / "Processed"
    
    if not data_dir.exists():
        return None
    
    # Load data
    RTs, io_users = import_data(data_dir)
    return RTs, set(io_users)


def main(base_data_dir=None):
    """Main function.
    
    Args:
        base_data_dir: Path to the directory containing dataset folders. 
                      If None, defaults to '../data' relative to this script.
    """
    if base_data_dir is None:
        base_data_dir = Path(__file__).parent.parent / "data"
    else:
        base_data_dir = Path(base_data_dir)
    
    # List of all datasets
    datasets = [
        "Honduras"
    #    #'Armenia', 'Bangladesh', 'Catalonia', 'China_1', 'China_2',
    #    'Cuba', 'Ecuador',
    #    'Egypt_UAE', 'Ghana_Nigeria',
    #    'Iran_1', 'Iran_2', 'Iran_3', 'Iran_4', 'Iran_5', 'Iran_6',
    #    'Qatar', 'Russia_1', 'Russia_2', 'Russia_3', 'Russia_4', 'Russia_5',
    #    'Spain', 'Thailand', 'UAE', 'Venezuela_1', 'Venezuela_2'
    ]
    
    # Print header
    print(f"{'Dataset':<20} {'No Filter':<30} {'Min 1 Coretweet':<35} {'Min 2 Coretweets':<35}")
    print("-" * 155)
    
    # Process each dataset
    for dataset in datasets:
        RTs, io_users = load_RTs_and_io_users(dataset, base_data_dir)

        users_rts = set([user for user, _, _ in RTs])
        u_rts, io_rts = count_users_and_io(users_rts, set(io_users))

        approach = CoRetweetsApproach(min_coactions=1)
        pairs = approach.compute_pairs_scores(RTs)
        u_coretweet, io_coretweet = get_users_and_io_users_from_pairs(pairs, io_users)
        conn_coretweet = count_connections(pairs)

        approach = CoRetweetsApproach(min_coactions=2)
        pairs = approach.compute_pairs_scores(RTs)
        u_coretweet_min2, io_coretweet_min2 = get_users_and_io_users_from_pairs(pairs, io_users)
        conn_coretweet_min2 = count_connections(pairs)
    
        no_filter_str = f"IO: {io_rts} ({u_rts})"
        cort_min1_str = f"IO: {io_coretweet} ({u_coretweet}) | Conn: {conn_coretweet}"
        cort_min2_str = f"IO: {io_coretweet_min2} ({u_coretweet_min2}) | Conn: {conn_coretweet_min2}"
        
        print(f"{dataset:<20} {no_filter_str:<30} {cort_min1_str:<35} {cort_min2_str:<35}")
    
    print("\nNote: Coretweets are pairs of users retweeting the same tweet within a 1-minute window.")


if __name__ == "__main__":
    main('datasets/CimaIO')
