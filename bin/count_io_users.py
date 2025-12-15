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
    
def load_RTs_and_io_users(dataset_name):
    data_dir = Path(__file__).parent.parent / "data" / dataset_name / "Processed"
    
    if not data_dir.exists():
        return None
    
    # Load data
    RTs, io_users = import_data(data_dir)
    return RTs, set(io_users)


def main():
    """Main function."""
    # List of all datasets
    datasets = [
        #'Armenia', 'Bangladesh', 'Catalonia', 'China_1', 'China_2',
        #'Cuba', 'Ecuador',
        'Egypt_UAE', 'Ghana_Nigeria',
        'Iran_1', 'Iran_2', 'Iran_3', 'Iran_4', 'Iran_5', 'Iran_6',
        'Qatar', 'Russia_1', 'Russia_2', 'Russia_3', 'Russia_4', 'Russia_5',
        'Spain', 'Thailand', 'UAE', 'Venezuela_1', 'Venezuela_2'
    ]
    
    # Print header
    print(f"{'Dataset':<20} {'No Filter':<20} {'Min 1 Coretweet':<25} {'Min 2 Coretweets':<25} {'Consistency Min2':<25}")
    print("-" * 125)
    
    # Process each dataset
    for dataset in datasets:
        RTs, io_users = load_RTs_and_io_users(dataset)

        users_rts = set([user for user, _, _ in RTs])
        u_rts, io_rts = count_users_and_io(users_rts, set(io_users))

        approach = CoRetweetsApproach(min_coactions=1)
        pairs = approach.compute_pairs_scores(RTs)
        u_coretweet, io_coretweet = get_users_and_io_users_from_pairs(pairs, io_users)

        approach = CoRetweetsApproach(min_coactions=2)
        pairs = approach.compute_pairs_scores(RTs)
        u_coretweet_min2, io_coretweet_min2 = get_users_and_io_users_from_pairs(pairs, io_users)

        approach = IgnoringTweetFastApproach(min_coactions=2)
        pairs = approach.compute_pairs_scores(RTs)
        u_synchdays_min2, io_synchdays_min2 = get_users_and_io_users_from_pairs(pairs, io_users)
    
        no_filter_str = f"{io_rts} ({u_rts})"
        cort_min1_str = f"{io_coretweet} ({u_coretweet})"
        cort_min2_str = f"{io_coretweet_min2} ({u_coretweet_min2})"
        synchdays_min2_str = f"{io_synchdays_min2} ({u_synchdays_min2})"
        
        print(f"{dataset:<20} {no_filter_str:<20} {cort_min1_str:<25} {cort_min2_str:<25} {synchdays_min2_str:<25}")
    
    print("\nNote: Coretweets are pairs of users retweeting the same tweet within a 1-minute window.")


if __name__ == "__main__":
    main()
