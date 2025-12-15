"""Optimized temporal synchronization ignoring tweet content approach using sliding window."""

from typing import Dict, Tuple, List
from collections import defaultdict, deque
from .base import PairsApproach


class IgnoringTweetFastApproach(PairsApproach):
    """
    Optimized temporal synchronization ignoring tweet content.

    Uses a sliding window algorithm with an active set to efficiently count
    days where two users posted within the time window, regardless of whether
    they interacted with the same content.

    Time complexity: O(R log R + R × k) where:
        R = total retweets
        k = average active set size << U (total users)

    This is significantly faster than the O(U²) all-pairs approach when
    users are not all active simultaneously.
    """

    def get_approach_name(self) -> str:
        return "Cons."

    def get_approach_key(self) -> str:
        return "ignoring_tweet_fast"

    def _compute_pairs_scores_impl(self, RTs: List[Tuple], **kwargs) -> Dict[Tuple, float]:
        """
        Count days users posted within time window using sliding window algorithm.

        Args:
            RTs: List of (user_id, tweet_id, timestamp) tuples
            **kwargs: Additional parameters (unused for this approach)

        Returns:
            Dictionary mapping user pairs to their coincidence counts
        """
        if not RTs:
            return {}

        # Get all unique users
        unique_users = sorted(set(user for user, _, _ in RTs))

        # Sort all events chronologically: O(R log R)
        events = [(ts, user) for user, _, ts in RTs]
        events.sort()

        # Active set: users whose last action is within the window
        active = deque()  # Contains (user, timestamp) tuples

        # Track unique (pair, day) combinations
        pair_days = {}

        # Process events chronologically
        for ts_curr, user_curr in events:
            # Remove users whose last action is outside the window
            while active and (ts_curr - active[0][1]).total_seconds() > self.window_sec:
                active.popleft()

            # Record day for all pairs with currently active users
            current_day = ts_curr.date()
            for user_active, ts_active in active:
                # Skip same user (can happen if user posts multiple times)
                if user_active != user_curr:
                    # Create canonical pair (sorted tuple)
                    pair = tuple(sorted((user_curr, user_active)))
                    # Add this day to the set of days for this pair
                    if pair not in pair_days:
                        pair_days[pair] = set()
                    pair_days[pair].add(current_day)

            # Add current user to active set
            active.append((user_curr, ts_curr))

        # Build result dictionary
        result = {}
        for i in range(len(unique_users)):
            for j in range(i + 1, len(unique_users)):
                pair = (unique_users[i], unique_users[j])
                if pair in pair_days and len(pair_days[pair]) >= self.min_coactions:
                    result[pair] = len(pair_days[pair])

        return result
