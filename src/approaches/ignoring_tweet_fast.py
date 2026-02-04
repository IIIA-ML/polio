"""Optimized temporal synchronization ignoring tweet content approach using sliding window."""

from typing import Dict, Tuple, List
from collections import defaultdict
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

        Optimized for large datasets with frequent temporal coincidences.

        Args:
            RTs: List of (user_id, tweet_id, timestamp) tuples
            **kwargs: Additional parameters (unused for this approach)

        Returns:
            Dictionary mapping user pairs to their coincidence counts
        """
        if not RTs:
            return {}

        # Sort all events chronologically: O(R log R)
        events = [(ts, user) for user, _, ts in RTs]
        events.sort()

        # Active set: users whose last action is within the window
        # Use dict for O(1) lookup and O(1) removal by user
        active = {}  # Maps user_id -> timestamp

        # Track unique (pair, day) combinations using defaultdict for efficiency
        # Maps pair -> day -> count (using int to track occurrences per day)
        pair_days = defaultdict(lambda: defaultdict(int))

        window_td = None  # Cache for window timedelta to avoid repeated conversion
        
        # Process events chronologically
        for ts_curr, user_curr in events:
            # Lazily create timedelta from window_sec (only once)
            if window_td is None:
                from datetime import timedelta
                window_td = timedelta(seconds=self.window_sec)
            
            # Remove users whose last action is outside the window
            # Iterate over a list of keys to avoid "dictionary changed size during iteration"
            expired_users = [u for u, ts in active.items() if (ts_curr - ts) > window_td]
            for u in expired_users:
                del active[u]

            # Record day for all pairs with currently active users
            current_day = ts_curr.date()
            for user_active, ts_active in active.items():
                # Skip same user (can happen if user posts multiple times)
                if user_active != user_curr:
                    # Create canonical pair (sorted tuple)
                    pair = tuple(sorted((user_curr, user_active)))
                    # Increment count for this pair on this day
                    # (use count to avoid repeated set operations)
                    pair_days[pair][current_day] += 1

            # Add current user to active set
            active[user_curr] = ts_curr

        # Build result dictionary: only include pairs meeting min_coactions threshold
        result = {}
        for pair, days_dict in pair_days.items():
            num_days = len(days_dict)
            if num_days >= self.min_coactions:
                result[pair] = num_days

        #result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True)[:2000])
        return result
