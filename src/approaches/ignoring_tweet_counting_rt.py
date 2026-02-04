"""Optimized temporal synchronization ignoring tweet content.

Computes for each user pair the mean number of total unique tweets
retweeted by either user (union) on days where they were active within
the time window.
"""

from typing import Dict, Tuple, List
from collections import defaultdict
from .base import PairsApproach


class IgnoringTweetCountingRTApproach(PairsApproach):
    """Temporal synchronization ignoring tweet content, scoring by total activity.

    Uses a sliding window with an active set to find days where two users
    were both active (within the configured temporal window). For those
    coinciding days, it counts the number of unique tweet IDs retweeted by
    either user (set union) and returns the mean union size across all
    coinciding days.

    Score(pair) = (1 / D) * Σ_{day in coinciding_days} |Tweets_u(day) ∪ Tweets_v(day)|
    where D = number of coinciding days for the pair.

    Time complexity: O(R log R + R × k + P × D × c) approximately, where:
        R = total retweets
        k = average active set size
        P = number of user pairs with at least one coinciding day
        D = average coinciding days per such pair
        c = average cost of set union (bounded by per-user tweets that day)

    Significantly faster than an O(U²) naive all-days all-pairs scan when
    user activity is sparse.
    """

    def get_approach_name(self) -> str:
        return "Prec."

    def get_approach_key(self) -> str:
        return "ignoring_tweet_counting_rt"

    def _compute_pairs_scores_impl(self, RTs: List[Tuple], **kwargs) -> Dict[Tuple, float]:
        """Compute mean total unique tweets (union) for user pairs on coinciding days.

        Args:
            RTs: List of (user_id, tweet_id, timestamp) tuples
            **kwargs: Additional parameters (unused for this approach)

        Returns:
            Dictionary mapping (user_i, user_j) to mean |union tweets| per coinciding day.
        """
        if not RTs:
            return {}

        # Sort all events chronologically: O(R log R)
        events = [(ts, user) for user, _, ts in RTs]
        events.sort()

        # Active set: users whose last action is within the window
        # Use dict for O(1) lookup/removal by user
        active = {}  # Maps user_id -> timestamp

        # Track unique (pair, day) combinations efficiently
        pair_days = defaultdict(set)

        # Track unique tweets per user per day: user -> day -> set(tweet_ids)
        user_day_tweets = defaultdict(lambda: defaultdict(set))

        for user, tweet_id, ts in RTs:
            day = ts.date()
            user_day_tweets[user][day].add(tweet_id)

        # Cache timedelta for window to avoid repeated conversion
        from datetime import timedelta
        window_td = timedelta(seconds=self.window_sec)

        # Process events chronologically
        for ts_curr, user_curr in events:
            # Remove users whose last action is outside the window
            expired = [u for u, ts in active.items() if (ts_curr - ts) > window_td]
            for u in expired:
                del active[u]

            # Record day for all pairs with currently active users
            current_day = ts_curr.date()
            for user_active, ts_active in active.items():
                # Skip same user (can happen if user posts multiple times)
                if user_active != user_curr:
                    # Create canonical pair (sorted tuple)
                    pair = tuple(sorted((user_curr, user_active)))
                    # Add this day to the set of days for this pair (unique days only)
                    pair_days[pair].add(current_day)

            # Add/update current user in active set
            active[user_curr] = ts_curr

        # Build result dictionary with mean total unique tweets (union) per coinciding day.
        # Pairs with no coinciding days get a score of 0.0
        result = {}
        for pair, days in pair_days.items():
            if len(days) < self.min_coactions:
                continue
            u_i, u_j = pair
            union_sum = 0
            for day in days:
                tweets_i = user_day_tweets[u_i][day]
                tweets_j = user_day_tweets[u_j][day]
                union_sum += len(tweets_i.union(tweets_j))

            # Preserve original scoring semantics (reciprocal of mean union size)
            result[pair] = round(1 / (union_sum / len(days)), 1)

        return result
