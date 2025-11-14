"""Fast co-retweet counting approach using sliding window algorithm."""

from typing import Dict, Tuple, List
from collections import Counter, defaultdict, deque
from .base import PairsApproach


class CoRetweetsFastApproach(PairsApproach):
    """
    Fast co-retweet counting approach using sliding window algorithm.

    Detects pairs of users who retweeted the same tweet within a time window.
    Uses a deque-based sliding window for O(n*k) complexity instead of O(n²),
    where k is the average number of retweets within the time window (typically << n).
    """

    def get_approach_name(self) -> str:
        return "Fast Co-retweets"

    def get_approach_key(self) -> str:
        return "coretweets_fast"

    def needs_filtered_data(self) -> bool:
        """Co-retweets doesn't need pre-filtered data."""
        return False

    @staticmethod
    def _group_RTs_by_tweet(RTs):
        """Group retweets by tweet, sorted by timestamp."""
        tweet_data = defaultdict(list)
        for user, tid, ts in RTs:
            tweet_data[tid].append((user, ts))
        # Sort each tweet's retweets chronologically
        for tid in tweet_data:
            tweet_data[tid].sort(key=lambda x: x[1])
        return tweet_data

    def _compute_pairs_scores_impl(self, RTs: List[Tuple], **kwargs) -> Dict[Tuple, float]:
        """
        Internal implementation of co-retweet counting using sliding window.

        Args:
            RTs: List of (user_id, tweet_id, timestamp) tuples
            **kwargs: Additional parameters (unused for this approach)

        Returns:
            Dictionary mapping user pairs to their co-retweet counts
        """
        by_tweet = self._group_RTs_by_tweet(RTs)
        y = Counter()

        # For each tweet, find pairs of users who retweeted within the time window
        # Using sliding window approach: O(n*k) where k = avg retweets in window
        for acc_times in by_tweet.values():
            active = deque()  # Deque of (user_id, timestamp) within current window

            for acc_curr, ts_curr in acc_times:
                # Remove retweets outside the time window from the front
                while active and (ts_curr - active[0][1]).total_seconds() > self.window_sec:
                    active.popleft()

                # Count pairs with all users in the active window
                for acc_prev, _ in active:
                    if acc_prev != acc_curr:
                        pair = tuple(sorted((acc_prev, acc_curr)))
                        y[pair] += 1

                # Add current retweet to active window
                active.append((acc_curr, ts_curr))
        # Filter pairs with at least min_coactions co-retweets
        return {k: v for k, v in y.items() if v >= self.min_coactions}
