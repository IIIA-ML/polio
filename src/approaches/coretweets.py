"""Co-retweet counting approach."""

from typing import Dict, Tuple, List
from collections import defaultdict
from .base import Approach


class CoRetweetsApproach(Approach):
    """
    Co-retweet counting approach.

    Detects pairs of users who retweeted the same tweet within a time window.
    This is the baseline approach that other methods build upon.
    """

    def get_approach_name(self) -> str:
        return "Co-retweets"

    def get_approach_key(self) -> str:
        return "coretweets"

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

    def compute_pairs_scores(self, RTs: List[Tuple], **kwargs) -> Dict[Tuple, float]:
        """
        Count pairs who retweeted same tweet within time window.

        Args:
            RTs: List of (user_id, tweet_id, timestamp) tuples
            **kwargs: Ignored for this approach

        Returns:
            Dictionary mapping user pairs to their co-retweet counts
        """
        by_tweet = self._group_RTs_by_tweet(RTs)
        y = {}

        # For each tweet, find pairs of users who retweeted within the time window
        for acc_times in by_tweet.values():
            n = len(acc_times)
            for i in range(n):
                acc_i, ts_i = acc_times[i]
                for j in range(i+1, n):
                    acc_j, ts_j = acc_times[j]
                    # Stop if time difference exceeds window
                    if (ts_j - ts_i).total_seconds() > self.window_sec:
                        break
                    if acc_i != acc_j:
                        pair = tuple(sorted((acc_i, acc_j)))
                        if pair not in y:
                            y[pair] = 1
                        else:
                            y[pair] += 1

        # Filter pairs with at least min_coactions co-retweets
        return {k: v for k, v in y.items() if v >= self.min_coactions}
