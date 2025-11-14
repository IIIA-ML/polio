"""Same tweet, same time approach."""

from typing import Dict, Tuple, List
from collections import defaultdict
from .base import PairsApproach


class SameTweetSameTimeApproach(PairsApproach):
    """
    Same tweet, same time approach.

    Counts distinct days where users co-retweeted the same tweet
    within the time window. This is the strongest indicator of
    coordination: same content, same time.
    """

    def get_approach_name(self) -> str:
        return "Same Tweet Same Time"

    def get_approach_key(self) -> str:
        return "same_tweet_same_time"

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
        Count days users co-retweeted same tweet within time window.

        Args:
            RTs: List of (user_id, tweet_id, timestamp) tuples
                 Should be filtered to users who have co-retweeted
            **kwargs: Ignored for this approach

        Returns:
            Dictionary mapping user pairs to their same-tweet-same-time counts
        """
        tweet_data = self._group_RTs_by_tweet(RTs)
        result = defaultdict(set)

        # For each tweet, find user pairs who retweeted within window
        for tid, events in tweet_data.items():
            for i in range(len(events)):
                for j in range(i+1, len(events)):
                    u1, t1 = events[i]
                    u2, t2 = events[j]
                    # Check if retweets are within time window
                    if abs((t1 - t2).total_seconds()) <= self.window_sec and u1 != u2:
                        pair = tuple(sorted((u1, u2)))
                        result[pair].add(max(t1, t2).date())
                    else:
                        break  # Events are sorted, no need to check further

        # Convert sets of days to counts
        return {pair: len(days) for pair, days in result.items()}
