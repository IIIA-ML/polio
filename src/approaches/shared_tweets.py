"""Temporal synchronization on shared content approach."""

from typing import Dict, Tuple, List
from collections import defaultdict
from .base import Approach


class SharedTweetsApproach(Approach):
    """
    Temporal synchronization on shared content.

    Counts days where two users retweeted shared content within the time window.
    Only considers tweets that both users interacted with.
    """

    def get_approach_name(self) -> str:
        return "Shared Tweets"

    def get_approach_key(self) -> str:
        return "shared_tweets"

    @staticmethod
    def _group_RTs_by_user(RTs):
        """Group retweets by user, sorted by timestamp."""
        user_data = defaultdict(list)
        for user, tid, ts in RTs:
            user_data[user].append((tid, ts))
        # Sort each user's retweets chronologically
        for u in user_data:
            user_data[u].sort(key=lambda x: x[1])
        return user_data

    @staticmethod
    def _count_daily_coincidences(times1, times2, window_sec):
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

    def compute_pairs_scores(self, RTs: List[Tuple], **kwargs) -> Dict[Tuple, float]:
        """
        Count days users retweeted shared content within time window.

        Args:
            RTs: List of (user_id, tweet_id, timestamp) tuples
                 Should be filtered to users who have co-retweeted
            **kwargs: Ignored for this approach

        Returns:
            Dictionary mapping user pairs to their shared content coincidence counts
        """
        user_data = self._group_RTs_by_user(RTs)
        users = list(user_data.keys())
        result = {}

        # Pre-store tweets per user for efficient lookup
        user_tweets = {u: set(tid for tid, _ in user_data[u]) for u in users}

        for i in range(len(users)):
            for j in range(i+1, len(users)):
                u1, u2 = users[i], users[j]

                # Find tweets both users retweeted
                shared = user_tweets[u1] & user_tweets[u2]
                if not shared:
                    result[(u1, u2)] = 0
                    continue

                # Extract timestamps for shared tweets only
                times1 = [ts for tid, ts in user_data[u1] if tid in shared]
                times2 = [ts for tid, ts in user_data[u2] if tid in shared]

                score = self._count_daily_coincidences(times1, times2, self.window_sec)
                pair = tuple(sorted((u1, u2)))
                result[pair] = score

        return result
