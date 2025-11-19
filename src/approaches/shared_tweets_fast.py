"""Optimized temporal synchronization on shared content using per-tweet sliding windows."""

from typing import Dict, Tuple, List
from collections import defaultdict, deque
from .base import PairsApproach


class SharedTweetsFastApproach(PairsApproach):
    """
    Optimized temporal synchronization on shared content.

    Matches EXACT semantics of the original SharedTweetsApproach:
    - Coincidences counted ONLY if the tweet_id is in BOTH users' tweet sets.
    - Uses sliding windows per tweet_id for major speedups.

    Complexity: O(R log R + R × k_t) where k_t is the number of users
    that retweeted a given tweet within the window.
    """

    def get_approach_name(self) -> str:
        return "Shared Tweets (Fast)"

    def get_approach_key(self) -> str:
        return "shared_tweets_fast"

    def _compute_pairs_scores_impl(self, RTs: List[Tuple], **kwargs) -> Dict[Tuple, float]:
        if not RTs:
            return {}

        window_sec = self.window_sec

        # Sort all RTs globally — allows chronological sliding per tweet
        RTs = sorted(RTs, key=lambda x: x[2])  # (user, tid, ts)

        # 1️⃣ Collect tweet sets per user (same as original)
        user_tweets = defaultdict(set)
        for user, tid, _ in RTs:
            user_tweets[user].add(tid)

        # 2️⃣ Track days per pair
        pair_days = defaultdict(set)

        # 3️⃣ Active sliding window PER tweet_id
        active_per_tid = defaultdict(deque)  # tid → deque[(user, ts)]

        # 4️⃣ Process events chronologically
        for user, tid, ts in RTs:
            dq = active_per_tid[tid]

            # Remove outdated events
            while dq and (ts - dq[0][1]).total_seconds() > window_sec:
                dq.popleft()

            day = ts.date()

            # For each other user who retweeted THIS SAME TWEET recently
            for other_user, ts2, tid2 in dq:
                if other_user == user:
                    continue

                # KEY SEMANTIC CHECK:
                # Count coincidence only if BOTH users retweeted this tweet at least once in dataset
                # i.e., tid ∈ intersection(user_tweets[u1], user_tweets[u2])
                # Since we're already processing tid events, we just need:
                if tid2 in user_tweets[user] and tid in user_tweets[other_user]:
                    pair = tuple(sorted((user, other_user)))
                    pair_days[pair].add(day)

            dq.append((user, ts, tid))

        # 5️⃣ Build final result (include zero counts)
        unique_users = sorted(set(user for user, _, _ in RTs))
        result = {}

        for i in range(len(unique_users)):
            for j in range(i + 1, len(unique_users)):
                pair = (unique_users[i], unique_users[j])
                result[pair] = len(pair_days.get(pair, set()))
        
        return result
