"""Optimized temporal synchronization ignoring tweet content.

Computes for each user pair the mean number of total unique tweets
retweeted by either user (union) on days where they synchronize within
the temporal window (regardless of what tweet triggered the sync).
"""

from typing import Dict, Tuple, List
from collections import defaultdict, deque
from .base import PairsApproach


class SharedTweetCountingRTApproach(PairsApproach):

    def get_approach_name(self) -> str:
        return "Shared Tweet Counting RT"

    def get_approach_key(self) -> str:
        return "shared_tweet_counting_rt"

    def _compute_pairs_scores_impl(self, RTs: List[Tuple], **kwargs) -> Dict[Tuple, float]:
        if not RTs:
            return {}

        window_sec = self.window_sec

        # Sort chronologically
        RTs = sorted(RTs, key=lambda x: x[2])  # (user, tid, ts)

        # Track tweets per user per day
        user_day_tweets = defaultdict(lambda: defaultdict(set))
        for user, tid, ts in RTs:
            user_day_tweets[user][ts.date()].add(tid)

        # Per-tweet active window — EXACT same structure as SharedTweetsFastApproach
        active_per_tid = defaultdict(deque)  # tid → deque[(user, ts)]

        # Track coinciding days
        pair_days = defaultdict(set)

        # Process retweets
        for user, tid, ts in RTs:
            dq = active_per_tid[tid]

            # Remove expired events
            while dq and (ts - dq[0][1]).total_seconds() > window_sec:
                dq.popleft()

            day = ts.date()

            # Each user in dq synchronizes with the new user on THIS tid
            for other_user, ts2, tid2 in dq:
                if other_user != user:
                    if tid2 in user_day_tweets[user].values() and tid in user_day_tweets[other_user].values():
                        pair = tuple(sorted((user, other_user)))
                        pair_days[pair].add(day)

            # Store (user, timestamp, tid)
            dq.append((user, ts, tid))

        # ---- Compute final scores: mean |union of tweets| over coinciding days ----

        users = sorted({u for u, _, _ in RTs})
        result = {}

        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                u1 = users[i]
                u2 = users[j]
                pair = (u1, u2)

                days = pair_days.get(pair, set())
                if not days:
                    result[pair] = 0.0
                    continue

                union_sum = 0
                for day in days:
                    t1 = user_day_tweets[u1][day]
                    t2 = user_day_tweets[u2][day]
                    union_sum += len(t1 | t2)

                # Correct mean
                result[pair] = round(1 / (union_sum / len(days)), 1)

        return result
