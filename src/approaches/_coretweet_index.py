"""
Shared preprocessing of the co-retweet approaches.

Grouping the retweets by tweet and sorting them chronologically is identical for every co-retweet approach,
so it is done once per RTs list and reused.

Each event additionally stores its timestamp as integer microseconds since the first retweet of the dataset,
so the time-window check becomes an integer comparison. For a whole number of seconds W:
    (t2 - t1).total_seconds() > W   <=>   us(t2) - us(t1) > W * 1_000_000
because timedelta arithmetic is exact and both sides stay far below 2**53.
"""

from collections import defaultdict
from datetime import timedelta

_ONE_MICROSECOND = timedelta(microseconds=1)

# Single-entry caches: (RTs list, ...). Holding a reference to the RTs list guarantees that the identity
# checks cannot match a different list; RTs lists are never mutated after loading.
_index_cache = None
_filter_cache = None


class CoRetweetIndex:
    """
    Retweets grouped by tweet, each group sorted by timestamp.

    Attributes:
        tweets: {tweet_id: [(user, microseconds, timestamp), ...]}, tweets in order of first retweet in RTs
    """

    __slots__ = ('tweets',)

    def __init__(self, RTs):
        tweet_data = defaultdict(list)
        for user, tid, ts in RTs:
            tweet_data[tid].append((user, ts))
        for events in tweet_data.values():
            events.sort(key=lambda event: event[1])      # stable: ties keep their order in RTs

        self.tweets = {}
        if tweet_data:
            ref = RTs[0][2]
            for tid, events in tweet_data.items():
                self.tweets[tid] = [(user, (ts - ref) // _ONE_MICROSECOND, ts) for user, ts in events]


def get_coretweet_index(RTs):
    """CoRetweetIndex of RTs, reused if RTs is the last list indexed."""
    global _index_cache
    if _index_cache is None or _index_cache[0] is not RTs:
        _index_cache = (RTs, CoRetweetIndex(RTs))
    return _index_cache[1]


def window_microseconds(window_sec):
    """Time window in microseconds, for comparison with event microsecond offsets."""
    return window_sec * 1_000_000


def filter_coretweeting_users(RTs, window_sec):
    """
    Retweets of the users with at least one synchronized co-retweet (None if there is none), reused if RTs
    and window_sec are those of the last call.
    """
    global _filter_cache
    if _filter_cache is None or _filter_cache[0] is not RTs or _filter_cache[1] != window_sec:
        window = window_microseconds(window_sec)
        users = set()
        for events in get_coretweet_index(RTs).tweets.values():
            n = len(events)
            for i in range(n):
                u1, t1, _ = events[i]
                for j in range(i + 1, n):
                    u2, t2, _ = events[j]
                    if t2 - t1 > window:
                        break
                    if u1 != u2:
                        users.update((u1, u2))
        filtered = [rt for rt in RTs if rt[0] in users] if users else None
        _filter_cache = (RTs, window_sec, filtered)
    return _filter_cache[2]


def clear_coretweet_index_cache():
    """Release the cached index and filtered retweets (and the RTs lists they reference)."""
    global _index_cache, _filter_cache
    _index_cache = _filter_cache = None
