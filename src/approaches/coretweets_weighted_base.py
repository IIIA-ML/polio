"""Shared algorithm of the period-weighted co-retweet approaches (12h, 1day, 2days)."""

import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from ._coretweet_index import get_coretweet_index, window_microseconds
from .base import PairsApproach

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_ONE_SECOND = timedelta(seconds=1)


class CoretweetsWeightedBase(PairsApproach):
    """
    Synchronized co-retweets weighted by the popularity of the tweet in the period of the co-retweet.

    The weight of a co-retweet of a tweet with m retweets in total is m_P / m, m_P its retweets in the period P
    of the co-retweet; when the two retweets fall in different periods P and P', (m_P + m_P') / (2 m).
    Subclasses define the periods (and may redefine the weight). Pair scores are summed with math.fsum, so they
    do not depend on the iteration order of the tweets.
    """

    # The timeline is partitioned into consecutive periods of PERIOD_SECONDS, starting PERIOD_OFFSET seconds
    # after a UTC midnight (e.g. 43200 and 5h: 05-17h and 17-05h UTC)
    PERIOD_SECONDS = 86400
    PERIOD_OFFSET = 0

    @classmethod
    def _period_key(cls, ts):
        """
        Period of a timestamp, as the UTC Unix timestamp (int) of the period start.

        Exact integer arithmetic on the UTC instant, independent of the timezone of the machine. Naive
        timestamps are taken as UTC.
        """
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        seconds = (ts - _EPOCH) // _ONE_SECOND
        return (seconds - cls.PERIOD_OFFSET) // cls.PERIOD_SECONDS * cls.PERIOD_SECONDS + cls.PERIOD_OFFSET

    @classmethod
    def _counted_periods(cls, date_set):
        """Periods whose popularity _popularity_fraction may read, given the synchronized ones."""
        return date_set

    def _popularity_fraction(self, day, tweet, date_tweet_popularity, tweet_retweets):
        """Weight of one co-retweet of `tweet` recorded under period `day` (or a pair of periods)."""
        if isinstance(day, tuple):
            return (date_tweet_popularity[day[0]][tweet] + date_tweet_popularity[day[1]][tweet]) / (2*tweet_retweets)
        return date_tweet_popularity[day][tweet] / tweet_retweets

    @classmethod
    def get_date_tweet_popularity(cls, RTs, date_set, synchronized_tweets):
        """{period: {tweet: retweets}} of the synchronized tweets in the periods of _counted_periods(date_set)."""
        counted = cls._counted_periods(date_set)
        period_key = cls._period_key
        date_tweet_popularity = defaultdict(lambda: defaultdict(int))
        for _, tid, ts in RTs:
            if tid in synchronized_tweets:
                period_seconds = period_key(ts)
                if period_seconds in counted:
                    date_tweet_popularity[period_seconds][tid] += 1
        return date_tweet_popularity

    def compute_pairs_scores(self, RTs):
        """
        1. For each tweet, the pairs of users who retweeted it within the time window, with the period of the
           co-retweet (or the pair of periods when the two retweets fall in different periods)
        2. Pairs with >= min_coactions co-retweets qualify; their periods and tweets are "synchronized"
        3. Retweets of each synchronized tweet in each synchronized period (and in any other period the
           weight reads, see _counted_periods)
        4. Score of a qualifying pair: sum of the weights of its co-retweets
        """
        index = get_coretweet_index(RTs)
        window = window_microseconds(self.window_sec)
        period_key = self._period_key
        periods_by_tweet = {tid: [period_key(ts) for _, _, ts in events] for tid, events in index.tweets.items()}
        pair_periods = defaultdict(lambda: defaultdict(set))

        for tweet, events in index.tweets.items():
            periods = periods_by_tweet[tweet]
            n = len(events)
            for i in range(n):
                u1, t1, _ = events[i]
                p1 = periods[i]
                for j in range(i+1, n):
                    u2, t2, _ = events[j]
                    if t2 - t1 > window:
                        break
                    if u1 != u2:
                        pair = (u1, u2) if u1 < u2 else (u2, u1)
                        p2 = periods[j]
                        pair_periods[pair][p1 if p1 == p2 else (p1, p2)].add(tweet)

        qualifying = []
        synchronized_days = set()
        synchronized_tweets = set()
        for pair, days_tweets in pair_periods.items():
            if sum(len(tweetset) for tweetset in days_tweets.values()) >= self.min_coactions:
                qualifying.append(pair)
                for day, tweets_set in days_tweets.items():
                    if isinstance(day, tuple):
                        synchronized_days.add(day[0])
                        synchronized_days.add(day[1])
                    else:
                        synchronized_days.add(day)
                    synchronized_tweets.update(tweets_set)

        date_tweet_popularity = self.get_date_tweet_popularity(RTs, synchronized_days, synchronized_tweets)

        result = {}
        fraction = self._popularity_fraction
        for pair in qualifying:
            # fsum is exactly rounded: the score does not depend on the (hash-based) order of the tweet sets
            weight = math.fsum(fraction(day, tweet, date_tweet_popularity, len(index.tweets[tweet]))
                               for day, tweets_set in pair_periods[pair].items()
                               for tweet in tweets_set)
            if weight > 0:
                result[pair] = weight
        return result
