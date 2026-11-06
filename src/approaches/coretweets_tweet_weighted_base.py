"""Shared algorithm of the co-retweet approaches weighted by a global property of the tweet (IDF, 1/m)."""

from abc import abstractmethod

from ._coretweet_index import get_coretweet_index
from .coretweets_weighted_base import CoretweetsWeightedBase


class CoretweetsTweetWeightedBase(CoretweetsWeightedBase):
    """
    Synchronized co-retweets weighted by a global (time-independent) property of the tweet.

    Pair score: s(u, v) = sum of w(tweet) over the synchronized co-retweets of (u, v). The pair search, the
    qualifying rule (min_coactions on the raw number of co-retweets) and the co-retweet units are those of
    CoretweetsWeightedBase, so these approaches differ from the period-weighted ones ONLY in the weight of a
    co-retweet. The periods do not enter the weight; they are those of the 12h approach so that the co-retweet
    units are exactly the same.

    The statistics of the weights are computed on the RTs the approach receives (the full dataset when
    need_filtering is False):
        N        number of distinct users with >= 1 retweet
        n(tweet) number of distinct users who retweeted the tweet
        m(tweet) number of retweets of the tweet
    """

    PERIOD_SECONDS = 12 * 3600
    PERIOD_OFFSET = 5 * 3600

    @classmethod
    def _counted_periods(cls, date_set):
        return frozenset()                 # the weight does not read any period popularity

    @abstractmethod
    def _tweet_weight(self, n_users, tweet_users, tweet_retweets):
        """Weight of one co-retweet of a tweet with n(tweet) users and m(tweet) retweets."""

    def _popularity_fraction(self, day, tweet, date_tweet_popularity, tweet_retweets):
        return self._weights[tweet]

    def compute_pairs_scores(self, RTs):
        index = get_coretweet_index(RTs)
        n_users = len({user for user, _, _ in RTs})
        self._weights = {tweet: self._tweet_weight(n_users, len({u for u, _, _ in events}), len(events))
                         for tweet, events in index.tweets.items()}
        try:
            return super().compute_pairs_scores(RTs)
        finally:
            del self._weights
