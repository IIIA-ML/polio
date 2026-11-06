"""IDF-weighted synchronized co-retweets (IDF as in Silva et al.)."""

import math

from .coretweets_tweet_weighted_base import CoretweetsTweetWeightedBase


class CoretweetsIDFApproach(CoretweetsTweetWeightedBase):
    """
    Synchronized co-retweets weighted by the inverse document frequency of the tweet, w = ln(1 + N / n(tweet)),
    with the IDF formula of Silva et al. (2026). Only their IDF is used, as the weight of each synchronized
    co-retweet (their method has no time window).
    """

    def get_approach_name(self):
        return "Co-Retweet-IDF"

    def get_approach_key(self):
        return "coretweets_idf"

    def _tweet_weight(self, n_users, tweet_users, tweet_retweets):
        return math.log1p(n_users / tweet_users)
