"""Synchronized co-retweets weighted by the global rarity of the tweet, 1 / m."""

from .coretweets_tweet_weighted_base import CoretweetsTweetWeightedBase


class CoretweetsInvMApproach(CoretweetsTweetWeightedBase):
    """Synchronized co-retweets weighted by w = 1 / m(tweet), m the number of retweets of the tweet."""

    def get_approach_name(self):
        return "Co-Retweet-InvM"

    def get_approach_key(self):
        return "coretweets_inv_m"

    def _tweet_weight(self, n_users, tweet_users, tweet_retweets):
        return 1.0 / tweet_retweets
