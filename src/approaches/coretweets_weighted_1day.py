"""Co-retweets weighted by the popularity of the tweet on the (UTC) day of the co-retweet."""

from .coretweets_weighted_base import CoretweetsWeightedBase


class CoretweetsWeighted1DayApproach(CoretweetsWeightedBase):
    """Periods: UTC calendar days."""

    PERIOD_SECONDS = 86400
    PERIOD_OFFSET = 0

    def get_approach_name(self):
        return "Co-Retweet-Weighted-1day"

    def get_approach_key(self):
        return "coretweets_weighted_1day"
