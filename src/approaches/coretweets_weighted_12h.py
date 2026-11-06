"""Co-retweets weighted by the popularity of the tweet in the 12-hour period of the co-retweet."""

from .coretweets_weighted_base import CoretweetsWeightedBase


class CoretweetsWeighted12hApproach(CoretweetsWeightedBase):
    """
    Periods: consecutive 12-hour windows, 5 AM to 5 PM and 5 PM to 5 AM of the next day (UTC); a retweet before
    5 AM belongs to the window started the previous day.
    """

    PERIOD_SECONDS = 12 * 3600
    PERIOD_OFFSET = 5 * 3600

    def get_approach_name(self):
        return "Co-Retweet-Weighted-12h"

    def get_approach_key(self):
        return "coretweets_weighted_12h"
