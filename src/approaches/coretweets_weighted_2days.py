"""Co-retweets weighted by the popularity of the tweet on the day of the co-retweet and the next one."""

from .coretweets_weighted_base import CoretweetsWeightedBase

ONE_DAY = 86400


class CoretweetsWeighted2DaysApproach(CoretweetsWeightedBase):
    """Periods: UTC calendar days; the popularity of a day adds that of the next one."""

    PERIOD_SECONDS = ONE_DAY
    PERIOD_OFFSET = 0

    def get_approach_name(self):
        return "Co-Retweet-Weighted-2days"

    def get_approach_key(self):
        return "coretweets_weighted_2days"

    @classmethod
    def _counted_periods(cls, date_set):
        """The synchronized days and the day after each of them."""
        return set(date_set) | {day + ONE_DAY for day in date_set}

    def _popularity_fraction(self, day, tweet, date_tweet_popularity, tweet_retweets):
        if isinstance(day, tuple):
            next_day_1 = day[0] + ONE_DAY
            next_day_2 = day[1] + ONE_DAY
            retweets = (date_tweet_popularity[day[0]][tweet] + date_tweet_popularity[next_day_1][tweet]
                        + date_tweet_popularity[day[1]][tweet] + date_tweet_popularity[next_day_2][tweet])
            return retweets / (2*tweet_retweets)
        next_day = day + ONE_DAY
        return (date_tweet_popularity[day][tweet] + date_tweet_popularity[next_day][tweet]) / tweet_retweets
