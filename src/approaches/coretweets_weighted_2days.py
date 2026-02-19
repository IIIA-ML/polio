"""Same tweet, same time approach."""

from datetime import datetime, time
from itertools import groupby
import numpy as np
from typing import Dict, Tuple, List
from collections import defaultdict
from .base import PairsApproach


class CoretweetsWeighted2DaysApproach(PairsApproach):
    """
    Weighted same-tweet, same-time coordination detection.

    This approach identifies coordinated behavior by analyzing when users retweet
    the same content within a specific time window. Unlike simple counting methods,
    it uses weighted scoring to account for tweet popularity:

    This method captures these intuitions by weighting co-retweets inversely to
    tweet popularity, rather than simply counting them. The weight for each
    co-retweet is: (number of retweets of that tweet on that day and next one) / (total 
    retweets of that tweet). This downweights co-retweets of viral tweets while
    emphasizing co-retweets of less popular content.
    """

    def get_approach_name(self) -> str:
        return "Co-Retweet-Weighted-2days"

    def get_approach_key(self) -> str:
        return "coretweets_weighted_2days"

    @staticmethod
    def _group_RTs_by_tweet(RTs):
        """Group retweets by tweet, sorted by timestamp."""
        tweet_data = defaultdict(list)
        for user, tid, ts in RTs:
            tweet_data[tid].append((user, ts))
        
        # Sort each tweet's retweets chronologically
        for tid in tweet_data:
            tweet_data[tid].sort(key=lambda x: x[1])
        return tweet_data
    
    @staticmethod
    def get_date_tweet_popularity(RTs, date_set, synchronized_tweets):
        """
        Calculate how many times each tweet was retweeted on each synchronized day.
        
        This helper computes tweet popularity on the days identified as having
        coordinated behavior. These popularity counts are used to normalize
        co-retweet weights: tweets shared by many users on a day contribute
        less weight (they're likely organic), while rare tweets contribute more
        weight (they're less likely to be coincidence).
        
        Args:
            RTs: List of (user_id, tweet_id, timestamp) tuples
            date_set: Set of days (as Unix timestamps) identified as synchronized
            synchronized_tweets: Set of tweet IDs involved in synchronized activity
            
        Returns:
            Dictionary mapping {date_seconds -> {tweet_id -> retweet_count}}
        """
        date_tweet_popularity = defaultdict(lambda: defaultdict(int))
        for _, tid, ts in RTs:
            date_seconds = np.int32(int(datetime.combine(ts.date(), time.min).timestamp()))
            if date_seconds in date_set and tid in synchronized_tweets:
                date_tweet_popularity[date_seconds][tid] += 1

        return date_tweet_popularity

    def _compute_pairs_scores_impl(self, RTs: List[Tuple], **kwargs) -> Dict[Tuple, float]:
        """
        Compute weighted coordination scores for user pairs based on co-retweets.

        Algorithm overview:
        1. Group retweets by tweet and sort chronologically
        2. For each tweet, identify pairs of users who retweeted it within the
           time window, recording which day they coordinated
        3. Identify "synchronized" days and tweets: pairs with sufficient
           co-retweets (>= min_coactions threshold) on the same day
        4. Compute popularity baseline: how many users retweeted each synchronized
           tweet on each synchronized day
        5. Weight each co-retweet by the inverse of tweet popularity on that day:
           weight = (daily retweets of that tweet) / (total retweets of that tweet)
           
        This weighting scheme ensures:
        - Viral tweets (many daily retweets) contribute low weight per co-retweet
        - Rare tweets (few daily retweets) contribute high weight per co-retweet
        - The approach only considers coordinated pairs meeting min_coactions threshold

        Args:
            RTs: List of (user_id, tweet_id, timestamp) tuples
                 Should be filtered to users who have co-retweeted
            **kwargs: Additional parameters (unused for this approach)

        Returns:
            Dictionary mapping user pairs to their weighted coordination scores
        """
        tweet_data = self._group_RTs_by_tweet(RTs)
        pair_days = defaultdict(lambda: defaultdict(set))

        # STEP 1: Find user pairs who co-retweeted within the time window
        # For each tweet, iterate through all retweets chronologically and identify
        # pairs whose retweets are within window_sec of each other
        for tweet, events in tweet_data.items():
            for i in range(len(events)):
                u1, t1 = events[i]
                for j in range(i+1, len(events)):
                    u2, t2 = events[j]
                    # Check if retweets are within time window
                    if (t2 - t1).total_seconds() > self.window_sec:
                        break
                    if u1 != u2:
                        pair = tuple(sorted((u1, u2)))
                        # Record: this pair co-retweeted this tweet on this day
                        date_only_t1 = datetime.combine(t1.date(), time.min)
                        current_day_seconds_t1 = np.int32(int(date_only_t1.timestamp()))
                        date_only_t2 = datetime.combine(t2.date(), time.min)
                        current_day_seconds_t2 = np.int32(int(date_only_t2.timestamp()))
                        if current_day_seconds_t1 == current_day_seconds_t2:
                            current_day_seconds = current_day_seconds_t1
                        else:
                            current_day_seconds = (current_day_seconds_t1, current_day_seconds_t2)

                        pair_days[pair][current_day_seconds].add(tweet)
        
        # STEP 2: Identify synchronized days and tweets
        # A day is "synchronized" if a pair has >= min_coactions co-retweets on that day
        # Pairs below this threshold are filtered out (likely organic sharing)
        synchronized_days = set()
        synchronized_tweets = set()
        for days_tweets in pair_days.values():
            # Count total co-retweets (not distinct days, but sum of tweets per day)
            if sum(len(tweetset) for tweetset in days_tweets.values()) >= self.min_coactions:
                for day, tweets_set in days_tweets.items():
                    if isinstance(day, tuple):
                        synchronized_days.add(day[0])
                        synchronized_days.add(day[1])
                    else:
                        synchronized_days.add(day)
                    synchronized_tweets.update(tweets_set)
        
        # STEP 3: Compute popularity baseline for synchronized content
        # For each tweet on each synchronized day, count how many users retweeted it
        # This will be used to normalize co-retweet weights
        date_tweet_popularity = self.get_date_tweet_popularity(RTs, synchronized_days, synchronized_tweets)
        
        # STEP 4: Compute weighted scores for coordinated pairs
        # Weight inversely to popularity: rare co-retweets score higher
        
        # Build result dictionary: only include pairs meeting min_coactions threshold
        result = {}
        for pair, days_tweets in pair_days.items():
            weight = 0
            # Only score pairs with sufficient coordinated co-retweets
            if sum(len(tweetset) for tweetset in days_tweets.values()) >= self.min_coactions:
                for day, tweets_set in days_tweets.items():
                    for tweet in tweets_set:
                        # Weight = daily tweet retweets / total tweet retweets
                        # Viral tweets: weight ~= 1/total (low)
                        # Rare tweets: weight ~= 1/daily ≈ 1/1 or 1/2 (high)
                        # This makes rare co-retweets more significant
                        if isinstance(day, tuple):
                            next_day_1 = day[0] + 86400
                            next_day_2 = day[1] + 86400
                            frac_popularity = (date_tweet_popularity[day[0]][tweet] + date_tweet_popularity[next_day_1][tweet] + date_tweet_popularity[day[1]][tweet] + date_tweet_popularity[next_day_2][tweet]) / (2*len(tweet_data[tweet]))
                        else:
                            next_day = day + 86400
                            frac_popularity = (date_tweet_popularity[day][tweet] + date_tweet_popularity[next_day][tweet]) / len(tweet_data[tweet])
                        #if frac_popularity > 0.9:
                        weight += frac_popularity

                if weight > 0:    
                    result[pair] = weight

        return result
