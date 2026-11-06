"""Co-retweet count: the baseline approach."""

from ._coretweet_index import get_coretweet_index, window_microseconds
from .base import PairsApproach


class CoRetweetsApproach(PairsApproach):
    """Pair score: number of distinct tweets that both users retweeted within window_sec of each other."""

    def get_approach_name(self):
        return "Co-Retweet"

    def get_approach_key(self):
        return "coretweets"

    def compute_pairs_scores(self, RTs):
        index = get_coretweet_index(RTs)
        window = window_microseconds(self.window_sec)
        # pair -> [number of distinct co-retweeted tweets, last tweet counted]. All pairs of one tweet are found
        # while scanning that tweet, so comparing with the last tweet counted is equivalent to a set of tweets.
        y = {}
        for tweet, acc_times in index.tweets.items():
            n = len(acc_times)
            for i in range(n):
                acc_i, us_i, _ = acc_times[i]
                for j in range(i + 1, n):
                    acc_j, us_j, _ = acc_times[j]
                    if us_j - us_i > window:
                        break
                    if acc_i != acc_j:
                        pair = (acc_i, acc_j) if acc_i < acc_j else (acc_j, acc_i)
                        entry = y.get(pair)
                        if entry is None:
                            y[pair] = [1, tweet]
                        elif entry[1] != tweet:
                            entry[0] += 1
                            entry[1] = tweet
        return {k: v[0] for k, v in y.items() if v[0] >= self.min_coactions}
