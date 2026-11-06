"""
Base classes of the detection approaches.

A pair approach scores every pair of co-retweeting users; the pair scores of a user are aggregated into a
user score (the 'ranking mode'), and users with equal scores form a tie group. A ranking is the list of tie
groups, best first. Rankings are stored in {output_dir}/results/{dataset}/{full approach key}.pkl.
"""

import math
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from statistics import median

import numpy as np
from scipy.stats import trim_mean

from ._coretweet_index import filter_coretweeting_users
from ._lp_centre import lp_centre

# Aggregations of the pair scores of a user; 'L<p>' with integer p >= 3 is the Lp centre:
#   L1 median, L2 mean, Linf midrange, max, min, TM25 interquartile mean, MH midhinge, Q90 0.9-quantile,
#   Linf_droptop / Linf_dropbottom midrange without the top / bottom score, sum strength, degree number of partners
RANKING_MODES = ('L1', 'L2', 'Linf', 'max', 'min', 'TM25', 'MH', 'Q90', 'Linf_droptop', 'Linf_dropbottom',
                 'sum', 'degree')
# Aggregations cheap enough not to be memoized on the sequence of scores of a user
_UNMEMOIZED_MODES = ('L1', 'L2', 'Linf', 'max', 'min', 'Linf_droptop', 'Linf_dropbottom', 'sum')


def check_ranking_mode(mode):
    """Raise ValueError unless mode is one of RANKING_MODES or 'L<p>' with integer p >= 3."""
    if mode in RANKING_MODES or (mode.startswith('L') and mode[1:].isdigit() and int(mode[1:]) >= 3):
        return
    raise ValueError(f"ranking_mode must be one of {', '.join(RANKING_MODES)} or 'L<p>' (p >= 3), got '{mode}'")


def group_scores_by_user(pairs_scores):
    """{user: nonzero scores of its pairs, in pairs_scores order}; independent of the ranking mode."""
    user_pair_scores = {}
    for pair, score in pairs_scores.items():
        if score != 0:
            for user in pair:
                user_pair_scores.setdefault(user, []).append(score)
    return user_pair_scores


def _aggregate_scores(scores, mode):
    """Aggregate the (two or more) pair scores of one user."""
    if mode == 'L1':
        return median(scores)
    if mode == 'L2':
        return sum(scores) / len(scores)
    if mode == 'Linf':
        return (min(scores) + max(scores)) / 2
    if mode == 'max':
        return max(scores)
    if mode == 'min':
        return min(scores)
    if mode == 'TM25':
        return float(trim_mean(scores, 0.25))       # trims floor(0.25 n) scores at each end
    if mode == 'MH':
        q1, q3 = np.quantile(scores, [0.25, 0.75])
        return float((q1 + q3) / 2)
    if mode == 'Q90':
        return float(np.quantile(scores, 0.9))
    if mode == 'Linf_droptop':
        rest = sorted(scores)[:-1]                   # one occurrence of the max removed
        return (min(rest) + max(rest)) / 2
    if mode == 'Linf_dropbottom':
        rest = sorted(scores)[1:]                    # one occurrence of the min removed
        return (min(rest) + max(rest)) / 2
    if mode == 'sum':
        return math.fsum(scores)                     # exactly rounded: independent of the order of the pairs
    return lp_centre(scores, int(mode[1:]))


def aggregate_user_scores(user_pair_scores, mode):
    """{user: aggregated score} from group_scores_by_user; a single pair score is the user score."""
    check_ranking_mode(mode)
    if mode == 'degree':
        return {user: float(len(scores)) for user, scores in user_pair_scores.items()}
    memoize = mode not in _UNMEMOIZED_MODES          # the aggregation is a function of the score sequence
    cache = {}
    user_scores = {}
    for user, scores in user_pair_scores.items():
        if len(scores) == 1:
            user_scores[user] = scores[0]
        elif memoize:
            key = tuple(scores)
            if key not in cache:
                cache[key] = _aggregate_scores(scores, mode)
            user_scores[user] = cache[key]
        else:
            user_scores[user] = _aggregate_scores(scores, mode)
    return user_scores


def rank_users(user_scores):
    """Tie groups of users by decreasing score, each group sorted by user id."""
    sorted_users = sorted(user_scores.items(), key=lambda item: item[1], reverse=True)
    groups = []
    idx, n = 0, len(sorted_users)
    while idx < n:
        score = sorted_users[idx][1]
        group = []
        while idx < n and sorted_users[idx][1] == score:
            group.append(sorted_users[idx][0])
            idx += 1
        groups.append(sorted(group))
    return groups


def get_ranks(suspicious):
    """{user: index of its tie group} of a ranking (0 = most suspicious)."""
    return {user: rank for rank, users in enumerate(suspicious) for user in users}


class Approach(ABC):
    """
    Detection approach.

    Args:
        window_sec: two retweets of the same tweet are synchronized if at most window_sec apart
        min_coactions: only pairs with at least this many synchronized co-retweets are scored
        ranking_mode: aggregation of the pair scores of a user (see RANKING_MODES)
        need_filtering: whether the retweets are first restricted to co-retweeting users (None: every
            approach but the co-retweet count); part of the storage key
    """

    def __init__(self, window_sec=60, min_coactions=1, ranking_mode='Linf', need_filtering=None):
        check_ranking_mode(ranking_mode)
        self.window_sec = window_sec
        self.min_coactions = min_coactions
        self.ranking_mode = ranking_mode
        self.need_filtering = need_filtering

    @abstractmethod
    def get_approach_name(self):
        """Human-readable name."""

    @abstractmethod
    def get_approach_key(self):
        """Key of the approach (e.g. 'coretweets')."""

    @abstractmethod
    def _compute_suspicious(self, RTs, **storage):
        """Ranking of the users; storage holds the arguments of get_suspicious that store rankings."""

    def _get_need_filtering(self):
        if self.need_filtering is None:
            return self.get_approach_key() != 'coretweets'
        return self.need_filtering

    def get_full_approach_key(self):
        """Storage key: key, '_min<k>' if min_coactions > 1, '_<ranking mode>', '_nofilter' if not filtering."""
        key = self.get_approach_key()
        if self.min_coactions > 1:
            key = f"{key}_min{self.min_coactions}"
        key = f"{key}_{self.ranking_mode}"
        if not self._get_need_filtering():
            key = f"{key}_nofilter"
        return key

    def get_metadata(self):
        return {
            'approach_class': self.__class__.__name__,
            'approach_name': self.get_approach_name(),
            'approach_key': self.get_approach_key(),
            'window_sec': self.window_sec,
            'min_coactions': self.min_coactions,
            'ranking_mode': self.ranking_mode,
        }

    def _result_path(self, output_dir, dataset):
        return Path(output_dir) / 'results' / dataset / f"{self.get_full_approach_key()}.pkl"

    def is_result_cached(self, output_dir, dataset):
        return self._result_path(output_dir, dataset).exists()

    def save_result(self, output_dir, dataset, io_users, suspicious_users):
        """Store a ranking with the IO users and the metadata of the approach."""
        result = {
            'dataset': dataset,
            'suspicious_users': suspicious_users,
            'io_users': io_users,
            'num_suspicious_groups': len(suspicious_users),
            'num_suspicious_users': sum(len(group) for group in suspicious_users),
            **self.get_metadata(),
        }
        path = self._result_path(output_dir, dataset)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(result, f)

    def get_suspicious(self, RTs, output_dir=None, dataset=None, io_users=None, force=False):
        """
        Ranking of the users: tie groups, best first.

        With output_dir and dataset, a stored ranking is reused (unless force) and a new non-empty one stored.
        """
        storage = {'output_dir': output_dir, 'dataset': dataset, 'io_users': io_users, 'force': force}
        stored = output_dir and dataset
        if stored and not force and self.is_result_cached(output_dir, dataset):
            with open(self._result_path(output_dir, dataset), 'rb') as f:
                return pickle.load(f)['suspicious_users']
        suspicious = self._compute_suspicious(RTs, **storage)
        if stored and suspicious:
            self.save_result(output_dir, dataset, io_users, suspicious)
        return suspicious


class PairsApproach(Approach):
    """Approach that scores pairs of users and ranks users by the aggregation of their pair scores."""

    @abstractmethod
    def compute_pairs_scores(self, RTs):
        """{(u, v) with u < v: score} of the pairs with at least min_coactions synchronized co-retweets."""

    def prepare_RTs(self, RTs):
        """The retweets the pair scores are computed on (restricted to co-retweeting users if filtering)."""
        if self._get_need_filtering():
            return filter_coretweeting_users(RTs, self.window_sec)
        return RTs

    def pairs_signature(self):
        """Everything the pair scores depend on: approaches with equal signatures share their pair scores."""
        return type(self), self.window_sec, self.min_coactions, self._get_need_filtering()

    def rank_from_user_pair_scores(self, user_pair_scores):
        """Ranking from group_scores_by_user, with the ranking mode of the approach."""
        return rank_users(aggregate_user_scores(user_pair_scores, self.ranking_mode))

    def _compute_suspicious(self, RTs, **_storage):
        pairs_scores = self.compute_pairs_scores(self.prepare_RTs(RTs))
        if not pairs_scores:
            return []
        return self.rank_from_user_pair_scores(group_scores_by_user(pairs_scores))
