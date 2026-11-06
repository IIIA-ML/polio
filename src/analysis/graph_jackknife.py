"""
Graph-level jackknife: the synchronized co-retweet graph recomputed without a group of tweets or of users.

A replicate gives every tweet a multiplicity k_tweet and every user a multiplicity c_u (the jackknife uses
k, c in {0, 1}: a removed group gets 0, everything else 1). The quantities of the weights are recomputed on
the replicate:
    m'(tweet)    sum of c_u over the events of the tweet
    m'_T(tweet)  the same, restricted to the period T
    N'           sum of c_u over the users with >= 1 event on a kept tweet (k_tweet > 0)
    n'(tweet)    sum of c_u over the distinct users of the tweet
and the pair score of users u != v with c_u, c_v > 0 is
    s'(u, v) = sum over tweets of k_tweet * sum over the synchronized pairs of (u, v) on the tweet of w'
with w' the weight of the approach computed with the primed quantities (w' = 1 for the count).

Fixed cohort: the cohort and its qualifying pairs (raw count >= min_coactions) are those of the full data and
are not recomputed, so every replicate ranks the same users. The user score aggregates s' over the qualifying
partners v with c_v > 0 and s'(u, v) > 0; a cohort user without such a partner gets BOTTOM_SCORE (tied last).

A ranker is a tuple of components (weight, aggregation): users are ranked by the first component, ties broken
by the next ones (the lexicographic approach). Aggregations: 'Linf' (midrange), 'L<p>' (integer p >= 2, the Lp
centre), 'max' and 'min', as Approach aggregates them. The pair search (window rule) and the periods are those
of the approaches; the weights are those of coretweets, coretweets_idf, coretweets_inv_m and
coretweets_weighted_12h. With every k = 1 and c = 1 a replicate reproduces the full-data rankings (up to TIE_TOL).
"""

import numpy as np

from approaches._coretweet_index import get_coretweet_index, window_microseconds
from approaches.coretweets_weighted_12h import CoretweetsWeighted12hApproach

from .metrics import block_metrics, blocks_from_score_tuples

# Pair weight of every approach whose pair scores are recomputed here
APPROACH_WEIGHTS = {'coretweets': 'count', 'coretweets_idf': 'idf', 'coretweets_inv_m': 'inv_m',
                    'coretweets_weighted_12h': 'wT12h'}
# Bisection steps of the Lp centre: 2^-64 is below the float resolution of [0, 1]
LP_ITERATIONS = 64
# Users whose scores differ by at most TIE_TOL are tied: the pair scores are summed in a different order than
# in the approaches (bincount vs math.fsum), which changes the last bits of a float
TIE_TOL = 1e-9
# Score of a cohort user without a live qualifying pair in a replicate (every live score is > 0)
BOTTOM_SCORE = -1.0


class SyncGraph:
    """
    Events and synchronized co-retweet units of one dataset, as integer arrays.

    Attributes (user and tweet indices refer to the sorted user and tweet ids):
        users, io               user ids and IO mask
        ev_user, ev_tweet       user and tweet of every retweet event
        ev_tp                   (tweet, 12h period) index of every event
        unit_tweet, unit_pair   tweet and pair of every synchronized unit
        unit_tp1, unit_tp2      (tweet, period) index of the two retweets of the unit
        pair_u, pair_v          users of every pair (u < v)
        pair_count              full-data count of every pair (its number of synchronized units)
        he_pair, he_starts      half-edges (one per pair end) sorted by user, and the start of the
        he_users                segment of every user with >= 1 pair
    """

    def __init__(self, RTs, io_users, window_sec=60, period_key=CoretweetsWeighted12hApproach._period_key):
        index = get_coretweet_index(RTs)
        window = window_microseconds(window_sec)
        users = sorted({u for u, _, _ in RTs})
        user_index = {u: i for i, u in enumerate(users)}
        tweets = sorted(index.tweets)

        ev_user, ev_tweet, ev_period, unit_a, unit_b = [], [], [], [], []
        offset = 0
        for t, tweet in enumerate(tweets):
            events = index.tweets[tweet]
            n = len(events)
            for u, _, ts in events:
                ev_user.append(user_index[u])
                ev_tweet.append(t)
                ev_period.append(period_key(ts))
            # Same scan as the approaches: events sorted by time, stop past the window
            for i in range(n):
                u1, t1, _ = events[i]
                for j in range(i + 1, n):
                    u2, t2, _ = events[j]
                    if t2 - t1 > window:
                        break
                    if u1 != u2:
                        unit_a.append(offset + i)
                        unit_b.append(offset + j)
            offset += n

        self.users = np.array(users)
        self.n_users, self.n_tweets = len(users), len(tweets)
        io_users = set(io_users)
        self.io = np.array([u in io_users for u in users], dtype=bool)
        self.ev_user = np.array(ev_user, dtype=np.int64)
        self.ev_tweet = np.array(ev_tweet, dtype=np.int64)
        if len(np.unique(self.ev_tweet * self.n_users + self.ev_user)) != len(self.ev_user):
            # The approaches count a pair once per tweet (count) or per (tweet, period) (weights), which equals
            # the number of synchronized event pairs only without repeated retweets
            raise ValueError("SyncGraph requires at most one retweet per (user, tweet)")

        periods, period_idx = np.unique(np.array(ev_period, dtype=np.int64), return_inverse=True)
        tp_keys, self.ev_tp = np.unique(self.ev_tweet * max(len(periods), 1) + period_idx, return_inverse=True)
        self.n_tp = len(tp_keys)

        unit_a = np.array(unit_a, dtype=np.int64)
        unit_b = np.array(unit_b, dtype=np.int64)
        ua, ub = self.ev_user[unit_a], self.ev_user[unit_b]
        pair_keys, self.unit_pair = np.unique(np.minimum(ua, ub) * self.n_users + np.maximum(ua, ub),
                                              return_inverse=True)
        self.n_pairs = len(pair_keys)
        self.pair_u, self.pair_v = pair_keys // self.n_users, pair_keys % self.n_users
        self.pair_count = np.bincount(self.unit_pair, minlength=self.n_pairs)
        self.unit_tweet = self.ev_tweet[unit_a]
        self.unit_tp1, self.unit_tp2 = self.ev_tp[unit_a], self.ev_tp[unit_b]

        he_user = np.concatenate([self.pair_u, self.pair_v])
        order = np.argsort(he_user, kind='stable')
        self.he_pair = np.concatenate([np.arange(self.n_pairs)] * 2)[order]
        self.he_users, self.he_starts = np.unique(he_user[order], return_index=True)

    @property
    def n_units(self):
        return len(self.unit_pair)


class FixedCohort:
    """
    Full-data cohort of one min_coactions rule, kept fixed in every replicate.

    Attributes:
        qualifying          mask of the pairs with full-data count >= min_coactions
        units               indices of the synchronized units of the qualifying pairs
        users, io           cohort user indices (sorted) and their IO mask
        he_pair, he_starts  half-edges of the qualifying pairs sorted by user, and the start of the segment of
                            every cohort user
        strata              user indices of the four strata of the jackknife groups, in order: IO and controls
                            of the cohort, then IO and controls outside it
    """

    def __init__(self, g, min_coactions):
        self.min_coactions = min_coactions
        self.qualifying = g.pair_count >= min_coactions
        self.units = np.flatnonzero(self.qualifying[g.unit_pair])
        he_user = np.repeat(g.he_users, np.diff(np.append(g.he_starts, len(g.he_pair))))
        keep = self.qualifying[g.he_pair]
        self.he_pair = g.he_pair[keep]
        self.users, self.he_starts = np.unique(he_user[keep], return_index=True)
        self.io = g.io[self.users]
        in_cohort = np.zeros(g.n_users, dtype=bool)
        in_cohort[self.users] = True
        self.strata = tuple(np.flatnonzero(m) for m in (in_cohort & g.io, in_cohort & ~g.io,
                                                        ~in_cohort & g.io, ~in_cohort & ~g.io))


def unit_multiplicities(g):
    """k = 1 for every tweet and c = 1 for every user: the replicate equal to the full data."""
    return np.ones(g.n_tweets), np.ones(g.n_users)


def primed_quantities(g, k, c):
    """{'m': m' per tweet, 'm_T': m'_T per (tweet, period), 'N': N', 'n': n' per tweet} of one replicate."""
    cw = c[g.ev_user]
    m = np.bincount(g.ev_tweet, weights=cw, minlength=g.n_tweets)
    drawn = (k[g.ev_tweet] > 0).astype(float)
    active = np.bincount(g.ev_user, weights=drawn, minlength=g.n_users) > 0
    return {'m': m,
            'm_T': np.bincount(g.ev_tp, weights=cw, minlength=g.n_tp),
            'N': float(c[active].sum()),
            'n': m}            # one retweet per (user, tweet) (checked by SyncGraph): n'(tweet) = m'(tweet)


def pair_scores(g, k, c, weights, units):
    """
    {weight: s'} of one replicate for every original pair, summing the given synchronized units; pairs with
    c_u = 0 or c_v = 0, or without a kept tweet, have score 0. weights: among 'count', 'idf', 'inv_m', 'wT12h'.
    """
    pair, tweet = g.unit_pair[units], g.unit_tweet[units]
    kt = k[tweet]
    live = (c[g.pair_u[pair]] > 0) & (c[g.pair_v[pair]] > 0) & (kt > 0)
    q = primed_quantities(g, k, c)
    out = {}
    with np.errstate(divide='ignore', invalid='ignore'):
        for weight in weights:
            if weight == 'count':
                w = 1.0
            elif weight == 'wT12h':
                w = (q['m_T'][g.unit_tp1[units]] + q['m_T'][g.unit_tp2[units]]) / (2.0 * q['m'][tweet])
            elif weight == 'idf':
                w = np.log1p(q['N'] / q['n'][tweet])
            elif weight == 'inv_m':
                w = 1.0 / q['m'][tweet]
            else:
                raise ValueError(f"Unknown weight: {weight}")
            out[weight] = np.bincount(pair, weights=np.where(live, kt * w, 0.0), minlength=g.n_pairs)
    return out


def lp_exponent(aggregation):
    """p of an 'L<p>' aggregation (integer p >= 2), None for 'Linf', 'max' and 'min'."""
    if aggregation in ('Linf', 'max', 'min'):
        return None
    if aggregation.startswith('L') and aggregation[1:].isdigit() and int(aggregation[1:]) >= 2:
        return int(aggregation[1:])
    raise ValueError(f"Aggregation {aggregation!r} is not supported by the graph-level resampling "
                     "('Linf', 'L<p>' with integer p >= 2, 'max', 'min')")


def ranker_from_spec(spec):
    """
    Ranker of an approach spec of the experiment configs: one component (weight, aggregation), or one per
    sub-approach for the lexicographic approach ('ranking_mode', or 'ranking_modes', must be in the spec).
    """
    if spec['name'] == 'lexicographic':
        names, modes = spec['approaches'], spec['ranking_modes']
        modes = [modes] * len(names) if isinstance(modes, str) else list(modes)
    else:
        names, modes = [spec['name']], [spec['ranking_mode']]
    if len(modes) != len(names):
        raise ValueError(f"{spec}: one aggregation per sub-approach is needed")
    missing = [n for n in names if n not in APPROACH_WEIGHTS]
    if missing:
        raise ValueError(f"{spec}: no graph-level weight for {missing} (known: {sorted(APPROACH_WEIGHTS)})")
    for mode in modes:
        lp_exponent(mode)
    return tuple((APPROACH_WEIGHTS[n], m) for n, m in zip(names, modes))


def lp_centres(v, live, starts, lo, hi, p, iterations=LP_ITERATIONS):
    """
    Lp centre argmin_c sum |s - c|^p (p > 1) of the live scores of every segment of v (segment i starts at
    starts[i]), as approaches._lp_centre.lp_centre: the scores are mapped to [0, 1] by the smallest (lo) and
    largest (hi) live score of the segment, and the root of the strictly decreasing sum sign(t - c) |t - c|^(p-1)
    is found by bisection on [0, 1], vectorized over the segments (lp_centre uses Brent's method; both stop at
    the float resolution of [0, 1]). A segment whose live scores are all equal gets that score, one without live
    scores NaN.
    """
    n = len(starts)
    seg = np.repeat(np.arange(n), np.diff(np.append(starts, len(v))))
    span = hi - lo
    finite = np.isfinite(span)
    solve = finite & (span > 0)
    centre = np.where(finite, lo, np.nan)
    sel = live & solve[seg]
    u = seg[sel]
    t = (v[sel] - lo[u]) / span[u]      # exactly 0 at the minimum and 1 at the maximum
    a, b = np.zeros(n), np.ones(n)
    for _ in range(iterations):
        mid = (a + b) / 2
        d = t - mid[u]
        up = np.bincount(u, weights=np.sign(d) * np.abs(d) ** (p - 1), minlength=n) > 0
        a, b = np.where(up, mid, a), np.where(up, b, mid)
    root = (a + b) / 2
    centre[solve] = np.minimum(np.maximum(lo[solve] + span[solve] * root[solve], lo[solve]), hi[solve])
    return centre


def aggregate(v, live, starts, aggregation):
    """User score of every segment of v from its live pair scores (NaN without live scores)."""
    hi = np.maximum.reduceat(np.where(live, v, -np.inf), starts)
    lo = np.minimum.reduceat(np.where(live, v, np.inf), starts)
    if aggregation == 'max':
        return hi
    if aggregation == 'min':
        return lo
    p = lp_exponent(aggregation)
    with np.errstate(invalid='ignore'):
        if p is None:
            return (lo + hi) / 2
        if p == 2:
            return np.add.reduceat(np.where(live, v, 0.0), starts) / np.add.reduceat(live.astype(float), starts)
    return lp_centres(v, live, starts, lo, hi, p)


def component_scores(cohort, scores, components):
    """
    User scores of the cohort in one replicate, one per component (weight, aggregation). A qualifying pair is
    live if its replicate count is > 0; a cohort user without a live pair gets BOTTOM_SCORE.

    Returns:
        (mask of the cohort users with >= 1 live pair, {component: scores aligned with cohort.users})
    """
    he_live = scores['count'][cohort.he_pair] > 0
    alive = np.logical_or.reduceat(he_live, cohort.he_starts)
    out = {}
    for weight, aggregation in components:
        values = aggregate(scores[weight][cohort.he_pair], he_live, cohort.he_starts, aggregation)
        out[(weight, aggregation)] = np.where(alive, values, BOTTOM_SCORE)
    return alive, out


def ranker_weights(rankers):
    """Pair weights needed by the rankers ('count' always: it defines the live pairs)."""
    return tuple(dict.fromkeys(['count'] + [w for r in rankers for w, _ in r]))


def replicate_metrics(g, cohort, k, c, rankers, metric_names, evaluated=None):
    """
    Metrics of one replicate: (array (rankers x metrics), {'dead': cohort users (with multiplicity) without a
    live pair, 'dead_io': the IO ones}). Users whose scores differ by <= TIE_TOL are tied. evaluated (boolean
    mask aligned with cohort.users, default all) restricts the users on which the metrics are computed; every
    user still enters the graph.
    """
    scores = pair_scores(g, k, c, ranker_weights(rankers), cohort.units)
    alive, by_component = component_scores(cohort, scores, dict.fromkeys(comp for r in rankers for comp in r))
    w = c[cohort.users] if evaluated is None else c[cohort.users] * evaluated
    out = np.array([block_metrics(*blocks_from_score_tuples([by_component[comp] for comp in r], cohort.io, w,
                                                            TIE_TOL), metric_names)
                    for r in rankers])
    dead = w * ~alive
    return out, {'dead': float(dead.sum()), 'dead_io': float(dead[cohort.io].sum())}


def jackknife_groups(rng, g, cohort, n_groups):
    """
    Random partitions of the delete-a-group jackknife: (group of every tweet, group of every user). Users are
    split stratum by stratum (cohort.strata), so every group removes about 1/n_groups of each stratum.
    """
    tweet_group = rng.permutation(np.arange(g.n_tweets) % n_groups)
    user_group = np.empty(g.n_users, dtype=np.int64)
    for stratum in cohort.strata:
        user_group[stratum] = rng.permutation(np.arange(len(stratum)) % n_groups)
    return tweet_group, user_group


def grouped_jackknife(g, cohort, rankers, metric_names, groups, n_groups, evaluated=None):
    """
    Delete-a-group jackknife of both factors: every tweet group, and separately every user group, of
    groups = jackknife_groups(...) is removed once (k = 0 or c = 0, everything else 1) and the weights, scores
    and metrics recomputed with the cohort fixed. evaluated: as in replicate_metrics.

    Returns:
        {'tweets': (n_groups, rankers, metrics), 'users': (n_groups, rankers, metrics)} replicate metrics
    """
    tweet_group, user_group = groups
    k1, c1 = unit_multiplicities(g)
    out = {'tweets': np.empty((n_groups, len(rankers), len(metric_names))),
           'users': np.empty((n_groups, len(rankers), len(metric_names)))}
    for j in range(n_groups):
        out['tweets'][j] = replicate_metrics(g, cohort, np.where(tweet_group == j, 0.0, k1), c1,
                                             rankers, metric_names, evaluated)[0]
        out['users'][j] = replicate_metrics(g, cohort, k1, np.where(user_group == j, 0.0, c1),
                                            rankers, metric_names, evaluated)[0]
    return out
