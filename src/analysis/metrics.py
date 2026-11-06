"""
Ranking metrics of a suspicious-user ranking against the known IO users.

compute_scores is the list-based reference (scikit-learn AUC and NDCG, tie-averaged AP). The tie-block
functions compute every metric from the IO and control counts of the tie blocks, optionally with user
multiplicities, which is what the bootstraps and the jackknife resample.

Tie blocks: a ranking is given by its tie blocks (maximal sets of users with equal score), best first:
pos[j] = a_j and neg[j] = b_j are the numbers of IO and control users of block j. They may be weighted by an
integer multiplicity per user (a user drawn twice is two tied copies); every metric then equals the metric of
the expanded list. Ties count as the expected value over a uniformly random order inside each block.
Notation: s_j = a_j + b_j, X_j = s_1 + ... + s_j, Y_j = a_1 + ... + a_j, X_0 = Y_0 = 0.

    auc    roc_auc_score (a tied IO / control pair counts 1/2)
    ap     expected AP over random orders inside the blocks (McSherry & Najork, 2008); equal to
           average_precision_score without ties (scikit-learn instead gives every IO user of a block the
           precision at the end of the block, which is not the expected value)
    ndcg   ndcg_score(ignore_ties=False): binary relevance, full list, log2 discount
    p<k>   P@k = (sum of a_j over the blocks with X_j <= k + (k - X_{j*-1}) a_j* / s_j*) / k, j* the first
           block with X_j > k
    ed     expected IO users before the first control: sum of s_j over the leading blocks with b_j = 0, plus
           a_j / (b_j + 1) for the first block with b_j > 0
    fc<r>  FC@r: t = r n_IO, j the first block with Y_j >= t, x = X_{j-1} + (t - Y_{j-1}) s_j / a_j: the users
           at which the expected recall curve (linear inside a block, see recall_curve) reaches r. block_metrics
           returns x (differences are taken before rounding); filtering_concentration rounds up to ceil(x)
"""

import math

import numpy as np
from scipy.special import digamma
from sklearn.metrics import ndcg_score, roc_auc_score

RANK_METRICS = ('auc', 'ap', 'ndcg')

# Tolerance of the comparisons of the Filtering Concentration (target vs cumulative IO, and the ceiling), so
# that e.g. 0.9 * 1610 = 1449.0000000000002 is treated as the integer it represents
EPS = 1e-9


def _labels_and_scores(suspicious_users, io_users):
    """(y_true, y_score) of the ranked users (a user seen in an earlier group is skipped), earlier groups higher."""
    y_true, y_score, seen = [], [], set()
    for group_idx, group in enumerate(suspicious_users):
        score = len(suspicious_users) - group_idx
        for u in group:
            if u not in seen:
                seen.add(u)
                y_true.append(1 if u in io_users else 0)
                y_score.append(score)
    return np.array(y_true), np.array(y_score)


def _auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_score))


def _ap(y_true, y_score):
    if np.sum(y_true) == 0:
        return 0.0
    return float(expected_average_precision(*blocks_from_scores(y_score, np.asarray(y_true) == 1)))


def _ndcg(y_true, y_score):
    if np.sum(y_true) == 0:
        return 0.0
    return float(ndcg_score([y_true], [y_score]))


_LIST_METRICS = {'auc': _auc, 'ap': _ap, 'ndcg': _ndcg}


def compute_scores(suspicious_users, io_users, metrics=RANK_METRICS):
    """{metric: score} of a ranking ('auc', 'ap', 'ndcg'; 0 without IO users, AUC 0 also without controls)."""
    y_true, y_score = _labels_and_scores(suspicious_users, io_users)
    return {m: _LIST_METRICS[m](y_true, y_score) for m in metrics}


def users_until_reaching_io_fraction(suspicious_users, io_users, fraction=0.9):
    """Filtering Concentration of a ranking: users to inspect to recover `fraction` of its IO users (0 if none)."""
    pos, neg = blocks_from_ranking(suspicious_users, io_users)
    if pos.sum() == 0:
        return 0
    return filtering_concentration(pos, neg, fraction)


def blocks_from_ranking(suspicious_users, io_users):
    """(pos, neg) of a ranking (tie groups, best first); users seen in an earlier group are skipped."""
    seen, pos, neg = set(), [], []
    for group in suspicious_users:
        new_users = [u for u in group if u not in seen]
        seen.update(new_users)
        if new_users:
            n_io = sum(1 for u in new_users if u in io_users)
            pos.append(n_io)
            neg.append(len(new_users) - n_io)
    return np.array(pos, dtype=float), np.array(neg, dtype=float)


def users_to_recall(X, Y, recall):
    """
    Users to inspect to reach each recall level (not rounded), from the recall_curve endpoints (X users, Y IO
    users, from (0, 0)): linear inside a block, as filtering_concentration.
    """
    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    t = np.asarray(recall, dtype=float) * Y[-1]
    j = np.maximum(np.searchsorted(Y, t - EPS), 1)       # first endpoint with Y >= t
    return X[j - 1] + (t - Y[j - 1]) * (X[j] - X[j - 1]) / (Y[j] - Y[j - 1])


def blocks_from_groups(group_idx, is_io, n_groups, weights=None):
    """(pos, neg) of users given by their tie-group index (0 = best), with optional multiplicities."""
    w = np.ones(len(group_idx)) if weights is None else np.asarray(weights, dtype=float)
    is_io = np.asarray(is_io, dtype=bool)
    return (np.bincount(group_idx, weights=w * is_io, minlength=n_groups),
            np.bincount(group_idx, weights=w * ~is_io, minlength=n_groups))


def blocks_from_scores(scores, is_io, weights=None, tol=0.0):
    """
    (pos, neg) of users given by their scores (higher = more suspicious), with optional multiplicities.
    Consecutive sorted scores whose gap is <= tol are tied (tol = 0: exact equality; a small tol absorbs
    floating-point summation noise).
    """
    return blocks_from_score_tuples([scores], is_io, weights, tol)


def blocks_from_score_tuples(score_columns, is_io, weights=None, tol=0.0):
    """
    (pos, neg) of users ranked lexicographically by several scores (primary first, higher = more suspicious):
    the dense rank of the score tuple. Two users are tied only if every component differs by <= tol.
    """
    columns = [np.asarray(s, dtype=float) for s in score_columns]
    if len(columns[0]) == 0:
        return np.zeros(0), np.zeros(0)
    order = np.lexsort([-s for s in reversed(columns)])     # the last key is the primary one
    new_block = np.zeros(len(order), dtype=bool)
    new_block[0] = True
    for s in columns:
        s = s[order]
        new_block[1:] |= np.abs(s[:-1] - s[1:]) > tol
    group_idx = np.cumsum(new_block) - 1
    w = None if weights is None else np.asarray(weights, dtype=float)[order]
    return blocks_from_groups(group_idx, np.asarray(is_io, dtype=bool)[order], int(group_idx[-1]) + 1, w)


def recall_curve(pos, neg):
    """Block endpoints (X_j, Y_j), j = 0..J, of the cumulative IO curve, starting at (0, 0)."""
    return (np.concatenate([[0.0], np.cumsum(pos + neg)]),
            np.concatenate([[0.0], np.cumsum(pos)]))


_disc_cum = np.zeros(1)


def _discount_cumsum(n):
    """[0, 1/log2(2), 1/log2(2) + 1/log2(3), ...] for at least n positions (cached, grown on demand)."""
    global _disc_cum
    if len(_disc_cum) <= n:
        size = max(n + 1, 2 * len(_disc_cum))
        _disc_cum = np.concatenate([[0.0], np.cumsum(1.0 / np.log2(np.arange(size - 1) + 2.0))])
    return _disc_cum


def expected_average_precision(pos, neg):
    """
    Expected AP over uniformly random orders inside the tie blocks (McSherry & Najork, 2008).

    For a block of size s with a IO users after X users and Y IO users, position r of the block (r = 1..s) is
    an IO user with probability a/s, and then the other a-1 IO users of the block lie in the other s-1
    positions uniformly, so the expected precision there is (Y + 1 + (r-1)(a-1)/(s-1)) / (X + r). Summing over r
    with harmonic numbers H:
        sum_r 1/(X+r) = H(X+s) - H(X) = h,    sum_r (r-1)/(X+r) = s - (X+1) h,
    and AP = sum over blocks of (a/s) [(Y+1) h + (a-1)/(s-1) (s - (X+1) h)] / (IO users).
    """
    pos, neg = np.asarray(pos, dtype=float), np.asarray(neg, dtype=float)
    P = pos.sum()
    if P <= 0:
        return np.nan
    size = pos + neg
    X = np.cumsum(size) - size
    Y = np.cumsum(pos) - pos
    j = pos > 0
    a, s, X, Y = pos[j], size[j], X[j], Y[j]
    h = digamma(X + s + 1.0) - digamma(X + 1.0)      # H(n) = digamma(n + 1) + Euler's constant
    slope = np.divide(a - 1.0, s - 1.0, out=np.zeros_like(a), where=s > 1)
    return float(np.sum(a / s * ((Y + 1.0) * h + slope * (s - (X + 1.0) * h))) / P)


def grouped_metrics(pos, neg):
    """AUC, AP and NDCG of tie blocks; NaN without IO users (AUC also NaN without controls)."""
    P, N = pos.sum(), neg.sum()
    if P <= 0:
        return np.full(3, np.nan)
    cum_pos, cum_neg = np.cumsum(pos), np.cumsum(neg)
    auc = np.sum(pos * ((N - cum_neg) + 0.5 * neg)) / (P * N) if N > 0 else np.nan
    seen = cum_pos + cum_neg
    ap = expected_average_precision(pos, neg)
    size = pos + neg
    end = np.rint(seen).astype(np.int64)
    start = end - np.rint(size).astype(np.int64)
    disc_cum = _discount_cumsum(int(end[-1]))
    relevance = np.divide(pos, size, out=np.zeros_like(pos), where=size > 0)
    ndcg = np.sum(relevance * (disc_cum[end] - disc_cum[start])) / disc_cum[int(round(P))]
    return np.array([auc, ap, ndcg])


def precision_at_k(pos, neg, k):
    """P@k of tie blocks; the denominator is k even if the ranking has fewer users. 0 if k <= 0."""
    if k <= 0:
        return 0.0
    size = pos + neg
    X = np.cumsum(size)
    inside = X <= k + EPS
    io = pos[inside].sum()
    j = int(np.count_nonzero(inside))
    if j < len(pos) and size[j] > 0:
        io += (k - (X[j] - size[j])) * pos[j] / size[j]     # IO share of the block straddling position k
    return float(io / k)


def early_detection(pos, neg):
    """Expected number of IO users before the first control (ED)."""
    mixed = np.flatnonzero(neg > 0)
    if len(mixed) == 0:
        return float(pos.sum())
    j = mixed[0]
    return float(pos[:j].sum() + pos[j] / (neg[j] + 1.0))


def filtering_concentration(pos, neg, fraction, rounded=True):
    """Users to inspect to recover `fraction` of the IO users (FC), linear inside the tie block: ceil(x) or x."""
    n_io = pos.sum()
    if n_io <= 0:
        return np.nan
    t = fraction * n_io
    Y = np.cumsum(pos)
    j = int(np.argmax(Y >= t - EPS))
    x = np.sum(pos[:j] + neg[:j]) + (t - (Y[j] - pos[j])) * (pos[j] + neg[j]) / pos[j]
    return ceil_users(x) if rounded else float(x)


def ceil_users(x):
    """ceil of a number of users, ignoring floating-point noise (17.000000000000004 -> 17)."""
    return int(math.ceil(x - EPS))


def block_metrics(pos, neg, names):
    """
    Array of the metrics `names` of one ranking given as tie blocks: 'auc', 'ap', 'ndcg', 'ed', 'p<k>' (P@k,
    e.g. 'p100') and 'fc<r>' (FC at r percent of the IO users, e.g. 'fc90', not rounded).
    """
    out = np.empty(len(names))
    grouped = None
    for i, name in enumerate(names):
        if name in RANK_METRICS:
            if grouped is None:
                grouped = grouped_metrics(pos, neg)
            out[i] = grouped[RANK_METRICS.index(name)]
        elif name == 'ed':
            out[i] = early_detection(pos, neg)
        elif name.startswith('fc') and name[2:].isdigit():
            out[i] = filtering_concentration(pos, neg, int(name[2:]) / 100.0, rounded=False)
        elif name.startswith('p') and name[1:].isdigit():
            out[i] = precision_at_k(pos, neg, int(name[1:]))
        else:
            raise ValueError(f"Unknown metric: {name}")
    return out
