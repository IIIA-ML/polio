"""
Uncertainty of ranking metrics: conditional (evaluation) bootstrap, graph-level jackknife intervals and paired
differences.

Conditional bootstrap: every ranking is computed once on the full data and kept fixed. In each replicate the
IO users and the controls of the cohort are resampled with replacement separately, each keeping its size; a
user drawn several times enters as tied copies (same score, same label). Every ranking of a block is scored on
the same draw, so differences are paired.
"""

import zlib

import numpy as np
from scipy.stats import t as student_t

from .metrics import block_metrics, blocks_from_groups

PARITY_TOL = 1e-9       # largest difference between the tie-block and the list-based metrics of a ranking


def block_seed(seed, *key):
    """SeedSequence(seed, spawn_key=key), with every string of the key replaced by its crc32."""
    return np.random.SeedSequence(seed, spawn_key=tuple(zlib.crc32(k.encode()) if isinstance(k, str) else k
                                                        for k in key))


def encode_rankings(rankings, io_users, sort_users=True):
    """
    Encode rankings of the same cohort on a common user index.

    Args:
        rankings: {key: suspicious groups (list of tie groups, best first)}
        io_users: known IO users
        sort_users: index the users by id; otherwise in the order of the first ranking

    Returns:
        (universe, encoded, io_mask): the indexed user ids, {key: (tie-group index of every user of the
        universe, number of groups)} and the IO mask of the universe

    Raises:
        ValueError: if a ranking repeats a user or the rankings do not rank the same users
    """
    user_sets = {}
    for key, groups in rankings.items():
        users = [u for g in groups for u in g]
        if len(set(users)) != len(users):
            raise ValueError(f"{key}: repeated users in the ranking")
        user_sets[key] = set(users)
    first = next(iter(rankings))
    reference = user_sets[first]
    for key, users in user_sets.items():
        if users != reference:
            raise ValueError(f"cohort invariance fails for {key}: {len(users - reference)} extra, "
                             f"{len(reference - users)} missing users")
    universe = np.array(sorted(reference) if sort_users else [u for g in rankings[first] for u in g])
    position = {u: i for i, u in enumerate(universe)}
    encoded = {}
    for key, groups in rankings.items():
        group_idx = np.empty(len(universe), dtype=np.int64)
        for g, group in enumerate(groups):
            group_idx[[position[u] for u in group]] = g
        encoded[key] = (group_idx, len(groups))
    io_users = set(io_users)
    return universe, encoded, np.array([u in io_users for u in universe], dtype=bool)


def checked_block_metrics(encoded, io_mask, metric_names, references, tol=PARITY_TOL):
    """
    Metrics of the encoded rankings with unit weights (the full data), checked against the list-based scores
    of the same rankings, so that the bootstrap describes the reported point estimates.

    Args:
        encoded, io_mask: as returned by encode_rankings
        metric_names: metrics to compute (block_metrics names)
        references: {key: {metric: value}} of every ranking to compute (compute_scores); the metrics in both
            metric_names and references are compared, a NaN agreeing only with a NaN
        tol: largest absolute difference allowed

    Returns:
        ({key: array of metric_names}, largest absolute difference)

    Raises:
        ValueError: if a difference exceeds tol
    """
    values, worst = {}, 0.0
    for key, reference in references.items():
        group_idx, n_groups = encoded[key]
        values[key] = block_metrics(*blocks_from_groups(group_idx, io_mask, n_groups), metric_names)
        shared = [i for i, m in enumerate(metric_names) if m in reference]
        got = values[key][shared]
        expected = np.array([reference[metric_names[i]] for i in shared], dtype=float)
        diff = np.where(np.isnan(got) & np.isnan(expected), 0.0, np.abs(got - expected))
        if not np.all(diff <= tol):     # also False for a NaN on one side only
            raise ValueError(f"{key}: tie-block metrics {got} differ from compute_scores {expected} "
                             f"(tolerance {tol:g})")
        worst = max(worst, float(diff.max(initial=0.0)))
    return values, worst


def multiplicities_by_strata(rng, n, strata):
    """
    Multiplicity of each of n units when every stratum (array of unit indices) is redrawn with replacement
    keeping its size, in the given order; empty strata are skipped.
    """
    c = np.zeros(n)
    for stratum in strata:
        if len(stratum):
            c[stratum] = np.bincount(rng.integers(0, len(stratum), len(stratum)), minlength=len(stratum))
    return c


def stratified_multiplicities(rng, io_mask):
    """Multiplicities when the IO users, then the controls, are each redrawn with replacement."""
    return multiplicities_by_strata(rng, len(io_mask), (np.flatnonzero(io_mask), np.flatnonzero(~io_mask)))


def conditional_bootstrap(encoded, io_mask, keys, metric_names, n_boot, rng):
    """(n_boot, len(keys), len(metric_names)) metrics of every ranking on paired, stratified resamples."""
    out = np.empty((n_boot, len(keys), len(metric_names)))
    for b in range(n_boot):
        c = stratified_multiplicities(rng, io_mask)
        for v, key in enumerate(keys):
            group_idx, n_groups = encoded[key]
            out[b, v] = block_metrics(*blocks_from_groups(group_idx, io_mask, n_groups, c), metric_names)
    return out


def percentile_interval(values, level=0.95):
    """(lo, hi) percentile interval of bootstrap values (linear interpolation of the quantiles)."""
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(values, [a, 1.0 - a], method='linear')
    return float(lo), float(hi)


def jackknife_variance(values):
    """
    Delete-a-group jackknife variance (K - 1) / K * sum_g (theta_g - mean)^2 of the K replicate values (axis
    0), each computed without one of K equal-sized random groups.
    """
    values = np.asarray(values, dtype=float)
    K = values.shape[0]
    return (K - 1) / K * np.sum((values - values.mean(axis=0)) ** 2, axis=0)


def crossed_jackknife_interval(estimate, tweet_values, user_values, level=0.95):
    """
    Graph-level interval of a statistic of crossed (tweet x user) data: the variances of the delete-a-group
    jackknives over tweets and over users are added. For a crossed random-effects mean (Y_ij = mu + a_i + b_j +
    e_ij) the tweet jackknife estimates var(a)/R + var(e)/N and the user one var(b)/C + var(e)/N, so the sum
    counts the interaction term twice, which errs on the wide side. Interval: estimate +- t_{K-1} * SE.

    Returns:
        (lo, hi, se)
    """
    se = float(np.sqrt(jackknife_variance(tweet_values) + jackknife_variance(user_values)))
    K = min(len(tweet_values), len(user_values))
    half = student_t.ppf(0.5 + level / 2.0, K - 1) * se
    return estimate - half, estimate + half, se


def contains_zero(lo, hi):
    return bool(np.isfinite(lo) and np.isfinite(hi) and lo <= 0.0 <= hi)


def interval_conclusion(lo, hi):
    """'+' if the interval is above 0, '-' if below, '0' if it contains 0."""
    if contains_zero(lo, hi):
        return '0'
    return '+' if lo > 0 else '-'


def paired_differences(values_a, values_b, metric_names, pp_metrics):
    """
    {metric: a - b} of metric arrays (last axis: metric_names): in percentage points (100 (a - b)) for
    pp_metrics, 'red90' = 100 (1 - FC_a / FC_b) for 'fc90' (users saved at 90% recall), a - b otherwise.
    """
    out = {}
    for i, m in enumerate(metric_names):
        a, b = np.asarray(values_a[..., i]), np.asarray(values_b[..., i])
        if m == 'fc90':
            out['red90'] = 100.0 * (1.0 - a / b)
        elif m in pp_metrics:
            out[m] = 100.0 * (a - b)
        else:
            out[m] = a - b
    return out
