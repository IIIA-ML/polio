"""
Central tendency of a list of scores for a p-norm (the centre used by the LX ranking modes).

The Lp centre of scores s_1..s_n is
    c_p = argmin_c  f(c),   f(c) = sum_i |s_i - c|^p
(p=1: median, p=2: mean, p=inf: midrange). For p > 1, f is strictly convex, so c_p is
unique, lies in [min s, max s], and is the only root of
    g(c) = sum_i sign(s_i - c) |s_i - c|^(p-1)      (f'(c) = -p g(c)),
which is strictly decreasing, with g(min s) > 0 > g(max s) when the scores are not all
equal. The centre is found as that root with Brent's method, to machine precision.

Numerics:
    - The scores are mapped to [0, 1] (t = (s - min) / (max - min)) before solving, so
      |t - c|^(p-1) cannot overflow or underflow for any score scale or large p; c_p is
      equivariant under this affine map.
    - The scores are sorted first, so the result depends only on the multiset of scores,
      not on their order: users with the same scores always tie exactly.
"""

from collections.abc import Sequence

import numpy as np
from scipy.optimize import brentq

_RTOL = 4 * np.finfo(np.float64).eps   # smallest relative tolerance brentq accepts
_MAXITER = 200


def lp_centre(scores: Sequence[float], p: float) -> float:
    """
    Lp centre (argmin_c sum |s_i - c|^p) of the scores, for p > 1.

    Args:
        scores: Non-empty sequence of finite scores
        p: Norm exponent, p > 1

    Returns:
        The centre, in [min(scores), max(scores)]
    """
    if not p > 1:
        raise ValueError(f"lp_centre needs p > 1, got {p}")
    values = np.sort(np.asarray(scores, dtype=np.float64))
    if values.size == 0:
        raise ValueError("lp_centre needs at least one score")
    lo, hi = values[0], values[-1]
    if lo == hi:
        return float(lo)

    span = hi - lo
    t = (values - lo) / span
    t[0], t[-1] = 0.0, 1.0      # exact endpoints, so the root is bracketed by [0, 1]
    q = p - 1

    def g(c):
        d = t - c
        return float(np.dot(np.sign(d), np.abs(d) ** q))

    root = brentq(g, 0.0, 1.0, xtol=1e-300, rtol=_RTOL, maxiter=_MAXITER)
    return float(min(max(lo + span * root, lo), hi))
