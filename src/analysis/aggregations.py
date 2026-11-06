"""Names of the aggregations (ranking modes) of the pair scores of a user: 'L<p>' p-norm centres and others."""

import math

AGGREGATION_TEXT = {'Linf': 'midrange', 'max': 'maximum', 'min': 'minimum'}


def p_value(mode):
    """p of a p-norm mode ('L3' -> 3, 'Linf' -> inf); NaN for any other aggregation."""
    if mode == 'Linf':
        return math.inf
    if mode.startswith('L') and mode[1:].isdigit():
        return float(mode[1:])
    return math.nan


def is_pnorm(mode):
    return not math.isnan(p_value(mode))


def p_label(mode):
    """The value of p as written in the text: 'L3' -> '3', 'Linf' -> '∞'."""
    return '∞' if mode == 'Linf' else mode[1:]


def p_tex(mode):
    """The value of p in LaTeX math: 'L3' -> '3', 'Linf' -> '\\infty'."""
    return '\\infty' if mode == 'Linf' else mode[1:]


def aggregation_text(mode):
    """Name of an aggregation in a caption: 'midrange', 'maximum', 'minimum' or '$p=<p>$'."""
    return AGGREGATION_TEXT.get(mode) or f"$p={mode[1:]}$"
