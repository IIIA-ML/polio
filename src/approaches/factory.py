"""
Approaches from the specs of the experiment configs.

Spec of an approach: {"name": ..., "ranking_mode": "Linf", "min_coactions": 1, "need_filtering": bool}; only
"name" is required. Spec of the lexicographic approach: {"name": "lexicographic", "approaches": [name, ...],
"ranking_modes": ..., "min_coactions": ..., "need_filtering": ...}, each of the last three either one value for
every sub-approach or a list with one value per sub-approach (defaults 'Linf', 1 and true).
"""

from .coretweets import CoRetweetsApproach
from .coretweets_idf import CoretweetsIDFApproach
from .coretweets_inv_m import CoretweetsInvMApproach
from .coretweets_weighted_1day import CoretweetsWeighted1DayApproach
from .coretweets_weighted_2days import CoretweetsWeighted2DaysApproach
from .coretweets_weighted_12h import CoretweetsWeighted12hApproach
from .lexicographic import LexicographicApproach

PAIR_APPROACHES = {
    'coretweets': CoRetweetsApproach,
    'coretweets_weighted_12h': CoretweetsWeighted12hApproach,
    'coretweets_weighted_1day': CoretweetsWeighted1DayApproach,
    'coretweets_weighted_2days': CoretweetsWeighted2DaysApproach,
    'coretweets_idf': CoretweetsIDFApproach,
    'coretweets_inv_m': CoretweetsInvMApproach,
}


def _check_min_coactions(value):
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"min_coactions must be a positive integer, got {value!r}")


def _check_need_filtering(value):
    if not isinstance(value, bool):
        raise ValueError(f"need_filtering must be a boolean, got {value!r}")


def _pair_approach(name, window_sec, min_coactions, ranking_mode, need_filtering):
    if name not in PAIR_APPROACHES:
        raise ValueError(f"Unknown approach '{name}'. Valid: {', '.join(PAIR_APPROACHES)}")
    _check_min_coactions(min_coactions)
    if need_filtering is not None:
        _check_need_filtering(need_filtering)
    # The co-retweet count is computed on the full data
    need_filtering = False if name == 'coretweets' else need_filtering
    return PAIR_APPROACHES[name](window_sec, min_coactions, ranking_mode, need_filtering)


def _per_sub_approach(spec, key, default, n):
    """Value of `key` for each of the n sub-approaches of a lexicographic spec."""
    value = spec.get(key, default)
    if not isinstance(value, list):
        return [value] * n
    if len(value) != n:
        raise ValueError(f"{key} length ({len(value)}) must match approaches length ({n})")
    return value


def create_approach(spec, window_sec=60):
    """Approach instance of a spec (dict, see the module docstring); ValueError if the spec is invalid."""
    if not isinstance(spec, dict) or not spec.get('name'):
        raise ValueError(f"An approach spec must be a dict with a 'name', got {spec!r}")
    if spec['name'] != 'lexicographic':
        return _pair_approach(spec['name'], window_sec, spec.get('min_coactions', 1),
                              spec.get('ranking_mode', 'Linf'), spec.get('need_filtering'))

    names = spec.get('approaches')
    if not isinstance(names, list) or len(names) < 2:
        raise ValueError("The lexicographic approach needs 'approaches', a list of at least 2 approaches")
    n = len(names)
    need_filtering = _per_sub_approach(spec, 'need_filtering', True, n)
    for value in need_filtering:
        _check_need_filtering(value)
    subs = [_pair_approach(name, window_sec, min_coactions, mode, need)
            for name, min_coactions, mode, need in zip(names, _per_sub_approach(spec, 'min_coactions', 1, n),
                                                       _per_sub_approach(spec, 'ranking_modes', 'Linf', n),
                                                       need_filtering)]
    return LexicographicApproach(subs, window_sec)
