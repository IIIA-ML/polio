"""
Configs and stored rankings of the experiments run on several scenarios (bin/article_main_comparison.py,
bin/pacheco_comparison.py, bin/benchmark_runtime.py, bin/preliminar_analysis_bootstrap.py and
bin/benchmark_dt_sensitivity.py).

Config fields shared by these experiments
    name, data_dir, output_dir, window_sec, datasets, force   as in run_experiments.py
    need_filtering     must be false: the weights use N, n and m of the full dataset
    min_coactions      list, one scenario per value (S1, S2, ... in this order): users with at least that many
                       co-retweets with the same partner, only such pairs scored
    approaches         approach specs, without min_coactions (set by the scenarios)
    analysis           settings of the experiment, checked by its script
A script may add its own sections (e.g. 'pacheco').

In the comparisons (load_config), every approach also has its own name and its aggregation ('ranking_mode', or
'ranking_modes' for the lexicographic approach; there is no default).

Rankings: the 'rank' step writes one run_experiments.py config per scenario to {output_dir}/{name}/configs/S<i>.json
and stores the rankings in {output_dir}/{name}/rankings/S<i>/results/. An experiment over several windows (Delta t)
ranks every window apart, under dt<window>_S<i> instead of S<i>.
"""

import copy
import json
from pathlib import Path

from approaches import create_approach
from approaches.base import get_ranks

from .graph_jackknife import ranker_from_spec
from .io_utils import apply_approach_defaults, load_stored_results, run_experiments, write_json
from .visualization import approach_style

CONFIG_REQUIRED = ('name', 'data_dir', 'output_dir', 'window_sec', 'min_coactions', 'need_filtering',
                   'approaches', 'datasets', 'analysis')
CONFIG_OPTIONAL = ('force',)

# Display of every approach: id, label (LaTeX for tables, mathtext for figures) and its description in table
# captions; colour and line style come from visualization.APPROACH_STYLES
DISPLAY = {
    'coretweets': ('count', 'Count', 'Count: number of synchronized co-retweets of the pair.'),
    'coretweets_idf': ('idf', 'IDF',
                       'IDF: synchronized co-retweets weighted by $\\ln(1+N/n_\\iota)$ (IDF as in Silva et al.).'),
    'coretweets_inv_m': ('inv_m', '$1/m$', '$1/m$: synchronized co-retweets weighted by $1/m(\\iota)$.'),
    'coretweets_weighted_12h': ('wT12h', '$w_T$',
                                '$w_T$: synchronized co-retweets weighted by '
                                '$(m_{T_k}(\\iota)+m_{T_{k\'}}(\\iota))/(2m(\\iota))$, $T=12$h.'),
    'lexicographic': ('lexi', 'Lexicographic', None),
}


class ComparisonError(Exception):
    """An invalid config, a missing ranking or a failed check of the stored rankings."""


def _check_keys(block, required, optional, what):
    missing = [k for k in required if k not in block]
    unknown = set(block) - set(required) - set(optional)
    if missing or unknown:
        raise ComparisonError(f"{what} misses {missing} or has unknown keys {sorted(unknown)}")


def _check_scenario_spec(spec):
    if 'min_coactions' in spec:
        raise ComparisonError(f"{spec}: min_coactions is set by the scenarios, not by the approach")


def load_scenario_config(path, analysis=None, sections=()):
    """
    The validated config, with the derived 'scenarios' [(label, min_coactions)].

    Args:
        analysis: (required keys, optional keys) of the 'analysis' block; None leaves it unchecked
        sections: further top-level sections of the experiment (e.g. 'pacheco'), all required
    """
    try:
        config = json.loads(Path(path).read_text(encoding='utf-8'))
    except FileNotFoundError:
        raise ComparisonError(f"config not found: {path}") from None
    _check_keys(config, CONFIG_REQUIRED + tuple(sections), CONFIG_OPTIONAL, "the config")
    if analysis is not None:
        _check_keys(config['analysis'], *analysis, "'analysis'")
    if config['need_filtering']:
        raise ComparisonError("need_filtering must be false (N, n and m on the full dataset)")
    min_co = config['min_coactions']
    if (not isinstance(min_co, list) or not min_co or len(set(min_co)) != len(min_co)
            or not all(isinstance(m, int) and m >= 1 for m in min_co)):
        raise ComparisonError("min_coactions must be a list of distinct positive integers (one scenario each)")
    config['scenarios'] = [(f"S{i + 1}", m) for i, m in enumerate(min_co)]
    if not config['approaches']:
        raise ComparisonError("no approaches")
    for spec in config['approaches']:
        _check_scenario_spec(spec)
    return config


def load_config(path, analysis=None, sections=()):
    """
    The validated config of a comparison (load_scenario_config), with the derived 'methods' (one dict per
    approach: id, spec, tex, ranker, style, description) and 'method_keys' {storage key: id}.
    """
    config = load_scenario_config(path, analysis, sections)
    methods = []
    for spec in config['approaches']:
        method = make_method(config, spec)
        if any(m['id'] == method['id'] for m in methods):
            raise ComparisonError(f"{spec['name']} appears twice; every approach must have its own name")
        methods.append(method)
    config['methods'] = methods
    config['method_keys'] = {m['key']: m['id'] for m in methods}
    return config


def make_method(config, spec):
    """Method of an approach spec: id, spec, storage key, tex, ranker, style and description (DISPLAY)."""
    _check_scenario_spec(spec)
    aggregation_key = 'ranking_modes' if spec.get('name') == 'lexicographic' else 'ranking_mode'
    if aggregation_key not in spec:
        raise ComparisonError(f"{spec}: '{aggregation_key}' is required (no default aggregation)")
    if spec['name'] not in DISPLAY:
        raise ComparisonError(f"no display for {spec['name']} (known: {sorted(DISPLAY)})")
    ident, tex, description = DISPLAY[spec['name']]
    try:
        key = create_approach(spec, config['window_sec']).get_full_approach_key()
        ranker = ranker_from_spec(spec)
    except ValueError as e:
        raise ComparisonError(f"{spec}: {e}") from None
    return {'id': ident, 'spec': spec, 'key': key, 'tex': tex, 'ranker': ranker,
            'style': {'label': tex, **approach_style(spec['name'])}, 'description': description}


def method_of(config, spec, what):
    """Id of the approach of the config that a spec designates."""
    try:
        return config['method_keys'][create_approach(spec, config['window_sec']).get_full_approach_key()]
    except (KeyError, ValueError, TypeError):
        raise ComparisonError(f"{what} {spec} is not one of the approaches") from None


def experiment_dir(config):
    return Path(config['output_dir']) / config['name']


def scenario_specs(config, min_coactions, specs=None):
    """Approach specs (default: those of the config) of one scenario, completed as run_experiments.py does."""
    scenario = {'min_coactions': min_coactions, 'need_filtering': config['need_filtering'],
                'approaches': copy.deepcopy(config['approaches'] if specs is None else specs)}
    return apply_approach_defaults(scenario)['approaches']


def run_name(label, window_sec=None):
    """Name of the rankings of a scenario: its label, or dt<window>_<label> for a given window (sweeps of Delta t)."""
    return label if window_sec is None else f"dt{window_sec}_{label}"


def results_dir(config, label, window_sec=None):
    return experiment_dir(config) / 'rankings' / run_name(label, window_sec) / 'results'


def variant_specs(config, names=None):
    """
    {(approach name, ranking mode): spec} of the approaches of the config (only those in `names`, if given), in
    config order: the ids of experiments that evaluate several aggregations of the same approach.
    """
    specs = [spec for spec in config['approaches'] if names is None or spec['name'] in names]
    variants = {(spec['name'], create_approach(spec, config['window_sec']).ranking_mode): spec for spec in specs}
    if len(variants) != len(specs):
        raise ComparisonError("an (approach, aggregation) appears twice in the config")
    return variants


def display_order(config):
    """Approaches in config order with the first one last (drawn on top, last row of the tables)."""
    methods = config['methods']
    return methods[1:] + methods[:1]


def rank(config, extra_specs=(), window_sec=None):
    """
    Rankings of the approaches and extra_specs in every scenario with run_experiments.py (stored reused), with
    the window of the config or, if given, window_sec (stored apart, see run_name).
    """
    for label, min_co in config['scenarios']:
        name = run_name(label, window_sec)
        path = experiment_dir(config) / 'configs' / f"{name}.json"
        write_json(path, {'name': name, 'data_dir': config['data_dir'],
                          'output_dir': str(experiment_dir(config) / 'rankings'),
                          'window_sec': config['window_sec'] if window_sec is None else window_sec,
                          'min_coactions': min_co,
                          'need_filtering': config['need_filtering'],
                          'approaches': config['approaches'] + list(extra_specs),
                          'datasets': config['datasets'], 'force': config.get('force', False)})
        run_experiments(path)


def _load_result(directory, approach, dataset, tag):
    key = approach.get_full_approach_key()
    result = load_stored_results(directory, key).get(dataset)
    if result is None:
        raise ComparisonError(f"[{tag}] no stored ranking {key} in {directory}. Run the 'rank' step first.")
    return result


def _check_lexicographic(approach, groups, directory, dataset, tag):
    """A lexicographic ranking must be the dense rank of the tuple of its sub-approach ranks."""
    ranks = [get_ranks(_load_result(directory, sub, dataset, tag)['suspicious_users'])
             for sub in approach.approaches]
    by_key = {}
    for u in ranks[0]:
        by_key.setdefault(tuple(r[u] for r in ranks), []).append(u)
    if [sorted(by_key[k]) for k in sorted(by_key)] != [sorted(g) for g in groups]:
        raise ComparisonError(f"[{tag}] the lexicographic ranking is not the dense rank of its sub-rankings")


def collect(config, extra=None):
    """collect_rankings of the methods and of the extra rankings ({id: spec})."""
    return collect_rankings(config, {**{m['id']: m['spec'] for m in config['methods']}, **(extra or {})})


def collect_rankings(config, specs, window_sec=None):
    """
    {(scenario, dataset): {id: stored result}} of the rankings {id: spec} in every scenario (window: as in
    rank), cells in the order of the scenarios and of config['datasets']. Checks that the rankings of a block
    share their IO users and rank the same users, and that a lexicographic ranking is the dense rank of its
    sub-rankings.
    """
    window = config['window_sec'] if window_sec is None else window_sec
    cells = {}
    for label, min_co in config['scenarios']:
        directory = results_dir(config, label, window_sec)
        for ident, spec in zip(specs, scenario_specs(config, min_co, list(specs.values()))):
            approach = create_approach(spec, window)
            for dataset in config['datasets']:
                tag = f"{run_name(label, window_sec)} / {dataset} / {ident}"
                result = _load_result(directory, approach, dataset, tag)
                if spec['name'] == 'lexicographic':
                    _check_lexicographic(approach, result['suspicious_users'], directory, dataset, tag)
                cells.setdefault((label, dataset), {})[ident] = result
    for cell, variants in cells.items():
        if len({frozenset(r['io_users']) for r in variants.values()}) != 1:
            raise ComparisonError(f"{cell}: the approaches do not share the same IO users")
        if len({frozenset(u for g in r['suspicious_users'] for u in g) for r in variants.values()}) != 1:
            raise ComparisonError(f"{cell}: the approaches do not rank the same users")
    return cells
