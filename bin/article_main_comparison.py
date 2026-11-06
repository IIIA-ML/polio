#!/usr/bin/env python3
"""
Comparison of detection approaches, each with its own aggregation, on one or more cohorts: figures, table and
differences with their intervals. The computational cost of the same approaches is measured by
bin/benchmark_runtime.py with the same config.

Usage:
    uv run bin/article_main_comparison.py all     experiments/article_main_comparison_Linf.json
    uv run bin/article_main_comparison.py rank    <config>   rankings, one run_experiments.py run per scenario
    uv run bin/article_main_comparison.py analyze <config>   figures, table1.tex and the differences

Config fields: those of every comparison (src/analysis/comparison.py: name, data_dir, output_dir, window_sec,
min_coactions, need_filtering, datasets, force, approaches), where the comparisons are the first approach minus
each of the others, and
    analysis
        n_boot, seed, ci_level   conditional bootstrap of the evaluated accounts
        jackknife_groups         groups of tweets and of accounts of the graph-level jackknife
        area_baseline            spec of the approach (one of approaches) against which the areas of
                                 figure1_old_zoomed.png are computed
        extra_comparisons        optional [[spec A, spec B], ...]: A minus B is also shown (A and B among
                                 approaches)
        supplementary_approaches optional approach specs compared with the first approach (bootstrap and
                                 jackknife), outside table1.tex and the figures (the global rarity 1/m)
        idf_aggregations         optional aggregations of coretweets_idf (one of approaches): the first
                                 approach against IDF at its best aggregation of each cell (bootstrap only)

Outputs ({output_dir}/{name}/)
    configs/S<i>.json, rankings/S<i>/results/   run_experiments.py configs and stored rankings
    analysis/figure1.png         cumulative IO recall against the users inspected (log scale)
    analysis/figure1_old_zoomed.png  IO users detected against users studied, with the area differences, the
                                 80/90/95% IO recall levels and, in the panels of FIGURE1_OLD_ZOOM, an inset
                                 zooming the curves of that region (the same figure without insets,
                                 figure1_old.png, is commented out in write_figures)
    analysis/figure2.png         differences in AUC, AP and NDCG with both intervals
    analysis/table1.tex          AUC, AP, NDCG, ED, P@100, P2500 and FC@80/90/95% of every approach
    analysis/scores.csv          every metric of every evaluated ranking
    analysis/differences.csv     the differences in AUC, AP and NDCG of figure2.png and of the supplementary
                                 approaches, their intervals and the claim (ED, P@k and FC: point estimates only)
    analysis/idf_best.csv        IDF at its best aggregation and the difference of the first approach with it
    analysis/fc_savings.csv      users saved by each comparison of the figures at recall 0.50 to 0.98 (point
                                 estimates)
    The robustness counts are printed.

Procedure
    Metrics: tie blocks count as a random order of the tied users (src/analysis/metrics.py).
    Conditional bootstrap: rankings fixed; IO and control accounts of the cohort redrawn separately with
        replacement, paired across approaches. Generator of a block: SeedSequence(seed, spawn_key=(scenario
        index, crc32(dataset))). 95% percentile intervals.
    Graph-level jackknife: each of K random groups of tweets and each of K random groups of accounts (stratified
        by label and cohort) is removed in turn, every weight, pair score and user score (with the aggregation
        of the approach) recomputed on the fixed cohort, and the two jackknife variances added; interval =
        estimate +- t_{K-1} SE. Generator: SeedSequence(seed, spawn_key=(scenario index, crc32(dataset), 2)).
    Claim: '+' or '-' if both intervals exclude 0 on the same side, '0' otherwise (hollow marker in figure2.png).
    IDF at its best aggregation: the best of the idf_aggregations in each cell (highest, lowest for FC),
        re-selected inside every bootstrap sample.
    The figures and table1.tex list the approaches in config order with the first one last (drawn on top).

Checks (the script aborts if one fails)
    1. Stored rankings (src/analysis/comparison.py): the approaches of a block rank the same users with the
       same IO users, and a lexicographic ranking is the dense rank of its sub-rankings.
    2. Metric parity: the tie-block AUC / AP / NDCG equal the list-based compute_scores.
    3. Graph identity: with every multiplicity 1 the jackknife path reproduces the cohort, the stored rankings
       (up to float-noise ties, TIE_TOL) and the metrics (IDENTITY_TOL).
"""

import argparse
import sys
import time

import matplotlib
import numpy as np
import pandas as pd

from analysis import graph_jackknife as gj
from analysis.aggregations import aggregation_text
from analysis.comparison import (
    DISPLAY,
    ComparisonError,
    collect,
    display_order,
    experiment_dir,
    load_config,
    make_method,
    method_of,
    rank,
)
from analysis.metrics import (
    RANK_METRICS,
    blocks_from_groups,
    ceil_users,
    compute_scores,
    recall_curve,
    users_to_recall,
)
from analysis.reporting import print_section, results_table_lines, write_lines
from analysis.statistics import (
    block_seed,
    checked_block_metrics,
    conditional_bootstrap,
    crossed_jackknife_interval,
    encode_rankings,
    interval_conclusion,
    paired_differences,
    percentile_interval,
)
from analysis.visualization import (
    plot_cumulative_recall_grid,
    # plot_detection_curves_with_areas,
    plot_detection_curves_with_areas_inset,
    plot_difference_forest,
)
from approaches._coretweet_index import clear_coretweet_index_cache
from data_loader import import_data

ANALYSIS_REQUIRED = ('n_boot', 'seed', 'ci_level', 'jackknife_groups', 'area_baseline')
ANALYSIS_OPTIONAL = ('extra_comparisons', 'supplementary_approaches', 'idf_aggregations')
# Markers of the comparisons, in order; a default comparison takes the colour of its second approach, an extra
# one the next of these greys
FOREST_MARKERS = ['o', 's', 'D', '^', 'v', 'P', 'X', '*']
EXTRA_COLOURS = ['#555555', '#999999', '#222222']

SCORE_METRICS = ['auc', 'ap', 'ndcg', 'ed', 'p100', 'p500', 'fc80', 'fc90', 'fc95']
TABLE1_METRICS = ['auc', 'ap', 'ndcg', 'ed', 'p100', 'p500', 'fc80', 'fc90', 'fc95']
DIFF_METRICS = ['auc', 'ap', 'ndcg', 'fc90']     # resampled metrics (FC@90 for the identity check and idf_best.csv)
FIGURE_METRICS = list(RANK_METRICS)              # differences with intervals; ED, P@k and FC are point estimates
IDENTITY_TOL = 1e-4              # largest metric change allowed from merging float-noise ties
FC_LEVELS = {'fc80': 0.8, 'fc90': 0.9, 'fc95': 0.95}
SAVING_RECALLS = np.linspace(0.5, 0.98, 481)     # recall levels of fc_savings.csv
AGGREGATION_NAMES = {'L1': 'median (p=1)', 'L2': 'mean (p=2)', 'Linf': 'midrange', 'max': 'maximum',
                     'min': 'minimum', 'TM25': 'trimmed mean (25%)', 'MH': 'midhinge', 'Q90': '0.9-quantile',
                     'Linf_droptop': 'midrange, top removed', 'Linf_dropbottom': 'midrange, bottom removed'}
# Labels of the figures that differ from the DISPLAY label of the approach (method id: mathtext label)
FIGURE_LABELS = {'count': 'Baseline co-retweets', 'wT12h': r'$wc^T_{\Delta t}$'}
# Line styles of figure1_old.png that replace those of visualization.APPROACH_STYLES (method id: style), same
# colours as the other figures: a distinct dash pattern per approach and thicker lines
FIGURE1_OLD_STYLES = {'count': {'linestyle': (0, (6, 3)), 'linewidth': 2.0},
                      'idf': {'linestyle': (0, (1, 1.5)), 'linewidth': 2.4},
                      'lexi': {'linestyle': (0, (4, 1.5, 1, 1.5)), 'linewidth': 2.0},
                      'wT12h': {'linewidth': 2.2}}
# Zoomed region of the panels (scenario, dataset) of figure1_old_zoomed.png: any subset of 'x_min', 'x_max',
# 'y_min' and 'y_max' (the missing limits: those of the full panel); the other panels are not zoomed
FIGURE1_OLD_ZOOM = {('S1', 'Honduras'): {'y_min': 1450, 'x_max': 50000},
                    ('S1', 'UAE'): {'y_min': 1150, 'x_max': 40000},
                    ('S2', 'Honduras'): {'y_min': 1350, 'x_min': 1300, 'x_max': 5000}}

FIGURE_FORMAT = dict(hollow_label='Hollow: not robust (one of its intervals contains 0)',
                     interval_labels=('95% bootstrap of the evaluated accounts', '95% graph-level jackknife'),
                     marker_size=6, fontsize=12)


def comparison_id(a, b):
    return f"{a} - {b}"


def difference_label(a, b):
    """Mathtext label of 'a minus b': '$w_T$', 'Count' -> '$w_T-$Count'."""
    left = a[:-1] if a.endswith('$') else a + '$'
    right = b[1:] if b.startswith('$') else '$' + b
    return f"{left}-{right}"


def figure_label(method):
    return FIGURE_LABELS.get(method['id'], method['tex'])


def aggregation_name(mode):
    return AGGREGATION_NAMES.get(mode, f"p={mode[1:]}")


def load_main_config(path):
    """
    The comparison config, with 'comparisons' [(a, b)] (those of the figures), 'area_baseline' (method id),
    'supplementary' (methods), 'supplementary_comparisons', 'idf_variants' ({id: spec} of the other IDF
    aggregations) and 'idf_best_ids' (the ids of every IDF aggregation, in config order).
    """
    config = load_config(path, (ANALYSIS_REQUIRED, ANALYSIS_OPTIONAL))
    methods = config['methods']
    if len(methods) < 2:
        raise ComparisonError("at least two approaches are needed")
    comparisons = [(methods[0]['id'], m['id']) for m in methods[1:]]
    for pair in config['analysis'].get('extra_comparisons', []):
        if len(pair) != 2:
            raise ComparisonError(f"an extra comparison must have two specs, got {pair}")
        a, b = (method_of(config, s, 'extra comparison') for s in pair)
        if a == b or (a, b) in comparisons:
            raise ComparisonError(f"extra comparison {a} - {b} is trivial or repeated")
        comparisons.append((a, b))
    config['comparisons'] = comparisons
    config['area_baseline'] = method_of(config, config['analysis']['area_baseline'], 'area_baseline')

    supplementary = [make_method(config, spec) for spec in config['analysis'].get('supplementary_approaches', [])]
    ids = [m['id'] for m in methods + supplementary]
    if len(set(ids)) != len(ids):
        raise ComparisonError("a supplementary approach repeats an approach")
    config['supplementary'] = supplementary
    config['supplementary_comparisons'] = [(methods[0]['id'], m['id']) for m in supplementary]

    config['idf_variants'], config['idf_best_ids'] = {}, []
    if config['analysis'].get('idf_aggregations'):
        idf = next((m for m in methods if m['spec']['name'] == 'coretweets_idf'), None)
        if idf is None:
            raise ComparisonError("idf_aggregations needs coretweets_idf among the approaches")
        for mode in config['analysis']['idf_aggregations']:
            if mode == idf['spec']['ranking_mode']:
                config['idf_best_ids'].append(idf['id'])
            else:
                config['idf_variants'][f"idf_{mode}"] = {'name': 'coretweets_idf', 'ranking_mode': mode}
                config['idf_best_ids'].append(f"idf_{mode}")
    return config


def extra_rankings(config):
    """{id: spec} of the rankings besides the approaches: supplementary approaches and IDF aggregations."""
    return {**{m['id']: m['spec'] for m in config['supplementary']}, **config['idf_variants']}


# ==========================================================================
# Full-data scores and conditional bootstrap
# ==========================================================================

def evaluate_block(config, scenario_index, scenario, dataset, variants):
    """
    Block description, full-data metrics and conditional-bootstrap replicates of every ranking (ids: the
    approaches, the supplementary approaches, then the IDF aggregations), and the recall curves of the
    approaches.
    """
    tag = f"{scenario} / {dataset}"
    ids = [m['id'] for m in config['methods'] + config['supplementary']] + list(config['idf_variants'])
    io_users = set(next(iter(variants.values()))['io_users'])
    references = {i: compute_scores(variants[i]['suspicious_users'], io_users) for i in ids}
    try:
        universe, encoded, io_mask = encode_rankings({i: variants[i]['suspicious_users'] for i in ids}, io_users)
        values, _ = checked_block_metrics(encoded, io_mask, SCORE_METRICS, references)
    except ValueError as e:
        raise ComparisonError(f"[{tag}]: {e}") from None
    point = {i: dict(zip(SCORE_METRICS, values[i])) for i in ids}
    curves = {m['id']: recall_curve(*blocks_from_groups(encoded[m['id']][0], io_mask, encoded[m['id']][1]))
              for m in config['methods']}
    block = {'scenario': scenario, 'dataset': dataset, 'n': len(universe), 'n_io': int(io_mask.sum())}
    analysis = config['analysis']
    print(f"  [{tag}] {block['n']} users ({block['n_io']} IO); conditional bootstrap "
          f"{len(ids)} rankings x {analysis['n_boot']} samples", flush=True)
    rng = np.random.default_rng(block_seed(analysis['seed'], scenario_index, dataset))
    boot = conditional_bootstrap(encoded, io_mask, ids, DIFF_METRICS, analysis['n_boot'], rng)
    return block, point, curves, boot


def idf_best(config, block, point, boot):
    """
    Rows of idf_best.csv: per metric, IDF at its best aggregation (highest, lowest for FC) and the difference of
    the first approach with it (pp for AUC, AP and NDCG, users for FC), re-selected inside every bootstrap
    sample.
    """
    ids = [m['id'] for m in config['methods'] + config['supplementary']] + list(config['idf_variants'])
    first = config['methods'][0]['id']
    idx = [ids.index(i) for i in config['idf_best_ids']]
    modes = [config['idf_variants'][i]['ranking_mode'] if i in config['idf_variants'] else
             next(m for m in config['methods'] if m['id'] == i)['spec']['ranking_mode'] for i in config['idf_best_ids']]
    rows = []
    for im, metric in enumerate(DIFF_METRICS):
        pick = np.min if metric.startswith('fc') else np.max
        scale = 100.0 if metric in RANK_METRICS else 1.0
        values = np.array([point[i][metric] for i in config['idf_best_ids']])
        value = float(pick(values))
        lo, hi = percentile_interval(scale * (boot[:, ids.index(first), im] - pick(boot[:, idx, im], axis=1)),
                                     config['analysis']['ci_level'])
        rows.append({'scenario': block['scenario'], 'dataset': block['dataset'], 'metric': metric,
                     'value': ceil_users(value) if metric.startswith('fc') else value, 'value_unrounded': value,
                     'modes': ' / '.join(aggregation_name(mode) for mode, v in zip(modes, values)
                                         if np.isclose(v, value, rtol=0, atol=1e-12)),
                     'estimate': scale * (point[first][metric] - value), 'lo': lo, 'hi': hi})
    return rows


# ==========================================================================
# Graph-level jackknife
# ==========================================================================

def identity_check(g, cohort, methods, variants, point, tag):
    """Check 3: the jackknife path with unit multiplicities reproduces the stored rankings."""
    k, c = gj.unit_multiplicities(g)
    rankers = [m['ranker'] for m in methods]
    ids = g.users[cohort.users]
    components = list(dict.fromkeys(comp for r in rankers for comp in r))
    alive, by_component = gj.component_scores(
        cohort, gj.pair_scores(g, k, c, gj.ranker_weights(rankers), cohort.units), components)
    unit, dead = gj.replicate_metrics(g, cohort, k, c, rankers, DIFF_METRICS)
    if not alive.all() or dead['dead'] != 0:
        raise ComparisonError(f"[{tag}]: cohort users without a pair with unit multiplicities")
    merged = {}
    for i, method in enumerate(methods):
        groups = variants[method['id']]['suspicious_users']
        group_of = {u: j for j, group in enumerate(groups) for u in group}
        if len(ids) != len(group_of) or any(u not in group_of for u in ids.tolist()):
            raise ComparisonError(f"[{tag}] {method['id']}: the graph cohort differs from the stored cohort "
                                  f"({len(ids)} vs {len(group_of)} users)")
        if len(method['ranker']) == 1:          # lexicographic rankings: through their metrics only
            stored = np.array([group_of[u] for u in ids.tolist()])
            order = np.argsort(stored, kind='stable')
            s, gi = by_component[method['ranker'][0]][order], stored[order]
            starts = np.flatnonzero(np.concatenate([[True], gi[1:] != gi[:-1]]))
            hi, lo = np.maximum.reduceat(s, starts), np.minimum.reduceat(s, starts)
            if np.any(hi - lo > gj.TIE_TOL) or np.any(lo[:-1] < hi[1:] - gj.TIE_TOL):
                raise ComparisonError(f"[{tag}] {method['id']}: the graph scores do not reproduce the stored ranking")
            merged[method['id']] = int(np.count_nonzero(lo[:-1] - hi[1:] <= gj.TIE_TOL))
        ref = np.array([point[method['id']][m] for m in DIFF_METRICS])
        diff = np.abs(unit[i] - ref)
        fc = DIFF_METRICS.index('fc90')
        diff[fc] /= max(ref[fc], 1.0)
        if np.max(diff) > IDENTITY_TOL:
            raise ComparisonError(f"[{tag}] {method['id']}: the graph metrics differ from the stored ones by "
                                  f"{np.max(diff):.2e} > {IDENTITY_TOL:g}")
    print(f"    [{tag}] identity check passed (float-noise ties merged: {merged})", flush=True)


def jackknife(config, blocks, cells, points):
    """
    {(scenario, dataset): {'tweets', 'users'}} replicate metrics (K x methods x DIFF_METRICS), with the
    approaches and the supplementary approaches.
    """
    analysis, labels = config['analysis'], [s for s, _ in config['scenarios']]
    min_co = dict(config['scenarios'])
    methods = config['methods'] + config['supplementary']
    rankers = [m['ranker'] for m in methods]
    K = analysis['jackknife_groups']
    out = {}
    for dataset in sorted({b['dataset'] for b in blocks}):
        RTs, io_users = import_data(f"{config['data_dir']}/{dataset}/Processed/")
        start = time.time()
        g = gj.SyncGraph(RTs, io_users, config['window_sec'])
        clear_coretweet_index_cache()
        del RTs
        print(f"  [{dataset}] graph: {g.n_users} users, {g.n_tweets} tweets, {g.n_units} synchronized "
              f"units, {g.n_pairs} pairs ({time.time() - start:.1f}s)", flush=True)
        for scenario in (b['scenario'] for b in blocks if b['dataset'] == dataset):
            tag = f"{scenario} / {dataset}"
            cohort = gj.FixedCohort(g, min_co[scenario])
            identity_check(g, cohort, methods, cells[(scenario, dataset)], points[(scenario, dataset)], tag)
            start = time.time()
            rng = np.random.default_rng(block_seed(analysis['seed'], labels.index(scenario), dataset, 2))
            groups = gj.jackknife_groups(rng, g, cohort, K)
            out[(scenario, dataset)] = gj.grouped_jackknife(g, cohort, rankers, DIFF_METRICS, groups, K)
            print(f"    [{tag}] jackknife: {K} tweet groups + {K} account groups ({time.time() - start:.0f}s)",
                  flush=True)
        del g
    return out


# ==========================================================================
# Differences
# ==========================================================================

def differences(config, blocks, points, boots, jk):
    """
    One row per comparison (of the figures, then supplementary), block and metric of FIGURE_METRICS, with both
    intervals and the claim.
    """
    ids = [m['id'] for m in config['methods'] + config['supplementary']]
    level = config['analysis']['ci_level']
    rows = []
    for block in blocks:
        cell = (block['scenario'], block['dataset'])
        point = {i: np.array([points[cell][i][m] for m in DIFF_METRICS]) for i in ids}
        for a, b in config['comparisons'] + config['supplementary_comparisons']:
            ia, ib = ids.index(a), ids.index(b)
            estimate = paired_differences(point[a], point[b], DIFF_METRICS, RANK_METRICS)
            boot = paired_differences(boots[cell][:, ia], boots[cell][:, ib], DIFF_METRICS, RANK_METRICS)
            by_tweets = paired_differences(jk[cell]['tweets'][:, ia], jk[cell]['tweets'][:, ib], DIFF_METRICS,
                                           RANK_METRICS)
            by_users = paired_differences(jk[cell]['users'][:, ia], jk[cell]['users'][:, ib], DIFF_METRICS,
                                          RANK_METRICS)
            for metric in FIGURE_METRICS:
                lo, hi = percentile_interval(boot[metric], level)
                jlo, jhi, se = crossed_jackknife_interval(float(estimate[metric]), by_tweets[metric],
                                                          by_users[metric], level)
                conclusion, jk_conclusion = interval_conclusion(lo, hi), interval_conclusion(jlo, jhi)
                rows.append({'comparison': comparison_id(a, b), 'scenario': cell[0], 'dataset': cell[1],
                             'metric': metric, 'estimate': float(estimate[metric]), 'lo': lo, 'hi': hi,
                             'jk_lo': jlo, 'jk_hi': jhi, 'jk_se': se,
                             'claim': conclusion if conclusion == jk_conclusion else '0'})
    return pd.DataFrame(rows)


# ==========================================================================
# Outputs
# ==========================================================================

def forest_styles(config):
    by_id = {m['id']: m for m in config['methods']}
    reference = config['methods'][0]['id']
    styles, extra = {}, 0
    for n, (a, b) in enumerate(config['comparisons']):
        if a == reference and n < len(config['methods']) - 1:
            colour = by_id[b]['style']['color']
        else:
            colour, extra = EXTRA_COLOURS[extra % len(EXTRA_COLOURS)], extra + 1
        styles[comparison_id(a, b)] = {'label': difference_label(figure_label(by_id[a]), figure_label(by_id[b])),
                                       'color': colour, 'marker': FOREST_MARKERS[n % len(FOREST_MARKERS)]}
    return styles


def write_figures(config, blocks, points, curves, diffs, out_dir):
    styles = {m['id']: {**m['style'], 'label': figure_label(m)} for m in display_order(config)}
    datasets = sorted({b['dataset'] for b in blocks})
    by_cell = {(b['scenario'], b['dataset']): b for b in blocks}
    panels = [[{'title': f"{s} · {d}", 'n': by_cell[(s, d)]['n'], 'n_io': by_cell[(s, d)]['n_io'],
                'curves': {m: curves[(s, d)][m] for m in styles},
                'fc': {m: points[(s, d)][m]['fc90'] for m in styles}} for d in datasets]
              for s, _ in config['scenarios']]
    plot_cumulative_recall_grid(panels, styles, out_dir / 'figure1.png')
    old_styles = {m: {**style, **FIGURE1_OLD_STYLES.get(m, {})} for m, style in styles.items()}
    # Full figure without zoom (uncomment, with its import, to generate figure1_old.png as well)
    # plot_detection_curves_with_areas(panels, old_styles, config['area_baseline'], out_dir / 'figure1_old.png')
    # plot_detection_curves_with_areas_zoomed (same arguments) cuts the y axis instead of adding an inset
    plot_detection_curves_with_areas_inset(panels, old_styles, config['area_baseline'],
                                           out_dir / 'figure1_old_zoomed.png',
                                           {f"{s} · {d}": zoom for (s, d), zoom in FIGURE1_OLD_ZOOM.items()})
    forest = diffs.rename(columns={'jk_lo': 'lo2', 'jk_hi': 'hi2'})
    forest['dagger'] = forest['claim'] == '0'              # hollow marker: not robust
    block_ids = [(b['scenario'], b['dataset']) for b in blocks]
    row_groups = [[(s, d) for s, _ in config['scenarios']] for d in datasets]   # one row per campaign
    comparisons = [comparison_id(a, b) for a, b in config['comparisons']]
    kwargs = dict(row_groups=row_groups, **FIGURE_FORMAT)
    plot_difference_forest(forest, block_ids, comparisons, forest_styles(config), list(RANK_METRICS),
                           out_dir / 'figure2.png', metric_labels={'auc': 'AUC', 'ap': 'AP', 'ndcg': 'NDCG'},
                           xlabel='Difference (pp)', panel_width=2.6, legend_ncol=3, **kwargs)


def table1_caption(config):
    descriptions = []
    for m in config['methods']:
        if m['spec']['name'] == 'lexicographic':
            subs = [DISPLAY[name][1] for name in m['spec']['approaches']]
            descriptions.append(f"Lexicographic: {subs[0]}, ties broken by {', then '.join(subs[1:])}.")
        else:
            descriptions.append(m['description'])
    aggregations = {m['id']: [aggregation_text(mode) for _, mode in m['ranker']] for m in config['methods']}
    distinct = {a for values in aggregations.values() for a in values}
    if len(distinct) == 1:
        aggregation = f"user scores use the {distinct.pop()} aggregation"
    else:
        aggregation = "user scores use the aggregation " + ', '.join(
            f"{m['tex']} {'/'.join(aggregations[m['id']])}" for m in display_order(config))
    return ("Ranking quality of the compared approaches. " + ' '.join(descriptions[1:] + descriptions[:1])
            + f" All rows of a block rank the same users; {aggregation}. ED: expected IO users before the "
            "first control; P@$k$: expected precision among the first $k$ users; FC@$r$: smallest number of "
            "users, in decreasing score order, at which the expected recall of the IO users of the block "
            "reaches $r$. Ties count as a random order of the tied users. Bold marks the best point estimate of each "
            "column and block only.")


def score_table(blocks, points):
    """scores.csv: every metric of every ranking (FC rounded up to whole users)."""
    return pd.DataFrame([{'method': i, 'scenario': b['scenario'], 'dataset': b['dataset'], 'metric': m,
                          'value': ceil_users(v) if m.startswith('fc') else v}
                         for b in blocks for i, p in points[(b['scenario'], b['dataset'])].items()
                         for m, v in p.items()])


def write_table1(config, blocks, scores, out_dir):
    write_lines(out_dir / 'table1.tex', results_table_lines(
        scores, blocks, TABLE1_METRICS, [(m['id'], m['tex']) for m in display_order(config)],
        table1_caption(config), 'tab:main_results'))


def fc_savings(config, points, curves):
    """
    Users saved, 100 (1 - FC_a(r) / FC_b(r)), by each comparison of the figures at every recall r of
    SAVING_RECALLS (point estimates). The interpolated curves must reproduce the FC of the scores.
    """
    rows = []
    for cell, cell_curves in curves.items():
        for method, (X, Y) in cell_curves.items():
            for metric, r in FC_LEVELS.items():
                if ceil_users(users_to_recall(X, Y, [r])[0]) != ceil_users(points[cell][method][metric]):
                    raise ComparisonError(f"{cell} {method}: {metric} of the recall curve differs from the score")
        users = {m: users_to_recall(*cell_curves[m], SAVING_RECALLS) for m in cell_curves}
        for a, b in config['comparisons']:
            saved = 100.0 * (1.0 - users[a] / users[b])
            rows += [{'comparison': comparison_id(a, b), 'scenario': cell[0], 'dataset': cell[1], 'recall': r,
                      'saved': s} for r, s in zip(SAVING_RECALLS, saved)]
    return pd.DataFrame(rows)


def print_claims(diffs):
    print_section("Robust differences (both intervals exclude 0 on the same side)")
    for comparison, sub in diffs.groupby('comparison', sort=False):
        for metric, cells in sub.groupby('metric', sort=False):
            claims = ''.join(cells['claim'])
            print(f"  {comparison:16s} {metric:5s} {claims}  (+ {claims.count('+')}, - {claims.count('-')}, "
                  f"not robust {claims.count('0')} of {len(claims)}; blocks "
                  f"{', '.join(cells['scenario'] + ' ' + cells['dataset'])})")


def print_robustness(config, diffs):
    """Robustness counts of the differences of the figures."""
    print_section("Robustness of the differences")
    figure = diffs[diffs['comparison'].isin(comparison_id(a, b) for a, b in config['comparisons'])]
    excludes = (figure['lo'] > 0) | (figure['hi'] < 0)
    lost = excludes & (figure['claim'] == '0')
    print(f"Differences in figure2.png: {len(figure)}; bootstrap interval excludes 0: {int(excludes.sum())}; "
          f"of these not robust (jackknife interval contains 0): {int(lost.sum())}")


def analyze(config):
    out_dir = experiment_dir(config) / 'analysis'
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [s for s, _ in config['scenarios']]
    cells = collect(config, extra_rankings(config))
    order = sorted(cells, key=lambda c: (labels.index(c[0]), c[1]))

    print_section(f"Full-data scores and conditional bootstrap ({config['analysis']['n_boot']} samples)")
    blocks, points, curves, boots, best = [], {}, {}, {}, []
    for scenario, dataset in order:
        block, point, curve, boot = evaluate_block(config, labels.index(scenario), scenario, dataset,
                                                    cells[(scenario, dataset)])
        blocks.append(block)
        points[(scenario, dataset)], curves[(scenario, dataset)], boots[(scenario, dataset)] = point, curve, boot
        if config['idf_best_ids']:
            best += idf_best(config, block, point, boot)

    print_section(f"Graph-level jackknife ({config['analysis']['jackknife_groups']} groups per factor)")
    jk = jackknife(config, blocks, cells, points)
    diffs = differences(config, blocks, points, boots, jk)

    print_section("Outputs")
    scores = score_table(blocks, points)
    tables = {'scores.csv': scores, 'differences.csv': diffs, 'fc_savings.csv': fc_savings(config, points, curves)}
    if best:
        tables['idf_best.csv'] = pd.DataFrame(best)
    for name, table in tables.items():
        table.to_csv(out_dir / name, index=False, float_format='%.6f')
        print(f"Saved: {out_dir / name}")
    write_table1(config, blocks, scores, out_dir)
    write_figures(config, blocks, points, curves, diffs, out_dir)
    print_claims(diffs)
    print_robustness(config, diffs)
    if best:
        print_section("IDF at its best aggregation")
        print(pd.DataFrame(best).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def main():
    matplotlib.use('Agg')
    parser = argparse.ArgumentParser(description="Comparison of detection approaches: figures, table and differences")
    parser.add_argument('step', choices=['rank', 'analyze', 'all'])
    parser.add_argument('config', help='Experiment JSON config')
    args = parser.parse_args()
    try:
        config = load_main_config(args.config)
        if args.step in ('rank', 'all'):
            rank(config, extra_rankings(config).values())
        if args.step in ('analyze', 'all'):
            analyze(config)
    except ComparisonError as e:
        sys.exit(f"ERROR: {e}")


if __name__ == '__main__':
    main()
