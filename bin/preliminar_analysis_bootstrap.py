#!/usr/bin/env python3
"""
Choice of the period T and of the aggregation of the weighted approach: figures and tables.

Usage:
    uv run bin/preliminar_analysis_bootstrap.py all     experiments/article_preliminar_T_and_p.json
    uv run bin/preliminar_analysis_bootstrap.py rank    <config>   rankings, one run_experiments.py run per scenario
    uv run bin/preliminar_analysis_bootstrap.py analyze <config>   figures, tables and summary numbers

Config fields: those of every experiment run on several scenarios (src/analysis/comparison.py: name, data_dir,
output_dir, window_sec, min_coactions, need_filtering, datasets, force, approaches), where min_coactions [1, 2]
gives Scenario 1 (S1, development scenario of the selection of T) and Scenario 2 (S2), and
    analysis   optional settings n_boot, seed, ci_level, tol_pp, plateau_min_p, oracle_scope (defaults in
               ANALYSIS_DEFAULTS); command-line options override them

Outputs ({output_dir}/{name}/)
    configs/S<i>.json, rankings/S<i>/results/   run_experiments.py configs and stored rankings
In analysis/ (or --out-dir):
    pnorm_plateau.png       score vs p (p >= 1), baseline and every T
    pnorm_plateau_alternative.png  same, with the alternative aggregations on the x-axis
    pnorm_gains.png         gains vs p with 95% bootstrap bands
    table_diff_same_{R}_{S}.png  same-aggregation gain per (dataset, T, aggregation) and metric
    table_diff_best_{R}_{S}.png  oracle gain per (dataset, T, aggregation) and metric
                                 S = scenario; R = 'pnorms' (p-norm rows) or 'alt' (min, trimmed
                                 mean, midhinge, 0.9-quantile, max, midrange top/bottom removed,
                                 strength and degree)
    table_oracle_baseline.png    aggregation attaining the oracle baseline per (scenario, dataset)
                                 and metric
    table_loco.tex          selection of T on each campaign of Scenario 1
    table_aggregation.tex   alternative aggregations with the focus T
Summary numbers are printed to stdout (no file is written).

Notation
    s_b(a), s_T(a)   score of the baseline / weighted approach with period T under aggregation a
    cell             (scenario, dataset, metric)
    oracle baseline  max_a s_b(a) per cell, over oracle_scope: 'all' (default) = every evaluated
                     aggregation, 'pnorms' = p-norms only
    plateau          p-norms with p >= plateau_min_p (default 3), p = inf included
    same-norm gain   100 * (s_T(a) - s_b(a))           [pp]
    oracle gain      100 * (s_T(a) - oracle baseline)  [pp]
    worst gain       minimum of a gain over the plateau

Selection of T (table_loco.tex): on Scenario 1, for each campaign and period T, the worst oracle gain
for AUC, AP and NDCG. T is retained if all three are > tol_pp, and the retained T with the largest
mean is selected.

Scores: computed on every user ranked by the evaluated variant (all variants of a cell rank the same users).

Evaluation bootstrap (n_boot samples, one generator seeded with seed): rankings are computed once
and kept fixed. In each sample, users are resampled with replacement within IO and within control
accounts, and every variant of a (scenario, dataset) is scored on the same sample (paired). The
cells are drawn in the order of the scenarios, datasets sorted by name. Every derived quantity
(gains, oracle baseline, worst gain, selection of T, differences between aggregations) is
recomputed inside each sample, and percentile intervals at ci_level are reported. A dagger
marks every value whose interval contains zero.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgb
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker
from matplotlib.patches import Patch, Rectangle
from scipy.stats import kendalltau

from analysis.aggregations import is_pnorm, p_label, p_value
from analysis.comparison import (
    ComparisonError,
    collect_rankings,
    experiment_dir,
    load_scenario_config,
    rank,
    results_dir,
    variant_specs,
)
from analysis.metrics import RANK_METRICS, compute_scores
from analysis.reporting import print_section, table_lines, tex_escape, write_lines
from analysis.statistics import checked_block_metrics, conditional_bootstrap, contains_zero, encode_rankings
from analysis.visualization import APPROACH_STYLES as APPROACH_COLOURS
from analysis.visualization import save_figure

# ==========================================================================
# Constants
# ==========================================================================

METRICS = list(RANK_METRICS)
METRIC_LABELS = {'auc': 'AUC', 'ap': 'AP', 'ndcg': 'NDCG'}
KEYS = ['scenario', 'dataset', 'metric']

BASELINE = 'coretweets'
FOCUS = 'coretweets_weighted_12h'
OTHERS = ['coretweets_weighted_1day', 'coretweets_weighted_2days']
GAINS_APPROACHES = ['coretweets_weighted_12h', 'coretweets_weighted_1day']   # curves of pnorm_gains.png

# Rows of table_aggregation.tex, in groups separated by \addlinespace
AGG_GROUPS = [
    [('min', 'Minimum'), ('L1', 'Median ($p=1$)'), ('TM25', 'Trimmed mean (25\\%)'), ('MH', 'Midhinge')],
    [('L2', 'Mean ($p=2$)'), ('Q90', '0.9-quantile'), ('max', 'Maximum')],
    [('Linf', 'Midrange (adopted)'), ('Linf_droptop', 'Midrange, top removed'),
     ('Linf_dropbottom', 'Midrange, bottom removed')],
    [('sum', 'Strength'), ('degree', 'Degree')],
]
AGGREGATIONS = [a for group in AGG_GROUPS for a in group]
AGG_LABELS = dict(AGGREGATIONS)
AGG_GROUP_OF = {m: i for i, group in enumerate(AGG_GROUPS) for m, _ in group}
REFERENCE_MODE = 'Linf'                # adopted aggregation, reference of "vs. midrange"
TAU_MODES = ('Linf', 'Linf_droptop')   # rankings compared with Kendall tau (printed only)

ORACLE = 'oracle baseline'
FOCUS_SYMBOL = '$w_T$'                 # name of the focus approach in the captions

# Line width and drawing order of every approach; colour and line style from visualization.APPROACH_STYLES
LINE_WEIGHTS = {
    'coretweets':                {'linewidth': 2.0, 'zorder': 4},
    'coretweets_weighted_12h':   {'linewidth': 2.2, 'zorder': 5},
    'coretweets_weighted_1day':  {'linewidth': 1.5, 'zorder': 3},
    'coretweets_weighted_2days': {'linewidth': 1.5, 'zorder': 3},
}
APPROACH_STYLES = {name: {**APPROACH_COLOURS[name], **weights} for name, weights in LINE_WEIGHTS.items()}
APPROACH_LABELS = {
    'coretweets': 'Co-retweet (baseline)',
    'coretweets_weighted_12h': 'Weighted, $T=12$h',
    'coretweets_weighted_1day': 'Weighted, $T=1$d',
    'coretweets_weighted_2days': 'Weighted, $T=2$d',
}
FALLBACK_COLORS = ['#555555', '#999999', '#222222']   # approaches without a style: greys, no approach colour
PLATEAU_COLOR = '#dbe9f6'
BAND_ALPHA = 0.15
# Oracle baseline: dashed line in pnorm_plateau.png and oracle-gain curve in pnorm_gains.png, grey (not an
# approach colour)
ORACLE_COLOR = '#7f7f7f'
ORACLE_LINE_COLOR = ORACLE_COLOR
# Font sizes of the score-vs-aggregation grids: pnorm_gains.png (FONTS) and pnorm_plateau.png (PLATEAU_FONTS)
FONTS = {'tick': 8, 'alt_tick': 7, 'label': 9, 'title': 11, 'legend': 8.5}
PLATEAU_FONTS = {'tick': 12, 'alt_tick': 11, 'label': 13, 'title': 15, 'legend': 12.5}
# Off-scale aggregations in pnorm_plateau.png and pnorm_plateau_alternative.png: a column is cut when a gap
# above it exceeds OFFSCALE_GAP of the panel range; it is left out of the y-limits and its values go in a box at
# the bottom of the panel
OFFSCALE_GAP = 0.3
OFFSCALE_FONTSIZE = 8.5
OFFSCALE_EDGE_COLOR = '0.5'
OFFSCALE_FORCED = ['L1']   # p-norms always cut in the p-norm version (p = 1 is far below in AP)
DAGGER = '^{\\dagger}'

# Difference tables (heatmaps): gain column -> (file suffix, title, colorbar label, mark best row)
DIFF_TABLES = {
    'gain_same': ('same', "Difference vs baseline at the same aggregation",
                  "score - baseline score (same aggregation)  [pp]", False),
    'gain_best': ('best', "Difference vs oracle baseline (maximum over {scope})",
                  "score - oracle baseline score  [pp]", True),
}
# Rows of the alternative-aggregation difference tables, in this order
ALT_AGGREGATIONS = ['min', 'TM25', 'MH', 'Q90', 'max', 'Linf_droptop', 'Linf_dropbottom', 'sum', 'degree']
ORACLE_SCOPE_TEXT = {'all': 'all aggregations', 'pnorms': 'p-norms'}
DIFF_VMIN, DIFF_VMAX = -30.0, 10.0     # color scale limits (pp); values outside are saturated
NEG_COLORS = ('#67001f', '#f4846f')    # (extreme, next to 0): dark red -> salmon red
POS_COLORS = ('#08306b', '#6baed6')    # (extreme, next to 0): dark blue -> clear blue
BEST_BORDER_COLOR = '#ffd700'          # best p-norm per (dataset, approach, metric)


# ==========================================================================
# Names and labels
# ==========================================================================

def t_label(approach):
    """'Weighted, $T=12$h' -> '12h'."""
    return APPROACH_LABELS.get(approach, tex_escape(approach)).replace('Weighted, $T=', '').replace('$', '')


def t_math(approach):
    """'Weighted, $T=12$h' -> '$T=12$h'."""
    return APPROACH_LABELS.get(approach, tex_escape(approach)).replace('Weighted, ', '')


def plain_label(approach):
    return APPROACH_LABELS.get(approach, approach).replace('$', '').replace('T=', 'T = ')


def scenario_name(label):
    """'S1' -> 'Scenario~1'; any other label is kept (LaTeX-escaped)."""
    if label[:1] == 'S' and label[1:].isdigit():
        return f"Scenario~{label[1:]}"
    return tex_escape(label)


def agg_name(label):
    """Name of an aggregation inside a sentence: 'Trimmed mean (25\\%)' -> 'trimmed mean'."""
    name = label.split(' (')[0]
    return name[0].lower() + name[1:]


# ==========================================================================
# Loading
# ==========================================================================

def collect(config, approaches):
    """
    Score every (approach, aggregation) variant of every scenario and dataset.

    Returns:
        scores:   DataFrame (scenario, dataset, approach, mode, p, metric, score)
        rankings: {(scenario, dataset, approach, mode): suspicious groups} for TAU_MODES
        cells:    {(scenario, dataset): {(approach, mode): stored result}}, scenarios in config order and
                  datasets sorted by name, which is the order of the bootstrap draws
    """
    specs = variant_specs(config, approaches)
    missing = set(approaches) - {name for name, _ in specs}
    if missing:
        raise ComparisonError(f"no approach spec for {sorted(missing)} in the config")
    stored = collect_rankings(config, specs)

    # Datasets sorted, so the order of the cells (and of the bootstrap draws) does not depend on the config
    labels = [label for label, _ in config['scenarios']]
    datasets = sorted(config['datasets'])
    cells = {(label, dataset): stored[(label, dataset)] for label in labels for dataset in datasets}
    rows, rankings = [], {}
    for label, min_coactions in config['scenarios']:
        print(f"[{label}] min_coactions={min_coactions}  ->  {results_dir(config, label)}")
        for name, mode in specs:
            for dataset in datasets:
                result = cells[(label, dataset)][(name, mode)]
                scores = compute_scores(result['suspicious_users'], set(result['io_users']), METRICS)
                for metric, score in scores.items():
                    rows.append({'scenario': label, 'dataset': dataset, 'approach': name,
                                 'mode': mode, 'p': p_value(mode), 'metric': metric, 'score': score})
                if mode in TAU_MODES:
                    rankings[(label, dataset, name, mode)] = result['suspicious_users']

    scores = pd.DataFrame(rows)
    scores['scenario'] = pd.Categorical(scores['scenario'], categories=labels, ordered=True)
    scores['metric'] = pd.Categorical(scores['metric'], categories=METRICS, ordered=True)
    return scores, rankings, cells


# ==========================================================================
# Analysis
# ==========================================================================

def compute_gains(scores, baseline, extra_keys=(), with_mode=True, oracle_scope='all'):
    """
    For every non-baseline variant: baseline at the same aggregation (base_same), oracle
    baseline (base_best, base_best_mode) and the gains in pp. extra_keys=['rep'] computes
    everything separately inside each bootstrap sample.
    """
    keys = KEYS + list(extra_keys)
    base = scores[scores['approach'] == baseline]
    pool = base if oracle_scope == 'all' else base[base['mode'].map(is_pnorm)]
    best = pool.groupby(keys, observed=True)['score'].max().rename('base_best')
    base_same = base.set_index(keys + ['mode'])['score'].rename('base_same')

    gains = (scores[scores['approach'] != baseline]
             .join(base_same, on=keys + ['mode'])
             .join(best, on=keys))
    if with_mode:
        tagged = pool.join(best, on=keys)
        best_mode = (tagged[np.isclose(tagged['score'], tagged['base_best'])]
                     .groupby(keys, observed=True)['mode'].agg('/'.join).rename('base_best_mode'))
        gains = gains.join(best_mode, on=keys)
    gains['gain_same'] = 100.0 * (gains['score'] - gains['base_same'])
    gains['gain_best'] = 100.0 * (gains['score'] - gains['base_best'])
    return gains


def plateau(df, plateau_min_p):
    return df[df['p'] >= plateau_min_p]   # NaN (non p-norm aggregations) is excluded


def loco_selection(gains, weighted, dev_scenario, plateau_min_p, tol_pp):
    """Selection of T: worst oracle gain per (campaign, T, metric), retained and selected periods."""
    sub = plateau(gains[(gains['scenario'] == dev_scenario) & gains['approach'].isin(weighted)],
                  plateau_min_p)
    table = (sub.groupby(['dataset', 'approach', 'metric'], observed=True)['gain_best'].min()
             .unstack('metric')[METRICS])
    table = table.reindex(pd.MultiIndex.from_product(
        [table.index.get_level_values('dataset').unique(), weighted],
        names=['dataset', 'approach']))
    table['mean'] = table[METRICS].mean(axis=1)
    table['retained'] = (table[METRICS] > tol_pp).all(axis=1)
    table['selected'] = False
    for dataset in table.index.get_level_values('dataset').unique():
        kept = table.loc[dataset][table.loc[dataset]['retained']]
        if not kept.empty:
            table.loc[(dataset, kept['mean'].idxmax()), 'selected'] = True
    return table


def aggregation_differences(scores, focus, extra_keys=()):
    """100 * (s_T(a) - s_T(midrange)) of the focus approach (paired with extra_keys=['rep'])."""
    keys = KEYS + list(extra_keys)
    fs = scores[(scores['approach'] == focus) & scores['mode'].isin(AGG_LABELS)]
    reference = fs[fs['mode'] == REFERENCE_MODE].set_index(keys)['score'].rename('ref_score')
    d = fs[fs['mode'] != REFERENCE_MODE].join(reference, on=keys)
    d['diff'] = 100.0 * (d['score'] - d['ref_score'])
    return d


def aggregation_summary(gains, boot_stats, focus):
    """
    Data of the alternative-aggregation table, one row per strategy of the focus approach:
      min_gain, n_cells, n_positive, n_ci_above   same-aggregation gain over the baseline
      exceptions                                  cells whose interval is not above zero:
                                                  [(scenario, dataset, metric, gain)]
      above, zero, below, above_by, below_by      cells above / indistinguishable from / below
                                                  the midrange (paired intervals)
    """
    sub = gains[(gains['approach'] == focus) & gains['mode'].isin(AGG_LABELS)]
    sub = sub.join(boot_stats['cells'][['gain_same_lo', 'gain_same_hi']],
                   on=KEYS + ['approach', 'mode'])
    dci = boot_stats['agg_diff'].reset_index()
    rows = []
    for mode, label in AGGREGATIONS:
        s = sub[sub['mode'] == mode]
        if s.empty:
            continue
        row = {'mode': mode, 'label': label, 'n_cells': len(s),
               'min_gain': s['gain_same'].min(),
               'identical': bool((s['gain_same'] == 0).all()),   # same ranking for both approaches (degree)
               'n_positive': int((s['gain_same'] > 0).sum()),
               'n_ci_above': int((s['gain_same_lo'] > 0).sum()),
               'exceptions': [(r.scenario, r.dataset, r.metric, r.gain_same)
                              for r in s.itertuples() if not r.gain_same_lo > 0]}
        if mode != REFERENCE_MODE:
            d = dci[dci['mode'] == mode]
            above, below = d['diff_lo'] > 0, d['diff_hi'] < 0
            row.update(above=int(above.sum()), zero=int((~above & ~below).sum()),
                       below=int(below.sum()),
                       above_by=dict(Counter(d.loc[above, 'dataset'])),
                       below_by=dict(Counter(d.loc[below, 'dataset'])))
        rows.append(row)
    return pd.DataFrame(rows).set_index('mode')


def kendall_taus(rankings, approaches):
    """Kendall tau-b between the midrange rankings with and without the top score."""
    rows = []
    keys = {(s, d, a) for (s, d, a, _) in rankings}
    for scenario, dataset, approach in sorted(keys, key=str):
        if approach not in approaches:
            continue
        a = rankings.get((scenario, dataset, approach, TAU_MODES[0]))
        b = rankings.get((scenario, dataset, approach, TAU_MODES[1]))
        if a is None or b is None:
            continue
        ra = {u: i for i, group in enumerate(a) for u in group}
        rb = {u: i for i, group in enumerate(b) for u in group}
        users = sorted(set(ra) & set(rb))
        tau = kendalltau([ra[u] for u in users], [rb[u] for u in users]).statistic
        rows.append({'scenario': scenario, 'dataset': dataset, 'approach': approach,
                     'tau': tau, 'n_users': len(users)})
    return pd.DataFrame(rows)


# ==========================================================================
# Evaluation bootstrap (rankings fixed, users resampled, all variants paired)
# ==========================================================================

def bootstrap_scores(cells, scores, n_boot, seed):
    """
    Long DataFrame (scenario, dataset, approach, mode, p, metric, rep, score). Checks first that, with unit
    weights, the tie-block metrics reproduce the point scores (checked_block_metrics); aborts if they do not. One
    generator for all the cells, drawn in the order of `cells`.
    """
    point = scores.set_index(['scenario', 'dataset', 'approach', 'mode', 'metric'])['score']
    rng = np.random.default_rng(seed)
    frames, worst_parity = [], 0.0
    for (scenario, dataset), variants in cells.items():
        keys = list(variants)
        io_users = set().union(*(r['io_users'] for r in variants.values()))
        references = {k: {m: point[(scenario, dataset, *k, m)] for m in METRICS} for k in keys}
        try:
            _, encoded, io_mask = encode_rankings({k: variants[k]['suspicious_users'] for k in keys}, io_users,
                                                  sort_users=False)
            _, worst = checked_block_metrics(encoded, io_mask, METRICS, references)
        except ValueError as e:
            raise ComparisonError(f"[{scenario} / {dataset}]: {e}") from None
        worst_parity = max(worst_parity, worst)

        out = conditional_bootstrap(encoded, io_mask, keys, METRICS, n_boot, rng)
        print(f"  Bootstrap [{scenario} / {dataset}]: {len(keys)} variants x {n_boot} samples")

        V, M = len(keys), len(METRICS)
        frame = pd.DataFrame({
            'rep': np.repeat(np.arange(n_boot), V * M),
            'approach': np.tile(np.repeat([k[0] for k in keys], M), n_boot),
            'mode': np.tile(np.repeat([k[1] for k in keys], M), n_boot),
            'metric': np.tile(METRICS, n_boot * V),
            'score': out.reshape(-1)})
        frame['scenario'], frame['dataset'] = scenario, dataset
        frames.append(frame)

    print(f"  Metric parity with compute_scores: max abs difference {worst_parity:.1e}")
    boot = pd.concat(frames, ignore_index=True)
    boot['p'] = boot['mode'].map(p_value)
    boot['scenario'] = pd.Categorical(boot['scenario'], categories=scores['scenario'].cat.categories,
                                      ordered=True)
    boot['metric'] = pd.Categorical(boot['metric'], categories=METRICS, ordered=True)
    return boot


def quantile_ci(df, by, columns, level):
    """Percentile interval of `columns` over samples, columns '<col>_lo' and '<col>_hi'."""
    a = (1.0 - level) / 2.0
    q = df.groupby(by, observed=True)[columns].quantile([a, 1.0 - a]).unstack(-1)
    q.columns = [f"{c}_{'lo' if np.isclose(qq, a) else 'hi'}" for c, qq in q.columns]
    return q


def bootstrap_stats(boot, baseline, focus, weighted, dev_scenario, plateau_min_p, tol_pp, level,
                    oracle_scope='all'):
    """Bootstrap intervals of every quantity of the figures, tables and printed summary."""
    bg = compute_gains(boot, baseline, extra_keys=['rep'], with_mode=False,
                       oracle_scope=oracle_scope)
    n_boot = boot['rep'].nunique()

    # Every variant: same-norm and oracle gains (gain figure and aggregation table)
    cells = quantile_ci(bg, KEYS + ['approach', 'mode'], ['gain_same', 'gain_best'], level)

    # Same-norm worst gain of every T (printed)
    pw = plateau(bg[bg['approach'].isin(weighted)], plateau_min_p)
    sw = (pw.groupby(KEYS + ['approach', 'rep'], observed=True)['gain_same'].min()
          .rename('same_worst').reset_index())
    same_worst = quantile_ci(sw, KEYS + ['approach'], ['same_worst'], level)

    # Differences between aggregations of the focus approach, paired (aggregation table)
    diffs = aggregation_differences(boot, focus, extra_keys=['rep'])
    agg_diff = quantile_ci(diffs, KEYS + ['mode'], ['diff'], level)

    # Selection of T inside each sample (selection table)
    sub = plateau(bg[(bg['scenario'] == dev_scenario) & bg['approach'].isin(weighted)],
                  plateau_min_p)
    t = (sub.groupby(['dataset', 'approach', 'rep', 'metric'], observed=True)['gain_best'].min()
         .unstack('metric')[METRICS])
    t['mean'] = t[METRICS].mean(axis=1)
    t['retained'] = (t[METRICS] > tol_pp).all(axis=1)
    loco = quantile_ci(t.reset_index(), ['dataset', 'approach'], METRICS + ['mean'], level)
    kept = t.loc[t['retained'], 'mean']
    datasets = list(t.index.get_level_values('dataset').unique())
    if kept.empty:
        choice = pd.DataFrame(index=range(n_boot), columns=datasets, dtype=object)
    else:
        winners = kept.groupby(level=['dataset', 'rep']).idxmax()
        choice = (pd.Series([ix[1] for ix in winners.to_numpy()], index=winners.index)
                  .unstack('dataset').reindex(index=range(n_boot), columns=datasets))
    freq = {(d, a): float((choice[d] == a).mean()) for d in datasets for a in weighted}
    loco['selected_freq'] = pd.Series(freq).reindex(loco.index).fillna(0.0)
    none_freq = {d: float(choice[d].isna().mean()) for d in datasets}
    agree_freq = float((choice.notna().all(axis=1) & (choice.nunique(axis=1) == 1)).mean())

    return {'cells': cells, 'same_worst': same_worst, 'agg_diff': agg_diff, 'loco': loco,
            'none_freq': none_freq, 'agree_freq': agree_freq, 'n_boot': n_boot, 'level': level}


def ci_flag(table, key, column):
    """True if the interval of `column` at row `key` of a quantile_ci table contains zero."""
    if key not in table.index:
        return False
    r = table.loc[key]
    return contains_zero(r[f'{column}_lo'], r[f'{column}_hi'])


def ci_counts(boot_stats, weighted, plateau_min_p):
    """Plateau cells (oracle gain) whose interval is above zero / contains zero / below zero."""
    cells = boot_stats['cells'].reset_index()
    cells = cells[cells['approach'].isin(weighted) & (cells['mode'].map(p_value) >= plateau_min_p)]
    lo, hi = cells['gain_best_lo'], cells['gain_best_hi']
    cells = cells.assign(above=lo > 0, below=hi < 0, overlap=(lo <= 0) & (hi >= 0))
    return (cells.groupby(['approach', 'scenario'], observed=True)[['above', 'overlap', 'below']]
            .sum().reindex(weighted, level='approach'))


# ==========================================================================
# Figures
# ==========================================================================

def approach_style(approach, index):
    return APPROACH_STYLES.get(approach, {'color': FALLBACK_COLORS[index % len(FALLBACK_COLORS)],
                                          'linestyle': '--', 'linewidth': 1.5, 'zorder': 3})


def line_kwargs(style):
    return {k: v for k, v in style.items() if k != 'zorder'}


def pnorm_modes(df, min_p=2):
    return sorted({m for m in df['mode'] if is_pnorm(m) and p_value(m) >= min_p}, key=p_value)


def figure_rows(df):
    return list(df[['scenario', 'dataset']].drop_duplicates()
                .sort_values(['scenario', 'dataset']).itertuples(index=False))


def shade_plateau(ax, modes, plateau_min_p):
    first = next((i for i, m in enumerate(modes) if p_value(m) >= plateau_min_p), None)
    if first is not None:
        ax.axvspan(first - 0.5, len(modes) - 0.5, color=PLATEAU_COLOR, alpha=0.6, zorder=0, lw=0)


def finish_axes(ax, modes, alternative=False, fonts=FONTS):
    ax.set_xticks(np.arange(len(modes)))
    if alternative:
        ax.set_xticklabels([mode_label(m) for m in modes], fontsize=fonts['alt_tick'],
                           rotation=45, ha='right', rotation_mode='anchor')
    else:
        ax.set_xticklabels([p_label(m) for m in modes], fontsize=fonts['tick'])
    ax.set_xlim(-0.5, len(modes) - 0.5)
    ax.tick_params(axis='y', labelsize=fonts['tick'])
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')


def label_grid(axes, rows, single, xlabel='Aggregation norm $p$', fonts=FONTS):
    for r, (scenario, dataset) in enumerate(rows):
        axes[r, 0].set_ylabel(dataset if single else f"{scenario}\n{dataset}",
                              fontsize=fonts['label'], fontweight='bold')
    for c, metric in enumerate(METRICS):
        axes[0, c].set_title(METRIC_LABELS[metric], fontsize=fonts['title'], fontweight='bold')
        axes[-1, c].set_xlabel(xlabel, fontsize=fonts['label'])


def offscale_columns(curves, forced=()):
    """
    Columns of `curves` (approach x column) far below the rest of the panel: the `forced` columns,
    then, while the largest gap between consecutive sorted values exceeds OFFSCALE_GAP of their
    range, the columns entirely below the gap. The oracle baseline is left out, otherwise a line
    far above every curve would cut them all.
    """
    kept = [j for j in range(curves.shape[1]) if j not in forced]
    while kept:
        v = curves[:, kept].ravel()
        v = np.sort(v[~np.isnan(v)])
        if len(v) < 2:
            break
        gaps = np.diff(v)
        i = int(np.argmax(gaps))
        if gaps[i] <= OFFSCALE_GAP * (v[-1] - v[0]):
            break
        below = [j for j in kept if np.nanmax(curves[:, j]) <= v[i]]
        if not below:
            break
        kept = [j for j in kept if j not in below]
    return [j for j in range(curves.shape[1]) if j not in kept]


def offscale_box(ax, x, values, colors):
    """
    Box inside the panel, at the bottom of column x: a down triangle (values below the axis), then
    the value of every curve, in legend order and color.
    """
    lines = [TextArea('▼', textprops={'color': OFFSCALE_EDGE_COLOR, 'fontsize': OFFSCALE_FONTSIZE})]
    lines += [TextArea(f"{v:.3f}", textprops={'color': color, 'fontsize': OFFSCALE_FONTSIZE})
              for v, color in zip(values, colors)]
    box = AnnotationBbox(VPacker(children=lines, align='center', pad=0, sep=1.5), (x, 0.03),
                         xycoords=ax.get_xaxis_transform(), box_alignment=(0.5, 0), pad=0.25,
                         bboxprops={'facecolor': 'white', 'edgecolor': OFFSCALE_EDGE_COLOR,
                                    'linewidth': 0.6})
    box.set_zorder(10)
    ax.add_artist(box)
    return box


def clear_offscale_boxes(fig, clear):
    """Lower the y-limit wherever a point drawn inside the axis would be hidden by its box."""
    if not clear:
        return
    fig.canvas.draw()   # box sizes are only known once drawn
    renderer = fig.canvas.get_renderer()
    for ax, box, y_low in clear:
        top = (box.get_window_extent(renderer).y1 - ax.bbox.y0) / ax.bbox.height + 0.04
        ylo, yhi = ax.get_ylim()
        if (y_low - ylo) / (yhi - ylo) < top:
            ax.set_ylim((y_low - top * yhi) / (1 - top), yhi)


def plot_pnorm_plateau(scores, gains, approaches, plateau_min_p, path, alternative=False):
    """
    Scores against the aggregation. Rows = (scenario, dataset), columns = metrics; p-norms with p >= 1. The
    dashed line is the oracle baseline of the cell (base_best of compute_gains), so the oracle gain
    of a curve at p is its distance to the line. alternative=True puts the alternative
    aggregations (ALT_AGGREGATIONS) on the x-axis instead of the p-norms (no plateau shading).
    Columns far below the rest of a panel (offscale_columns) and, in the p-norm version, the
    OFFSCALE_FORCED p-norms are cut and their values boxed.
    """
    if alternative:
        present = set(scores['mode'])
        modes = [m for m in ALT_AGGREGATIONS if m in present]
    else:
        modes = pnorm_modes(scores, min_p=1)
    forced = [modes.index(m) for m in OFFSCALE_FORCED if m in modes] if not alternative else []
    data = scores[scores['mode'].isin(modes)]
    rows = figure_rows(data)
    single = len({s for s, _ in rows}) == 1
    x = np.arange(len(modes))
    oracle = gains.groupby(KEYS, observed=True)['base_best'].first()

    # Wide panels: room for adjacent boxes and as wide as the legend at PLATEAU_FONTS
    fig, axes = plt.subplots(len(rows), len(METRICS),
                             figsize=(4.6 * len(METRICS), 2.5 * len(rows)), squeeze=False)
    any_cut = False
    clear = []   # (ax, box, lowest point of the column drawn inside the axis)
    for r, (scenario, dataset) in enumerate(rows):
        for c, metric in enumerate(METRICS):
            ax = axes[r, c]
            panel = data[(data['scenario'] == scenario) & (data['dataset'] == dataset)
                         & (data['metric'] == metric)]
            if not alternative:
                shade_plateau(ax, modes, plateau_min_p)
            curves = np.array([panel[panel['approach'] == approach].set_index('mode')['score']
                               .reindex(modes).to_numpy(dtype=float) for approach in approaches])
            for idx, (approach, y) in enumerate(zip(approaches, curves)):
                style = approach_style(approach, idx)
                ax.plot(x, y, marker='o', markersize=3.5, **style)
                if 'Linf' in modes:
                    i = modes.index('Linf')
                    ax.plot(x[i], y[i], marker='o', markersize=7, color=style['color'],
                            markeredgecolor='white', markeredgewidth=0.8, zorder=style['zorder'] + 1)
            base_best = oracle.get((scenario, dataset, metric), np.nan)
            ax.axhline(base_best, color=ORACLE_LINE_COLOR, linestyle=(0, (4, 2)), linewidth=1.1,
                       zorder=2)
            cut = offscale_columns(curves, forced)
            kept = [j for j in range(len(modes)) if j not in cut]
            values = np.append(curves[:, kept].ravel(), base_best)
            lo, hi = np.nanmin(values), np.nanmax(values)
            pad = max(0.12 * (hi - lo), 0.003)
            ax.set_ylim(lo - pad, hi + pad)
            for j in cut:   # lines leave through the bottom, values boxed inside the panel
                below = curves[:, j] < lo - pad
                if not below.any():
                    continue
                box = offscale_box(ax, x[j], curves[below, j],
                                   [approach_style(a, i)['color']
                                    for i, a in enumerate(approaches) if below[i]])
                visible = curves[~below, j]
                if not np.isnan(visible).all():   # forced column, partly inside the axis
                    clear.append((ax, box, np.nanmin(visible)))
            any_cut = any_cut or bool(cut)
            finish_axes(ax, modes, alternative, PLATEAU_FONTS)
            if r < len(rows) - 1:   # common x-axis: tick labels on the bottom row only
                ax.tick_params(axis='x', labelbottom=False)
    label_grid(axes, rows, single, 'Aggregation' if alternative else 'Aggregation norm $p$',
               PLATEAU_FONTS)

    handles = [Line2D([], [], marker='o', markersize=4, label=plain_label(a),
                      **line_kwargs(approach_style(a, i)))
               for i, a in enumerate(approaches)]
    handles.append(Line2D([], [], color=ORACLE_LINE_COLOR, linestyle=(0, (4, 2)), linewidth=1.1,
                          label='Oracle baseline (best baseline aggregation)'))
    if not alternative:
        handles += [Line2D([], [], marker='o', markersize=7, color='0.4', linestyle='none',
                           markeredgecolor='white', label='p = ∞ (adopted)'),
                    Patch(facecolor=PLATEAU_COLOR, label=f'Plateau (p ≥ {plateau_min_p:g})')]
    if any_cut:
        handles.append(Patch(facecolor='white', edgecolor=OFFSCALE_EDGE_COLOR,
                             label='▼ Below the axis (values in legend order)'))
    fig.legend(handles=handles, loc='lower center', ncol=-(-len(handles) // 2),
               fontsize=PLATEAU_FONTS['legend'], frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    clear_offscale_boxes(fig, clear)
    save_figure(fig, path)


def plot_pnorm_gains(gains, boot_stats, gain_approaches, focus, plateau_min_p, path):
    """
    Gains against p: same-norm gain of every period in gain_approaches (solid / dashed) and oracle
    gain of the focus period (dotted) as a function of p, each with its bootstrap band.
    """
    modes = pnorm_modes(gains)
    data = gains[gains['mode'].isin(modes)]
    cells = boot_stats['cells']
    rows = figure_rows(data)
    single = len({s for s, _ in rows}) == 1
    x = np.arange(len(modes))
    curves = [(a, 'gain_same', None) for a in gain_approaches] + [(focus, 'gain_best', ':')]

    def curve_style(approach, ls, idx):
        style = dict(approach_style(approach, idx))
        if ls is not None:
            style.update(linestyle=ls, linewidth=1.6, color=ORACLE_COLOR)
        return style

    def band(scenario, dataset, metric, approach, column):
        keys = [(scenario, dataset, metric, approach, m) for m in modes]
        lo = [cells.loc[k, f'{column}_lo'] if k in cells.index else np.nan for k in keys]
        hi = [cells.loc[k, f'{column}_hi'] if k in cells.index else np.nan for k in keys]
        return np.array(lo, dtype=float), np.array(hi, dtype=float)

    fig, axes = plt.subplots(len(rows), len(METRICS),
                             figsize=(4.0 * len(METRICS), 2.5 * len(rows)), squeeze=False)
    for r, (scenario, dataset) in enumerate(rows):
        for c, metric in enumerate(METRICS):
            ax = axes[r, c]
            shade_plateau(ax, modes, plateau_min_p)
            ax.axhline(0.0, color='0.3', linewidth=0.9, zorder=1)
            panel = data[(data['scenario'] == scenario) & (data['dataset'] == dataset)
                         & (data['metric'] == metric)]
            for idx, (approach, column, ls) in enumerate(curves):
                y = (panel[panel['approach'] == approach].set_index('mode')[column]
                     .reindex(modes).to_numpy(dtype=float))
                style = curve_style(approach, ls, idx)
                lo, hi = band(scenario, dataset, metric, approach, column)
                ax.fill_between(x, lo, hi, color=style['color'], alpha=BAND_ALPHA, lw=0,
                                zorder=style['zorder'] - 1)
                ax.plot(x, y, marker='o', markersize=3, **style)
            finish_axes(ax, modes)
    label_grid(axes, rows, single)
    for ax in axes[:, 0]:
        ax.set_ylabel(ax.get_ylabel() + '\ngain (pp)', fontsize=9, fontweight='bold')

    handles, labels = [], []
    for idx, (approach, column, ls) in enumerate(curves):
        style = line_kwargs(curve_style(approach, ls, idx))
        handles.append((Patch(facecolor=style['color'], alpha=BAND_ALPHA, lw=0),
                        Line2D([], [], marker='o', markersize=4, **style)))
        labels.append(f"{plain_label(approach)} vs baseline, same p" if column == 'gain_same'
                      else f"{plain_label(approach)} vs {ORACLE}")
    handles.append(Patch(facecolor=PLATEAU_COLOR))
    labels.append(f'Plateau (p ≥ {plateau_min_p:g})')
    fig.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=1)},
               loc='lower center', ncol=2, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.0),
               title=f"Bands: {100 * boot_stats['level']:g}% bootstrap interval of each curve",
               title_fontsize=8.5)
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    save_figure(fig, path)


# ==========================================================================
# Difference tables (heatmaps)
# ==========================================================================

def difference_table(gains, column, scenario, approaches, modes):
    """
    Gain `column` of one scenario as a table: rows (dataset, approach, mode) restricted to `modes`,
    sorted by dataset, approach and mode (in the order of `approaches` and `modes`); one column
    per metric.
    """
    df = gains[(gains['scenario'] == scenario) & gains['mode'].isin(modes)
               & gains['approach'].isin(approaches)]
    table = df.pivot_table(index=['dataset', 'approach', 'mode'], columns='metric',
                           values=column, observed=True).reindex(columns=METRICS)
    order = sorted(table.index,
                   key=lambda i: (str(i[0]), approaches.index(i[1]), modes.index(i[2])))
    return table.loc[order]


def mode_label(mode):
    """Row label of an aggregation: 'p = 3' for p-norms, the plain name otherwise."""
    if is_pnorm(mode):
        return f"p = {p_label(mode)}"
    return AGG_LABELS.get(mode, mode).replace('\\%', '%')


def uncertain_cells(table, cells, column, scenario):
    """Boolean array: True where the bootstrap interval of the cell contains zero."""
    return np.array([[ci_flag(cells, (scenario, dataset, metric, approach, mode), column)
                      for metric in table.columns]
                     for dataset, approach, mode in table.index], dtype=bool)


def best_cell_positions(table):
    """(row, col) positions of the best row per (dataset, approach) and metric, ties kept."""
    positions = set()
    groups = [(d, a) for d, a, _ in table.index]
    values = table.to_numpy(dtype=float)
    for group in dict.fromkeys(groups):
        rows = [i for i, g in enumerate(groups) if g == group]
        for j in range(values.shape[1]):
            col = values[rows, j]
            if np.all(np.isnan(col)):
                continue
            positions.update((rows[k], j) for k in np.flatnonzero(np.isclose(col, np.nanmax(col))))
    return positions


def diverging_cmap():
    """Red (< 0) / blue (> 0) colormap with a sharp jump at 0, so tiny gains still show the sign."""
    def channel(i):
        return [(0.0, to_rgb(NEG_COLORS[0])[i], to_rgb(NEG_COLORS[0])[i]),
                (0.5, to_rgb(NEG_COLORS[1])[i], to_rgb(POS_COLORS[1])[i]),
                (1.0, to_rgb(POS_COLORS[0])[i], to_rgb(POS_COLORS[0])[i])]
    return LinearSegmentedColormap('red_blue_split',
                                   {'red': channel(0), 'green': channel(1), 'blue': channel(2)})


def plot_difference_table(table, uncertain, title, cbar_label, level, path, mark_best=True,
                          row_kind='p-norm'):
    """
    Heatmap of a difference table (pp). Bold value + gold border (if mark_best): best row
    (`row_kind`) per (dataset, approach, metric). Dagger: bootstrap interval contains zero.
    Values outside [DIFF_VMIN, DIFF_VMAX] are saturated in color, the printed number is kept.
    """
    values = table.to_numpy(dtype=float)
    n_rows, n_cols = values.shape
    norm = TwoSlopeNorm(vmin=DIFF_VMIN, vcenter=0.0, vmax=DIFF_VMAX)
    cmap = diverging_cmap()
    best = best_cell_positions(table) if mark_best else set()

    fig, ax = plt.subplots(figsize=(2 + 1.6 * n_cols, 1.5 + 0.28 * n_rows))
    im = ax.imshow(values, cmap=cmap, norm=norm, aspect='auto')

    for i in range(n_rows):
        for j in range(n_cols):
            v = values[i, j]
            if np.isnan(v):
                continue
            r, g, b, _ = cmap(norm(np.clip(v, DIFF_VMIN, DIFF_VMAX)))
            color = 'white' if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else 'black'
            ax.text(j, i, f"{v:+.2f}", ha='center', va='center', color=color,
                    fontsize=8, fontweight='bold' if (i, j) in best else 'normal')
            if uncertain[i, j]:   # dagger apart, at the right edge of the cell
                ax.text(j + 0.42, i, '†', ha='right', va='center', color=color,
                        fontsize=11, fontweight='bold')
    for i, j in best:
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                               edgecolor=BEST_BORDER_COLOR, linewidth=3, zorder=3))

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([METRIC_LABELS[m] for m in table.columns])
    ax.xaxis.tick_top()
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([f"{d} | {plain_label(a)} | {mode_label(m)}" for d, a, m in table.index],
                       fontsize=8)
    ax.tick_params(which='both', length=0)

    # Separate (dataset, approach) groups with a thin line and datasets with a thick one
    for i in range(1, n_rows):
        (d0, a0, _), (d1, a1, _) = table.index[i - 1], table.index[i]
        if (d0, a0) != (d1, a1):
            ax.axhline(i - 0.5, color='black', linewidth=2.0 if d0 != d1 else 1.0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(cbar_label)
    notes = ([f"gold border = best {row_kind} per dataset, approach and metric"]
             if mark_best else [])
    notes.append(f"† = {100 * level:g}% bootstrap interval contains 0")
    ax.set_title(f"{title}\n({'; '.join(notes)})", pad=28, fontweight='bold', fontsize=10)
    fig.tight_layout()
    save_figure(fig, path)


def plot_difference_tables(gains, boot_stats, approaches, scenarios, oracle_scope, out_dir):
    """
    Both difference tables (same aggregation, oracle baseline) for every scenario, each split in
    two figures: p-norm rows and alternative-aggregation rows (ALT_AGGREGATIONS).
    """
    row_sets = {
        'pnorms': ('p-norm', sorted({m for m in gains['mode'] if is_pnorm(m)}, key=p_value)),
        'alt': ('aggregation', ALT_AGGREGATIONS),
    }
    for scenario in scenarios:
        for column, (suffix, title, cbar_label, mark_best) in DIFF_TABLES.items():
            title = title.format(scope=ORACLE_SCOPE_TEXT[oracle_scope])
            for rows, (row_kind, modes) in row_sets.items():
                table = difference_table(gains, column, scenario, approaches, modes)
                if table.empty:
                    print(f"  WARNING: no {rows} variants for the {column} table of {scenario}")
                    continue
                uncertain = uncertain_cells(table, boot_stats['cells'], column, scenario)
                plot_difference_table(table, uncertain, f"{title}, {scenario}", cbar_label,
                                      boot_stats['level'],
                                      out_dir / f'table_diff_{suffix}_{rows}_{scenario}.png',
                                      mark_best, row_kind)


def mode_order(mode):
    """Sort key: p-norms by p first, then the other aggregations in AGGREGATIONS order."""
    if is_pnorm(mode):
        return (0, p_value(mode))
    order = [m for m, _ in AGGREGATIONS]
    return (1, order.index(mode) if mode in order else len(order))


def plot_oracle_table(gains, oracle_scope, path):
    """
    Table figure: aggregation(s) attaining the oracle baseline (ties joined by '/') and its score,
    one row per (scenario, dataset) and one column per metric.
    """
    oracle = (gains.groupby(KEYS, observed=True)[['base_best', 'base_best_mode']].first()
              .reset_index())
    rows = figure_rows(oracle)
    cell_text = []
    for scenario, dataset in rows:
        line = []
        for metric in METRICS:
            r = oracle[(oracle['scenario'] == scenario) & (oracle['dataset'] == dataset)
                       & (oracle['metric'] == metric)]
            if r.empty:
                line.append('—')
                continue
            modes = sorted(r['base_best_mode'].iloc[0].split('/'), key=mode_order)
            line.append(' / '.join(mode_label(m) for m in modes)
                        + f"\n(score {r['base_best'].iloc[0]:.4f})")
        cell_text.append(line)

    fig, ax = plt.subplots(figsize=(2 + 2.6 * len(METRICS), 1.0 + 0.6 * len(rows)))
    ax.axis('off')
    table = ax.table(cellText=cell_text, rowLabels=[f"{s} | {d}" for s, d in rows],
                     colLabels=[METRIC_LABELS[m] for m in METRICS], loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.8)
    for (i, j), cell in table.get_celld().items():
        if i == 0 or j == -1:
            cell.set_text_props(fontweight='bold')
            cell.set_facecolor('#eeeeee')
    ax.set_title(f"Aggregation attaining the oracle baseline ({BASELINE}, maximum over "
                 f"{ORACLE_SCOPE_TEXT[oracle_scope]})", fontweight='bold', fontsize=10)
    fig.tight_layout()
    save_figure(fig, path)


# ==========================================================================
# LaTeX tables
# ==========================================================================

def signed(value, uncertain=False):
    """Signed 2-decimal value in math mode, with a dagger if its interval contains zero."""
    if not np.isfinite(value):
        return '--'
    return f"${value:+.2f}{DAGGER if uncertain else ''}$"


def write_loco_table(loco, boot_stats, dev_scenario, plateau_min_p, path):
    """Table of the selection of T on each campaign of dev_scenario."""
    boot = boot_stats['loco']
    body = []
    for k, dataset in enumerate(loco.index.get_level_values('dataset').unique()):
        if k:
            body.append('\\midrule')
        for i, (approach, r) in enumerate(loco.loc[dataset].iterrows()):
            key = (dataset, approach)
            name = f"\\textbf{{{t_label(approach)}}}" if r.selected else t_label(approach)
            cells = [signed(r[m], ci_flag(boot, key, m)) for m in METRICS + ['mean']]
            body.append(' & '.join([tex_escape(dataset) if i == 0 else '', name] + cells) + ' \\\\')
    # Retained periods and selection frequencies are printed to stdout, not written to the table
    header = ['Campaign & $T$ & AUC & AP & NDCG & Mean \\\\']
    caption = (f"Selection of $T$ on each campaign of {scenario_name(dev_scenario)}. Values: worst "
               f"oracle gain (pp) of the weighted approach over the plateau "
               f"$p\\geq{plateau_min_p:g}$. A period is retained if this value is positive for the "
               f"three metrics, and the retained period with the largest mean is selected (bold). "
               f"$^{{\\dagger}}$: the {100 * boot_stats['level']:g}\\% interval of the worst gain, "
               f"recomputed in each bootstrap sample, contains zero.")
    write_lines(path, table_lines(caption, 'tab:loco', 'llrrrr', header, body, env='table'))


def exceptions_text(agg_sum):
    """
    Sentence of the alternative-aggregation table caption on the same-aggregation intervals not above zero. Aggregations
    that give both approaches the same ranking (the degree) are named apart and not starred.
    """
    identical = [agg_name(r.label) for _, r in agg_sum.iterrows() if r.identical]
    lead = "All these gains"
    if identical:
        verb = 'is' if len(identical) == 1 else 'are'
        lead = (f"Except for the {' and the '.join(identical)}, which {verb} the same for both approaches, "
                f"all these gains")
    starred = [(r.label, r.min_gain, e) for _, r in agg_sum.iterrows() if not r.identical for e in r.exceptions]
    if not starred:
        return f"{lead} have an interval above zero."

    def where(scenario, dataset, metric, gain):
        return f"{tex_escape(scenario)} {tex_escape(dataset)} {METRIC_LABELS[metric]} with ${gain:+.2f}$~pp"

    if len(starred) == 1:
        label, min_gain, cell = starred[0]
        text = (f"{lead} have an interval above zero except one cell of the "
                f"{agg_name(label)}, marked $^{{*}}$ ({where(*cell)})")
        if f"{cell[3]:+.2f}" != f"{min_gain:+.2f}":
            text += ", which is not the cell with its smallest gain"
        return text + "."
    listed = ', '.join(f"{agg_name(label)} {where(*cell)}" for label, _, cell in starred)
    return f"{lead} have an interval above zero except {len(starred)} cells, marked $^{{*}}$ ({listed})."


def write_aggregation_table(agg_sum, focus, level, path):
    """Table of the alternative aggregations of the focus approach."""
    body, previous = [], None
    for mode, r in agg_sum.iterrows():
        if previous is not None and AGG_GROUP_OF.get(mode) != AGG_GROUP_OF.get(previous):
            body.append('\\addlinespace')
        label = r.label + ('$^{*}$' if r.exceptions and not r.identical else '')
        if pd.isna(r.get('above', np.nan)):
            compare = ['--'] * 3
        else:
            compare = [str(int(r.above)), str(int(r.zero)), str(int(r.below))]
        min_gain = '$0$' if r.identical else signed(r.min_gain)
        body.append(' & '.join([label, min_gain] + compare) + ' \\\\')
        previous = mode
    header = [' & & \\multicolumn{3}{c}{vs.\\ midrange} \\\\', '\\cmidrule(lr){3-5}',
              'Aggregation & Min gain & Above & Zero & Below \\\\']
    caption = (f"Alternative aggregations with {t_math(focus)}, over the "
               f"{int(agg_sum['n_cells'].max())} cells. Min gain: smallest same-aggregation gain (pp) of "
               f"{FOCUS_SYMBOL}. {exceptions_text(agg_sum)} Above, Zero, Below: number of cells in which "
               f"{FOCUS_SYMBOL} with this aggregation scores above the midrange, cannot be distinguished "
               f"from it, or scores below it (paired {100 * level:g}\\% intervals).")
    write_lines(path, table_lines(caption, 'tab:aggregation', 'lrrrr', header, body, env='table'))


# ==========================================================================
# Summary numbers (stdout only)
# ==========================================================================

def print_numbers(scores, gains, boot_stats, cic, agg_sum, diffs, taus, baseline, focus,
                  weighted, dev_scenario, plateau_min_p):
    def f2(v):
        return f"{v:+.2f}"

    pd.set_option('display.width', 220)
    pd.set_option('display.max_columns', None)
    pmin = f"L{plateau_min_p:g}"
    cells = boot_stats['cells']

    print_section(f"Behaviour across norms: AP of the baseline and of the focus T ({dev_scenario})")
    sub = scores[(scores['scenario'] == dev_scenario) & (scores['metric'] == 'ap')
                 & scores['mode'].isin([pmin, 'Linf']) & scores['approach'].isin([baseline, focus])]
    print(sub.pivot_table(index=['dataset', 'approach'], columns='mode', values='score',
                          observed=True).to_string(float_format=lambda v: f"{v:.4f}"))

    print_section("Selection of T: bootstrap frequencies")
    print(f"No period retained: {boot_stats['none_freq']}; both campaigns select the same "
          f"period: {100 * boot_stats['agree_freq']:.0f}%")
    print("Percentage of samples in which the rule selects each period:")
    print((100 * boot_stats['loco']['selected_freq']).unstack('approach').to_string(
        float_format=lambda v: f"{v:.0f}"))

    print_section("Same-norm worst gain over the plateau, every T: intervals vs zero")
    sw = boot_stats['same_worst'].reset_index()
    for approach in weighted:
        s = sw[sw['approach'] == approach]
        above = s['same_worst_lo'] > 0
        print(f"{t_label(approach)}: {int(above.sum())}/{len(s)} cells above zero; not above: "
              f"{[tuple(r) for r in s.loc[~above, KEYS].itertuples(index=False)]}")

    print_section("Oracle gain on the plateau, every T: cells by interval (above / zero / below)")
    print(cic.to_string())

    print_section("Focus T on the plateau: same-norm and oracle gains")
    fp = plateau(gains[gains['approach'] == focus], plateau_min_p)
    fp = fp.join(cells, on=KEYS + ['approach', 'mode'])
    for column, name in (('gain_same', 'same-norm'), ('gain_best', 'oracle')):
        above, below = fp[f'{column}_lo'] > 0, fp[f'{column}_hi'] < 0
        zero = ~above & ~below
        i = fp[column].idxmin()
        print(f"{name} gain: {len(fp)} cells, {int(above.sum())} above zero, {int(zero.sum())} "
              f"containing zero, {int(below.sum())} below zero; smallest {fp.loc[i, column]:+.2f} pp "
              f"at {tuple(fp.loc[i, KEYS + ['mode']])}")
        if (zero | below).any():
            print(fp.loc[zero | below, KEYS + ['mode', column]].to_string(index=False, float_format=f2))

    print_section(f"Focus T, same-norm and oracle gains at p = {plateau_min_p:g} and p = inf (pp)")
    fs = gains[(gains['approach'] == focus) & gains['mode'].isin([pmin, 'Linf'])]
    print(fs.pivot_table(index=KEYS, columns='mode', values=['gain_same', 'gain_best'],
                         observed=True).to_string(float_format=f2))
    worst = fp.loc[fp.groupby(KEYS, observed=True)['gain_best'].idxmin(), KEYS + ['mode']]
    print("\nNorm with the smallest oracle gain on the plateau:")
    print(worst.set_index(KEYS)['mode'].unstack('metric').to_string())

    print_section("Aggregation(s) attaining the oracle baseline")
    print(gains.groupby(KEYS, observed=True)['base_best_mode'].first().unstack('metric')
          .to_string())

    print_section("Alternative aggregations of the focus T")
    print(f"Same-aggregation gain: {int(agg_sum['n_cells'].sum())} cells, "
          f"{int(agg_sum['n_positive'].sum())} positive, {int(agg_sum['n_ci_above'].sum())} with "
          f"interval above zero")
    for _, r in agg_sum.iterrows():
        for s, d, m, g in r.exceptions:
            print(f"  interval not above zero: {r.label}, {s} {d} {m}: {g:+.2f} pp")
    cols = ['label', 'min_gain', 'above', 'zero', 'below', 'above_by', 'below_by']
    print("\nVersus the midrange (paired intervals), with the datasets of the above / below cells:")
    print(agg_sum[cols].to_string(float_format=f2))

    top = gains[(gains['approach'] == focus) & (gains['mode'] == 'Linf_droptop')]
    top = top.join(cells, on=KEYS + ['approach', 'mode'])
    print(f"\nMidrange, top removed: {int((top['gain_best'] > 0).sum())}/{len(top)} oracle gains "
          f"positive; intervals not above zero:")
    print(top.loc[~(top['gain_best_lo'] > 0), KEYS + ['gain_best']].to_string(index=False,
                                                                              float_format=f2))
    bottom = diffs[diffs['mode'] == 'Linf_dropbottom']['diff']
    print(f"Midrange, bottom removed: lower in {int((bottom < 0).sum())}/{len(bottom)} cells "
          f"(point estimates), max |difference| {bottom.abs().max():.2f} pp")

    print_section("Kendall tau_b between the midrange rankings with and without the top score")
    print(taus.to_string(index=False, float_format=lambda v: f"{v:.2f}"))


# ==========================================================================
# Entry point
# ==========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Figures and tables of the preliminary analysis (choice of T and of p)")
    parser.add_argument('step', choices=['rank', 'analyze', 'all'])
    parser.add_argument('config', help='Experiment JSON config')
    # Settings of the 'analyze' step. Defaults: the "analysis" section of the config, then ANALYSIS_DEFAULTS
    parser.add_argument('--oracle-scope', choices=['all', 'pnorms'], default=None,
                        help="Aggregations over which the oracle baseline is the maximum")
    parser.add_argument('--plateau-min-p', type=float, default=None)
    parser.add_argument('--tol', type=float, default=None,
                        help='|gain| <= tol (pp) is treated as zero (config key tol_pp)')
    parser.add_argument('--n-boot', type=int, default=None, help='Bootstrap samples')
    parser.add_argument('--ci-level', type=float, default=None,
                        help='Confidence level of the percentile intervals')
    parser.add_argument('--seed', type=int, default=None, help='Bootstrap seed')
    parser.add_argument('--out-dir', default=None, help='Default: {output_dir}/{name}/analysis/')
    return parser.parse_args()


# Settings of the analysis when neither the command line nor the "analysis" section of the config gives them:
# {argument: (config key, default)}
ANALYSIS_DEFAULTS = {'n_boot': ('n_boot', 1000), 'seed': ('seed', 0), 'ci_level': ('ci_level', 0.95),
                     'tol': ('tol_pp', 0.005), 'plateau_min_p': ('plateau_min_p', 3.0),
                     'oracle_scope': ('oracle_scope', 'all')}
ANALYSIS_KEYS = tuple(key for key, _ in ANALYSIS_DEFAULTS.values())


def resolve_settings(args, config):
    """Fill the settings not given on the command line from config['analysis'], then ANALYSIS_DEFAULTS."""
    for arg, (key, default) in ANALYSIS_DEFAULTS.items():
        if getattr(args, arg) is None:
            setattr(args, arg, config['analysis'].get(key, default))
    if args.oracle_scope not in ORACLE_SCOPE_TEXT:
        raise ComparisonError(f"oracle_scope must be one of {list(ORACLE_SCOPE_TEXT)}, got {args.oracle_scope!r}")
    if args.n_boot < 1:
        raise ComparisonError("n_boot must be positive (every table and figure uses the intervals)")
    return args


def analyze(config, args):
    """Every output of the preliminary analysis from the stored rankings of the scenarios."""
    weighted = [FOCUS] + OTHERS
    approaches = [BASELINE] + weighted
    labels = [label for label, _ in config['scenarios']]
    dev_scenario = labels[0]

    args = resolve_settings(args, config)
    print(f"Settings: n_boot={args.n_boot}, seed={args.seed}, ci_level={args.ci_level}, tol={args.tol} pp, "
          f"plateau_min_p={args.plateau_min_p:g}, oracle_scope={args.oracle_scope}")
    out_dir = Path(args.out_dir) if args.out_dir else experiment_dir(config) / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    scores, rankings, cells = collect(config, approaches)
    gains = compute_gains(scores, BASELINE, oracle_scope=args.oracle_scope)
    loco = loco_selection(gains, weighted, dev_scenario, args.plateau_min_p, args.tol)
    diffs = aggregation_differences(scores, FOCUS)

    print_section(f"Evaluation bootstrap: {args.n_boot} samples, seed {args.seed}")
    boot = bootstrap_scores(cells, scores, args.n_boot, args.seed)
    boot_stats = bootstrap_stats(boot, BASELINE, FOCUS, weighted, dev_scenario,
                                 args.plateau_min_p, args.tol, args.ci_level, args.oracle_scope)
    agg_sum = aggregation_summary(gains, boot_stats, FOCUS)
    cic = ci_counts(boot_stats, weighted, args.plateau_min_p)
    taus = kendall_taus(rankings, [BASELINE, FOCUS])

    print_numbers(scores, gains, boot_stats, cic, agg_sum, diffs, taus, BASELINE, FOCUS,
                  weighted, dev_scenario, args.plateau_min_p)

    print_section("Figures and tables")
    plot_pnorm_plateau(scores, gains, approaches, args.plateau_min_p, out_dir / 'pnorm_plateau.png')
    plot_pnorm_plateau(scores, gains, approaches, args.plateau_min_p,
                       out_dir / 'pnorm_plateau_alternative.png', alternative=True)
    plot_pnorm_gains(gains, boot_stats, GAINS_APPROACHES, FOCUS, args.plateau_min_p, out_dir / 'pnorm_gains.png')
    plot_difference_tables(gains, boot_stats, weighted, labels, args.oracle_scope, out_dir)
    plot_oracle_table(gains, args.oracle_scope, out_dir / 'table_oracle_baseline.png')
    write_loco_table(loco, boot_stats, dev_scenario, args.plateau_min_p, out_dir / 'table_loco.tex')
    write_aggregation_table(agg_sum, FOCUS, args.ci_level, out_dir / 'table_aggregation.tex')


def main():
    matplotlib.use('Agg')
    args = parse_args()
    try:
        config = load_scenario_config(args.config, ((), ANALYSIS_KEYS))
        if args.step in ('rank', 'all'):
            rank(config)
        if args.step in ('analyze', 'all'):
            analyze(config, args)
    except ComparisonError as e:
        sys.exit(f"ERROR: {e}")


if __name__ == '__main__':
    main()
