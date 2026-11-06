#!/usr/bin/env python3
"""
Sensitivity of the weighted co-retweet gain to the synchronization window Delta t.

The period T, the aggregations (midrange and p = 3) and every other setting are fixed; only Delta t varies.
The reference window (window_sec) is only marked in the outputs, never selected from them.

Usage:
    uv run bin/benchmark_dt_sensitivity.py all     experiments/article_dt_sensitivity.json
    uv run bin/benchmark_dt_sensitivity.py rank    <config>   one run_experiments.py run per (Delta t, scenario)
    uv run bin/benchmark_dt_sensitivity.py analyze <config>   tables, figure and summary counts

  rank     writes one run_experiments.py config per (Delta t, scenario) to
           {output_dir}/{name}/configs/dt<window>_S<i>.json and runs it (rankings stored in
           {output_dir}/{name}/rankings/dt<window>_S<i>/results/)
  analyze  reads the stored rankings and writes the outputs below

Config fields: those of every experiment run on several scenarios (src/analysis/comparison.py: name, data_dir,
output_dir, window_sec, min_coactions, need_filtering, datasets, force, approaches), where window_sec is the
reference window (marked in the outputs, never selected), and
    windows_sec            Delta t values (s); must include window_sec
    approaches             every aggregation (ranking_mode) of the baseline and of the focus approach, the same
                           for both; must include 'Linf'
    analysis
        baseline, focus        approach names (baseline co-retweets, weighted T = 12h)
        n_boot, ci_level, seed evaluation bootstrap (same procedure as bin/preliminar_analysis_bootstrap.py)
        fc_fraction            recall target of Filtering Concentration

Outputs ({output_dir}/{name}/analysis/)
    table_dt_main.tex               pairs, IO found and IO share of every dataset, scenario and Delta t
    table_app_dt_gains.tex          gains with intervals, all scenarios and aggregations
    table_app_dt_scores.tex         scores and Filtering Concentration, midrange
    dt_sensitivity_gains.{png,pdf}  gain vs Delta t with 95% bands
    dt_scores.csv, dt_gains.csv, dt_graph.csv   supporting data
    The report printed to stdout ends with the counts of the gain intervals by sign and the scores at the
    reference window, which must coincide with table1.tex of bin/article_main_comparison.py.

Uncertainty
    Within each (Delta t, scenario, dataset) the rankings are fixed and users are resampled with replacement
    inside IO and control accounts (stratified); baseline and weighted rankings are re-scored on the same
    resample (paired), with the tie-block metrics of src/analysis/metrics.py. Generator of a cell:
    SeedSequence(seed, spawn_key=(Delta t, scenario index, crc32(dataset))). Cohorts differ across Delta t, so
    intervals are only paired within a window. Percentile intervals, not corrected for multiple comparisons.
"""

import argparse
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.aggregations import p_label, p_tex
from analysis.comparison import (
    ComparisonError,
    collect_rankings,
    experiment_dir,
    load_scenario_config,
    rank,
    variant_specs,
)
from analysis.descriptives import coretweet_pair_statistics
from analysis.metrics import RANK_METRICS, compute_scores, users_until_reaching_io_fraction
from analysis.reporting import print_section, table_lines, tex_escape, write_lines
from analysis.statistics import (
    block_seed,
    checked_block_metrics,
    conditional_bootstrap,
    contains_zero,
    encode_rankings,
    percentile_interval,
)
from analysis.visualization import save_figure

ANALYSIS_KEYS = ('baseline', 'focus', 'n_boot', 'ci_level', 'seed', 'fc_fraction')
METRICS = list(RANK_METRICS)
METRIC_LABELS = {'auc': 'AUC', 'ap': 'AP', 'ndcg': 'NDCG'}
# Every curve is a gain of the focus approach over the baseline; aggregations told apart by colour, line style
# and marker
MODE_STYLES = {'Linf': {'color': '#0072B2', 'linestyle': '-', 'marker': 'o', 'label': 'p = ∞ (adopted)'},
               'L3': {'color': '#D62728', 'linestyle': '--', 'marker': 's', 'label': 'p = 3'}}
FALLBACK_STYLES = [{'color': '#009E73', 'linestyle': '-.', 'marker': '^'},
                   {'color': '#CC79A7', 'linestyle': ':', 'marker': 'D'}]
BAND_ALPHA = 0.15
DAGGER = '$^{\\dagger}$'


# ==========================================================================
# Configuration, rankings and loading
# ==========================================================================

def load_dt_config(path):
    """
    The config (load_scenario_config), with windows_sec sorted, 'variants' {(approach, mode): spec} and
    'ranking_modes' (the aggregations, in config order).
    """
    config = load_scenario_config(path, (ANALYSIS_KEYS, ()), sections=('windows_sec',))
    windows = config['windows_sec']
    if (not isinstance(windows, list) or not windows or len(set(windows)) != len(windows)
            or not all(isinstance(w, int) and w > 0 for w in windows)):
        raise ComparisonError("windows_sec must be a list of distinct positive integers")
    if config['window_sec'] not in windows:
        raise ComparisonError(f"window_sec ({config['window_sec']}) must be one of windows_sec")
    config['windows_sec'] = sorted(windows)
    config['variants'] = variant_specs(config)
    config['ranking_modes'] = list(dict.fromkeys(mode for _, mode in config['variants']))
    approaches = (config['analysis']['baseline'], config['analysis']['focus'])
    if set(config['variants']) != {(a, m) for a in approaches for m in config['ranking_modes']}:
        raise ComparisonError(f"the approaches must be the same aggregations of {approaches[0]} and {approaches[1]}")
    if 'Linf' not in config['ranking_modes']:
        raise ComparisonError("the aggregations must include 'Linf'")
    return config


def scenario_labels(config):
    return [label for label, _ in config['scenarios']]


def rank_windows(config):
    """Rankings of every (Delta t, scenario) with run_experiments.py (stored reused)."""
    for window in config['windows_sec']:
        rank(config, window_sec=window)


def load_cells(config):
    """{(window, scenario, dataset): {(approach, mode): stored result}}."""
    cells = {}
    for window in config['windows_sec']:
        for (label, dataset), variants in collect_rankings(config, config['variants'], window).items():
            cells[(window, label, dataset)] = variants
    return cells


def graph_statistics(config):
    """Co-retweet graph per (window, dataset), from the baseline counts (c >= 1)."""
    cases = [(config['data_dir'], dataset, window)
             for window in config['windows_sec'] for dataset in config['datasets']]
    ties = coretweet_pair_statistics(cases).rename(columns={'window_sec': 'window'})
    return ties[['window', 'dataset', 'pairs_all', 'pairs_io', 'c1_share_all', 'io_total']]


def evaluate_cell(config, window, scenario, dataset, variants):
    """Score rows (one per variant) and gain rows (one per mode and metric) of one cell."""
    tag = f"dt={window}s / {scenario} / {dataset}"
    analysis = config['analysis']
    keys = list(variants)
    io_users = set().union(*(r['io_users'] for r in variants.values()))
    point = {k: compute_scores(variants[k]['suspicious_users'], set(variants[k]['io_users']), METRICS)
             for k in keys}
    try:
        universe, encoded, io_mask = encode_rankings({k: variants[k]['suspicious_users'] for k in keys}, io_users,
                                                     sort_users=False)
        checked_block_metrics(encoded, io_mask, METRICS, point)
    except ValueError as e:
        raise ComparisonError(f"[{tag}] {e}") from None

    scores = []
    for key in keys:
        result = variants[key]
        fc = users_until_reaching_io_fraction(result['suspicious_users'], io_users=set(result['io_users']),
                                              fraction=analysis['fc_fraction'])
        scores.append({'window': window, 'scenario': scenario, 'dataset': dataset,
                       'approach': key[0], 'mode': key[1], 'cohort': len(universe), 'n_io': int(io_mask.sum()),
                       **{m: point[key][m] for m in METRICS}, 'fc': fc})

    seed = block_seed(analysis['seed'], window, scenario_labels(config).index(scenario), dataset)
    boot = conditional_bootstrap(encoded, io_mask, keys, METRICS, analysis['n_boot'], np.random.default_rng(seed))
    print(f"  Bootstrap [{tag}]: {len(keys)} variants x {analysis['n_boot']} replicates")

    gains = []
    for mode in config['ranking_modes']:
        ib, iw = keys.index((analysis['baseline'], mode)), keys.index((analysis['focus'], mode))
        for im, metric in enumerate(METRICS):
            lo, hi = percentile_interval(100.0 * (boot[:, iw, im] - boot[:, ib, im]), analysis['ci_level'])
            gains.append({'window': window, 'scenario': scenario, 'dataset': dataset, 'mode': mode,
                          'metric': metric, 'gain': 100.0 * (point[keys[iw]][metric] - point[keys[ib]][metric]),
                          'lo': lo, 'hi': hi})
    return scores, gains


# ==========================================================================
# Tables and figure
# ==========================================================================

def fint(value):
    return f"{int(value):,}".replace(',', '\\,') if np.isfinite(value) else '--'


def fgain(row):
    """Signed gain with its interval; dagger if the interval contains 0."""
    value, lo, hi = float(row['gain']), float(row['lo']), float(row['hi'])
    if not np.isfinite(value):
        return '--'
    dagger = '^{\\dagger}' if contains_zero(lo, hi) else ''
    return f"${value:+.2f}\\ [{lo:+.2f}, {hi:+.2f}]{dagger}$"


def window_cell(window, config):
    return f"\\textbf{{{window}}}" if window == config['window_sec'] else str(window)


def interval_note(config):
    return (f" Brackets: {100 * config['analysis']['ci_level']:g}\\% bootstrap interval;"
            f" {DAGGER}: the interval contains zero.")


def group_header(lead_cols, groups, width):
    """One header row with group names spanning `width` columns each, plus its cmidrules."""
    top = ' & ' * lead_cols + ' & '.join(f"\\multicolumn{{{width}}}{{c}}{{{g}}}" for g in groups)
    rules = ' '.join(f"\\cmidrule(lr){{{lead_cols + 1 + width * i}-{lead_cols + width * (i + 1)}}}"
                     for i in range(len(groups)))
    return [top + ' \\\\', rules]


def write_main_table(scores, config, path):
    """
    Table of what the window changes in the data, for every dataset, scenario and Delta t: the size of the
    co-retweet graph (pairs with at least one co-retweet), the share of the IO accounts of the dataset that enter
    the cohort (IO found) and the share of IO accounts among the cohort users (IO share). The gains of the
    weighted approach are in the figure of plot_gains.
    """
    s = scores[(scores['approach'] == config['analysis']['baseline']) & (scores['mode'] == 'Linf')].set_index(
        ['scenario', 'dataset', 'window'])
    scenarios = scenario_labels(config)
    header = group_header(3, [tex_escape(label) for label in scenarios], 2)
    header.append('Dataset & $\\Delta t$ (s) & Pairs (M) & '
                  + ' & '.join(['IO found (\\%) & IO share (\\%)'] * len(scenarios)) + ' \\\\')
    body = []
    for i, dataset in enumerate(config['datasets']):
        if i:
            body.append('\\midrule')
        for j, window in enumerate(config['windows_sec']):
            pairs = s.loc[(scenarios[0], dataset, window), 'pairs_all']
            cells = [tex_escape(dataset) if j == 0 else '', window_cell(window, config), f"{pairs / 1e6:.2f}"]
            for label in scenarios:
                r = s.loc[(label, dataset, window)]
                cells += [f"{100 * r['coverage']:.1f}", f"{100 * r['prevalence']:.2f}"]
            body.append(' & '.join(cells) + ' \\\\')
    caption = (f"Effect of the synchronization window $\\Delta t$ on the data. Pairs: pairs of users with at "
               f"least one co-retweet, in millions, which set the size of the co-retweet graph and the cost of "
               f"the computation. IO found: share of the IO accounts of the dataset that enter the cohort. IO "
               f"share: share of IO accounts among the users of the cohort. The reference window "
               f"({config['window_sec']}~s) is in bold. The gains of $w_T$ over the count for every "
               f"$\\Delta t$ are in Figure~\\ref{{fig:dt_sensitivity}}.")
    write_lines(path, table_lines(caption, 'tab:dt_sensitivity', 'llr' + 'rr' * len(scenarios), header, body))


def write_app_gains_table(gains, config, path):
    """Table of the gains with intervals for every scenario and aggregation."""
    g = gains.set_index(['scenario', 'mode', 'dataset', 'window', 'metric'])
    body = []
    for scen in scenario_labels(config):
        for mode in config['ranking_modes']:
            if body:
                body.append('\\midrule')
            for i, dataset in enumerate(config['datasets']):
                for j, window in enumerate(config['windows_sec']):
                    lead = [tex_escape(scen) if i == 0 and j == 0 else '',
                            f"$p={p_tex(mode)}$" if i == 0 and j == 0 else '',
                            tex_escape(dataset) if j == 0 else '', window_cell(window, config)]
                    cells = [fgain(g.loc[(scen, mode, dataset, window, m)]) for m in METRICS]
                    body.append(' & '.join(lead + cells) + ' \\\\')
    header = ['Scenario & Aggregation & Dataset & $\\Delta t$ (s) & AUC & AP & NDCG \\\\']
    caption = ("Gain (pp) of the weighted approach over the baseline as a function of "
               "$\\Delta t$, for every scenario and aggregation." + interval_note(config))
    write_lines(path, table_lines(caption, 'tab:app_dt_gains', 'llllccc', header, body))


def write_app_scores_table(scores, config, path):
    """Table of the scores and Filtering Concentration, midrange aggregation."""
    s = scores[scores['mode'] == 'Linf'].set_index(['scenario', 'dataset', 'window', 'approach'])
    groups = ['AUC', 'AP', 'NDCG', f"FC@{100 * config['analysis']['fc_fraction']:g}\\%"]
    header = group_header(5, groups, 2)
    header.append('Scenario & Dataset & $\\Delta t$ (s) & Cohort & Prev.\\ (\\%) & '
                  + ' & '.join(['Base & Wtd'] * len(groups)) + ' \\\\')
    body = []
    for scen in scenario_labels(config):
        if body:
            body.append('\\midrule')
        for i, dataset in enumerate(config['datasets']):
            for j, window in enumerate(config['windows_sec']):
                b = s.loc[(scen, dataset, window, config['analysis']['baseline'])]
                w = s.loc[(scen, dataset, window, config['analysis']['focus'])]
                cells = [tex_escape(scen) if i == 0 and j == 0 else '',
                         tex_escape(dataset) if j == 0 else '', window_cell(window, config),
                         fint(b['cohort']), f"{100 * b['prevalence']:.2f}"]
                for m in METRICS:
                    cells += [f"{b[m]:.4f}", f"{w[m]:.4f}"]
                cells += [fint(round(b['fc'])), fint(round(w['fc']))]
                body.append(' & '.join(cells) + ' \\\\')
    caption = (f"Scores of the baseline (Base) and the weighted approach (Wtd) as a function of "
               f"$\\Delta t$, midrange aggregation. \\emph{{Prev.}}: share of IO accounts in the "
               f"cohort. \\emph{{FC}}: Filtering Concentration, users to inspect to recover "
               f"{100 * config['analysis']['fc_fraction']:g}\\% of the IO accounts of the cohort.")
    write_lines(path, table_lines(caption, 'tab:app_dt_scores', 'lllrr' + 'rr' * 4, header, body))


def plot_gains(gains, config, stem):
    """Gains against Delta t: rows (scenario, dataset), columns metrics, one curve per aggregation."""
    rows = [(s, d) for s in scenario_labels(config) for d in config['datasets']]
    windows = config['windows_sec']
    x = np.arange(len(windows))
    ref = windows.index(config['window_sec'])
    fig, axes = plt.subplots(len(rows), len(METRICS), figsize=(4.0 * len(METRICS), 2.5 * len(rows)),
                             squeeze=False)
    for r, (scen, dataset) in enumerate(rows):
        for c, metric in enumerate(METRICS):
            ax = axes[r, c]
            ax.axhline(0.0, color='0.3', linewidth=0.9, zorder=1)
            ax.axvline(ref, color='0.6', linestyle=':', linewidth=1.0, zorder=1)
            for i, mode in enumerate(config['ranking_modes']):
                style = MODE_STYLES.get(mode, {**FALLBACK_STYLES[i % len(FALLBACK_STYLES)],
                                               'label': f"p = {p_label(mode)}"})
                sub = (gains[(gains['scenario'] == scen) & (gains['dataset'] == dataset)
                             & (gains['metric'] == metric) & (gains['mode'] == mode)]
                       .set_index('window').reindex(windows))
                ax.fill_between(x, sub['lo'], sub['hi'], color=style['color'], alpha=BAND_ALPHA, lw=0)
                ax.plot(x, sub['gain'], markersize=3.5, linewidth=1.8, **style)
            ax.set_xticks(x)
            ax.set_xticklabels([str(w) for w in windows], fontsize=8)
            ax.tick_params(axis='y', labelsize=8)
            ax.grid(True, axis='y', alpha=0.3, linestyle='--')
            if r == 0:
                ax.set_title(METRIC_LABELS[metric], fontsize=11, fontweight='bold')
            if r == len(rows) - 1:
                ax.set_xlabel('Δt (s)', fontsize=9)
        axes[r, 0].set_ylabel(f"{scen}\n{dataset}\ngain (pp)", fontsize=9, fontweight='bold')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    title = (f"Bands: {100 * config['analysis']['ci_level']:g}% bootstrap interval.   Dotted line: "
             f"Δt = {config['window_sec']} s")
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, 0.0), title=title, title_fontsize=8.5)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    save_figure(fig, f"{stem}.png", f"{stem}.pdf", dpi=200)


# ==========================================================================
# Report
# ==========================================================================

def print_report(scores, gains, graph, config):
    print_section("Co-retweet graph per Delta t")
    print(graph.to_string(index=False))
    print_section("Gain intervals by sign")
    for (scen, mode), g in gains.groupby(['scenario', 'mode'], sort=False):
        above, below = int((g['lo'] > 0).sum()), int((g['hi'] < 0).sum())
        i = g['gain'].idxmin()
        print(f"{scen}, p={p_label(mode)}: {len(g)} cells (windows x datasets x metrics); "
              f"interval above zero {above}, containing zero {len(g) - above - below}, "
              f"below zero {below}; smallest gain {g.loc[i, 'gain']:+.2f} pp at "
              f"Delta t = {g.loc[i, 'window']} s, {g.loc[i, 'dataset']}, {g.loc[i, 'metric']}")
        notabove = g[~(g['lo'] > 0)]
        if not notabove.empty:
            print(notabove[['window', 'dataset', 'metric', 'gain', 'lo', 'hi']]
                  .to_string(index=False, float_format=lambda v: f"{v:+.2f}"))
    print_section(f"Check: Delta t = {config['window_sec']} s, midrange "
                  f"(must equal table1.tex of bin/article_main_comparison.py)")
    ref = scores[(scores['window'] == config['window_sec']) & (scores['mode'] == 'Linf')]
    print(ref[['scenario', 'dataset', 'approach', 'cohort', 'n_io'] + METRICS + ['fc']]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))


# ==========================================================================
# Entry point
# ==========================================================================

def analyze(config):
    out_dir = experiment_dir(config) / 'analysis'
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = load_cells(config)
    graph = graph_statistics(config)

    score_rows, gain_rows = [], []
    for window, scen, dataset in sorted(cells, key=str):
        s, g = evaluate_cell(config, window, scen, dataset, cells[(window, scen, dataset)])
        score_rows += s
        gain_rows += g

    scores = pd.DataFrame(score_rows).merge(graph, on=['window', 'dataset'], how='left')
    scores['prevalence'] = scores['n_io'] / scores['cohort']
    scores['coverage'] = scores['n_io'] / scores['io_total']
    gains = pd.DataFrame(gain_rows)

    print_report(scores, gains, graph, config)
    write_main_table(scores, config, out_dir / 'table_dt_main.tex')
    write_app_gains_table(gains, config, out_dir / 'table_app_dt_gains.tex')
    write_app_scores_table(scores, config, out_dir / 'table_app_dt_scores.tex')
    plot_gains(gains, config, out_dir / 'dt_sensitivity_gains')
    scores.to_csv(out_dir / 'dt_scores.csv', index=False, float_format='%.6f')
    gains.to_csv(out_dir / 'dt_gains.csv', index=False, float_format='%.6f')
    graph.to_csv(out_dir / 'dt_graph.csv', index=False, float_format='%.6f')
    print(f"\nOutputs saved to: {out_dir}")


def main():
    matplotlib.use('Agg')
    parser = argparse.ArgumentParser(description="Sensitivity of the gain to Delta t")
    parser.add_argument('step', choices=['rank', 'analyze', 'all'])
    parser.add_argument('config', help='Experiment JSON config')
    args = parser.parse_args()
    try:
        config = load_dt_config(args.config)
        if args.step in ('rank', 'all'):
            rank_windows(config)
        if args.step in ('analyze', 'all'):
            analyze(config)
    except ComparisonError as e:
        sys.exit(f"ERROR: {e}")


if __name__ == '__main__':
    main()
