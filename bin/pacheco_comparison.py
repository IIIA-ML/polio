#!/usr/bin/env python3
"""
The approaches of a comparison config against the co-retweet TF-IDF cosine similarity of Pacheco et al. (2021),
on the accounts that both can rank.

Usage:
    uv run bin/pacheco_comparison.py all     experiments/article_pacheco_comparison.json
    uv run bin/pacheco_comparison.py rank    <config>   rankings, one run_experiments.py run per scenario
    uv run bin/pacheco_comparison.py analyze <config>   scores, differences and tables

Config fields: those of every comparison (src/analysis/comparison.py: name, data_dir, output_dir, window_sec,
min_coactions, need_filtering, datasets, force, approaches), where the comparisons are every approach minus the
baseline, and
    pacheco     parameters of the baseline: support_threshold, min_df, top_n, edge_quantile
    analysis    n_boot, seed, ci_level (conditional bootstrap), jackknife_groups (graph-level jackknife)

Baseline (src/baselines/pacheco.py, as in the released code of Pacheco et al.): accounts with more than
support_threshold retweets and more than one distinct retweeted tweet, TF-IDF vectors of their retweeted tweets
(min_df), cosine similarity with the top_n neighbours of every account, no time window. Account score: its
largest similarity over its listed edges, which is the ranking implied by the edge filter of the method (an
account is flagged by the edge_quantile filter if and only if its score reaches the threshold).

Evaluated accounts of a block (scenario, dataset): the cohort of the scenario intersected with the accounts that
the baseline scores. The approaches keep their full-data scores (computed with all their partners, with the
aggregation of the config) and are only evaluated on these accounts; the baseline uses its own filtered
network, as published.

Statistics, as in bin/article_main_comparison.py:
    conditional bootstrap: the evaluated accounts redrawn (IO and controls separately), rankings fixed, paired;
        SeedSequence(seed, spawn_key=(scenario index, crc32(dataset), 3))
    graph-level jackknife: the tweet and account groups of bin/article_main_comparison.py (SeedSequence(seed,
        spawn_key=(scenario index, crc32(dataset), 2))); in every replicate the approaches
        (src/analysis/graph_jackknife.py) and the baseline are recomputed on the same reduced data; evaluated
        accounts fixed, removed accounts not evaluated, accounts left without a score ranked last
    claim: '+' or '-' if both 95% intervals exclude 0 on the same side, '0' otherwise
Operating point of the baseline: the accounts it flags with its edge filter on the full data; on the evaluated
accounts their number k_F gives the column P@k_F (the precision of the flagged accounts for the baseline, and of
the first k_F accounts for every approach).

Outputs ({output_dir}/{name}/)
    configs/S<i>.json, rankings/S<i>/results/   run_experiments.py configs and stored rankings
    analysis/scores.csv, diffs.csv (both intervals and the claim), blocks.csv (sizes and operating point),
    table_scores.tex, table_diffs.tex, summary.md, metadata.json,
    figure_diffs.png (differences in AUC, AP and NDCG with both intervals, as figure2.png of
    bin/article_main_comparison.py)
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from analysis import graph_jackknife as gj
from analysis.comparison import ComparisonError, collect, display_order, experiment_dir, load_config, rank
from analysis.metrics import block_metrics, blocks_from_groups, blocks_from_scores, ceil_users
from analysis.reporting import block_header, interval_pair_table_lines, results_table_lines, write_lines
from analysis.statistics import (
    block_seed,
    conditional_bootstrap,
    contains_zero,
    crossed_jackknife_interval,
    encode_rankings,
    interval_conclusion,
    paired_differences,
    percentile_interval,
)
from analysis.visualization import plot_difference_forest
from approaches._coretweet_index import clear_coretweet_index_cache
from baselines import pacheco
from data_loader import import_data

ANALYSIS_REQUIRED = ('n_boot', 'seed', 'ci_level', 'jackknife_groups')
PACHECO_KEYS = ('support_threshold', 'min_df', 'top_n', 'edge_quantile')
BASELINE = 'pacheco'
BASELINE_TEX = 'Pacheco et al.'
SCORE_METRICS = ['auc', 'ap', 'ndcg', 'ed', 'p100', 'p500', 'pF', 'fc90']
DIFF_METRICS = ['auc', 'ap', 'ndcg', 'pF', 'red90']
PP_METRICS = ('auc', 'ap', 'ndcg', 'p100', 'p500', 'pF')     # differences in percentage points
FIGURE_METRICS = ['auc', 'ap', 'ndcg']
FOREST_MARKERS = ['o', 's', 'D', '^', 'v', 'P', 'X', '*']
# Labels of the figure that differ from the tex label of the approach (method id: mathtext label), as in
# bin/article_main_comparison.py
FIGURE_LABELS = {'count': 'Baseline co-retweets', 'wT12h': r'$wc^T_{\Delta t}$'}


def comparison_id(a, b):
    return f"{a} - {b}"


def difference_label(a, b):
    """Mathtext label of 'a minus b', as in bin/article_main_comparison.py: '$w_T$', 'Count' -> '$w_T-$Count'."""
    left = a[:-1] if a.endswith('$') else a + '$'
    right = b[1:] if b.startswith('$') else '$' + b
    return f"{left}-{right}"


def metric_names(k_flagged):
    """Metric names of block_metrics, with 'pF' as P@k_F."""
    return [f"p{k_flagged}" if m == 'pF' else m for m in SCORE_METRICS]


def load_pacheco_config(path):
    """
    The comparison config, with 'methods_order' (the baseline, then the approaches in display order),
    'comparisons' (every approach, in config order, minus the baseline) and 'tex' (label of every method).
    """
    config = load_config(path, (ANALYSIS_REQUIRED, ()), sections=('pacheco',))
    missing = [k for k in PACHECO_KEYS if k not in config['pacheco']]
    unknown = set(config['pacheco']) - set(PACHECO_KEYS)
    if missing or unknown:
        raise ComparisonError(f"'pacheco' misses {missing} or has unknown keys {sorted(unknown)}")
    config['methods_order'] = [BASELINE] + [m['id'] for m in display_order(config)]
    config['comparisons'] = [(m['id'], BASELINE) for m in config['methods']]
    config['tex'] = {BASELINE: BASELINE_TEX, **{m['id']: m['tex'] for m in config['methods']}}
    return config


# ==========================================================================
# Rankings and baseline
# ==========================================================================

def groups_from_scores(scores, users, tol=gj.TIE_TOL):
    """Tie groups (best first) of `users` by their scores, merging scores within tol."""
    users = sorted(users)
    s = np.array([scores[u] for u in users])
    order = np.argsort(-s, kind='stable')
    groups, last = [], None
    for i in order:
        if last is None or last - s[i] > tol:
            groups.append([])
            last = s[i]
        groups[-1].append(users[i])
    return groups


def restrict(groups, keep):
    """Tie groups restricted to the users in keep (empty groups dropped)."""
    out = [[u for u in g if u in keep] for g in groups]
    return [g for g in out if g]


def baseline_scores(RTs, params):
    """({account: score} of the baseline, its edge list (accounts, u1, u2, similarity))."""
    edges = pacheco.interactions(RTs, params['support_threshold'], params['min_df'], params['top_n'])
    return pacheco.account_scores(*edges), edges


def baseline_replicate_metrics(RTs, keep_rt, evaluated, io, weights, params, names):
    """Metrics of the baseline on the retweets keep_rt, evaluated on `evaluated` with `weights`."""
    scores, _ = baseline_scores([rt for rt, k in zip(RTs, keep_rt) if k], params)
    s = np.array([scores.get(u, gj.BOTTOM_SCORE) for u in evaluated])
    return block_metrics(*blocks_from_scores(s, io, weights, gj.TIE_TOL), names)


# ==========================================================================
# One block
# ==========================================================================

def baseline_jackknife(RTs, g, groups, universe, io_mask, params, names, K):
    """{'tweets', 'users'}: (K, metrics) baseline metrics without each tweet group / account group."""
    tweet_group, user_group = groups
    user_pos = {u: i for i, u in enumerate(g.users.tolist())}
    tweet_pos = {t: i for i, t in enumerate(sorted({t for _, t, _ in RTs}))}
    if len(tweet_pos) != g.n_tweets:
        raise ComparisonError("the tweets of the graph are not those of the retweets")
    rt_user = np.array([user_pos[u] for u, _, _ in RTs])
    rt_tweet = np.array([tweet_pos[t] for _, t, _ in RTs])
    eval_idx = np.array([user_pos[u] for u in universe])
    out = {'tweets': np.empty((K, len(names))), 'users': np.empty((K, len(names)))}
    for j in range(K):
        out['tweets'][j] = baseline_replicate_metrics(RTs, tweet_group[rt_tweet] != j, universe, io_mask, None,
                                                      params, names)
        removed = user_group[eval_idx] == j
        out['users'][j] = baseline_replicate_metrics(RTs, user_group[rt_user] != j, universe, io_mask,
                                                     (~removed).astype(float), params, names)
    return out


def evaluate_block(config, index, label, min_coactions, dataset, variants, RTs, base_scores, flagged, g):
    tag = f"{label} / {dataset}"
    settings = config['analysis']
    methods = config['methods_order']
    approaches = [m['id'] for m in config['methods']]
    io_users = set(next(iter(variants.values()))['io_users'])
    rankings = {i: variants[i]['suspicious_users'] for i in approaches}
    cohort_users = {u for grp in rankings[approaches[0]] for u in grp}
    evaluated = sorted(cohort_users & set(base_scores))
    keep = set(evaluated)
    groups = {m: restrict(r, keep) for m, r in rankings.items()}
    groups[BASELINE] = groups_from_scores(base_scores, evaluated)
    universe, encoded, io_mask = encode_rankings(groups, io_users)
    flagged_eval = flagged & keep
    k_f = len(flagged_eval)
    names = metric_names(k_f)
    block = {'scenario': label, 'dataset': dataset, 'cohort': len(cohort_users),
             'cohort_io': len(cohort_users & io_users), 'n': len(universe), 'n_io': int(io_mask.sum()),
             'baseline_scored': len(base_scores), 'flagged': len(flagged), 'flagged_io': len(flagged & io_users),
             'flagged_in_evaluated': k_f, 'flagged_in_evaluated_io': len(flagged_eval & io_users)}
    print(f"  [{tag}] evaluated {block['n']} of {block['cohort']} cohort accounts ({block['n_io']} of "
          f"{block['cohort_io']} IO); baseline flags {k_f} of them ({block['flagged_in_evaluated_io']} IO)")

    point = {m: block_metrics(*blocks_from_groups(encoded[m][0], io_mask, encoded[m][1]), names) for m in methods}
    rows = [{'scenario': label, 'dataset': dataset, 'method': m, 'metric': name,
             'value': ceil_users(v) if name.startswith('fc') else float(v)}
            for m in methods for name, v in zip(SCORE_METRICS, point[m])]
    # The flagged accounts are the top k_F of the baseline ranking (ties at the threshold included)
    if k_f and not np.isclose(point[BASELINE][SCORE_METRICS.index('pF')], block['flagged_in_evaluated_io'] / k_f):
        raise ComparisonError(f"[{tag}] P@k_F of the baseline is not the precision of its flagged accounts")

    rng = np.random.default_rng(block_seed(settings['seed'], index, dataset, 3))
    boot = conditional_bootstrap(encoded, io_mask, methods, names, settings['n_boot'], rng)

    # Graph-level jackknife: the groups of bin/article_main_comparison.py, everything recomputed
    K = settings['jackknife_groups']
    cohort = gj.FixedCohort(g, min_coactions)
    if set(g.users[cohort.users].tolist()) != cohort_users:
        raise ComparisonError(f"[{tag}] the graph cohort differs from the stored cohort")
    start = time.time()
    jk_groups = gj.jackknife_groups(np.random.default_rng(block_seed(settings['seed'], index, dataset, 2)),
                                    g, cohort, K)
    eval_mask = np.isin(g.users[cohort.users], universe)
    jk = gj.grouped_jackknife(g, cohort, [m['ranker'] for m in config['methods']], names, jk_groups, K, eval_mask)
    base_jk = baseline_jackknife(RTs, g, jk_groups, universe, io_mask, config['pacheco'], names, K)
    print(f"    [{tag}] jackknife: {K} tweet groups + {K} account groups ({time.time() - start:.0f}s)")

    diffs = []
    for a, b in config['comparisons']:
        ia, ib = methods.index(a), methods.index(b)
        est = paired_differences(point[a], point[b], SCORE_METRICS, PP_METRICS)
        rep = paired_differences(boot[:, ia], boot[:, ib], SCORE_METRICS, PP_METRICS)
        by_t = paired_differences(jk['tweets'][:, approaches.index(a)], base_jk['tweets'], SCORE_METRICS, PP_METRICS)
        by_u = paired_differences(jk['users'][:, approaches.index(a)], base_jk['users'], SCORE_METRICS, PP_METRICS)
        for m in DIFF_METRICS:
            lo, hi = percentile_interval(rep[m], settings['ci_level'])
            jlo, jhi, se = crossed_jackknife_interval(float(est[m]), by_t[m], by_u[m], settings['ci_level'])
            concl_c, concl_j = interval_conclusion(lo, hi), interval_conclusion(jlo, jhi)
            diffs.append({'comparison': comparison_id(a, b), 'scenario': label, 'dataset': dataset, 'metric': m,
                          'estimate': float(est[m]), 'lo': lo, 'hi': hi, 'dagger': contains_zero(lo, hi),
                          'jk_lo': jlo, 'jk_hi': jhi, 'jk_se': se,
                          'claim': concl_c if concl_c == concl_j else '0',
                          'ddagger': concl_c != '0' and concl_c != concl_j})
    return block, rows, diffs


# ==========================================================================
# Outputs
# ==========================================================================

def scores_header(block, n_cols):
    return block_header(block, n_cols, unit='accounts', extra=f", $k_F={block['flagged_in_evaluated']}$")


def write_figure(config, blocks, diffs, out_dir):
    """
    figure_diffs.png: the differences of FIGURE_METRICS with both intervals, hollow if not robust, in the format
    of figure2.png of bin/article_main_comparison.py (each comparison in the colour of its approach).
    """
    colours = {m['id']: m['style']['color'] for m in config['methods']}
    labels = {**config['tex'], **FIGURE_LABELS}
    styles = {comparison_id(a, b): {'label': difference_label(labels[a], labels[b]),
                                    'color': colours[a],
                                    'marker': FOREST_MARKERS[n % len(FOREST_MARKERS)]}
              for n, (a, b) in enumerate(config['comparisons'])}
    forest = diffs.rename(columns={'jk_lo': 'lo2', 'jk_hi': 'hi2'})
    forest['dagger'] = forest['claim'] == '0'
    datasets = sorted({b['dataset'] for b in blocks})
    row_groups = [[(s, d) for s, _ in config['scenarios']] for d in datasets]    # one row per campaign
    plot_difference_forest(forest, [(b['scenario'], b['dataset']) for b in blocks], list(styles), styles,
                           FIGURE_METRICS, out_dir / 'figure_diffs.png',
                           metric_labels={'auc': 'AUC', 'ap': 'AP', 'ndcg': 'NDCG'}, xlabel='Difference (pp)',
                           hollow_label='Hollow: not robust (one of its intervals contains 0)',
                           interval_labels=('95% bootstrap of the evaluated accounts', '95% graph-level jackknife'),
                           row_groups=row_groups, marker_size=6, panel_width=2.6, fontsize=12, legend_ncol=3)


def write_outputs(config, config_path, blocks, scores, diffs, out_dir):
    params, tex = config['pacheco'], config['tex']
    pd.DataFrame(blocks).to_csv(out_dir / 'blocks.csv', index=False)
    scores.to_csv(out_dir / 'scores.csv', index=False, float_format='%.6f')
    diffs.to_csv(out_dir / 'diffs.csv', index=False, float_format='%.6f')

    write_lines(out_dir / 'table_scores.tex', results_table_lines(
        scores, blocks, SCORE_METRICS, [(m, tex[m]) for m in config['methods_order']],
        "Comparison with the co-retweet TF-IDF cosine similarity of Pacheco et al.\\ on the accounts that both "
        f"rank (cohort accounts with more than {params['support_threshold']} retweets and at least one similarity "
        f"edge). $k_F$: number of these accounts flagged by its {params['edge_quantile']} edge filter, so its "
        "P@$k_F$ is the precision of the flagged accounts.",
        'tab:pacheco_scores', header_row=scores_header))
    graph = diffs.drop(columns=['lo', 'hi', 'dagger']).rename(columns={'jk_lo': 'lo', 'jk_hi': 'hi'})
    graph['dagger'] = [contains_zero(lo, hi) for lo, hi in zip(graph['lo'], graph['hi'])]
    write_lines(out_dir / 'table_diffs.tex', interval_pair_table_lines(
        diffs, graph, blocks, [(comparison_id(a, b), tex[a]) for a, b in config['comparisons']],
        DIFF_METRICS, ('Accounts', 'Graph'), "AUC, AP, NDCG and P@$k_F$ in percentage points.",
        f"Differences of every approach minus the baseline of {BASELINE_TEX}", 'tab:pacheco_diffs'))
    write_figure(config, blocks, diffs, out_dir)

    lines = ['# Comparison with Pacheco et al. (2021)', '', '## Blocks', '',
             pd.DataFrame(blocks).to_string(index=False), '', '## Differences (claim: both intervals)', '',
             diffs.to_string(index=False, float_format=lambda v: f'{v:.3f}'), '']
    (out_dir / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    meta = {'command': sys.argv, 'config': json.loads(Path(config_path).read_text(encoding='utf-8')),
            'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__,
            'pandas': pd.__version__}
    (out_dir / 'metadata.json').write_text(json.dumps(meta, indent=2, default=str) + '\n', encoding='utf-8')
    print(f"Saved outputs to {out_dir}")
    print('\n'.join(lines))


def analyze(config, config_path):
    params = config['pacheco']
    out_dir = experiment_dir(config) / 'analysis'
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = collect(config)

    blocks, score_rows, diff_rows = [], [], []
    for dataset in config['datasets']:
        RTs, _ = import_data(f"{config['data_dir']}/{dataset}/Processed/")
        start = time.time()
        base_scores, edges = baseline_scores(RTs, params)
        flagged, threshold = pacheco.flagged_accounts(*edges, params['edge_quantile'])
        print(f"[{dataset}] baseline: {len(edges[0])} accounts with a document, {len(edges[3])} listed edges, "
              f"{len(base_scores)} scored, {len(flagged)} flagged (similarity >= {threshold:.4f}) "
              f"({time.time() - start:.0f}s)")
        io_users = set().union(*[next(iter(cells[(label, dataset)].values()))['io_users']
                                 for label, _ in config['scenarios']])
        g = gj.SyncGraph(RTs, io_users, config['window_sec'])
        clear_coretweet_index_cache()
        for index, (label, min_coactions) in enumerate(config['scenarios']):
            block, rows, diffs = evaluate_block(config, index, label, min_coactions, dataset, cells[(label, dataset)],
                                                RTs, base_scores, flagged, g)
            block['edge_threshold'] = threshold
            blocks.append(block)
            score_rows += rows
            diff_rows += diffs
        del g, RTs

    order = {label: i for i, (label, _) in enumerate(config['scenarios'])}
    blocks.sort(key=lambda b: (order[b['scenario']], b['dataset']))
    write_outputs(config, config_path, blocks, pd.DataFrame(score_rows), pd.DataFrame(diff_rows), out_dir)


def main():
    matplotlib.use('Agg')
    parser = argparse.ArgumentParser(description="Approaches of a config against Pacheco et al. (2021)")
    parser.add_argument('step', choices=['rank', 'analyze', 'all'])
    parser.add_argument('config', help='Experiment JSON config')
    args = parser.parse_args()
    try:
        config = load_pacheco_config(args.config)
        if args.step in ('rank', 'all'):
            rank(config)
        if args.step in ('analyze', 'all'):
            analyze(config, args.config)
    except ComparisonError as e:
        sys.exit(f"ERROR: {e}")


if __name__ == '__main__':
    main()
