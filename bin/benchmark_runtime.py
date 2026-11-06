#!/usr/bin/env python3
"""
Runtime and memory of the approaches of a comparison config.

Usage:
    uv run bin/benchmark_runtime.py experiments/article_main_comparison_Linf.json [--repeats 3]

The config is that of a comparison (src/analysis/comparison.py, e.g. bin/article_main_comparison.py; its
'analysis' block is not used): every approach, with its aggregation, is measured on every dataset and scenario
(min_coactions). Each (scenario, dataset, approach, repeat) runs in a fresh Python process, so caches and memory
do not carry over. The process loads the retweets (not timed), then computes the pair scores and the ranking
with the approach (timed). It reports the wall time of that step and the peak resident memory of the process
(data included). Results are printed and written to {output_dir}/{name}/analysis/runtime.csv and
runtime_summary.csv (medians over the repeats).
"""

import argparse
import json
import platform
import subprocess
import sys

import pandas as pd

from analysis.comparison import ComparisonError, experiment_dir, load_config, scenario_specs

CHILD = r"""
import json, resource, time
from approaches import create_approach
from data_loader import import_data
RTs, io_users = import_data({data!r})
approach = create_approach({spec!r}, {window})
start = time.perf_counter()
groups = approach.get_suspicious(RTs)
seconds = time.perf_counter() - start
print(json.dumps({{'seconds': seconds, 'peak_rss_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  'n_retweets': len(RTs), 'n_users_ranked': sum(len(g) for g in groups)}}))
"""


def measure(data_dir, dataset, spec, window):
    """Measurements of one run of an approach, in a fresh process."""
    code = CHILD.format(data=f"{data_dir}/{dataset}/Processed/", spec=spec, window=window)
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def main():
    parser = argparse.ArgumentParser(description="Runtime and memory of the approaches of a comparison config")
    parser.add_argument('config', help='Main-comparison JSON config (see bin/article_main_comparison.py)')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    try:
        config = load_config(args.config)
    except ComparisonError as e:
        sys.exit(f"ERROR: {e}")

    rows = []
    for label, min_coactions in config['scenarios']:
        for dataset in config['datasets']:
            for method, spec in zip(config['methods'], scenario_specs(config, min_coactions)):
                for r in range(args.repeats):
                    row = measure(config['data_dir'], dataset, spec, config['window_sec'])
                    rows.append({'scenario': label, 'dataset': dataset, 'method': method['id'], 'repeat': r, **row})
                    print(f"{label} {dataset:9s} {method['id']:6s} run {r}: {row['seconds']:.1f} s, "
                          f"{row['peak_rss_mb']:.0f} MB", flush=True)
    df = pd.DataFrame(rows)
    summary = (df.groupby(['scenario', 'dataset', 'method'], sort=False)
               .agg(n_retweets=('n_retweets', 'first'), users_ranked=('n_users_ranked', 'first'),
                    seconds_median=('seconds', 'median'), peak_rss_mb_median=('peak_rss_mb', 'median'))
               .reset_index())
    print(summary.to_string(index=False))
    out_dir = experiment_dir(config) / 'analysis'
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / 'runtime.csv', index=False)
    summary.to_csv(out_dir / 'runtime_summary.csv', index=False)
    print(f"Saved: {out_dir / 'runtime.csv'} ({platform.processor() or platform.machine()}, "
          f"{platform.python_version()})")


if __name__ == '__main__':
    main()
