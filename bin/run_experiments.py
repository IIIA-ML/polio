#!/usr/bin/env python3
"""
Rankings of every (dataset, approach) of an experiment config, stored in
{output_dir}/{name}/results/{dataset}/{approach key}.pkl (output_dir null: next to the config).

Stored rankings are reused unless "force" is true. Approaches that differ only in their aggregation share their
pair scores, which are computed once per dataset.

Usage:
    uv run bin/run_experiments.py <config.json>

Config fields
    name, data_dir, datasets, approaches   required (approach specs: src/approaches/factory.py)
    output_dir, window_sec, force          optional (null, 60, false)
    min_coactions, ranking_mode, need_filtering   optional defaults of every approach spec
"""

import argparse
from collections import Counter

from analysis.io_utils import experiment_dir, load_experiment_config
from approaches import PairsApproach, create_approach
from approaches._coretweet_index import clear_coretweet_index_cache
from approaches.base import group_scores_by_user
from data_loader import import_data

STATUS_TEXT = {'processed': '✓ Processed and saved', 'skipped': '⊘ Skipped (already processed)',
               'empty': '✗ Skipped (no results)'}


class DatasetLoader:
    """Loads a dataset at most once, and only when a ranking has to be computed."""

    def __init__(self, processed_dir):
        self.processed_dir = processed_dir
        self._data = None

    def load(self):
        """(RTs, io_users)."""
        if self._data is None:
            self._data = import_data(self.processed_dir)
        return self._data


def group_approaches(approaches):
    """Approaches grouped by their pair scores (pairs_signature), in order; any other approach alone."""
    groups = {}
    for i, approach in enumerate(approaches):
        key = approach.pairs_signature() if isinstance(approach, PairsApproach) else ('single', i)
        groups.setdefault(key, []).append(approach)
    return list(groups.values())


def rank_group(approaches, dataset, loader, output_dir, force):
    """Yield (approach, status) for a group of group_approaches, computing and storing the missing rankings."""
    pending = []
    for approach in approaches:
        if not force and approach.is_result_cached(output_dir, dataset):
            yield approach, 'skipped'
        else:
            pending.append(approach)
    if not pending:
        return
    RTs, io_users = loader.load()

    if not isinstance(pending[0], PairsApproach):
        approach = pending[0]
        suspicious = approach.get_suspicious(RTs, output_dir=output_dir, dataset=dataset, io_users=io_users,
                                             force=force)
        yield approach, 'processed' if suspicious else 'empty'
        return

    first = pending[0]
    print(f"\n  Computing pair scores of {first.get_approach_name()} (shared by {len(pending)} ranking mode(s))")
    pairs_scores = first.compute_pairs_scores(first.prepare_RTs(RTs))
    user_pair_scores = group_scores_by_user(pairs_scores) if pairs_scores else None
    for approach in pending:
        suspicious = approach.rank_from_user_pair_scores(user_pair_scores) if user_pair_scores else []
        if suspicious:
            approach.save_result(output_dir, dataset, io_users, suspicious)
        yield approach, 'processed' if suspicious else 'empty'


def load_run_config(path):
    config = load_experiment_config(path)
    missing = [k for k in ('data_dir', 'datasets', 'approaches') if k not in config]
    if missing:
        raise SystemExit(f"Error: {path} misses {missing}")
    config.setdefault('output_dir', None)
    config.setdefault('window_sec', 60)
    config.setdefault('force', False)
    return config


def main():
    parser = argparse.ArgumentParser(description="Compute and store the rankings of an experiment config")
    parser.add_argument('config', help='Path to the JSON configuration file')
    args = parser.parse_args()

    config = load_run_config(args.config)
    output_dir = experiment_dir(config, args.config)
    output_dir.mkdir(parents=True, exist_ok=True)
    approaches = [create_approach(spec, config['window_sec']) for spec in config['approaches']]

    print("=" * 80 + "\nRUNNING EXPERIMENTS\n" + "=" * 80)
    print(f"\nExperiment name: {config['name']}\n\nConfiguration:")
    print(f"  Data directory: {config['data_dir']}")
    print(f"  Output directory: {output_dir}")
    print(f"  Window: {config['window_sec']} seconds")
    print(f"  Force reprocess: {config['force']}")
    print(f"  Datasets: {len(config['datasets'])}")
    print(f"  Approaches: {len(approaches)}")
    for approach in approaches:
        print(f"    {approach.get_approach_name()}: ranking_mode={approach.ranking_mode}, "
              f"min_coactions={approach.min_coactions}")

    counts = Counter()
    groups = group_approaches(approaches)
    for dataset in config['datasets']:
        print(f"\n{'=' * 80}\nDataset: {dataset}\n{'=' * 80}")
        loader = DatasetLoader(f"{config['data_dir']}/{dataset}/Processed/")
        for group in groups:
            for approach, status in rank_group(group, dataset, loader, output_dir, config['force']):
                print(f"\n  Approach: {approach.get_approach_name()}\n    {STATUS_TEXT[status]}")
                counts[status] += 1
        clear_coretweet_index_cache()      # release the per-dataset caches before the next dataset

    print(f"\n{'=' * 80}\nSUMMARY\n{'=' * 80}")
    print(f"  Processed: {counts['processed']}")
    print(f"  Skipped (already done): {counts['skipped']}")
    print(f"  Skipped (no results): {counts['empty']}")
    print(f"  Total combinations: {len(config['datasets']) * len(approaches)}")


if __name__ == '__main__':
    main()
