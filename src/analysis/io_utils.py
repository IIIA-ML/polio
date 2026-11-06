"""Experiment configs, stored rankings (written by bin/run_experiments.py) and runs of that script."""

import json
import pickle
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Top-level config keys that act as defaults for every approach spec
APPROACH_DEFAULT_KEYS = ('min_coactions', 'ranking_mode', 'need_filtering')


def apply_approach_defaults(config):
    """
    Fill the top-level min_coactions / ranking_mode / need_filtering into every approach spec that does not set
    them (ranking_mode not for the lexicographic approach, whose modes are per sub-approach). Returns config,
    modified in place.
    """
    defaults = {k: config[k] for k in APPROACH_DEFAULT_KEYS if config.get(k) is not None}
    if not defaults or not config.get('approaches'):
        return config
    specs = []
    for spec in config['approaches']:
        spec = dict(spec)
        for key, value in defaults.items():
            if not (key == 'ranking_mode' and spec.get('name') == 'lexicographic'):
                spec.setdefault(key, value)
        specs.append(spec)
    config['approaches'] = specs
    return config


def load_experiment_config(config_path):
    """Config of bin/run_experiments.py, with the top-level defaults applied to the approach specs."""
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    if 'name' not in config:
        raise ValueError(f"{config_path}: the config must include a 'name'")
    return apply_approach_defaults(config)


def experiment_dir(config, config_path):
    """{output_dir}/{name}, or {directory of the config}/{name} if output_dir is null."""
    if config.get('output_dir') is None:
        return Path(config_path).resolve().parent / config['name']
    return Path(config['output_dir']) / config['name']


def load_stored_results(results_dir, key):
    """{dataset: stored result of the approach with storage key `key`}, datasets in directory order."""
    results = {}
    results_path = Path(results_dir)
    if not results_path.exists():
        return results
    for dataset_dir in results_path.iterdir():
        path = dataset_dir / f"{key}.pkl"
        if path.exists():
            with open(path, 'rb') as f:
                results[dataset_dir.name] = pickle.load(f)
    return results


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')


def run_experiments(config_path):
    """Run bin/run_experiments.py on a config (stored rankings are reused)."""
    command = [sys.executable, str(PROJECT_ROOT / 'bin' / 'run_experiments.py'), str(config_path)]
    print(f"\n$ {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)
