# Batch Processing Scripts

This directory contains command-line scripts for batch processing and analysis of coordinated behavior detection experiments.

## Scripts Overview

### 1. `convert_to_pickle.py` - Dataset Cache Conversion

Converts text-based datasets to pickle format for 10-100x faster loading.

```bash
# Convert all datasets
uv run bin/convert_to_pickle.py

# Convert specific datasets
uv run bin/convert_to_pickle.py --datasets Armenia Catalonia Spain

# List cache status for all datasets
uv run bin/convert_to_pickle.py --list

# Force reconversion even if cache exists
uv run bin/convert_to_pickle.py --force

# Custom data directory
uv run bin/convert_to_pickle.py --data-dir /path/to/data
```

**When to use:**
- Before running large-scale experiments
- To speed up repeated dataset access
- Automatically creates `data_cache.pkl` in each dataset's Processed/ directory

### 2. `run_experiments.py` - Batch Experiment Runner

Processes multiple datasets with multiple detection approaches in batch mode.

```bash
# Run all experiments (all datasets × all approaches)
uv run bin/run_experiments.py

# Process specific datasets only
uv run bin/run_experiments.py --datasets Armenia Catalonia Spain

# Process specific approaches only
uv run bin/run_experiments.py --approaches coretweets ignoring_tweet

# List status of all combinations
uv run bin/run_experiments.py --list

# Force reprocessing (ignore existing results)
uv run bin/run_experiments.py --force

# Custom parameters
uv run bin/run_experiments.py --window-sec 120 --min-coactions 2

# Custom directories
uv run bin/run_experiments.py --data-dir ../data --output-dir my_results
```

**Output structure:**
```
results/
├── Armenia/
│   ├── coretweets.pkl
│   ├── ignoring_tweet.pkl
│   ├── shared_tweets.pkl
│   └── same_tweet_same_time.pkl
├── Catalonia/
│   └── ...
└── ...
```

Each `.pkl` file contains:
- Detection scores for all user pairs
- IO (inauthentic) user list
- Approach metadata and parameters
- Dataset name and statistics

### 3. `analyze_results.py` - Statistical Analysis

Performs comprehensive statistical analysis on experiment results.

```bash
# Analyze all results
uv run bin/analyze_results.py

# Custom directories
uv run bin/analyze_results.py --results-dir results --output-dir analysis

# Skip individual dataset plots (faster)
uv run bin/analyze_results.py --skip-plots

# Analyze specific datasets
uv run bin/analyze_results.py --datasets Armenia Catalonia Spain
```

**Generates:**
- Per-dataset comparison plots
- Overall ranking using Friedman test
- Nemenyi post-hoc analysis
- Critical difference diagrams
- Statistical significance reports

## Typical Workflow

1. **Convert datasets to pickle format** (one-time, optional but recommended):
   ```bash
   uv run bin/convert_to_pickle.py
   ```

2. **Run experiments** across all datasets and approaches:
   ```bash
   uv run bin/run_experiments.py
   ```

3. **Analyze results** with statistical tests:
   ```bash
   uv run bin/analyze_results.py
   ```

## Performance Tips

- **Use pickle caching**: Dramatically speeds up dataset loading
- **Run experiments in parallel**: Process different datasets simultaneously
- **Incremental processing**: `run_experiments.py` tracks completed combinations
- **Selective reprocessing**: Use `--datasets` and `--approaches` flags

## Available Approaches

- `coretweets`: Co-retweet counting (baseline)
- `ignoring_tweet`: Temporal synchronization ignoring content
- `shared_tweets`: Temporal synchronization on shared content
- `same_tweet_same_time`: Same tweet, same time (strongest signal)

## Available Datasets

Default datasets (21 total):
- Armenia, Catalonia, Ghana_Nigeria, Iran_1-6, Russia_1-5, Spain
- Venezuela_2, Thailand, Ecuador, Qatar, China_1-2

See `run_experiments.py` for complete list.
