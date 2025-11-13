# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains research code for detecting coordinated inauthentic behavior on social media through synchronous repeated actions analysis. The primary focus is analyzing retweet patterns to identify users who repeatedly interact with content at similar times, which may indicate coordinated manipulation campaigns.

## Project Structure

- **[src/](src/)**: Core analysis module containing the detection algorithms
  - [synchronous_repeated_detection.py](src/synchronous_repeated_detection.py): Main library with all detection functions
  - [approaches/](src/approaches/)**: Detection approach implementations using factory pattern
    - [factory.py](src/approaches/factory.py): ApproachFactory for creating approach instances
    - [coretweets.py](src/approaches/coretweets.py), [ignoring_tweet_fast.py](src/approaches/ignoring_tweet_fast.py), etc.
- **[bin/](bin/)**: Command-line scripts for running experiments and analysis
  - [run_experiments.py](bin/run_experiments.py): Batch experiment execution using JSON configs
  - [analyze_results.py](bin/analyze_results.py): Statistical analysis and visualization of results
- **[experiments/](experiments/)**: Experiment configuration files and results
  - JSON config files define experiment parameters
  - Results stored in `{experiment_name}/results/` subdirectories
  - Analysis outputs in `{experiment_name}/analysis/` subdirectories
- **[nb/](nb/)**: Jupyter notebooks for interactive exploration
  - [synchronous_repeated_detection.ipynb](nb/synchronous_repeated_detection.ipynb): Main experimental notebook

## Core Architecture

### Data Format

The code expects preprocessed data in a specific directory structure (`../../Labeled_Datasets/{dataset}/Processed/`) containing:
- `data.txt`: User ID mappings (index_to_accountid dictionary)
- `io_users.txt`: List of known inauthentic operation (IO) users
- `RTs.txt`: Retweet data as list of tuples (user_id, tweet_id, timestamp)

### Detection Approaches

The codebase implements multiple detection methods using a factory pattern ([approaches/factory.py](src/approaches/factory.py)):

#### Available Approaches

1. **`coretweets`** - Co-retweet Counting
   - Detects user pairs who retweeted the same tweet within a time window
   - Classic baseline approach
   - Returns groups of users sorted by co-retweet score

2. **`coretweets_numpy`** - Co-retweet Counting (NumPy Optimized)
   - Same algorithm as `coretweets` but with NumPy vectorization
   - Provides 2-5x speedup on large datasets with viral tweets
   - Recommended for performance-critical experiments

3. **`ignoring_tweet`** - Temporal Synchronization (Content-Agnostic)
   - Groups users by days where they posted within a time window
   - Ignores tweet content entirely
   - Detects temporal coordination patterns

4. **`ignoring_tweet_fast`** - Fast Temporal Synchronization
   - Optimized version of `ignoring_tweet`
   - Filters to users who appeared in `coretweets` results first
   - Significantly faster while maintaining detection quality

5. **`shared_tweets`** - Temporal Synchronization on Shared Content
   - Days where users retweeted shared content within a time window
   - Combines content and timing signals

6. **`same_tweet_same_time`** - Same Tweet, Same Time
   - Strongest signal: users who co-retweeted identical tweets within a time window
   - Groups by specific tweet and day
   - Most precise but may miss broader coordination patterns

7. **`coretweets_fast`** - Fast Co-retweet Variant
   - Similar optimization strategy as `ignoring_tweet_fast`

**Performance Notes:**
- `_fast` variants pre-filter using coretweets results for improved performance
- `_numpy` variants use vectorized operations for computational efficiency
- Benchmark different approaches with `examples/benchmark_*.py` scripts

### Two-Pointer Temporal Matching

The core temporal coincidence detection uses an efficient two-pointer algorithm (`count_daily_coincidences`) that scans through sorted timestamp lists to find temporal overlaps within a configurable window (default 60 seconds).

### Evaluation Methodology

Detection quality is evaluated using a progressive user discovery curve:
- User pairs are sorted by detection score (descending)
- Users are collected greedily from high-scoring pairs
- Progress is measured as "IO users detected" vs "total users examined"
- The `_compute_curve` function handles equal-score pairs correctly by aggregating users introduced at each score level
- Performance is measured by area under the curve compared to an ideal detector

The `plot_comparison` function compares two detection methods and calculates:
- Total area difference between curves
- Negative area (where baseline outperforms the test method)

### Neural Network Approach

The notebook includes a cross-validation experiment training a linear model to combine three detection approaches:
- Score = a × ignoring_tweet + b × shared_tweets + 1.0 × same_tweet_same_time
- Loss function: `mean(score(non_io)) - mean(score(io)) - 0.5 × mean(score(half_io))`
  - Minimizes scores for IO user pairs
  - Maximizes scores for non-IO pairs
  - Half-IO pairs (one IO, one legitimate) get intermediate treatment
- Uses 3-dataset training sets with remaining datasets as test sets
- Evaluates using normalized area difference to ideal detector

## Environment Setup with UV

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management.

### Initial Setup

Create the virtual environment and install all dependencies:
```bash
# Create virtual environment and install dependencies in one step
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On Linux/macOS
# Or on Windows: .venv\Scripts\activate
```

### Installing Additional Dependencies

```bash
# Add a new dependency
uv add <package-name>

# Add a development dependency
uv add --dev <package-name>

# Install with development dependencies
uv sync --all-extras
```

### Syncing Dependencies

After pulling changes or modifying `pyproject.toml`:
```bash
uv sync
```

## Running the Code

### Dataset Loading Performance

The codebase implements automatic pickle caching for dataset loading, providing **10-100x speedup** for repeated loads:

**How it works:**
- First load: Parses text files (slow, uses `eval()`)
- Creates `data_cache.pkl` automatically
- Subsequent loads: Uses pickle cache (very fast)
- Cache automatically invalidated if source files change

**Usage:**
```python
from synchronous_repeated_detection import import_data

# Default: automatic caching (recommended)
RTs, io_users = import_data("../data/Armenia/Processed/")

# Disable caching (use original text parsing)
RTs, io_users = import_data("../data/Armenia/Processed/", use_cache=False)

# Force reload from text files and recreate cache
RTs, io_users = import_data("../data/Armenia/Processed/", force_reload=True)
```

**Batch conversion:** Convert all datasets to pickle format upfront:
```bash
# Convert all datasets
uv run bin/convert_to_pickle.py

# Convert specific datasets
uv run bin/convert_to_pickle.py --datasets Armenia Catalonia Spain

# List cache status
uv run bin/convert_to_pickle.py --list

# Force reconversion
uv run bin/convert_to_pickle.py --force
```

**Test performance improvement:**
```bash
uv run examples/test_pickle_performance.py ../data/Armenia/Processed/
```

This is especially beneficial when running experiments with multiple approaches, as each approach loads the same dataset.

### Experiment Workflow (Recommended)

The recommended workflow uses JSON configuration files to define and run experiments. This provides:
- Reproducible experiment definitions
- Organized results with clear experiment naming
- Automatic directory structure creation
- Integration with statistical analysis tools

#### 1. Create Experiment Configuration

Create a JSON file in the `experiments/` directory (e.g., `experiments/baseline_comparison.json`):

```json
{
  "name": "baseline_comparison",
  "data_dir": "data",
  "window_sec": 60,
  "min_coactions": 1,
  "approaches": ["coretweets", "ignoring_tweet_fast"],
  "datasets": ["Armenia", "Catalonia", "Spain"],
  "force": false
}
```

**Configuration fields:**
- `name` (required): Experiment name (creates directory with this name)
- `data_dir` (optional): Path to datasets directory (default: `"../data"`)
- `output_dir` (optional): Custom output directory (default: `null` creates `{json_dir}/{name}/`)
- `window_sec` (optional): Time window in seconds (default: 60)
- `min_coactions` (optional): Minimum co-actions threshold (default: 1)
- `approaches` (optional): List of approaches to run (default: `null` = all approaches)
- `datasets` (optional): List of datasets to process (default: `null` = all datasets)
- `force` (optional): Reprocess existing results (default: `false`)

#### 2. Run Experiment

```bash
# Run the experiment defined in the JSON file
uv run bin/run_experiments.py experiments/baseline_comparison.json

# Check which dataset+approach combinations have been processed
uv run bin/run_experiments.py experiments/baseline_comparison.json --list
```

**Experiment Directory Structure:**

When `output_dir` is `null`, the script creates the following structure in the same directory as the JSON file:

```
experiments/
├── baseline_comparison.json          # Experiment configuration
└── baseline_comparison/               # Created automatically
    └── results/
        ├── Armenia/
        │   ├── coretweets.pkl
        │   └── ignoring_tweet_fast.pkl
        ├── Catalonia/
        │   └── ...
        └── Spain/
            └── ...
```

Each `.pkl` file contains:
- `dataset`: Dataset name
- `suspicious_users`: List of lists of suspicious user groups
- `io_users`: List of known inauthentic users
- `num_suspicious_groups`: Number of score-based groups
- `num_suspicious_users`: Total suspicious users detected
- Plus approach-specific metadata (approach name, parameters, etc.)

#### 3. Analyze Results

The `bin/analyze_results.py` script performs comprehensive statistical analysis on experiment results:

```bash
# Analyze the experiment results
uv run bin/analyze_results.py experiments/baseline_comparison.json

# Skip individual dataset plots (faster)
uv run bin/analyze_results.py experiments/baseline_comparison.json --no-plots
```

The script:
1. Loads the experiment configuration from the JSON file
2. Reads results from `{json_dir}/{experiment_name}/results/`
3. Saves analysis to `{json_dir}/{experiment_name}/analysis/`

**Final Directory Structure:**

```
experiments/
├── baseline_comparison.json
└── baseline_comparison/
    ├── results/                # Created by run_experiments.py
    │   ├── Armenia/
    │   │   ├── coretweets.pkl
    │   │   └── ignoring_tweet_fast.pkl
    │   ├── Catalonia/
    │   └── Spain/
    └── analysis/               # Created by analyze_results.py
        ├── plots/
        │   ├── Armenia_comparison.png
        │   ├── Catalonia_comparison.png
        │   └── Spain_comparison.png
        ├── critical_difference_diagram.png  # Only for 3+ methods
        └── summary.txt
```

**Analysis Outputs:**

1. **Individual Dataset Plots** (`analysis/plots/{dataset}_comparison.png`):
   - Comparison of each method vs baseline (first method)
   - Performance rankings by area under curve (AUC)
   - Dataset statistics

2. **Statistical Analysis**:
   - **For 2 methods**: Wilcoxon signed-rank test (paired comparison on AUC scores)
   - **For 3+ methods**: Friedman test + Nemenyi post-hoc test (rank-based)
   - Results include p-values, test statistics, and significance indicators

3. **Summary Report** (`analysis/summary.txt`):
   - Experiment name and configuration
   - Average AUC scores and rankings
   - Statistical test results
   - Winner identification (for 2-method comparisons)

**Note:** The critical difference diagram is only generated when comparing 3 or more methods. For 2-method comparisons, the Wilcoxon test provides a direct statistical comparison.

### Jupyter Notebook

To run the main experiments with visualization:
```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Start Jupyter
jupyter notebook nb/synchronous_repeated_detection.ipynb
```

The notebook is configured to iterate over multiple datasets defined in the `datasets` tuple. It compares different detection approaches and generates comparison plots.

The `data_dir` variable at the top of the notebook can be modified to point to your dataset location (default: `../data`).

### Python Module

The detection functions can be imported and used directly:
```python
from synchronous_repeated_detection import (
    import_data,
    count_coretweets,
    coincide_ignoring_tweet,
    plot_comparison
)

# Load data
RTs, io_users = import_data("../data/Armenia/Processed/")

# Compare detection methods
pairs_coretweets = count_coretweets(RTs, window_sec=60, min_coactions=1)
pairs_ignoring = coincide_ignoring_tweet(RTs, window_sec=60)

plot_comparison(pairs_ignoring, pairs_coretweets, io_users)
```

## Key Parameters

- `window_sec`: Time window in seconds for considering actions as synchronous (default: 60)
- `min_coactions`: Minimum number of co-retweets required to consider a user pair (default: 1)

## Dependencies

Dependencies are managed via [pyproject.toml](pyproject.toml):
- numpy >= 1.20.0
- matplotlib >= 3.3.0
- scipy >= 1.7.0 (for statistical analysis)
- torch >= 2.0.0 (for neural network experiments)
- jupyter >= 1.0.0
- ipykernel >= 6.0.0

Development dependencies (optional):
- pytest >= 7.0.0
- ruff >= 0.1.0

## Data Path Configuration

Datasets are expected in the following structure:
```
{data_dir}/
  ├── Armenia/Processed/
  │   ├── data.txt
  │   ├── io_users.txt
  │   └── RTs.txt
  ├── Catalonia/Processed/
  │   └── ...
  └── ...
```

The default data directory is `../data` (relative to project root). You can customize this:
- In the notebook: modify the `data_dir` variable in the second cell
- In experiment JSON: set the `data_dir` field
- In Python code: pass the full path to `import_data()`

## Quick Reference

### Running a Complete Experiment

```bash
# 1. Create experiment configuration
cat > experiments/my_experiment.json << 'EOF'
{
  "name": "my_experiment",
  "data_dir": "data",
  "approaches": ["coretweets", "ignoring_tweet_fast"],
  "datasets": null,
  "window_sec": 60,
  "min_coactions": 1
}
EOF

# 2. Run the experiment
uv run bin/run_experiments.py experiments/my_experiment.json

# 3. Analyze results
uv run bin/analyze_results.py experiments/my_experiment.json

# Results will be in:
# - experiments/my_experiment/results/
# - experiments/my_experiment/analysis/
```

### Available Approach Keys

Use these in the `approaches` field of your experiment JSON:
- `coretweets` - Standard co-retweet counting
- `coretweets_numpy` - Optimized co-retweet counting
- `coretweets_fast` - Fast co-retweet variant
- `ignoring_tweet` - Temporal sync (content-agnostic)
- `ignoring_tweet_fast` - Fast temporal sync
- `shared_tweets` - Temporal sync on shared content
- `same_tweet_same_time` - Same tweet, same time

### Common Workflows

**Compare two methods across all datasets:**
```json
{
  "name": "method_comparison",
  "approaches": ["coretweets", "ignoring_tweet_fast"],
  "datasets": null
}
```

**Test one method on specific datasets:**
```json
{
  "name": "single_method_test",
  "approaches": ["coretweets_numpy"],
  "datasets": ["Armenia", "Catalonia", "Spain"]
}
```

**Parameter sensitivity analysis:**
```json
{
  "name": "window_60sec",
  "approaches": ["coretweets"],
  "window_sec": 60
}
```

Then create `window_120sec.json`, `window_180sec.json`, etc. and compare results.
