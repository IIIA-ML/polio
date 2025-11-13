# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains research code for detecting coordinated inauthentic behavior on social media through synchronous repeated actions analysis. The primary focus is analyzing retweet patterns to identify users who repeatedly interact with content at similar times, which may indicate coordinated manipulation campaigns.

## Project Structure

- **[src/](src/)**: Core analysis module containing the detection algorithms
  - [synchronous_repeated_detection.py](src/synchronous_repeated_detection.py): Main library with all detection functions
- **[nb/](nb/)**: Jupyter notebooks for experiments and visualization
  - [synchronous_repeated_detection.ipynb](nb/synchronous_repeated_detection.ipynb): Main experimental notebook comparing detection approaches

## Core Architecture

### Data Format

The code expects preprocessed data in a specific directory structure (`../../Labeled_Datasets/{dataset}/Processed/`) containing:
- `data.txt`: User ID mappings (index_to_accountid dictionary)
- `io_users.txt`: List of known inauthentic operation (IO) users
- `RTs.txt`: Retweet data as list of tuples (user_id, tweet_id, timestamp)

### Detection Approaches

The codebase implements and compares multiple detection methods:

1. **Co-retweet counting** (`coretweets`): Pairs of users who retweeted the same tweet within a time window
2. **Co-retweet counting (NumPy optimized)** (`coretweets_numpy`): Same as above but with NumPy vectorization for 2-5x speedup
3. **Temporal synchronization ignoring content** (`ignoring_tweet`): Days where users posted within a time window regardless of tweet content
4. **Temporal synchronization on shared content** (`shared_tweets`): Days where users retweeted shared content within a time window
5. **Same tweet, same time** (`same_tweet_same_time`): Strongest signal - days where users co-retweeted identical tweets within a time window

**Performance Optimization**: The `coretweets_numpy` approach uses NumPy vectorization for improved performance on large datasets, especially those with viral tweets (many retweets per tweet). Benchmark with `examples/benchmark_coretweets.py` to compare.

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

### Batch Experiments Script

The `bin/run_experiments.py` script processes datasets in batch mode without plotting. It automatically tracks which dataset+approach combinations have been processed and stores results as pickle files (one per combination).

```bash
# Run all datasets and approaches with default settings
uv run bin/run_experiments.py

# Specify data directory
uv run bin/run_experiments.py --data-dir /path/to/data

# Run specific datasets only
uv run bin/run_experiments.py --datasets Armenia Catalonia Spain

# Run specific approaches only
uv run bin/run_experiments.py --approaches coretweets ignoring_tweet

# List processing status of all combinations
uv run bin/run_experiments.py --list

# Reprocess even if results exist
uv run bin/run_experiments.py --force

# Custom parameters
uv run bin/run_experiments.py --window-sec 120 --min-coactions 2 --output-dir my_results
```

**File Structure**: Results are stored in the `results/` directory (or custom `--output-dir`) with the following structure:
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
- `dataset`: Dataset name
- `approach`: Approach name
- `pairs_scores`: Dictionary of user pair scores
- `io_users`: List of known inauthentic users
- `window_sec`, `min_coactions`: Parameters used
- `num_pairs`: Number of pairs detected

### Results Analysis Script

The `bin/analyze_results.py` script performs comprehensive statistical analysis on experiment results:

```bash
# Analyze all results with default settings
uv run bin/analyze_results.py

# Specify custom directories
uv run bin/analyze_results.py --results-dir results --output-dir analysis

# Skip individual dataset plots (faster)
uv run bin/analyze_results.py --no-plots
```

The script performs:

1. **Individual Dataset Analysis**: For each dataset, generates a 2×2 comparison plot showing:
   - Each method vs co-retweet baseline
   - Performance rankings by area under curve (AUC)
   - Dataset statistics

2. **Statistical Analysis**:
   - **Friedman Test**: Tests for significant differences across methods
   - **Nemenyi Post-hoc Test**: Pairwise comparisons between methods
   - **Critical Difference Diagram**: Visual representation of significant differences

3. **Outputs** (saved to `analysis/` directory):
   - `plots/{dataset}_comparison.png`: Individual dataset comparisons
   - `critical_difference_diagram.png`: CD diagram showing method rankings
   - `summary.txt`: Statistical test results and rankings

The critical difference diagram shows methods connected by blue lines as not significantly different (p > 0.05).

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
- In the script: use the `--data-dir` argument
- In Python code: pass the full path to `import_data()`
