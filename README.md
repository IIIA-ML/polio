# Detecting Coordination in Information Operations Campaigns

This repository contains research code for detecting coordinated behavior on IO campaigns on Twitter/X through synchronous retweets analysis. Using labeled datasets from the ["Labeled Datasets for Research on Information Operations"](https://ojs.aaai.org/index.php/ICWSM/article/view/35958) article, we present a novel feature for identifying users engaged in coordinated campaigns with strong empirical results.

## Key Results

Our approach **outperforms the co-retweet baseline** on a critical metric: **ranking IO (inauthentic operation) users first**. By analyzing retweet patterns and temporal synchronization, we achieve superior detection of coordinated manipulation campaigns across 23 real-world datasets.

## Repository Structure

```
polio/
├── bin/                 # Command-line scripts for experiments and analysis
│   ├── generate_RTs_files.py      # Convert raw Twitter data to retweet format
│   ├── count_io_users.py          # Count IO users with different filtering criteria
│   ├── benchmark_pnorm.py         # Benchmark different Lp-norm central tendency aggregation modes
│   ├── run_experiments.py         # Batch runner for all experiments
│   └── analyze_results.py         # Statistical analysis and visualization
├── src/                 # Core libraries and utilities
│   ├── synchronous_repeated_detection.py  # Main detection library
│   ├── data_loader.py                     # Data loading utilities
│   ├── approaches/                        # Detection approach implementations
│   │   ├── base.py                        # Base approach class
│   │   ├── coretweets.py                  # Co-retweet counting (baseline)
│   │   ├── ignoring_tweet_counting_rt.py  # Precission approach (counting unique tweets on synchronized days)
│   │   ├── ignoring_tweet_fast.py         # Optimized Synchronous Days (novel approach)
│   │   ├── factory.py                     # Class to create approaches instances
│   │   └── lexicographic.py               # Lexicographic approach, hierarchical refinement using multiple approaches
│   └── analysis/            # Analysis utilities
│       ├── metrics.py                 # Evaluation metrics (AUC, AP, NDCG)
│       ├── statistics.py              # Statistical tests (Ranking users, Friedman, Wilcoxon, Nemenyi)
│       ├── reporting.py               # Report generation (summaries and results)
│       ├── visualization.py           # Plot generation
│       └── io_utils.py                # IO accounts analysis utilities
└── experiments/         # Experiment configurations and results
    ├── {experiment_name}.json     # Experiment configuration
    ├── {experiment_name}/
    │   ├── results/               # Raw detection results (pickles)
    │   └── analysis/              # Analysis outputs (metrics, plots, stats)
    └── approaches.json            # Approach example used for Experiment 2
```

## Quick Start

### Prerequisites

- Python 3.12+
- [uv] package manager (or pip)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd polio

# Install dependencies
uv sync
```

## Scripts Guide

### 1. `bin/generate_RTs_files.py` - Data Preparation

Converts raw Twitter/X data into standardized retweet format for analysis.

**Input Formats:**
- JSONL files from Twitter API v2
- Parquet files (multiple file support)
- Anonymized datasets

**Usage:**
```bash
uv run bin/generate_RTs_files.py
```

### 2. `bin/count_io_users.py` - Dataset Statistics

Analyzes IO user distribution in each dataset with different filtering thresholds.

**Usage:**
```bash
uv run bin/count_io_users.py
```

### 3. `bin/benchmark_pnorm.py` - Benchmarking Aggregation Norms

Benchmarks different Lp-norm central tendency aggregation modes for pair-based approaches accross all datasets, evaluating on different metrics (AUC, AP and NDCG). It needs to have already run an experiment (bin/`run_experiments.py`).

**Usage:**
```bash
uv run bin/benchmark_pnorm.py experiments/experiment_name.json approach_key
```

### 4. `bin/run_experiments.py` - Experiment Execution

Batch runner that executes experiments across multiple datasets and approaches. First you need to define an experiment .json in `experiments/` such as follows:

```json
{
  "name": "experiment_name",
  "approaches": ["coretweets", "ignoring_tweet_fast"],
  "datasets": ["Armenia", "Catalonia", "Spain"],
  "parameters": {
    "time_window_seconds": 3600
  }
}
{
  "name": "experiment_name",
  "data_dir": "data", // Where the RTs data is located
  "window_sec": 60,
  "min_coactions": 1, // Filtering to minimum co-retweets
  "approaches": [
    "coretweets[Linf]",     // The Linf indicates which aggregation norm you want to use (from [1,...,inf)
    "ignoring_tweet_fast[Linf]",
    "lexicographic:coretweets[Linf]+ignoring_tweet_fast[Linf]"
  ],
  "datasets": [
    "Armenia",
    "Catalonia",
    "Spain",
    "Iran_1",
    "Russia_1",
    "Venezuela_1"
  ],
  "force": false    // True if you want to overwrite all the results saved
}
```

**Usage:**
```bash
# Run all approaches on all datasets
uv run bin/run_experiments.py experiments/experiment_name.json

# Run only specific approaches
uv run bin/run_experiments.py experiments/experiment_name.json \
  --approaches coretweets ignoring_tweet_fast

# Force reprocessing
uv run bin/run_experiments.py experiments/experiment_name.json --force
```

### 5. `bin/analyze_results.py` - Results Analysis

Performs statistical analysis and generates comparison visualizations on the desired evaluation metric (AUC, AP or NDCG). If the experiment only have 2 approaches, Wilcoxon Signed-Rank test is performed, if more, Nemenyi test with Critical Diagram plot is generated.

**Usage:**
```bash
# Analyze experiment and generate all outputs
uv run bin/analyze_results.py experiments/coretweets_vs_approaches.json --metric all

# Analyze single metric
uv run bin/analyze_results.py experiments/coretweets_vs_approaches.json --metric AUC
```

## Typical Workflow

```bash
# Generate retweet files from parquet files downloaded from Labeled Datasets on IO article
uv run bin/generate_RTs_files.py

# Execute all approaches on all datasets
uv run bin/run_experiments.py experiments/my_experiment.json

# Generate metrics, plots, and statistical tests
uv run bin/analyze_results.py experiments/my_experiment.json --metric all
```

## Performance Optimization

The codebase includes several performance optimizations:

1. **Pickle Caching**: Convert text-based datasets to binary pickle format
   - 10-100x faster loading for repeated experiments
   - Automatic with `convert_to_pickle.py` or when running `run_experiments.py`

2. **Result Caching**: `run_experiments.py` caches computed results
   - Avoids reprocessing unchanged (dataset, approach) pairs
   - Force reprocessing with `--force` flag
