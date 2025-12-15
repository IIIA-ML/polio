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
uv run bin/benchmark_pnorm.py experiments/experiment_name.json approach_key1 approach_key2
```

### 4. `bin/run_experiments.py` - Experiment Execution

Batch runner that executes experiments across multiple datasets and approaches. First you need to define an experiment .json in `experiments/` such as follows:

```json
{
  "name": "experiment_name",
  "data_dir": "data",
  "output_dir": null,
  "window_sec": 60,
  "filter_min_coactions": 1,
  "approaches": [
    {
      "name": "coretweets",
      "min_coactions": 1,
      "ranking_mode": "Linf"
    },
    {
      "name": "ignoring_tweet_fast",
      "min_coactions": 1,
      "ranking_mode": "Linf",
      "need_filtering": true
    },
    {
      "name": "coretweets",
      "min_coactions": 2,
      "ranking_mode": "Linf"
    },
    {
      "name": "lexicographic",
      "approaches": ["coretweets", "ignoring_tweet_fast"],
      "ranking_modes": ["Linf", "Linf"],
      "min_coactions": [1, 1],
      "need_filtering": [false, false]
    }
  ],
  "datasets": [
    "Armenia",
    "Catalonia",
    "Spain",
    "Iran_1",
    "Russia_1",
    "Venezuela_1"
  ],
  "force": false
}
```

**Configuration Parameters:**
- `name`: Experiment identifier
- `data_dir`: Directory containing processed retweet data (default: "data")
- `output_dir`: Directory for results (default: null, uses experiment name in same dir as config)
- `window_sec`: Time window in seconds for synchronous actions (default: 60)
- `filter_min_coactions`: The filtering of all users to only those with minimum X coretweets (default: 1)
- `approaches`: List of approach specifications (each can be a dictionary with per-approach parameters)
- `datasets`: List of dataset names to process (default: null = all datasets)
- `force`: Whether to overwrite existing results (default: false)

**Approach Specification Formats:**
- **Dictionary format** (recommended):
  ```json
  {
    "name": "approach_name",
    "min_coactions": 1,
    "ranking_mode": "Linf",
    "need_filtering": true
  }
  ```
  - `name`: Required. Approach identifier
  - `min_coactions`: Optional. Minimum approach number threshold (default: 1)
  - `ranking_mode`: Optional. Ranking aggregation mode: L1, L2, Linf (default: Linf)
  - `need_filtering`: Optional. Whether to filter data by coretweets (with min_filter_coactions) before processing (default: true)

- **Lexicographic format** (for combining multiple approaches):
  ```json
  {
    "name": "lexicographic",
    "approaches": ["approach1", "approach2"],
    "ranking_modes": ["Linf", "L2"],
    "min_coactions": [1, 2],
    "need_filtering": [true, false]
  }
  ```
  - `approaches`: Required. List of sub-approach names (at least 2)
  - `ranking_modes`: Optional. List of ranking modes per approach (default: Linf for all)
  - `min_coactions`: Optional. Can be single integer (applied to all) or list (one per approach)
  - `need_filtering`: Optional. Can be single boolean (applied to all) or list (one per approach, default: true for all)

**Usage:**
```bash
# Run all approaches on all datasets
uv run bin/run_experiments.py experiments/experiment_name.json
```

### 5. `bin/analyze_results.py` - Results Analysis

Performs statistical analysis and generates comparison visualizations on the desired evaluation metric (AUC, AP or NDCG). If the experiment only have 2 approaches, Wilcoxon Signed-Rank test is performed, if more, Nemenyi test with Critical Diagram plot is generated.

**Usage:**
```bash
# Analyze experiment and generate all outputs
uv run bin/analyze_results.py experiments/coretweets_vs_approaches.json --metric all

# Analyze single metric
uv run bin/analyze_results.py experiments/coretweets_vs_approaches.json --metric AUC

# Generate plots without truncation (no area computation, full rank range)
uv run bin/analyze_results.py experiments/coretweets_vs_approaches.json --notruncation
```

**Options:**
- `--metric`: Evaluation metric (auc, ndcg, ap, all) (default: auc)
- `--notruncation`: Generate plots without truncating by x_max and without computing areas. Results saved to `experiments/{experiment_name}/analysis/no_truncation/`

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
