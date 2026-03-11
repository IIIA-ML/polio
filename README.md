# Detecting Coordination in Information Operations Campaigns

This repository contains research code for detecting coordinated behavior in IO campaigns on Twitter/X through different approaches (e.g., weighted co-retweets). Using labeled datasets from the ["Coordinated Behavior in Information Operations on Twitter"](https://ieeexplore.ieee.org/abstract/document/10508551) article, accessible in ["url"](https://zenodo.org/doi/10.5281/zenodo.10619747), we present improvements for the co-retweets feature for identifying users engaged in coordinated campaigns, consisting of weighting each co-retweet.

## Repository Structure

```
polio/
├── bin/                                      # Executable scripts
│   ├── analyze_results.py                    # Statistical analysis and visualization
│   ├── benchmark_pnorm.py                    # Benchmark Lp-norm aggregation modes
│   ├── bootstrap_experiments.py              # Bootstrap analysis with confidence intervals for metrics
│   ├── compare_datasets.ipynb                # Dataset comparison notebook (Seckin vs Cima datasets)
│   ├── convert_to_pickle.py                  # Convert RTs to pickle format for faster loading
│   ├── count_io_users.py                     # Count IO users with filtering criteria
│   ├── generate_RTs_files.py                 # Generate filtered datasets (calls src/extractors.py)
│   ├── plot_multiple_results_toghether.py    # Plot multiple approaches in the same Cumulative IO Discovery Plot
│   └── run_experiments.py                    # Batch experiment runner
├── src/                                      # Core libraries and utilities
│   ├── analysis/                             # Analysis utilities
│   │   ├── __init__.py
│   │   ├── io_utils.py                       # IO accounts analysis utilities
│   │   ├── metrics.py                        # Evaluation metrics (AUC, AP, NDCG)
│   │   ├── reporting.py                      # Report generation (summary.txt)
│   │   ├── statistics.py                     # Statistical tests (Wilcoxon, Nemenyi)
│   │   └── visualization.py                  # Plot generation
│   ├── approaches/                           # Detection approach implementations
│   │   ├── base.py                           # Base approach class
│   │   ├── factory.py                        # Approach factory (register new approaches here)
│   │   ├── lexicographic.py                  # Lexicographic approach (hierarchical)
│   │   ├── coretweets.py                     # Co-retweet counting (baseline)
│   │   ├── coretweets_weighted_1day.py       # 1-day weighted co-retweets
│   │   ├── coretweets_weighted_12h.py        # 12-hour weighted co-retweets
│   │   └── coretweets_weighted_2days.py      # 2-day weighted co-retweets
│   ├── data_loader.py                        # Load filtered datasets for experiments
│   ├── extractors.py                         # Extract and generate filtered data (retweets, tweets, hashtags)
│   └── synchronous_repeated_detection.py     # Main detection library
├── experiments/                              # Experiment configurations and results
│   └── CimaIO_coretweetweighted_ranking_modes.json  # Example: comparing metrics
├── output/                                   # Outputs for the experiments results and plots
│   └── CimaIO_coretweetweighted_rts_filtering/  # Example for an experiment output
│       └── analysis/
│           ├── plots/<plot>.png              # Cumulative IO Discovery Plot and Bootstrap Plots
│           ├── <metric>/summary.txt          # Results for each metric specified AUC, AP, NDCG
│           └── summary.txt                   # General evaluations (ED, P@100, P@500, Filtering Concentration)
├── pyproject.toml                            # Project dependencies
└── README.md
```

## Directory Overview

### `bin/` - Executable Scripts
Contains command-line tools for the entire workflow: data generation, experiment execution, and analysis.

### `src/` - Core Utilities
- **`analysis/`**: Utilities for metrics computation, visualization, statistical tests, and reporting
- **`approaches/`**: Detection approach implementations (baseline and weighted variants)
- **`extractors.py`**: Called by `generate_RTs_files.py` to extract and filter datasets (retweets, tweets, hashtags, etc.)
- **`data_loader.py`**: Loads filtered datasets when running experiments with `run_experiments.py`

### `experiments/` - Experiment Configurations
JSON files defining experiment scenarios, comparing different approaches on Cima datasets. This file also specifies the folder where the datasets has been stored.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager (recommended) or pip

## Installation

```bash
git clone <repository-url>
cd polio
uv sync
source .venv/bin/activate
```

## Defining New Approaches

To create a custom detection approach:

1. **Create approach file**: Use an existing approach in `src/approaches/` as a template (e.g., [coretweets.py](src/approaches/coretweets.py))
2. **Implement required methods**: Inherit from `BaseApproach` in [base.py](src/approaches/base.py)
3. **Register in factory**: Add your approach to [src/approaches/factory.py](src/approaches/factory.py) so it can be called by name in experiment configurations

Example registration in `factory.py`:
```python
from .my_new_approach import MyNewApproach

APPROACHES = {
    "coretweets": CoreTweets,
    "my_new_approach": MyNewApproach,  # Add your approach here
    # ...
}
```

## Workflow

### 1. Store Raw Datasets

Once you download the datasets you want to analyze, store it in the correspondant folder. 
Datasets from ["Coordinated Behavior in Information Operations on Twitter"](https://ieeexplore.ieee.org/abstract/document/10508551) article store them in the folder `datasets/CimaIO/`:
```
  datasets/CimaIO/Honduras/honduras-bad-anonymized.jsonl
```

If you want to compare with the datasets presented in the ["Labeled Datasets for Research on Information Operations"](https://ojs.aaai.org/index.php/ICWSM/article/view/35958) article, available in ["url"](https://zenodo.org/doi/10.5281/zenodo.14141549), place them in folder `datasets/`, for example:
```
  datasets/Catalonia/Catalonia_part_1.gzip.parquet
```

### 1. Generate Filtered Datasets

Extract retweets, tweets, and other features from raw data (e.g., Cima anonymized JSONL):

```bash
uv run bin/generate_RTs_files.py
```

This script calls [src/extractors.py](src/extractors.py) to generate filtered datasets in the appropriate format.

### 2. Define Approaches

Create detection approaches in [src/approaches/](src/approaches/). Remember to register new approaches in [factory.py](src/approaches/factory.py).

### 3. Define Experiment Configuration

Create an experiment JSON file in `experiments/`. This defines:
- Which approaches to compare
- Which datasets to use
- Time window parameters
- Ranking modes (metrics)
- If the approach needs a first filtering of a users cohort (users that presents minimum number of co-retweets)

**Important**: Different ranking modes (L1, L2,..., Linf) aggregate scores differently. To determine the optimal metric for your use case, first run an experiment comparing different ranking modes. For example:

```bash
uv run bin/run_experiments.py experiments/CimaIO_coretweetweighted_ranking_modes.json
uv run bin/benchmark_pnorm.py experiments/CimaIO_coretweetweighted_ranking_modes.json <approach_name>
```

This benchmarks metrics (AUC, AP, NDCG) across all datasets to help you choose the best aggregation mode.

**Example experiment configuration**:
```json
{
  "name": "my_experiment",
  "data_dir": "datasets/CimaIO/RTs/",
  "output_dir": "output/",
  "window_sec": 60,
  "approaches": [
    {
      "name": "coretweets",
      "min_coactions": 2,
      "ranking_mode": "Linf"
    },
    {
      "name": "coretweets_weighted_1day",
      "ranking_mode": "L2",
      "need_filtering": false
    }
  ],
  "datasets": ["Honduras"],
  "force": false
}
```

### 4. Run Experiment

Execute the experiment across all specified datasets and approaches:

```bash
uv run bin/run_experiments.py experiments/my_experiment.json
```

Results are saved to `experiments/my_experiment/results/`.

### 5. Analyze Results

#### Option A: Comprehensive Analysis

Generate metrics, statistical tests, and visualizations:

```bash
# Analyze all metrics
uv run bin/analyze_results.py experiments/my_experiment.json --metric all

# Analyze specific metric
uv run bin/analyze_results.py experiments/my_experiment.json --metric AUC

# Skip plot generation
uv run bin/analyze_results.py experiments/my_experiment.json --no-plots

# Generate plots without truncation (used when one approach is not based
# on co-retweets and end up with bigger suspicious users cohort)
uv run bin/analyze_results.py experiments/my_experiment.json --notruncation

# Compare approach results with ideal (catching all IO users first)
uv run bin/analyze_results.py experiments/my_experiment.json --ideal
```

**Statistical Tests** (need a lot of datasets):
- 2 approaches: Wilcoxon Signed-Rank test
- 3+ approaches: Nemenyi test with Critical Diagram

#### Option B: Custom Visualization

Plot multiple experiment results together:

```bash
uv run bin/plot_multiple_results_toghether.py experiments/my_experiment.json
```

### 6. Bootstrap Analysis (Optional)

For limited datasets, perform bootstrap resampling to compute confidence intervals (95% quantile) for metrics (AUC, AP, NDCG):

```bash
uv run bin/bootstrap_experiments.py experiments/my_experiment.json
```

This replaces users in the retweet data and computes confidence intervals across bootstrap samples.

## Complete Example Workflow

```bash
# 1. Generate datasets from Cima anonymized data
uv run bin/generate_RTs_files.py

# 2. Compare ranking modes to find optimal metric
uv run bin/run_experiments.py experiments/Ranking_modes_experiment.json
uv run bin/benchmark_pnorm.py experiments/Ranking_modes_experiment.json coretweets_weighted_1day

# 3. Define and run main experiment
uv run bin/run_experiments.py experiments/my_experiment.json

# 4. Analyze results
uv run bin/analyze_results.py experiments/my_experiment.json --metric all

# 5. (Optional) Bootstrap for confidence intervals
uv run bin/bootstrap_experiments.py experiments/my_experiment.json
```

## Additional Tools

- [count_io_users.py](bin/count_io_users.py): Analyze IO user distribution with different filtering thresholds
- [compare_datasets.ipynb](bin/compare_datasets.ipynb): Interactive dataset comparison notebook (Cima vs Seckin). Need both datasets downloaded.
- [convert_to_pickle.py](bin/convert_to_pickle.py): Convert filtered datasets to pickle format for faster loading (done automatically when running an experiment)
```
