# Detecting Coordination in Information Operations Campaigns

Research code for detecting coordinated behavior in information operations (IO) campaigns on Twitter/X by
weighting synchronized co-retweets.

Two users co-retweet when they retweet the same tweet within Δt = 60 seconds of each other. Each co-retweet is
weighted by how popular the tweet was in a period T (12 hours by default), the weighted co-retweets of each pair
of users give a pair score, and the pair scores of each user are aggregated into a user score. Users are ranked
by that score, and the ranking is evaluated against the known IO accounts.

The data are the labeled Honduras and UAE datasets of
["Coordinated Behavior in Information Operations on Twitter"](https://ieeexplore.ieee.org/abstract/document/10508551)
(Cima et al.), available on [Zenodo](https://zenodo.org/doi/10.5281/zenodo.10619747).

## Contents

- [Quick start](#quick-start)
- [Data](#data)
- [Running an experiment](#running-an-experiment)
- [Reproducing the article](#reproducing-the-article)
- [Repository structure](#repository-structure)
- [Adding an approach](#adding-an-approach)

## Quick start

Requirements: Python 3.12+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone <repository-url>
cd polio
uv sync
```

`uv sync` installs `src/` in editable mode, so the scripts, the notebooks and the tests can import its packages
(`approaches`, `analysis`, `baselines`, `data_loader`) directly. Run every script from the repository root.

## Data

### Raw datasets

Download the Cima et al. datasets and place the files like this:

```
datasets/CimaIO/
├── Honduras/
│   ├── honduras-bad-anonymized.jsonl    # IO accounts
│   └── honduras-good-anonymized.jsonl   # control accounts
└── UAE/
    ├── uae-bad-anonymized.jsonl
    └── uae-good-anonymized.jsonl
```

The dataset comparison notebook also uses the datasets of
["Labeled Datasets for Research on Information Operations"](https://ojs.aaai.org/index.php/ICWSM/article/view/35958)
(Seckin et al., [Zenodo](https://zenodo.org/doi/10.5281/zenodo.14141549)), downloaded them and place them in the datasets/ folder as well.

### Processed datasets

`bin/generate_RTs_files.py` turns the raw files into one processed folder per dataset:

```
datasets/CimaIO/RTs/<dataset>/Processed/
├── RTs.txt                    # retweets as (user, tweet, timestamp)
├── io_users.txt               # IO labels
├── index_to_accountid.txt     # user index -> account ID
└── data_cache.pkl             # created the first time the dataset is loaded
```

Each account keeps one retweet per tweet. The time of each retweet is decoded from its Snowflake tweet ID, because
the time field of the released data is truncated to the minute.

## Running an experiment

An experiment is a JSON config in `experiments/`. This is the start of
[experiments/article_main_comparison_Linf.json](experiments/article_main_comparison_Linf.json):

```json
{
  "name": "article_main_comparison_Linf",
  "data_dir": "datasets/CimaIO/RTs",
  "output_dir": "output/",
  "window_sec": 60,
  "min_coactions": [1, 2],
  "need_filtering": false,
  "datasets": ["UAE", "Honduras"],
  "force": false,
  "approaches": [
    {"name": "coretweets_weighted_12h", "ranking_mode": "Linf"},
    {"name": "coretweets", "ranking_mode": "Linf"},
    ...
  ]
}
```

To compute the ranking of every (dataset, approach) pair of a config:

```bash
uv run bin/run_experiments.py experiments/article_main_comparison_Linf.json
```

Each ranking is stored in `<output_dir>/<name>/results/<dataset>/<approach key>.pkl` and reused on later runs
unless the config sets `"force": true`. The analysis scripts of the next section call this step themselves, so
you rarely need to run it alone.

### Config fields

| Field | Meaning |
|---|---|
| `window_sec` | Synchronization window Δt, in seconds |
| `min_coactions` | Scenarios to run. `1` (S1): users with at least one synchronized co-retweet. `2` (S2): users with at least two co-retweets with the same partner |
| `datasets` | Folders under `data_dir` |
| `approaches` | List of approach specs (see below) |
| `force` | Recompute rankings that are already stored |

`min_coactions`, `ranking_mode` and `need_filtering` set at the top level are defaults for every approach (see
[src/approaches/factory.py](src/approaches/factory.py)). The comparison configs of the article set
`ranking_mode` on each approach and have no default.

### Approaches

An approach spec is a dict with the approach name and its aggregation:

```json
{"name": "coretweets_weighted_12h", "ranking_mode": "Linf"}
```

The lexicographic approach ranks by the first approach and breaks ties with the next ones:

```json
{"name": "lexicographic", "approaches": ["coretweets", "coretweets_weighted_12h"], "ranking_modes": "Linf"}
```

| Name | Pair score |
|---|---|
| `coretweets` | Number of synchronized co-retweets |
| `coretweets_weighted_12h`, `_1day`, `_2days` | Co-retweets weighted by tweet popularity in a period T of 12 h, 1 day or 2 days |
| `coretweets_idf` | Co-retweets weighted by IDF |
| `coretweets_inv_m` | Co-retweets weighted by 1/m, with m the number of retweets of the tweet |
| `lexicographic` | Ranking by one approach, ties broken by the next ones |

### Aggregations (`ranking_mode`)

How the pair scores of a user are combined into the user score:

| Value | Aggregation |
|---|---|
| `L1` | Median |
| `L2` | Mean |
| `L<p>`, p ≥ 3 | Lp centre |
| `Linf` | Midrange (default in the article) |
| `max`, `min` | Maximum, minimum |
| `TM25` | Interquartile mean |
| `MH` | Midhinge |
| `Q90` | 0.9-quantile |
| `Linf_droptop`, `Linf_dropbottom` | Midrange without the top or bottom score |
| `sum` | Strength |
| `degree` | Degree |

## Reproducing the article results

Run the commands below in this order. Together they produce every table, figure and number of the article. The
seeds and resampling settings (bootstrap samples, jackknife groups, interval level) come from the configs.

Most analysis scripts take a step and a config:

```bash
uv run bin/<script>.py {rank|analyze|all} <config>
```

`rank` computes and stores the rankings of every scenario, `analyze` computes the metrics, intervals, tables and
figures from the stored rankings, and `all` runs both.

### 1. Data preparation

```bash
uv run bin/generate_RTs_files.py
```

Writes the retweets of Honduras and UAE to `datasets/CimaIO/RTs/<dataset>/Processed/`.

### 2. Dataset comparison

Retweets and active days per user of the IO and control accounts, for the Cima and Seckin datasets. Open the
notebook in Jupyter or run it headless:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace nb/compare_datasets.ipynb
```

### 3. Preliminary analysis (choice of T and of the aggregation)

Ranks users with every period and aggregation in S1 and S2, then computes the figures, tables and bootstrap
intervals. The numbers quoted in the text are printed to the terminal.

```bash
uv run bin/preliminar_analysis_bootstrap.py all experiments/article_preliminar_T_and_p.json
```

Outputs in `output/article_preliminar_T_and_p/analysis/`:

- `pnorm_plateau.png`, `pnorm_plateau_alternative.png`, `pnorm_gains.png`
- `table_loco.tex`, `table_aggregation.tex` and the gain tables of each cell

### 4. Sensitivity to Δt

```bash
uv run bin/benchmark_dt_sensitivity.py all experiments/article_dt_sensitivity.json
```

Outputs in `output/article_dt_sensitivity/analysis/`: `table_dt_main.tex`, `table_app_dt_gains.tex`,
`table_app_dt_scores.tex`, `dt_sensitivity_gains.png`.

### 5. Main results

Compares w_T against the count, IDF and the lexicographic ranking, and the lexicographic ranking against the count.
The `analyze` step computes the scores, a conditional bootstrap over the evaluated accounts, a graph-level
jackknife over tweets and accounts, the figures and Table 1.

```bash
uv run bin/article_main_comparison.py all experiments/article_main_comparison_Linf.json
uv run bin/article_main_comparison.py all experiments/article_main_comparison_L3.json   # appendix, p = 3
```

A difference counts as robust when both 95% intervals exclude zero on the same side (filled markers in Figure 2).
Intervals are computed for AUC, AP and NDCG. ED, P@k and FC are point estimates. The robustness counts are printed
to the terminal.

Outputs in `output/article_main_comparison_Linf/analysis/` (and `..._L3/analysis/`):

| File | Content |
|---|---|
| `figure1.png` | Cumulative IO recall |
| `figure1_old_zoomed.png` | IO users detected against users studied, with the area differences and the 80/90/95% IO recall levels |
| `figure2.png` | Differences in AUC, AP and NDCG |
| `table1.tex` | Table 1 |
| `scores.csv` | Every metric of every evaluated ranking |
| `differences.csv` | Both intervals and the verdict of every AUC, AP and NDCG difference, including those of 1/m |
| `fc_savings.csv` | Users saved at IO recall 0.50 to 0.98 |
| `idf_best.csv` | IDF at its best aggregation (midrange config only) |

### 6. Computational cost

Time and memory of every approach of a comparison config, in every scenario and campaign. Each measurement is
repeated three times in fresh processes (change it with `--repeats`).

```bash
uv run bin/benchmark_runtime.py experiments/article_main_comparison_Linf.json
```

Outputs: `output/article_main_comparison_Linf/analysis/runtime.csv` and `runtime_summary.csv`.

### 7. Comparison with Pacheco et al. (2021)

The baseline is the co-retweet TF-IDF cosine similarity of Pacheco et al., reimplemented from their released code
in [src/baselines/pacheco.py](src/baselines/pacheco.py). It only scores accounts with more than 10 retweets, so
the comparison is restricted to those accounts. The config has the fields of the main comparison plus the baseline
parameters under `pacheco`. It takes about 30 minutes, most of it in the graph-level jackknife.

```bash
uv run bin/pacheco_comparison.py all experiments/article_pacheco_comparison.json
```

Outputs in `output/article_pacheco_comparison/analysis/`: `table_scores.tex`, `table_diffs.tex`, `scores.csv`,
`diffs.csv`, `blocks.csv`, `summary.md`.

## Repository structure

```
polio/
├── bin/                                  # One script per experiment of the article
│   ├── generate_RTs_files.py             # Data preparation (uses src/extractors.py)
│   ├── run_experiments.py                # Rankings of every (dataset, approach) of a config
│   ├── preliminar_analysis_bootstrap.py  # Choice of the period T and of the aggregation
│   ├── benchmark_dt_sensitivity.py       # Sensitivity to Δt
│   ├── article_main_comparison.py        # Main results: Figures 1-3 and Table 1
│   ├── benchmark_runtime.py              # Time and memory
│   └── pacheco_comparison.py             # Comparison with Pacheco et al. (2021)
├── nb/
│   └── compare_datasets.ipynb            # Cima vs Seckin statistics
├── experiments/                          # JSON configs of the article
├── output/                               # One folder per experiment: results/ and analysis/
├── src/
│   ├── approaches/
│   │   ├── base.py                       # Pair approaches, aggregations, stored rankings
│   │   ├── factory.py                    # Builds approaches from config specs
│   │   ├── coretweets.py                 # Co-retweet count (baseline)
│   │   ├── coretweets_weighted_base.py   # Period-weighted co-retweets (shared algorithm)
│   │   ├── coretweets_weighted_{12h,1day,2days}.py
│   │   ├── coretweets_tweet_weighted_base.py   # Per-tweet weights (shared algorithm)
│   │   ├── coretweets_idf.py             # IDF weights
│   │   ├── coretweets_inv_m.py           # 1/m weights
│   │   ├── lexicographic.py              # Ranking with tie-breaking approaches
│   │   ├── _coretweet_index.py           # Retweets grouped by tweet
│   │   └── _lp_centre.py                 # Lp centre of the p-norm aggregations
│   ├── baselines/pacheco.py              # Pacheco et al. (2021)
│   ├── analysis/
│   │   ├── metrics.py                    # AUC, AP, NDCG, P@k, ED and FC, with tied scores
│   │   ├── statistics.py                 # Conditional bootstrap, jackknife intervals, paired differences
│   │   ├── graph_jackknife.py            # Graph-level jackknife over tweets and accounts
│   │   ├── comparison.py                 # Comparison configs and their stored rankings
│   │   ├── descriptives.py               # Single co-retweet pairs of the co-retweet graph
│   │   ├── aggregations.py               # Aggregation names
│   │   ├── io_utils.py                   # Configs, stored rankings, runs of run_experiments.py
│   │   ├── reporting.py                  # LaTeX tables and console output
│   │   └── visualization.py              # Figures of the main comparison
│   ├── data_loader.py                    # Dataset loading with a pickle cache
│   └── extractors.py                     # (account, object, time) events from raw data
└── pyproject.toml
```

## Adding an approach

1. Subclass `PairsApproach` in [src/approaches/base.py](src/approaches/base.py), using
   [src/approaches/coretweets.py](src/approaches/coretweets.py) as a template.
2. Implement `get_approach_name`, `get_approach_key` and `compute_pairs_scores`.
3. Register the class in `PAIR_APPROACHES` in [src/approaches/factory.py](src/approaches/factory.py) so configs can
   use it by name.
