# Everybody Out of the Pool! A Predictive Assessment of Models of Policy Diffusion in the U.S. States

This repository includes the replication data and code for Hemauer, Saunders, and Desmarais.

**Abstract:** <br>
Three decades of policy diffusion research have advanced our theoretical understanding of how and why policies spread. Yet, the predictive performance of the field’s dominant modeling approach, event history analysis, has not been systematically evaluated. Benchmarking replication data from ten published studies, we find that standard models predict out-of-sample policy adoption at rates slightly above chance. We also find that covariate relationships are substantially unstable across time periods, suggesting that diffusion dynamics are strongly defined by historical and time-contingencies. Nonlinear machine learning methods predict meaningfully better across nearly all applications and reveal non-monotonic covariate relationships that linear models cannot capture. These findings indicate that the dynamics of policy diffusion are richer and more complex than existing approaches have detected, and point toward a productive research agenda combining benchmarking with nonlinear modeling to deepen theoretical understanding of how policies spread.

---

## System Requirements

All scripts were designed for high-performance computing. Running most of the experiment scripts on a personal computer is not possible.

- **Cores:** 24-48
- **RAM:** 192–336 GB (varies by job)
- **Wall time:** Up to 2 weeks per job

---

## Repository Structure

```
pred_diffusion_2025/
├── data/                        # Input datasets (raw and processed)
├── ml_adoption_timing/          # Adoption timing experiment
├── ml_covariate_analysis/       # Coefficient stability, feature importance, and LRT
│   ├── ml_coef_split/           # Stata: split-sample logit coefficient comparison
│   ├── ml_feature_importance/   # Python: random forest feature importance
│   └── ml_lrt/                  # R: likelihood ratio tests
├── ml_forecast/                 # Temporal forecasting experiment
├── ml_hyperparameter/           # Hyperparameter tuning
│   ├── ml_hyperparameter_prune/ # Grid pruning (run first)
│   └── ml_best_random_hyperparameters/ # Final hyperparameter search
├── ml_pdp/                      # Partial dependence plot analysis (R + Python)
├── ml_policy/                   # Policy-specific prediction experiment
├── ml_random/                   # Random split experiment
└── ml_state/                    # State-specific prediction experiment
```

### General Folder Conventions

- The prefix `ml_` designates folders containing a nested analysis.
- Most `ml_` folders include a `bin/` subfolder with a SLURM Bash script to submit jobs to the HPC.
- Some folders include a `full_models/` subfolder. These scripts are **not meant to be run**; they were used to split large jobs into manageable scripts for the HPC and are provided for reference.
- All `ml_` folders include a `figures/` subfolder where generated figures and result files are written.
- `ml_state/` and `ml_policy/` include `_split.py` scripts that use the SLURM `--array` flag to distribute jobs across states/policies. After these run, use `figures/aggregate_state.py` and `figures/aggregate_policy.py` respectively to combine results.

---

## Data (`data/`)

The `data/` folder contains the input datasets for all 10 policy diffusion studies examined in this article.

| File | Study | Format |
|---|---|---|
| `berry_berry1990.txt` | Berry & Berry (1990) | Space-delimited text |
| `berry_berry1990_processed.csv` | Berry & Berry (1990) | Processed CSV referenced in berry_berry_coef.do|
| `boehmke2017.dta` | Boehmke et al. (2017) | Stata |
| `boushey2016.dta` | Boushey (2016) | Stata |
| `bricker_lacombe2021.dta` | Bricker & Lacombe (2021) | Stata |
| `karch2016.dta` | Karch et al. (2016) | Stata |
| `kreitzer_boehmke2016.dta` | Kreitzer & Boehmke (2016) | Stata |
| `lacombe_boehmke2021.dta` | LaCombe & Boehmke (2021) | Stata |
| `lacombe_boehmke_2021_processed.csv` | LaCombe & Boehmke (2021) | Processed CSV referenced in lacombe_boehmke_data_sim.R|
| `mallinson2019.csv` | Mallinson (2021) | CSV |
| `mallinson2019_processed.csv` | Mallinson (2021) | Processed CSV referenced in mallinson_coef.do |
| `mallinson_lovell2022.csv` | Mallinson & Lovell (2022) | CSV |
| `parinandi2020.dta` | Parinandi (2020) | Stata |

Mallinson2019 should be Mallinson2021, but there are too many references to change.

**Note on processed files:** The `_processed.csv` files are the same datasets as their raw counterparts but have been converted to CSV format, had column names standardized, or had character values converted to numeric.

## Experiments and Folder Details

### Hyperparameter Tuning (`ml_hyperparameter/`)

**Run these first, before any main experiments.** Hyperparameter results are consumed by `ml_random/` and inform all model runs.

#### `ml_hyperparameter_prune/`
Prunes large hyperparameter grids to smaller, computationally feasible grids using F-tests.

- **Scripts:** `*_hyperparameter_prune.py` per dataset
- **Outputs:** `figures/<dataset>/` — TXT files with RF and XGBoost candidate hyperparameter sets

#### `ml_best_random_hyperparameters/`
Performs the final hyperparameter search on the pruned grids for the random split experiment.

- **Scripts:** `*_random_hyperparameters.py` per dataset
- **Outputs:** `figures/<dataset>/` — PNG plots comparing optimized vs. unoptimized models; TXT files with best hyperparameters for logistic regression, RF, and XGBoost

---

### Random Split Experiment (`ml_random/`)

Baseline experiment using a standard random train/test split with optimized hyperparameters from `ml_hyperparameter/`. The scripts also optimize precision/recall output.

- **Scripts:** `*_random_threshold.py` per dataset
- **Outputs:** `figures/<dataset>/` — PNG performance comparison plots; TXT files with detailed results for each model type

---

### Temporal Forecasting Experiment (`ml_forecast/`)

Evaluates out-of-sample prediction when training on earlier time periods and forecasting 1, 5, and 10 years ahead.

- **Scripts:** Multiple `*_forecast_*.py` scripts per dataset organized by lead time (t+1, t+5, t+10)
- **Outputs:** `figures/<dataset>/` — TXT files with forecast metrics per lead time; CSV files with time-series predictions; PNG visualization plots

---

### Policy-Specific Experiment (`ml_policy/`)

Evaluates prediction when an entire policy is held out of training (leave-one-policy-out cross-validation).

- **Scripts:** `*_policy_logit.py`, `*_policy_rf.py`, `*_policy_xgb.py` for applicable datasets
- **Outputs:** `figures/<dataset>/` — CSV files per model type; `figures/<dataset>/policy_files/` — individual array job outputs (one file per policy)
- **Post-processing:** After all array jobs finish, run `figures/aggregate_policy.py` to combine per-policy CSVs into summary statistics

---

### State-Specific Experiment (`ml_state/`)

Evaluates prediction when all observations for one state are held out of training (leave-one-state-out cross-validation).

- **Scripts:** `*_state_logit.py`, `*_state_rf.py`, `*_state_xgb.py` per dataset
- **Outputs:** `figures/<dataset>/` — CSV files per model; `figures/<dataset>/state_files/` — individual array job outputs (one file per state, up to 50 files)
- **Post-processing:** After all array jobs finish, run `figures/aggregate_state.py` to combine per-state CSVs into summary statistics

---

### Adoption Timing Experiment (`ml_adoption_timing/`)

Tests model ability to predict the timing of policy adoption using MAE.

- **Scripts:** `*_cdf_logit.py`, `*_cdf_reglogit.py`, `*_cdf_rf.py`, `*_cdf_xgb.py` for applicable datasets
- **Outputs:** `figures/<dataset>/` — CSV files with CDF/survival analysis results; TXT files with MAE metrics per model

---

### Covariate Analysis (`ml_covariate_analysis/`)

Three nested sub-analyses examining coefficient stability and variable importance.

#### `ml_coef_split/` — Logistic Regression Coefficient Stability
Splits each dataset at the median year and compares logit coefficients between the two halves to assess temporal stability.

- **Scripts:** `*_coef.do` per dataset
- **Outputs:** `figures/<dataset>/` — PNG coefficient plots showing split-sample comparisons

#### `ml_feature_importance/` — Random Forest Feature Importance
Computes and ranks feature importances from random forest models trained on each half of the data.

- **Scripts:** `*_split_feature.py` per dataset
- **Outputs:** `figures/<dataset>/` — PNG feature importance visualizations

#### `ml_lrt/` — Likelihood Ratio Tests
Tests coefficient stability using likelihood ratio tests, comparing a simple model against a model with time-period interaction terms.

- **Scripts:** `*_lrt.R` per dataset
- **Outputs:** `figures/<dataset>/` — TXT files with LRT statistics and p-values

---

### Partial Dependence Plots (`ml_pdp/`)

Two-stage analysis.

#### Stage 1: Data Simulation
Simulates logistic regression data from the original study's model coefficients to create linear synthetic datasets for PDP analysis.

- **Scripts:** `*_data_sim.R` per dataset
- **Outputs:** `figures/<dataset>/` — CSV files of simulated data

#### Stage 2: Feature Simulation
Computes partial dependence plots using the simulated data from Stage 1 and trained XGBoost or RF models (selected by best random-split performance).

- **Scripts:** `*_feature_sim.py` per dataset
- **Outputs:** `figures/<dataset>/` — PNG partial dependence plot visualizations

---