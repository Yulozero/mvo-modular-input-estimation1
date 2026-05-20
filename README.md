# Modular Input Estimation for Mean-Variance Portfolio Choice

This repository contains the code and configuration files used for the experiments in the paper on modular input estimation for mean-variance optimization (MVO).

The empirical design estimates the main MVO inputs separately:

- expected returns through an Attention Mean module;
- volatility scales through DWT-LSTM and related volatility baselines;
- cross-asset dependence through Pearson, rank-based, distance-correlation, and signed distance-correlation estimators;
- portfolio weights through a common long-only constrained MVO allocator.

## Repository Structure

```text
configs/      YAML files for the main experiment, module comparisons, ablations, robustness checks, and generalization tests
scripts/      asset-universe and auxiliary experiment scripts
src/          experiment engine, data loading, models, optimization, backtesting, and evaluation code
tests/        protocol and implementation tests
data/         local data placeholder; CSV market data are not committed by default
```

Large intermediate outputs, raw market data, processed market data, Python caches, and LaTeX build products are excluded by `.gitignore`.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

## Data

The main 30-asset experiment expects the following processed files:

```text
data/processed/pool30/returns.csv
data/processed/pool30/close.csv
data/processed/pool30/market_cap.csv
```

Additional robustness and generalization runs use the data paths documented in `data/README.md` and the YAML files under `configs/experiment_design/`.

Market data files are not committed by default because source providers may impose redistribution restrictions. The repository is intended to make the code, configurations, preprocessing logic, and experiment protocol transparent.

## Run the Main Model

```bash
python -m src.runners.run_experiment --base configs/experiment_design/base_design.yaml --preset configs/experiment_design/presets/design_main_full_model.yaml
```

## Run the Paper Experiment Suite

```bash
python -m src.runners.run_suite --suite configs/experiment_design/suites/design_suite_full.yaml
```

Run a selected experiment group:

```bash
python -m src.runners.run_suite --suite configs/experiment_design/suites/design_suite_full.yaml --group "Mean Comparison"
python -m src.runners.run_suite --suite configs/experiment_design/suites/design_suite_full.yaml --group "Vol Comparison"
python -m src.runners.run_suite --suite configs/experiment_design/suites/design_suite_full.yaml --group "Corr Comparison"
python -m src.runners.run_suite --suite configs/experiment_design/suites/design_suite_full.yaml --group "Ablation"
```

## Tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## Reproducibility

See `REPRODUCIBILITY.md` for the full reproducibility guide, including the expected data paths, suite commands, output structure, and leakage-control notes.
