# Reproducibility Guide

This repository contains the experiment code and configuration files for the paper on modular input estimation for mean-variance portfolio choice.

## Environment

The experiments were developed for Python 3.9 or later. A fresh environment can be prepared with:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Linux or macOS, replace the activation command with:

```bash
source .venv/bin/activate
```

## Data

The main configuration expects processed CSV files under:

```text
data/processed/pool30/
data/processed/pool50/
data/processed/pool100/
data/processed/market2/
```

At minimum, the main 30-asset experiment requires:

```text
data/processed/pool30/returns.csv
data/processed/pool30/close.csv
data/processed/pool30/market_cap.csv
```

Raw and processed market data are intentionally excluded from version control. If the data provider's terms permit redistribution, place the CSV snapshots in the paths above. Otherwise, keep the local data outside GitHub and use the repository to document the asset universe, preprocessing rules, and experiment configuration.

## Main Experiment

Run the main modular model:

```bash
python -m src.runners.run_experiment --base configs/experiment_design/base_design.yaml --preset configs/experiment_design/presets/design_main_full_model.yaml
```

Run the full experiment suite used for the paper tables:

```bash
python -m src.runners.run_suite --suite configs/experiment_design/suites/design_suite_full.yaml
```

Selected groups can be run separately:

```bash
python -m src.runners.run_suite --suite configs/experiment_design/suites/design_suite_full.yaml --group "Main Result"
python -m src.runners.run_suite --suite configs/experiment_design/suites/design_suite_full.yaml --group "Mean Comparison"
python -m src.runners.run_suite --suite configs/experiment_design/suites/design_suite_full.yaml --group "Vol Comparison"
python -m src.runners.run_suite --suite configs/experiment_design/suites/design_suite_full.yaml --group "Corr Comparison"
python -m src.runners.run_suite --suite configs/experiment_design/suites/design_suite_full.yaml --group "Ablation"
python -m src.runners.run_suite --suite configs/experiment_design/suites/design_suite_full.yaml --group "Robustness"
python -m src.runners.run_suite --suite configs/experiment_design/suites/design_suite_full.yaml --group "Generalization"
```

## Output Structure

By default, the design suite writes results to:

```text
outputs_experiment_design/
```

Each experiment directory contains exported summaries, raw portfolio paths, weights, predictions, and metadata. These outputs are ignored by Git because they can be regenerated from the code, configurations, and data snapshots.

## Validation Tests

Run the available protocol tests with:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

The tests cover the rolling splitter protocol, optimizer weight-cap feasibility, and DWT no-future-leakage behavior.

## Reproducibility Notes

- Random seeds are specified in the YAML configurations.
- Forward return and volatility labels are generated inside each rolling block to avoid forward-label leakage.
- DWT features are built from the visible historical window only.
- The main suite uses long-only constrained MVO with the same transaction-cost convention across comparable strategies.
- If raw data cannot be redistributed, the GitHub repository should still include the exact code, YAML configurations, asset-universe construction scripts, and data-availability statement.
