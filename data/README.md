# Data Directory

This directory is reserved for local market data snapshots used by the portfolio experiments.

The GitHub repository intentionally excludes raw and processed data files by default. Some market data sources may impose licensing or redistribution restrictions. To reproduce the experiments, place the required CSV files in the paths referenced by the YAML configurations.

Main 30-asset experiment:

```text
data/processed/pool30/returns.csv
data/processed/pool30/close.csv
data/processed/pool30/market_cap.csv
```

Generalization and robustness experiments may also use:

```text
data/processed/pool50/
data/processed/pool100/
data/processed/market2/
data/processed_initialdate/
data/processed_growthstyle/
data/processed_balanced3m/
data/processed_liquidity3m/
data/processed_liquidity6m/
```

All CSV files should use dates as the first column and asset identifiers as column names, matching the conventions expected by `src/data/loader.py`.

