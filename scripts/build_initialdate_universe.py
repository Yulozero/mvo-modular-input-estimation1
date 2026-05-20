import json
from pathlib import Path

import numpy as np
import pandas as pd


CN_RENAME = {
    "股票代码": "code",
    "股票名称": "name",
    "日期": "date",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
    "成交额": "amount",
    "流通股本": "float_shares",
    "换手率": "turnover",
}


def _read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to decode {path}")


def _to_wide(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    wide = df.pivot(index="date", columns="code", values=value_col)
    return wide.sort_index().sort_index(axis=1)


def build_initialdate_cn_universe(
    raw_path: str,
    output_dir: str,
    pool_sizes=(30, 50, 100),
    missing_threshold: float = 0.10,
    start_date: str = "2019-01-02",
    end_date: str = "2024-12-31",
):
    raw = _read_csv(Path(raw_path)).rename(columns=CN_RENAME).copy()
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["code"] = raw["code"].astype(str).str.strip().str.replace(".0", "", regex=False).str.zfill(6)
    raw["name"] = raw["name"].astype(str).str.strip()

    numeric_cols = ["open", "high", "low", "close", "volume", "amount", "float_shares", "turnover"]
    for col in numeric_cols:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw = raw.dropna(subset=["date"]).drop_duplicates(subset=["code", "date"]).sort_values(["code", "date"])
    raw = raw.loc[(raw["date"] >= pd.to_datetime(start_date)) & (raw["date"] <= pd.to_datetime(end_date))].copy()

    for col in ["open", "high", "low", "close"]:
        raw.loc[raw[col] <= 0, col] = np.nan
    for col in ["volume", "amount", "float_shares", "turnover"]:
        raw.loc[raw[col] < 0, col] = np.nan

    raw["market_cap"] = raw["close"] * raw["float_shares"]
    total_days = raw["date"].nunique()

    stats = []
    for code, g in raw.groupby("code"):
        stats.append(
            {
                "code": code,
                "name": g["name"].iloc[0],
                "traded_days": int(g["date"].nunique()),
                "total_days": int(total_days),
                "coverage_ratio": float(g["date"].nunique() / max(total_days, 1)),
                "missing_ratio": float(g["close"].isna().mean()),
                "initial_market_cap": float(g.sort_values("date")["market_cap"].dropna().iloc[0]) if g["market_cap"].notna().any() else np.nan,
                "avg_market_cap": float(g["market_cap"].mean()) if g["market_cap"].notna().any() else np.nan,
            }
        )

    stats_df = pd.DataFrame(stats)
    filtered = stats_df.loc[
        (stats_df["coverage_ratio"] >= 1.0 - missing_threshold)
        & (stats_df["missing_ratio"] <= missing_threshold)
        & stats_df["initial_market_cap"].notna()
    ].copy()
    filtered = filtered.sort_values(["initial_market_cap", "coverage_ratio"], ascending=[False, False]).reset_index(drop=True)

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(out_root / "filtered_stock_stats.csv", index=False, encoding="utf-8-sig")

    policy = {
        "universe_selection_policy": "largest_initial_market_cap_after_quality_filter",
        "missing_threshold": missing_threshold,
        "start_date": start_date,
        "end_date": end_date,
    }
    with open(out_root / "processing_policy.json", "w", encoding="utf-8") as f:
        json.dump(policy, f, ensure_ascii=False, indent=2)

    for size in pool_sizes:
        if len(filtered) < size:
            continue
        selected = filtered.head(size).copy()
        selected_codes = selected["code"].tolist()
        sub = raw.loc[raw["code"].isin(selected_codes)].copy()

        pool_dir = out_root / f"pool{size}"
        pool_dir.mkdir(parents=True, exist_ok=True)

        close_wide = _to_wide(sub, "close").ffill()
        open_wide = _to_wide(sub, "open").ffill()
        high_wide = _to_wide(sub, "high").ffill()
        low_wide = _to_wide(sub, "low").ffill()
        volume_wide = _to_wide(sub, "volume")
        amount_wide = _to_wide(sub, "amount")
        turnover_wide = _to_wide(sub, "turnover")
        float_shares_wide = _to_wide(sub, "float_shares").ffill()
        market_cap_wide = close_wide * float_shares_wide
        returns_wide = close_wide.pct_change().replace([np.inf, -np.inf], np.nan)

        close_wide.to_csv(pool_dir / "close.csv", encoding="utf-8-sig")
        open_wide.to_csv(pool_dir / "open.csv", encoding="utf-8-sig")
        high_wide.to_csv(pool_dir / "high.csv", encoding="utf-8-sig")
        low_wide.to_csv(pool_dir / "low.csv", encoding="utf-8-sig")
        volume_wide.to_csv(pool_dir / "volume.csv", encoding="utf-8-sig")
        amount_wide.to_csv(pool_dir / "amount.csv", encoding="utf-8-sig")
        turnover_wide.to_csv(pool_dir / "turnover.csv", encoding="utf-8-sig")
        float_shares_wide.to_csv(pool_dir / "float_shares.csv", encoding="utf-8-sig")
        market_cap_wide.to_csv(pool_dir / "market_cap.csv", encoding="utf-8-sig")
        returns_wide.to_csv(pool_dir / "returns.csv", encoding="utf-8-sig")
        selected.to_csv(pool_dir / "selected_stocks.csv", index=False, encoding="utf-8-sig")

        summary = {
            "raw_path": str(raw_path),
            "pool_size": int(size),
            "selection_policy": "largest_initial_market_cap_after_quality_filter",
            "n_assets": int(size),
            "n_dates": int(returns_wide.shape[0]),
            "date_start": str(close_wide.index.min().date()),
            "date_end": str(close_wide.index.max().date()),
        }
        with open(pool_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    build_initialdate_cn_universe(
        raw_path=r"D:\Python_program\a1\data\raw\沪深300_全部数据_2019-2024.csv",
        output_dir=r"D:\Python_program\a1\portfolio_research\data\processed_initialdate",
    )
