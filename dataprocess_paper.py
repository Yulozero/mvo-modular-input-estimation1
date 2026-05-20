import json
import os
from typing import Iterable

import numpy as np
import pandas as pd


def _read_raw_csv(raw_path: str) -> pd.DataFrame:
    for encoding in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return pd.read_csv(raw_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("csv", b"", 0, 1, f"Unable to decode {raw_path}")


def _rename_columns(df: pd.DataFrame, rename_map: dict) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    return df.rename(columns=rename_map)


def _to_wide(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    wide = df.pivot(index="date", columns="code", values=value_col)
    return wide.sort_index().sort_index(axis=1)


def _safe_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _summarize_asset_quality(sub: pd.DataFrame, total_days: int) -> pd.DataFrame:
    rows = []
    for code, g in sub.groupby("code"):
        close = g["close"]
        returns = close.pct_change()
        rows.append({
            "code": code,
            "name": g["name"].iloc[0] if "name" in g.columns else code,
            "n_rows": int(len(g)),
            "coverage_ratio": float(g["date"].nunique() / max(total_days, 1)),
            "close_na_ratio": float(close.isna().mean()),
            "return_na_ratio": float(returns.isna().mean()),
            "avg_market_cap": float(g["market_cap"].mean()) if "market_cap" in g.columns else np.nan,
            "avg_amount": float(g["amount"].mean()) if "amount" in g.columns else np.nan,
        })
    return pd.DataFrame(rows)


def _build_return_outlier_report(returns_wide: pd.DataFrame) -> pd.DataFrame:
    stacked = returns_wide.stack(dropna=True).rename("return").reset_index()
    if stacked.empty:
        return pd.DataFrame(columns=["date", "code", "return", "abs_return", "rule"])

    stacked["abs_return"] = stacked["return"].abs()
    threshold = max(float(stacked["abs_return"].quantile(0.999)), 0.20)
    outliers = stacked.loc[stacked["abs_return"] >= threshold].copy()
    outliers["rule"] = f"abs_return >= max(q99.9, 0.20), threshold={threshold:.6f}"
    return outliers.sort_values(["abs_return", "date"], ascending=[False, True]).reset_index(drop=True)


def prepare_stock_pool_and_process_paper(
    raw_path: str,
    output_base_dir: str = "data/processed_paper",
    pool_sizes=(30, 50, 100),
    missing_threshold: float = 0.10,
    code_col="股票代码",
    name_col="股票名称",
    date_col="日期",
    open_col="开盘",
    high_col="最高",
    low_col="最低",
    close_col="收盘",
    volume_col="成交量",
    amount_col="成交额",
    float_shares_col="流通股本",
    turnover_col="换手率",
    start_date=None,
    end_date=None,
):
    os.makedirs(output_base_dir, exist_ok=True)

    raw = _read_raw_csv(raw_path)
    raw = _rename_columns(
        raw,
        {
            code_col: "code",
            name_col: "name",
            date_col: "date",
            open_col: "open",
            high_col: "high",
            low_col: "low",
            close_col: "close",
            volume_col: "volume",
            amount_col: "amount",
            float_shares_col: "float_shares",
            turnover_col: "turnover",
        },
    )

    required_cols = ["code", "name", "date", "open", "high", "low", "close", "volume"]
    missing_cols = [c for c in required_cols if c not in raw.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    for col in ["amount", "float_shares", "turnover"]:
        if col not in raw.columns:
            raw[col] = np.nan

    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["code"] = raw["code"].astype(str).str.strip().str.replace(".0", "", regex=False)
    raw["name"] = raw["name"].astype(str).str.strip()
    raw = _safe_numeric(raw, ["open", "high", "low", "close", "volume", "amount", "float_shares", "turnover"])
    raw = raw.dropna(subset=["date"]).drop_duplicates(subset=["code", "date"]).sort_values(["code", "date"])

    if start_date is not None:
        raw = raw.loc[raw["date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        raw = raw.loc[raw["date"] <= pd.to_datetime(end_date)]
    if raw.empty:
        raise ValueError("No data left after date filtering.")

    for col in ["open", "high", "low", "close"]:
        raw.loc[raw[col] <= 0, col] = np.nan
    for col in ["volume", "amount", "float_shares", "turnover"]:
        raw.loc[raw[col] < 0, col] = np.nan

    if raw["amount"].isna().all():
        raw["amount"] = raw["close"] * raw["volume"]
    else:
        mask = raw["amount"].isna() & raw["close"].notna() & raw["volume"].notna()
        raw.loc[mask, "amount"] = raw.loc[mask, "close"] * raw.loc[mask, "volume"]

    raw["market_cap"] = raw["close"] * raw["float_shares"]
    total_days = raw["date"].nunique()
    quality = _summarize_asset_quality(raw, total_days)
    filtered = quality.loc[
        (quality["coverage_ratio"] >= 1.0 - missing_threshold)
        & (quality["close_na_ratio"] <= missing_threshold)
    ].copy()

    if filtered["avg_market_cap"].notna().any():
        filtered = filtered.sort_values(["avg_market_cap", "coverage_ratio"], ascending=[False, False])
        universe_policy = "largest_average_market_cap_after_quality_filter"
    else:
        filtered = filtered.sort_values(["avg_amount", "coverage_ratio"], ascending=[False, False])
        universe_policy = "largest_average_amount_after_quality_filter"

    filtered = filtered.reset_index(drop=True)

    processing_policy = {
        "price_fill_policy": "forward_fill_only",
        "return_fill_policy": "preserve_nan_except_structural_first_diff_nan",
        "strict_mode": True,
        "universe_selection_policy": universe_policy,
        "missing_threshold": missing_threshold,
        "price_backfill_allowed": False,
        "global_return_fillna_zero": False,
    }

    with open(os.path.join(output_base_dir, "processing_policy.json"), "w", encoding="utf-8") as f:
        json.dump(processing_policy, f, ensure_ascii=False, indent=2)

    quality.to_csv(os.path.join(output_base_dir, "asset_quality_report.csv"), index=False, encoding="utf-8-sig")

    for size in sorted(set(pool_sizes)):
        if len(filtered) < size:
            continue

        selected_codes = filtered.head(size)["code"].tolist()
        pool_dir = os.path.join(output_base_dir, f"pool{size}")
        os.makedirs(pool_dir, exist_ok=True)

        sub = raw.loc[raw["code"].isin(selected_codes)].copy()
        close_wide = _to_wide(sub, "close").ffill()
        open_wide = _to_wide(sub, "open").ffill()
        high_wide = _to_wide(sub, "high").ffill()
        low_wide = _to_wide(sub, "low").ffill()
        volume_wide = _to_wide(sub, "volume")
        amount_wide = _to_wide(sub, "amount")
        turnover_wide = _to_wide(sub, "turnover")
        float_shares_wide = _to_wide(sub, "float_shares").ffill()
        market_cap_wide = close_wide * float_shares_wide

        returns_wide = close_wide.pct_change()
        returns_wide = returns_wide.replace([np.inf, -np.inf], np.nan)
        outlier_report = _build_return_outlier_report(returns_wide)
        outlier_report.to_csv(os.path.join(pool_dir, "return_outlier_report.csv"), index=False, encoding="utf-8-sig")

        close_wide.to_csv(os.path.join(pool_dir, "close.csv"), encoding="utf-8-sig")
        open_wide.to_csv(os.path.join(pool_dir, "open.csv"), encoding="utf-8-sig")
        high_wide.to_csv(os.path.join(pool_dir, "high.csv"), encoding="utf-8-sig")
        low_wide.to_csv(os.path.join(pool_dir, "low.csv"), encoding="utf-8-sig")
        volume_wide.to_csv(os.path.join(pool_dir, "volume.csv"), encoding="utf-8-sig")
        amount_wide.to_csv(os.path.join(pool_dir, "amount.csv"), encoding="utf-8-sig")
        turnover_wide.to_csv(os.path.join(pool_dir, "turnover.csv"), encoding="utf-8-sig")
        float_shares_wide.to_csv(os.path.join(pool_dir, "float_shares.csv"), encoding="utf-8-sig")
        market_cap_wide.to_csv(os.path.join(pool_dir, "market_cap.csv"), encoding="utf-8-sig")
        returns_wide.to_csv(os.path.join(pool_dir, "returns.csv"), encoding="utf-8-sig")
        filtered.head(size).to_csv(os.path.join(pool_dir, "selected_stocks.csv"), index=False, encoding="utf-8-sig")

        summary = {
            "raw_path": raw_path,
            "output_base_dir": output_base_dir,
            "pool_size": int(size),
            "price_fill_policy": processing_policy["price_fill_policy"],
            "return_fill_policy": processing_policy["return_fill_policy"],
            "strict_mode": processing_policy["strict_mode"],
            "universe_selection_policy": processing_policy["universe_selection_policy"],
            "date_start": str(sub["date"].min().date()),
            "date_end": str(sub["date"].max().date()),
            "n_assets": int(size),
            "n_dates": int(returns_wide.shape[0]),
        }
        with open(os.path.join(pool_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    prepare_stock_pool_and_process_paper(
        raw_path=r"D:\Python_program\a1\data\raw\沪深300_全部数据_2019-2024.csv",
        output_base_dir="data/processed_paper",
        pool_sizes=(30, 50, 100),
        missing_threshold=0.10,
    )
