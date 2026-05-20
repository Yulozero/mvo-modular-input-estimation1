import json
import time
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to decode {path}")


def _standardize_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    expected = [
        "code",
        "name",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "float_shares",
        "turnover",
    ]
    if len(df.columns) < len(expected):
        raise ValueError(f"Unexpected raw column count: {len(df.columns)}")
    rename_map = {old: new for old, new in zip(df.columns[: len(expected)], expected)}
    return df.rename(columns=rename_map)


def _to_wide(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    wide = df.pivot(index="date", columns="code", values=value_col)
    return wide.sort_index().sort_index(axis=1)


def _safe_industry_lookup(code: str, sleep_seconds: float = 0.3) -> str:
    try:
        info = ak.stock_individual_info_em(symbol=code)
        industry_rows = info.loc[info["item"] == "行业", "value"]
        if not industry_rows.empty:
            industry = str(industry_rows.iloc[0]).strip()
            if industry:
                time.sleep(sleep_seconds)
                return industry
    except Exception:
        pass
    time.sleep(sleep_seconds)
    return "未知行业"


def _attach_industries(stats_df: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        cache = pd.read_csv(cache_path, encoding="utf-8-sig")
    else:
        cache = pd.DataFrame(columns=["code", "industry"])

    cache["code"] = cache["code"].astype(str).str.strip().str.zfill(6)
    industry_map = dict(zip(cache["code"], cache["industry"]))

    missing_codes = [c for c in stats_df["code"].tolist() if c not in industry_map]
    if missing_codes:
        new_rows = []
        for code in missing_codes:
            industry = _safe_industry_lookup(code)
            new_rows.append({"code": code, "industry": industry})
        if new_rows:
            cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
            cache = cache.drop_duplicates(subset=["code"], keep="last")
            cache.to_csv(cache_path, index=False, encoding="utf-8-sig")
            industry_map = dict(zip(cache["code"], cache["industry"]))

    out = stats_df.copy()
    out["industry"] = out["code"].map(industry_map).fillna("未知行业")
    return out


def _rank_score(series: pd.Series) -> pd.Series:
    return series.rank(method="average", pct=True)


def build_initial_balanced_scored_cn_universe(
    raw_path: str,
    output_dir: str,
    score_days: int = 63,
    pool_sizes=(30, 50, 100),
    missing_threshold: float = 0.10,
    start_date: str = "2019-01-02",
    end_date: str = "2024-12-31",
    industry_cap: int = 3,
):
    raw = _standardize_raw_columns(_read_csv(Path(raw_path))).copy()
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
    score_dates = sorted(raw["date"].dropna().unique())[:score_days]
    initial_cut = raw.loc[raw["date"].isin(score_dates)].copy()

    stats = []
    for code, g in raw.groupby("code"):
        init_g = initial_cut.loc[initial_cut["code"] == code]
        stats.append(
            {
                "code": code,
                "name": g["name"].iloc[0],
                "traded_days": int(g["date"].nunique()),
                "total_days": int(total_days),
                "coverage_ratio": float(g["date"].nunique() / max(total_days, 1)),
                "missing_ratio": float(g["close"].isna().mean()),
                "avg_initial_amount": float(init_g["amount"].mean()) if init_g["amount"].notna().any() else np.nan,
                "avg_initial_turnover": float(init_g["turnover"].mean()) if init_g["turnover"].notna().any() else np.nan,
                "avg_initial_float_mcap": float((init_g["close"] * init_g["float_shares"]).mean())
                if init_g["close"].notna().any() and init_g["float_shares"].notna().any()
                else np.nan,
                "initial_obs": int(init_g["close"].notna().sum()),
            }
        )

    stats_df = pd.DataFrame(stats)
    min_initial_obs = max(20, int(score_days * 0.7))
    filtered = stats_df.loc[
        (stats_df["coverage_ratio"] >= 1.0 - missing_threshold)
        & (stats_df["missing_ratio"] <= missing_threshold)
        & (stats_df["initial_obs"] >= min_initial_obs)
        & stats_df["avg_initial_amount"].notna()
        & stats_df["avg_initial_turnover"].notna()
        & stats_df["avg_initial_float_mcap"].notna()
    ].copy()

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    industry_cache = out_root / "industry_cache.csv"
    filtered = _attach_industries(filtered, industry_cache)

    filtered["score_amount"] = _rank_score(filtered["avg_initial_amount"])
    filtered["score_float_mcap"] = _rank_score(filtered["avg_initial_float_mcap"])
    filtered["score_turnover"] = _rank_score(filtered["avg_initial_turnover"])
    filtered["composite_score"] = (
        0.45 * filtered["score_amount"]
        + 0.35 * filtered["score_float_mcap"]
        + 0.20 * filtered["score_turnover"]
    )
    filtered = filtered.sort_values(
        ["composite_score", "score_amount", "score_float_mcap"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    filtered.to_csv(out_root / "filtered_stock_stats.csv", index=False, encoding="utf-8-sig")

    policy = {
        "universe_selection_policy": "first_3m_industry_balanced_composite_score_after_quality_filter",
        "score_days": score_days,
        "missing_threshold": missing_threshold,
        "start_date": start_date,
        "end_date": end_date,
        "industry_cap": industry_cap,
        "score_weights": {
            "avg_initial_amount": 0.45,
            "avg_initial_float_mcap": 0.35,
            "avg_initial_turnover": 0.20,
        },
    }
    with open(out_root / "processing_policy.json", "w", encoding="utf-8") as f:
        json.dump(policy, f, ensure_ascii=False, indent=2)

    for size in pool_sizes:
        if len(filtered) < size:
            continue

        selected_rows = []
        industry_counts = {}
        for _, row in filtered.iterrows():
            industry = row["industry"]
            if industry_counts.get(industry, 0) >= industry_cap:
                continue
            selected_rows.append(row)
            industry_counts[industry] = industry_counts.get(industry, 0) + 1
            if len(selected_rows) >= size:
                break

        if len(selected_rows) < size:
            selected_codes = {row["code"] for row in selected_rows}
            for _, row in filtered.iterrows():
                if row["code"] in selected_codes:
                    continue
                selected_rows.append(row)
                selected_codes.add(row["code"])
                if len(selected_rows) >= size:
                    break

        selected = pd.DataFrame(selected_rows).reset_index(drop=True)
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
            "selection_policy": "first_3m_industry_balanced_composite_score_after_quality_filter",
            "score_days": int(score_days),
            "industry_cap": int(industry_cap),
            "n_assets": int(size),
            "n_dates": int(returns_wide.shape[0]),
            "date_start": str(close_wide.index.min().date()),
            "date_end": str(close_wide.index.max().date()),
            "n_industries": int(selected["industry"].nunique()),
        }
        with open(pool_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    build_initial_balanced_scored_cn_universe(
        raw_path=r"D:\Python_program\a1\data\raw\沪深300_全部数据_2019-2024.csv",
        output_dir=r"D:\Python_program\a1\portfolio_research\data\processed_balanced3m",
        score_days=63,
        industry_cap=3,
    )
