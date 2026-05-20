import os
import json
import pandas as pd
import numpy as np


def prepare_stock_pool_and_process(
    raw_path: str,
    output_base_dir: str = "data/processed/market2",
    pool_sizes=(30, 50, 100),
    missing_threshold=0.10,
    code_col="股票代码",
    name_col="公司名称",
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
    """
    读取原始股票日频数据，筛选股票池并生成宽表文件。
    支持生成 pool30 / pool50 / pool100 等多个股票池。

    兼容以下情况：
    1. 原始数据只有 OHLCV + 公司名称，没有成交额/流通股本/换手率
    2. 若缺少成交额，则使用 close * volume 近似生成 amount
    3. 若缺少流通股本，则不按市值排序，改按成交额/成交量排序
    4. 若缺少换手率，则自动补 0

    输出文件包括：
    - returns.csv
    - close.csv / open.csv / high.csv / low.csv
    - volume.csv / amount.csv / turnover.csv
    - float_shares.csv / market_cap.csv
    - selected_stocks.csv
    - daily_long_cleaned.csv
    - all_stock_stats.csv
    - filtered_stock_stats.csv
    - data_quality_report.csv
    """

    os.makedirs(output_base_dir, exist_ok=True)

    # =========================
    # 1. 读取原始数据
    # =========================
    try:
        df = pd.read_csv(raw_path, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(raw_path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(raw_path, encoding="gbk")

    # 去掉列名前后空格，避免隐藏空格导致识别失败
    df.columns = df.columns.astype(str).str.strip()

    # =========================
    # 2. 重命名字段
    # =========================
    rename_map = {
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
    }
    df = df.rename(columns=rename_map)

    # 核心必需字段
    required_cols = [
        "code", "name", "date", "open", "high", "low", "close", "volume"
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"原始数据缺少必要字段: {missing_cols}\n"
            f"当前实际字段为: {list(df.columns)}"
        )

    # 可选字段自动补空
    for col in ["amount", "float_shares", "turnover"]:
        if col not in df.columns:
            df[col] = np.nan

    # =========================
    # 3. 数据类型处理
    # =========================
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["code"] = df["code"].astype(str).str.strip()
    df["name"] = df["name"].astype(str).str.strip()

    # 统一股票代码格式，避免 000001.0 这种情况
    df["code"] = df["code"].str.replace(".0", "", regex=False)

    num_cols = ["open", "high", "low", "close", "volume", "amount", "float_shares", "turnover"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date"])
    df = df.drop_duplicates(subset=["code", "date"])
    df = df.sort_values(["code", "date"]).reset_index(drop=True)

    # 可选时间截面
    if start_date is not None:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df["date"] <= pd.to_datetime(end_date)]

    if df.empty:
        raise ValueError("筛选日期后数据为空，请检查 start_date / end_date 设置。")

    # =========================
    # 4. 对明显不合理值先做清理
    # =========================
    # 价格不应 <= 0
    for col in ["open", "high", "low", "close"]:
        df.loc[df[col] <= 0, col] = np.nan

    # 成交量/成交额/股本/换手率不应为负
    for col in ["volume", "amount", "float_shares", "turnover"]:
        df.loc[df[col] < 0, col] = np.nan

    # 如果 amount 缺失，则用 close * volume 近似生成
    if df["amount"].isna().all():
        df["amount"] = df["close"] * df["volume"]
    else:
        fill_mask = df["amount"].isna() & df["close"].notna() & df["volume"].notna()
        df.loc[fill_mask, "amount"] = df.loc[fill_mask, "close"] * df.loc[fill_mask, "volume"]

    # turnover 缺失先保留，后续宽表时补 0
    # float_shares 若缺失，不强制补值，因为可能根本没有该信息

    # =========================
    # 5. 计算原始流通市值
    # =========================
    if df["float_shares"].notna().any():
        df["market_cap"] = df["close"] * df["float_shares"]
    else:
        df["market_cap"] = np.nan

    # =========================
    # 6. 统计每只股票的数据完整性
    # =========================
    all_dates = sorted(df["date"].dropna().unique())
    total_days = len(all_dates)

    stock_stats = []
    for code, g in df.groupby("code"):
        name = g["name"].iloc[0] if "name" in g.columns else code

        traded_days = g["date"].nunique()
        coverage_ratio = traded_days / total_days if total_days > 0 else 0.0
        missing_ratio = 1.0 - coverage_ratio

        avg_market_cap = g["market_cap"].mean()
        median_market_cap = g["market_cap"].median()
        avg_volume = g["volume"].mean()
        avg_amount = g["amount"].mean()

        # 只把真正核心的 OHLCV 纳入缺失率统计
        core_missing = g[["close", "open", "high", "low", "volume"]].isna().mean().mean()

        stock_stats.append({
            "code": code,
            "name": name,
            "traded_days": traded_days,
            "total_days": total_days,
            "coverage_ratio": coverage_ratio,
            "missing_ratio": missing_ratio,
            "core_missing_ratio": core_missing,
            "avg_market_cap": avg_market_cap,
            "median_market_cap": median_market_cap,
            "avg_volume": avg_volume,
            "avg_amount": avg_amount,
        })

    stats_df = pd.DataFrame(stock_stats)

    # =========================
    # 7. 过滤股票
    # =========================
    filtered = stats_df[
        (stats_df["missing_ratio"] <= missing_threshold) &
        (stats_df["core_missing_ratio"] <= missing_threshold)
    ].copy()

    # 优先按市值，其次成交额，再次成交量
    if not filtered.empty:
        if filtered["avg_market_cap"].notna().any():
            filtered = filtered.sort_values(
                ["avg_market_cap", "coverage_ratio"],
                ascending=[False, False]
            )
        elif filtered["avg_amount"].notna().any():
            filtered = filtered.sort_values(
                ["avg_amount", "coverage_ratio"],
                ascending=[False, False]
            )
        else:
            filtered = filtered.sort_values(
                ["avg_volume", "coverage_ratio"],
                ascending=[False, False]
            )

    filtered = filtered.reset_index(drop=True)

    print("原始股票数：", len(stats_df))
    print("过滤后股票数：", len(filtered))
    print(filtered.head(20))

    if len(filtered) < min(pool_sizes):
        raise ValueError(
            f"过滤后股票数不足，当前只有 {len(filtered)} 只，"
            f"但最小股票池需要 {min(pool_sizes)} 只。"
        )

    # =========================
    # 8. 宽表函数
    # =========================
    def to_wide(sub_df, value_col):
        wide = sub_df.pivot(index="date", columns="code", values=value_col)
        wide = wide.sort_index().sort_index(axis=1)
        return wide

    # =========================
    # 9. 单个股票池处理
    # =========================
    quality_rows = []

    def process_one_pool(selected_codes, pool_dir):
        os.makedirs(pool_dir, exist_ok=True)

        sub = df[df["code"].isin(selected_codes)].copy()
        sub = sub.sort_values(["code", "date"]).reset_index(drop=True)

        close_wide = to_wide(sub, "close")
        open_wide = to_wide(sub, "open")
        high_wide = to_wide(sub, "high")
        low_wide = to_wide(sub, "low")
        volume_wide = to_wide(sub, "volume")
        amount_wide = to_wide(sub, "amount")
        turnover_wide = to_wide(sub, "turnover")
        float_shares_wide = to_wide(sub, "float_shares")

        # 价格类：前向 + 后向
        close_wide = close_wide.ffill().bfill()
        open_wide = open_wide.ffill().bfill()
        high_wide = high_wide.ffill().bfill()
        low_wide = low_wide.ffill().bfill()

        # 成交类：补0
        volume_wide = volume_wide.fillna(0)
        amount_wide = amount_wide.fillna(0)
        turnover_wide = turnover_wide.fillna(0)

        # 股本：前向 + 后向；若全缺失则最终补0
        float_shares_wide = float_shares_wide.ffill().bfill()

        # 市值
        market_cap_wide = close_wide * float_shares_wide

        # 收益率
        returns_wide = close_wide.pct_change()
        returns_wide = returns_wide.replace([np.inf, -np.inf], np.nan)
        returns_wide = returns_wide.fillna(0)

        # 再兜底清理
        close_wide = close_wide.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
        open_wide = open_wide.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
        high_wide = high_wide.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
        low_wide = low_wide.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

        volume_wide = volume_wide.replace([np.inf, -np.inf], 0).fillna(0)
        amount_wide = amount_wide.replace([np.inf, -np.inf], 0).fillna(0)
        turnover_wide = turnover_wide.replace([np.inf, -np.inf], 0).fillna(0)

        float_shares_wide = (
            float_shares_wide.replace([np.inf, -np.inf], np.nan)
            .ffill()
            .bfill()
            .fillna(0)
        )
        market_cap_wide = (
            market_cap_wide.replace([np.inf, -np.inf], np.nan)
            .ffill()
            .bfill()
            .fillna(0)
        )

        returns_wide = returns_wide.replace([np.inf, -np.inf], 0).fillna(0)

        # 长表保存用 return
        sub = sub.sort_values(["date", "code"]).copy()
        returns_long = returns_wide.stack().rename("return").reset_index()
        returns_long.columns = ["date", "code", "return"]

        sub = sub.merge(returns_long, on=["date", "code"], how="left")
        sub["return"] = sub["return"].fillna(0)

        # 保存宽表
        returns_wide.to_csv(os.path.join(pool_dir, "returns.csv"), encoding="utf-8-sig")
        close_wide.to_csv(os.path.join(pool_dir, "close.csv"), encoding="utf-8-sig")
        open_wide.to_csv(os.path.join(pool_dir, "open.csv"), encoding="utf-8-sig")
        high_wide.to_csv(os.path.join(pool_dir, "high.csv"), encoding="utf-8-sig")
        low_wide.to_csv(os.path.join(pool_dir, "low.csv"), encoding="utf-8-sig")
        volume_wide.to_csv(os.path.join(pool_dir, "volume.csv"), encoding="utf-8-sig")
        amount_wide.to_csv(os.path.join(pool_dir, "amount.csv"), encoding="utf-8-sig")
        turnover_wide.to_csv(os.path.join(pool_dir, "turnover.csv"), encoding="utf-8-sig")
        float_shares_wide.to_csv(os.path.join(pool_dir, "float_shares.csv"), encoding="utf-8-sig")
        market_cap_wide.to_csv(os.path.join(pool_dir, "market_cap.csv"), encoding="utf-8-sig")

        # 保存股票池名单
        selected_info = filtered[filtered["code"].isin(selected_codes)].copy()
        sort_col = "avg_market_cap" if selected_info["avg_market_cap"].notna().any() else (
            "avg_amount" if selected_info["avg_amount"].notna().any() else "avg_volume"
        )
        selected_info = selected_info.sort_values(sort_col, ascending=False)
        selected_info.to_csv(
            os.path.join(pool_dir, "selected_stocks.csv"),
            index=False,
            encoding="utf-8-sig"
        )

        # 保存清洗后长表
        sub.to_csv(
            os.path.join(pool_dir, "daily_long_cleaned.csv"),
            index=False,
            encoding="utf-8-sig"
        )

        # 质量报告
        quality_row = {
            "pool_dir": pool_dir,
            "n_assets": len(selected_codes),
            "n_dates": len(returns_wide.index),
            "close_na": int(close_wide.isna().sum().sum()),
            "float_shares_na": int(float_shares_wide.isna().sum().sum()),
            "market_cap_na": int(market_cap_wide.isna().sum().sum()),
            "returns_na": int(returns_wide.isna().sum().sum()),
            "returns_inf": int(np.isinf(returns_wide.values).sum()),
            "avg_abs_return": float(np.abs(returns_wide.values).mean()),
            "max_abs_return": float(np.abs(returns_wide.values).max()),
        }
        quality_rows.append(quality_row)

        print(f"\n股票池已生成：{pool_dir}")
        print("returns shape:", returns_wide.shape)
        print("股票数:", len(selected_codes))
        print("样本日期数:", len(returns_wide.index))
        print("close 缺失值总数:", quality_row["close_na"])
        print("float_shares 缺失值总数:", quality_row["float_shares_na"])
        print("market_cap 缺失值总数:", quality_row["market_cap_na"])
        print("returns 缺失值总数:", quality_row["returns_na"])
        print("returns inf 数量:", quality_row["returns_inf"])

    # =========================
    # 10. 针对每个股票池分别生成
    # =========================
    for size in sorted(set(pool_sizes)):
        if len(filtered) < size:
            print(f"\n跳过 pool{size}：过滤后股票不足 {size} 只（当前 {len(filtered)} 只）")
            continue

        selected = filtered.head(size).copy()
        selected_codes = selected["code"].tolist()

        pool_dir = os.path.join(output_base_dir, f"pool{size}")
        process_one_pool(selected_codes, pool_dir)

    # =========================
    # 11. 保存统计信息
    # =========================
    stats_df.to_csv(
        os.path.join(output_base_dir, "all_stock_stats.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    filtered.to_csv(
        os.path.join(output_base_dir, "filtered_stock_stats.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    quality_df = pd.DataFrame(quality_rows)
    quality_df.to_csv(
        os.path.join(output_base_dir, "data_quality_report.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    summary = {
        "raw_path": raw_path,
        "output_base_dir": output_base_dir,
        "pool_sizes": list(pool_sizes),
        "missing_threshold": missing_threshold,
        "n_raw_stocks": int(len(stats_df)),
        "n_filtered_stocks": int(len(filtered)),
        "date_start": str(df["date"].min().date()) if not df.empty else None,
        "date_end": str(df["date"].max().date()) if not df.empty else None,
        "has_amount_data": bool(df["amount"].notna().any()),
        "has_float_shares_data": bool(df["float_shares"].notna().any()),
        "has_turnover_data": bool(df["turnover"].notna().any()),
    }
    with open(os.path.join(output_base_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n全部完成。")


if __name__ == "__main__":
    prepare_stock_pool_and_process(
        raw_path=r"D:\Python_program\a1\data\raw\标普500_全部数据_2019-2024.csv",
        output_base_dir="data/processed/market2",
        pool_sizes=(30, 50, 100),
        missing_threshold=0.10,
        name_col="公司名称"
    )
