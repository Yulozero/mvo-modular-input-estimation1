import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _read_csv_if_exists(path: Path):
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def _read_json_if_exists(path: Path):
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def collect_experiment_dirs(root_dir: str) -> List[Path]:
    root = Path(root_dir)
    if not root.exists():
        return []

    excluded_prefixes = {"_", "."}
    out = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if any(p.name.startswith(x) for x in excluded_prefixes):
            continue
        if (p / "meta" / "experiment_info.json").exists() or (p / "paper" / "paper_main_result.csv").exists():
            out.append(p)

    return sorted(out)


def load_experiment_bundle(exp_dir: Path) -> Dict:
    return {
        "exp_dir": str(exp_dir),
        "meta": _read_json_if_exists(exp_dir / "meta" / "experiment_info.json"),
        "config": _read_json_if_exists(exp_dir / "meta" / "config_used.json"),
        "summary_json": _read_json_if_exists(exp_dir / "summary" / "summary.json"),
        "summary_flat": _read_csv_if_exists(exp_dir / "summary" / "summary_flat.csv"),
        "portfolio_metrics": _read_csv_if_exists(exp_dir / "summary" / "portfolio_metrics.csv"),
        "mean_metrics": _read_csv_if_exists(exp_dir / "summary" / "mean_metrics.csv"),
        "vol_metrics": _read_csv_if_exists(exp_dir / "summary" / "vol_metrics.csv"),
        "cov_metrics": _read_csv_if_exists(exp_dir / "summary" / "cov_metrics.csv"),
        "paper_main": _read_csv_if_exists(exp_dir / "paper" / "paper_main_result.csv"),
        "paper_module_comparison": _read_csv_if_exists(exp_dir / "paper" / "paper_module_comparison.csv"),
        "decisions": _read_csv_if_exists(exp_dir / "raw" / "decisions.csv"),
        "weights": _read_csv_if_exists(exp_dir / "raw" / "weights.csv"),
        "predictions": _read_csv_if_exists(exp_dir / "raw" / "predictions.csv"),
        "rebalance_panel": _read_csv_if_exists(exp_dir / "raw" / "rebalance_panel.csv"),
        "raw_diagnostics": _read_json_if_exists(exp_dir / "raw" / "diagnostics.json"),
    }


def build_master_table(bundles: List[Dict]) -> pd.DataFrame:
    rows = []

    for b in bundles:
        paper_main = b.get("paper_main")
        meta = b.get("meta") or {}
        exp_dir = b.get("exp_dir")

        if paper_main is not None and not paper_main.empty:
            row = paper_main.iloc[0].to_dict()
        else:
            summary_json = b.get("summary_json") or {}
            row = {
                "preset_name": meta.get("preset_name"),
                "experiment_group": meta.get("experiment_group"),
                "variant": meta.get("variant"),
                "paper_tag": meta.get("paper_tag"),
                "mean_model": (meta.get("mean_model") or {}).get("name"),
                "vol_model": (meta.get("vol_model") or {}).get("name"),
                "corr_model": (meta.get("corr_model") or {}).get("name"),
                "optimizer": (meta.get("optimizer") or {}).get("name"),
                "annual_return": summary_json.get("net_annual_return_holding"),
                "annual_volatility": summary_json.get("net_annual_vol_holding"),
                "sharpe_ratio": summary_json.get("net_sharpe_holding"),
                "sortino_ratio": summary_json.get("net_sortino_holding"),
                "calmar_ratio": summary_json.get("net_calmar_holding"),
                "max_drawdown": summary_json.get("net_max_drawdown_holding"),
                "avg_turnover": summary_json.get("avg_turnover"),
                "net_annual_return": summary_json.get("net_annual_return_holding"),
                "net_sharpe_ratio": summary_json.get("net_sharpe_holding"),
                "mean_ic": (summary_json.get("mean_module_metrics") or {}).get("global_ic"),
                "mean_rank_ic": (summary_json.get("mean_module_metrics") or {}).get("global_rank_ic"),
                "vol_qlike": (summary_json.get("vol_module_metrics") or {}).get("qlike"),
                "cov_frobenius": (summary_json.get("cov_module_metrics") or {}).get("avg_frobenius_distance"),
            }

        row["exp_dir"] = exp_dir
        raw_diag = b.get("raw_diagnostics") or {}
        row["fallback_used"] = raw_diag.get("n_optimizer_fallback_steps", 0)
        row["solver_success_steps"] = raw_diag.get("n_optimizer_solver_success_steps", 0)
        if row.get("solver_success_steps") is not None and raw_diag.get("n_rebalances"):
            row["solver_success_rate"] = float(row["solver_success_steps"]) / float(raw_diag.get("n_rebalances"))
        else:
            row["solver_success_rate"] = None
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def split_by_group(master_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if master_df.empty or "experiment_group" not in master_df.columns:
        return {}

    result = {}
    for group, df_g in master_df.groupby("experiment_group"):
        result[group] = df_g.reset_index(drop=True)

    return result


def build_paper_package(master_df: pd.DataFrame) -> Dict:
    package = {
        "study_title": "基于均值、波动率与依赖结构分解的模块化投资组合实验设计（融合版，无KG）",
        "n_experiments": int(len(master_df)),
        "main_experiment": [],
        "baseline_results": [],
        "mean_comparison": [],
        "vol_comparison": [],
        "corr_comparison": [],
        "ablation_results": [],
        "robustness_results": [],
        "generalization_results": [],
        "main_conclusions": [],
    }

    if master_df.empty:
        package["main_conclusions"].append("当前没有可用实验结果。")
        return package

    def rec(df):
        return df.to_dict(orient="records")

    if "experiment_group" in master_df.columns:
        package["main_experiment"] = rec(master_df[master_df["experiment_group"] == "Main"])
        package["baseline_results"] = rec(master_df[master_df["experiment_group"] == "Baseline"])
        package["mean_comparison"] = rec(master_df[master_df["experiment_group"] == "Mean Comparison"])
        package["vol_comparison"] = rec(master_df[master_df["experiment_group"] == "Vol Comparison"])
        package["corr_comparison"] = rec(master_df[master_df["experiment_group"] == "Corr Comparison"])
        package["ablation_results"] = rec(master_df[master_df["experiment_group"] == "Ablation"])
        package["robustness_results"] = rec(master_df[master_df["experiment_group"] == "Robustness"])
        package["generalization_results"] = rec(master_df[master_df["experiment_group"] == "Generalization"])

    if "net_sharpe_ratio" in master_df.columns:
        dfv = master_df.dropna(subset=["net_sharpe_ratio"])
        if not dfv.empty:
            best = dfv.sort_values("net_sharpe_ratio", ascending=False).iloc[0]
            package["main_conclusions"].append(
                f"全样本中净夏普最高的实验为 {best.get('variant')}（{best.get('experiment_group')}），"
                f"对应均值模块={best.get('mean_model')}、波动率模块={best.get('vol_model')}、"
                f"依赖结构模块={best.get('corr_model')}、优化器={best.get('optimizer')}，"
                f"净夏普为 {best.get('net_sharpe_ratio')}。"
            )

    for group in ["Baseline", "Mean Comparison", "Vol Comparison", "Corr Comparison", "Ablation", "Robustness", "Generalization"]:
        if "experiment_group" not in master_df.columns:
            continue

        df_g = master_df[master_df["experiment_group"] == group]
        if df_g.empty or "net_sharpe_ratio" not in df_g.columns:
            continue

        dfv = df_g.dropna(subset=["net_sharpe_ratio"])
        if dfv.empty:
            continue

        row = dfv.sort_values("net_sharpe_ratio", ascending=False).iloc[0]
        package["main_conclusions"].append(
            f"{group} 组中表现最优的是 {row.get('variant')}，净夏普为 {row.get('net_sharpe_ratio')}。"
        )

    return package
