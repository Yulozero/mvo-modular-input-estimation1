import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.utils.config import flatten_dict


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], path: str):
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)


def save_dataframe(df: pd.DataFrame, path: str):
    ensure_dir(str(Path(path).parent))
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _json_default(x):
    if isinstance(x, (np.integer, )):
        return int(x)
    if isinstance(x, (np.floating, )):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if hasattr(x, "isoformat"):
        try:
            return x.isoformat()
        except Exception:
            pass
    if is_dataclass(x):
        return asdict(x)
    return str(x)


def _adapt_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    mean_m = summary.get("mean_module_metrics", {}) or {}
    vol_m = summary.get("vol_module_metrics", {}) or {}
    cov_m = summary.get("cov_module_metrics", {}) or {}

    portfolio_metrics = {
        "annual_return": summary.get("net_annual_return_holding"),
        "net_of_cost_annual_return": summary.get("net_of_cost_annual_return_holding"),
        "annual_volatility": summary.get("net_annual_vol_holding"),
        "sharpe_ratio": summary.get("net_sharpe_holding"),
        "sortino_ratio": summary.get("net_sortino_holding"),
        "calmar_ratio": summary.get("net_calmar_holding"),
        "max_drawdown": summary.get("net_max_drawdown_holding"),
        "avg_turnover": summary.get("avg_turnover"),
        "avg_transaction_cost": summary.get("avg_transaction_cost"),
        "avg_cost_drag": summary.get("avg_cost_drag"),
        "gross_minus_net_annual_return_gap": summary.get("gross_minus_net_annual_return_gap"),
        "gross_minus_net_sharpe_gap": summary.get("gross_minus_net_sharpe_gap"),
        "net_annual_return": summary.get("net_annual_return_holding"),
        "net_sharpe_ratio": summary.get("net_sharpe_holding"),
        "avg_expected_risk": summary.get("avg_expected_risk"),
        "avg_concentration_hhi": summary.get("avg_concentration_hhi"),
        "avg_effective_n_assets": summary.get("avg_effective_n_assets"),
        "avg_max_weight": summary.get("avg_max_weight"),
        "n_rebalances": summary.get("n_rebalances"),
    }

    mean_metrics = {
        "mse": mean_m.get("mse"),
        "mae": mean_m.get("mae"),
        "hit_ratio": mean_m.get("hit_ratio"),
        "ic": mean_m.get("global_ic"),
        "rank_ic": mean_m.get("global_rank_ic"),
        "avg_cross_sectional_ic": mean_m.get("avg_cross_sectional_ic"),
        "avg_cross_sectional_rank_ic": mean_m.get("avg_cross_sectional_rank_ic"),
        "median_cross_sectional_ic": mean_m.get("median_cross_sectional_ic"),
        "median_cross_sectional_rank_ic": mean_m.get("median_cross_sectional_rank_ic"),
        "n_obs": mean_m.get("n_obs"),
        "n_rebalances_used": mean_m.get("n_rebalances_used"),
    }

    vol_metrics = {
        "mse": vol_m.get("mse"),
        "rmse": vol_m.get("rmse"),
        "mae": vol_m.get("mae"),
        "median_abs_error": vol_m.get("median_abs_error"),
        "qlike": vol_m.get("qlike"),
        "n_obs": vol_m.get("n_obs"),
    }

    cov_metrics = {
        "frobenius": cov_m.get("avg_frobenius_distance"),
        "median_frobenius_distance": cov_m.get("median_frobenius_distance"),
        "avg_ex_ante_variance": cov_m.get("avg_ex_ante_variance"),
        "avg_ex_post_variance": cov_m.get("avg_ex_post_variance"),
        "avg_abs_risk_bias": cov_m.get("avg_abs_risk_bias"),
        "n_rebalances_used": cov_m.get("n_rebalances_used"),
    }

    return {
        "portfolio_metrics": portfolio_metrics,
        "mean_metrics": mean_metrics,
        "vol_metrics": vol_metrics,
        "cov_metrics": cov_metrics,
    }


def export_config_and_meta(config: Dict[str, Any], output_dir: str):
    meta_dir = Path(output_dir) / "meta"
    ensure_dir(str(meta_dir))

    save_json(config, str(meta_dir / "config_used.json"))

    exp_info = {
        "preset_name": config.get("_meta", {}).get("preset_name"),
        "description": config.get("_meta", {}).get("description", ""),
        "experiment_group": config.get("experiment", {}).get("group"),
        "variant": config.get("experiment", {}).get("variant"),
        "paper_tag": config.get("experiment", {}).get("paper_tag"),
        "seed": config.get("seed"),
        "data": config.get("data", {}),
        "split": config.get("split", {}),
        "targets": config.get("targets", {}),
        "backtest": config.get("backtest", {}),
        "mean_model": config.get("mean_model", {}),
        "vol_model": config.get("vol_model", {}),
        "corr_model": config.get("corr_model", {}),
        "optimizer": config.get("optimizer", {}),
        "kg": config.get("kg", config.get("kg_data", {})),
    }
    save_json(exp_info, str(meta_dir / "experiment_info.json"))


def export_summary_files(summary: Dict[str, Any], output_dir: str):
    summary_dir = Path(output_dir) / "summary"
    ensure_dir(str(summary_dir))

    save_json(summary, str(summary_dir / "summary.json"))

    flat = flatten_dict(summary)
    flat_df = pd.DataFrame([flat])
    save_dataframe(flat_df, str(summary_dir / "summary_flat.csv"))

    adapted = _adapt_summary(summary)

    save_dataframe(pd.DataFrame([adapted["portfolio_metrics"]]), str(summary_dir / "portfolio_metrics.csv"))
    save_dataframe(pd.DataFrame([adapted["mean_metrics"]]), str(summary_dir / "mean_metrics.csv"))
    save_dataframe(pd.DataFrame([adapted["vol_metrics"]]), str(summary_dir / "vol_metrics.csv"))
    save_dataframe(pd.DataFrame([adapted["cov_metrics"]]), str(summary_dir / "cov_metrics.csv"))


def _normalize_results(engine_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    decisions = []
    weights = []
    predictions = []
    attention_weights = []
    rebalance_panel = []
    diagnostics = {
        "n_rebalances": len(engine_results),
        "n_optimizer_fallback_steps": 0,
        "n_optimizer_solver_success_steps": 0,
        "optimizer_status_counts": {},
    }

    for r in engine_results:
        d = r.get("decision")
        if d is not None:
            row = {
                "date": getattr(d, "date", None),
                "expected_return": getattr(d, "expected_return", None),
                "expected_risk": getattr(d, "expected_risk", None),
                "turnover": getattr(d, "turnover", None),
                "transaction_cost": getattr(d, "transaction_cost", None),
                "cost_drag": (getattr(d, "meta", {}) or {}).get("cost_drag"),
                "initial_cost_charged": (getattr(d, "meta", {}) or {}).get("initial_cost_charged"),
                "realized_return": getattr(d, "realized_return", None),
                "realized_net_return": getattr(d, "realized_net_return", None),
                "n_test_days": (getattr(d, "meta", {}) or {}).get("n_test_days"),
                "concentration_hhi": (getattr(d, "meta", {}) or {}).get("concentration_hhi"),
                "effective_n_assets": (getattr(d, "meta", {}) or {}).get("effective_n_assets"),
                "max_weight": (getattr(d, "meta", {}) or {}).get("max_weight"),
                "gross_cumulative_path": (getattr(d, "meta", {}) or {}).get("gross_cumulative_path"),
                "net_cumulative_path": (getattr(d, "meta", {}) or {}).get("net_cumulative_path"),
                "rebalance_date": r.get("rebalance_date"),
                "predict_date": r.get("predict_date"),
            }
            decisions.append(row)

            asset_list = r.get("asset_list") or []
            w = np.asarray(getattr(d, "weights", []), dtype=float).reshape(-1)
            for asset, weight in zip(asset_list, w):
                weights.append({
                    "date": getattr(d, "date", None),
                    "rebalance_date": r.get("rebalance_date"),
                    "predict_date": r.get("predict_date"),
                    "asset": asset,
                    "weight": float(weight),
                })

        mu_hat = r.get("mu_hat")
        sigma_hat = r.get("sigma_hat")
        asset_list = r.get("asset_list") or []

        if mu_hat is not None:
            mu_hat = np.asarray(mu_hat, dtype=float).reshape(-1)
            for asset, val in zip(asset_list, mu_hat):
                predictions.append({
                    "predict_date": r.get("predict_date"),
                    "asset": asset,
                    "pred_type": "mean",
                    "pred_value": float(val),
                })

        if sigma_hat is not None:
            sigma_hat = np.asarray(sigma_hat, dtype=float).reshape(-1)
            for asset, val in zip(asset_list, sigma_hat):
                predictions.append({
                    "predict_date": r.get("predict_date"),
                    "asset": asset,
                    "pred_type": "vol",
                    "pred_value": float(val),
                })

        mean_diag = r.get("mean_diag") or {}
        last_attention = mean_diag.get("last_attention")
        if last_attention is not None:
            seq_len = len(last_attention[0]) if last_attention else 0
            weighted_raw = mean_diag.get("last_weighted_raw_mean") or [None] * len(asset_list)
            context_norm = mean_diag.get("last_weighted_context_norm") or [None] * len(asset_list)
            for asset_idx, asset in enumerate(asset_list):
                asset_attention = last_attention[asset_idx] if asset_idx < len(last_attention) else []
                asset_weighted_raw = weighted_raw[asset_idx] if asset_idx < len(weighted_raw) else None
                asset_context_norm = context_norm[asset_idx] if asset_idx < len(context_norm) else None
                for step_idx, alpha in enumerate(asset_attention):
                    attention_weights.append({
                        "predict_date": r.get("predict_date"),
                        "rebalance_date": r.get("rebalance_date"),
                        "asset": asset,
                        "asset_idx": asset_idx,
                        "seq_pos": step_idx,
                        "relative_lag": step_idx - seq_len + 1,
                        "attention_weight": float(alpha),
                        "mean_model": mean_diag.get("model"),
                        "weighted_raw_mean": asset_weighted_raw,
                        "weighted_context_norm": asset_context_norm,
                    })

        rebalance_panel.append({
            "rebalance_date": r.get("rebalance_date"),
            "predict_date": r.get("predict_date"),
            "n_assets": len(asset_list),
            "mean_model": mean_diag.get("model"),
            "vol_model": (r.get("vol_diag") or {}).get("model"),
            "corr_model": (r.get("corr_diag") or {}).get("model"),
            "optimizer": (r.get("optimizer_diag") or {}).get("optimizer"),
            "optimizer_fallback_count": (r.get("optimizer_diag") or {}).get("fallback_count"),
            "optimizer_solver_success_count": (r.get("optimizer_diag") or {}).get("solver_success_count"),
            "optimizer_solve_count": (r.get("optimizer_diag") or {}).get("solve_count"),
            "optimizer_last_status": ((r.get("optimizer_diag") or {}).get("last_diagnostics") or {}).get("status"),
        })

        optimizer_diag = r.get("optimizer_diag") or {}
        if optimizer_diag.get("used_fallback"):
            diagnostics["n_optimizer_fallback_steps"] += 1
        if optimizer_diag.get("solver_success"):
            diagnostics["n_optimizer_solver_success_steps"] += 1
        solver_name = optimizer_diag.get("solver_name") or "fallback"
        diagnostics["optimizer_status_counts"][solver_name] = diagnostics["optimizer_status_counts"].get(solver_name, 0) + 1

    return {
        "decisions": decisions,
        "weights": weights,
        "predictions": predictions,
        "attention_weights": attention_weights,
        "rebalance_panel": rebalance_panel,
        "diagnostics": diagnostics,
    }


def export_raw_results(engine_results: List[Dict[str, Any]], output_dir: str):
    raw_dir = Path(output_dir) / "raw"
    ensure_dir(str(raw_dir))

    results = _normalize_results(engine_results)

    decisions = pd.DataFrame(results["decisions"])
    weights = pd.DataFrame(results["weights"])
    predictions = pd.DataFrame(results["predictions"])
    attention_weights = pd.DataFrame(results["attention_weights"])
    rebalances = pd.DataFrame(results["rebalance_panel"])
    diagnostics = results["diagnostics"]

    if not decisions.empty:
        save_dataframe(decisions, str(raw_dir / "decisions.csv"))
    if not weights.empty:
        save_dataframe(weights, str(raw_dir / "weights.csv"))
    if not predictions.empty:
        save_dataframe(predictions, str(raw_dir / "predictions.csv"))
    if not attention_weights.empty:
        save_dataframe(attention_weights, str(raw_dir / "attention_weights.csv"))
    if not rebalances.empty:
        save_dataframe(rebalances, str(raw_dir / "rebalance_panel.csv"))

    save_json(diagnostics, str(raw_dir / "diagnostics.json"))


def export_paper_tables(config: Dict[str, Any], summary: Dict[str, Any], output_dir: str):
    paper_dir = Path(output_dir) / "paper"
    ensure_dir(str(paper_dir))

    adapted = _adapt_summary(summary)
    portfolio_metrics = adapted["portfolio_metrics"]
    mean_metrics = adapted["mean_metrics"]
    vol_metrics = adapted["vol_metrics"]
    cov_metrics = adapted["cov_metrics"]

    row = {
        "preset_name": config.get("_meta", {}).get("preset_name"),
        "experiment_group": config.get("experiment", {}).get("group"),
        "variant": config.get("experiment", {}).get("variant"),
        "paper_tag": config.get("experiment", {}).get("paper_tag"),
        "mean_model": config.get("mean_model", {}).get("name"),
        "vol_model": config.get("vol_model", {}).get("name"),
        "corr_model": config.get("corr_model", {}).get("name"),
        "optimizer": config.get("optimizer", {}).get("name"),
        "annual_return": portfolio_metrics.get("annual_return"),
        "annual_volatility": portfolio_metrics.get("annual_volatility"),
        "sharpe_ratio": portfolio_metrics.get("sharpe_ratio"),
        "sortino_ratio": portfolio_metrics.get("sortino_ratio"),
        "calmar_ratio": portfolio_metrics.get("calmar_ratio"),
        "max_drawdown": portfolio_metrics.get("max_drawdown"),
        "avg_turnover": portfolio_metrics.get("avg_turnover"),
        "avg_expected_risk": portfolio_metrics.get("avg_expected_risk"),
        "avg_concentration_hhi": portfolio_metrics.get("avg_concentration_hhi"),
        "avg_effective_n_assets": portfolio_metrics.get("avg_effective_n_assets"),
        "avg_max_weight": portfolio_metrics.get("avg_max_weight"),
        "avg_transaction_cost": portfolio_metrics.get("avg_transaction_cost"),
        "avg_cost_drag": portfolio_metrics.get("avg_cost_drag"),
        "gross_minus_net_annual_return_gap": portfolio_metrics.get("gross_minus_net_annual_return_gap"),
        "gross_minus_net_sharpe_gap": portfolio_metrics.get("gross_minus_net_sharpe_gap"),
        "net_annual_return": portfolio_metrics.get("net_annual_return"),
        "net_sharpe_ratio": portfolio_metrics.get("net_sharpe_ratio"),
        "mean_ic": mean_metrics.get("ic"),
        "mean_rank_ic": mean_metrics.get("rank_ic"),
        "median_cross_sectional_ic": mean_metrics.get("median_cross_sectional_ic"),
        "median_cross_sectional_rank_ic": mean_metrics.get("median_cross_sectional_rank_ic"),
        "vol_qlike": vol_metrics.get("qlike"),
        "vol_median_abs_error": vol_metrics.get("median_abs_error"),
        "cov_frobenius": cov_metrics.get("frobenius"),
        "cov_median_frobenius_distance": cov_metrics.get("median_frobenius_distance"),
    }

    df = pd.DataFrame([row])
    save_dataframe(df, str(paper_dir / "paper_main_result.csv"))
    save_dataframe(df, str(paper_dir / "paper_module_comparison.csv"))

    narrative = {
        "experiment_name": config.get("_meta", {}).get("experiment_name", config.get("_meta", {}).get("preset_name")),
        "main_findings": row,
        "text_hint": (
            "该文件用于后续交给AI生成论文结果分析。"
            "重点关注 mean/vol/corr 三个模块对 net_sharpe_ratio、max_drawdown、avg_turnover 的影响。"
        ),
    }
    save_json(narrative, str(paper_dir / "paper_narrative.json"))

    md = f"""# 实验自动摘要

- 预设：{row['preset_name']}
- 实验组：{row['experiment_group']}
- 变体：{row['variant']}
- Mean：{row['mean_model']}
- Vol：{row['vol_model']}
- Corr：{row['corr_model']}
- Optimizer：{row['optimizer']}

## 组合指标
- 年化收益：{row['annual_return']}
- 年化波动：{row['annual_volatility']}
- 夏普：{row['sharpe_ratio']}
- 净夏普：{row['net_sharpe_ratio']}
- 最大回撤：{row['max_drawdown']}
- 平均换手：{row['avg_turnover']}
- 平均预测风险：{row['avg_expected_risk']}
- 平均集中度HHI：{row['avg_concentration_hhi']}
- 平均有效资产数：{row['avg_effective_n_assets']}

## 模块指标
- Mean IC：{row['mean_ic']}
- Mean RankIC：{row['mean_rank_ic']}
- Vol QLIKE：{row['vol_qlike']}
- Cov Frobenius：{row['cov_frobenius']}

## 建议解释
该实验结果可用于论文中的主结果或模块比较部分。建议同时从收益、风险、换手、集中度与模块预测误差四个层面进行解释。
"""
    with open(paper_dir / "auto_summary.md", "w", encoding="utf-8") as f:
        f.write(md)
