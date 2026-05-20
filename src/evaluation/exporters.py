from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional


def _safe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _to_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def extract_paper_metrics(summary: Dict[str, Any]) -> Dict[str, Any]:
    mean_m = summary.get("mean_module_metrics", {}) or {}
    vol_m = summary.get("vol_module_metrics", {}) or {}
    cov_m = summary.get("cov_module_metrics", {}) or {}

    return {
        "annual_return": _to_float(summary.get("net_annual_return_holding")),
        "annual_volatility": _to_float(summary.get("net_annual_vol_holding")),
        "sharpe_ratio": _to_float(summary.get("net_sharpe_holding")),
        "net_sharpe_ratio": _to_float(summary.get("net_sharpe_holding")),
        "sortino_ratio": _to_float(summary.get("net_sortino_holding")),
        "calmar_ratio": _to_float(summary.get("net_calmar_holding")),
        "max_drawdown": _to_float(summary.get("net_max_drawdown_holding")),
        "avg_turnover": _to_float(summary.get("avg_turnover")),
        "avg_transaction_cost": _to_float(summary.get("avg_transaction_cost")),
        "avg_cost_drag": _to_float(summary.get("avg_cost_drag")),
        "avg_concentration_hhi": _to_float(summary.get("avg_concentration_hhi")),
        "avg_effective_n_assets": _to_float(summary.get("avg_effective_n_assets")),
        "avg_max_weight": _to_float(summary.get("avg_max_weight")),
        "gross_minus_net_annual_return_gap": _to_float(summary.get("gross_minus_net_annual_return_gap")),
        "gross_minus_net_sharpe_gap": _to_float(summary.get("gross_minus_net_sharpe_gap")),
        "mean_ic": _to_float(mean_m.get("global_ic")),
        "mean_rank_ic": _to_float(mean_m.get("global_rank_ic")),
        "median_cross_sectional_ic": _to_float(mean_m.get("median_cross_sectional_ic")),
        "median_cross_sectional_rank_ic": _to_float(mean_m.get("median_cross_sectional_rank_ic")),
        "vol_qlike": _to_float(vol_m.get("qlike")),
        "vol_median_abs_error": _to_float(vol_m.get("median_abs_error")),
        "cov_frobenius": _to_float(cov_m.get("avg_frobenius_distance")),
        "cov_median_frobenius_distance": _to_float(cov_m.get("median_frobenius_distance")),
    }


def build_paper_narrative(summary: Dict[str, Any], experiment_info: Dict[str, Any]) -> Dict[str, Any]:
    metrics = extract_paper_metrics(summary)
    return {
        "legacy_exporter": True,
        "note": "Legacy compatibility exporter. Prefer src.utils.paper_exporters for paper-mode outputs.",
        "experiment_name": experiment_info.get("preset_name")
        or experiment_info.get("output", {}).get("experiment_name")
        or "unknown_experiment",
        "main_findings": {
            "preset_name": experiment_info.get("preset_name"),
            "experiment_group": experiment_info.get("experiment_group"),
            "variant": experiment_info.get("variant"),
            "paper_tag": experiment_info.get("paper_tag"),
            "mean_model": _safe_get(experiment_info, "mean_model", "name"),
            "vol_model": _safe_get(experiment_info, "vol_model", "name"),
            "corr_model": _safe_get(experiment_info, "corr_model", "name"),
            "optimizer": _safe_get(experiment_info, "optimizer", "name"),
            **metrics,
        },
    }


def write_json(path: Path, obj: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_single_row_csv(path: Path, row: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def write_auto_summary_md(path: Path, summary: Dict[str, Any], experiment_info: Dict[str, Any]):
    metrics = extract_paper_metrics(summary)
    text = "\n".join(
        [
            "# Legacy Export Summary",
            "",
            "This file is retained for backward compatibility.",
            "Prefer the paper-mode outputs under `paper/` and `summary/`.",
            "",
            f"- preset: {experiment_info.get('preset_name', 'unknown')}",
            f"- group: {experiment_info.get('experiment_group', 'unknown')}",
            f"- variant: {experiment_info.get('variant', 'unknown')}",
            f"- mean_model: {_safe_get(experiment_info, 'mean_model', 'name', default='unknown')}",
            f"- vol_model: {_safe_get(experiment_info, 'vol_model', 'name', default='unknown')}",
            f"- corr_model: {_safe_get(experiment_info, 'corr_model', 'name', default='unknown')}",
            f"- optimizer: {_safe_get(experiment_info, 'optimizer', 'name', default='unknown')}",
            "",
            f"- net_sharpe_ratio: {metrics['net_sharpe_ratio']}",
            f"- max_drawdown: {metrics['max_drawdown']}",
            f"- avg_turnover: {metrics['avg_turnover']}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def export_result_bundle(
    output_dir: str | Path,
    summary: Dict[str, Any],
    config_used: Dict[str, Any],
    experiment_info: Dict[str, Any],
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = extract_paper_metrics(summary)
    mean_m = summary.get("mean_module_metrics", {}) or {}
    vol_m = summary.get("vol_module_metrics", {}) or {}
    cov_m = summary.get("cov_module_metrics", {}) or {}

    write_auto_summary_md(output_dir / "auto_summary.md", summary, experiment_info)
    write_json(output_dir / "paper_narrative.json", build_paper_narrative(summary, experiment_info))
    write_json(
        output_dir / "diagnostics.json",
        {
            "legacy_exporter": True,
            "summary_keys": list(summary.keys()),
            "config_keys": list(config_used.keys()),
        },
    )

    portfolio_row = {
        "gross_annual_return_holding": summary.get("gross_annual_return_holding"),
        "gross_annual_vol_holding": summary.get("gross_annual_vol_holding"),
        "gross_sharpe_holding": summary.get("gross_sharpe_holding"),
        "net_annual_return_holding": summary.get("net_annual_return_holding"),
        "net_annual_vol_holding": summary.get("net_annual_vol_holding"),
        "net_sharpe_holding": summary.get("net_sharpe_holding"),
        "avg_turnover": summary.get("avg_turnover"),
        "avg_transaction_cost": summary.get("avg_transaction_cost"),
        "avg_cost_drag": summary.get("avg_cost_drag"),
        "avg_concentration_hhi": summary.get("avg_concentration_hhi"),
        "avg_effective_n_assets": summary.get("avg_effective_n_assets"),
        "avg_max_weight": summary.get("avg_max_weight"),
        "gross_minus_net_annual_return_gap": summary.get("gross_minus_net_annual_return_gap"),
        "gross_minus_net_sharpe_gap": summary.get("gross_minus_net_sharpe_gap"),
    }
    write_single_row_csv(output_dir / "portfolio_metrics.csv", portfolio_row)

    write_single_row_csv(
        output_dir / "mean_metrics.csv",
        {
            "mse": mean_m.get("mse"),
            "mae": mean_m.get("mae"),
            "global_ic": mean_m.get("global_ic"),
            "global_rank_ic": mean_m.get("global_rank_ic"),
            "median_cross_sectional_ic": mean_m.get("median_cross_sectional_ic"),
            "median_cross_sectional_rank_ic": mean_m.get("median_cross_sectional_rank_ic"),
        },
    )
    write_single_row_csv(
        output_dir / "vol_metrics.csv",
        {
            "mse": vol_m.get("mse"),
            "rmse": vol_m.get("rmse"),
            "mae": vol_m.get("mae"),
            "median_abs_error": vol_m.get("median_abs_error"),
            "qlike": vol_m.get("qlike"),
        },
    )
    write_single_row_csv(
        output_dir / "cov_metrics.csv",
        {
            "avg_frobenius_distance": cov_m.get("avg_frobenius_distance"),
            "median_frobenius_distance": cov_m.get("median_frobenius_distance"),
            "avg_ex_ante_variance": cov_m.get("avg_ex_ante_variance"),
            "avg_ex_post_variance": cov_m.get("avg_ex_post_variance"),
        },
    )
    write_single_row_csv(output_dir / "paper_main_result.csv", metrics)
