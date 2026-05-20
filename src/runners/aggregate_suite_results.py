import argparse
import json
from pathlib import Path

import pandas as pd

from src.utils.paper_packager import (
    build_master_table,
    build_paper_package,
    collect_experiment_dirs,
    load_experiment_bundle,
    split_by_group,
)
from src.utils.stat_tests import compare_two_return_series


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_root", type=str, default="outputs")
    parser.add_argument("--save_dir", type=str, default="outputs/_aggregated")
    return parser.parse_args()


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data, path: str):
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_special_tables(master_df: pd.DataFrame, save_dir: Path):
    if master_df.empty:
        return

    group_map = {
        "Baseline": "paper_baseline.csv",
        "Mean Comparison": "paper_mean_comparison.csv",
        "Vol Comparison": "paper_vol_comparison.csv",
        "Corr Comparison": "paper_corr_comparison.csv",
        "Ablation": "paper_ablation.csv",
        "Robustness": "paper_robustness.csv",
        "Generalization": "paper_generalization.csv",
        "Main": "paper_main.csv",
    }

    for group_name, file_name in group_map.items():
        if "experiment_group" not in master_df.columns:
            continue
        df_g = master_df[master_df["experiment_group"] == group_name].copy()
        if not df_g.empty:
            df_g.to_csv(save_dir / file_name, index=False, encoding="utf-8-sig")

    if not master_df.empty:
        main_df = master_df[master_df.get("experiment_group") == "Main"].copy() if "experiment_group" in master_df.columns else pd.DataFrame()
        if not main_df.empty:
            main_df.to_csv(save_dir / "paper_main_result.csv", index=False, encoding="utf-8-sig")

        cols = [
            c for c in [
                "preset_name",
                "experiment_group",
                "variant",
                "paper_tag",
                "mean_model",
                "vol_model",
                "corr_model",
                "optimizer",
                "mean_ic",
                "mean_rank_ic",
                "median_cross_sectional_ic",
                "median_cross_sectional_rank_ic",
                "vol_qlike",
                "vol_median_abs_error",
                "cov_frobenius",
                "cov_median_frobenius_distance",
                "net_sharpe_ratio",
                "max_drawdown",
                "avg_turnover",
                "avg_transaction_cost",
                "avg_cost_drag",
                "avg_concentration_hhi",
                "avg_effective_n_assets",
                "avg_max_weight",
                "fallback_used",
                "solver_success_rate",
            ] if c in master_df.columns
        ]
        if cols:
            master_df[cols].to_csv(save_dir / "paper_module_comparison.csv", index=False, encoding="utf-8-sig")


def build_stat_tests(bundles, save_dir: Path):
    rows = []

    variant_map = {}
    for b in bundles:
        meta = b.get("meta") or {}
        variant = meta.get("variant")
        if variant:
            variant_map[variant] = b

    pairs = [
        ("Full Model", "Baseline"),
        ("Transformer Mean", "Hist Mean"),
        ("GARCH-LSTM", "EWMA Vol"),
        ("Distance Corr", "Pearson Corr"),
        ("Shrinkage MVO", "Traditional MVO"),
    ]

    for a, b in pairs:
        if a not in variant_map or b not in variant_map:
            continue

        dec_a = variant_map[a].get("decisions")
        dec_b = variant_map[b].get("decisions")

        if dec_a is None or dec_b is None:
            continue
        if "realized_net_return" not in dec_a.columns or "realized_net_return" not in dec_b.columns:
            continue

        rows.append(
            compare_two_return_series(
                dec_a["realized_net_return"],
                dec_b["realized_net_return"],
                name_a=a,
                name_b=b,
                metric_name="realized_net_return",
            )
        )

    df = pd.DataFrame(rows)
    df.to_csv(save_dir / "paper_stat_tests.csv", index=False, encoding="utf-8-sig")


def build_final_report(master_df: pd.DataFrame, save_dir: Path):
    lines = ["# Final Experiment Report", ""]

    if master_df.empty:
        lines.append("No experiment results found.")
    else:
        lines.append(f"- Number of experiments: {len(master_df)}")

        if "net_sharpe_ratio" in master_df.columns:
            dfv = master_df.dropna(subset=["net_sharpe_ratio"])
            if not dfv.empty:
                best = dfv.sort_values("net_sharpe_ratio", ascending=False).iloc[0]
                lines.append(
                    f"- Best experiment by net Sharpe: {best.get('variant')} "
                    f"({best.get('experiment_group')}), value={best.get('net_sharpe_ratio')}"
                )

        if "experiment_group" in master_df.columns:
            for group in ["Baseline", "Mean Comparison", "Vol Comparison", "Corr Comparison", "Ablation", "Robustness"]:
                df_g = master_df[master_df["experiment_group"] == group]
                if df_g.empty or "net_sharpe_ratio" not in df_g.columns:
                    continue
                dfv = df_g.dropna(subset=["net_sharpe_ratio"])
                if dfv.empty:
                    continue
                best = dfv.sort_values("net_sharpe_ratio", ascending=False).iloc[0]
                lines.append(
                    f"- Best in {group}: {best.get('variant')} | "
                    f"net_sharpe={best.get('net_sharpe_ratio')}"
                )

        if "avg_max_weight" in master_df.columns:
            lines.append(f"- Average of avg_max_weight across experiments: {master_df['avg_max_weight'].dropna().mean():.4f}")
        if "solver_success_rate" in master_df.columns:
            lines.append(f"- Average solver success rate: {master_df['solver_success_rate'].dropna().mean():.4f}")

    report_path = save_dir / "final_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    ensure_dir(str(save_dir))

    exp_dirs = collect_experiment_dirs(args.outputs_root)
    bundles = [load_experiment_bundle(d) for d in exp_dirs]
    master_df = build_master_table(bundles)

    master_df.to_csv(save_dir / "master_summary.csv", index=False, encoding="utf-8-sig")

    grouped = split_by_group(master_df)
    for group_name, df_g in grouped.items():
        safe = group_name.lower().replace(" ", "_")
        df_g.to_csv(save_dir / f"{safe}.csv", index=False, encoding="utf-8-sig")

    build_special_tables(master_df, save_dir)
    build_stat_tests(bundles, save_dir)

    package = build_paper_package(master_df)
    save_json(package, str(save_dir / "paper_package.json"))

    build_final_report(master_df, save_dir)

    print(f"[DONE] Aggregated results saved to: {save_dir}")


if __name__ == "__main__":
    main()
