import argparse
import json
from pathlib import Path

import pandas as pd

from src.utils.stat_tests import compare_two_return_series


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_root", type=str, default="outputs_paper_full_matrix")
    parser.add_argument("--save_dir", type=str, default="outputs_paper_full_matrix/_aggregated")
    parser.add_argument("--master_summary", type=str, default=None)
    return parser.parse_args()


def _load_master_summary(path: str | None, save_dir: Path) -> pd.DataFrame:
    if path:
        return pd.read_csv(path)
    return pd.read_csv(save_dir / "master_summary.csv")


def _find_row(df: pd.DataFrame, candidates):
    for candidate in candidates:
        mask = pd.Series(True, index=df.index)
        for key, value in candidate.items():
            mask = mask & (df[key].astype(str) == str(value))
        hit = df.loc[mask]
        if not hit.empty:
            return hit.iloc[0]
    return None


def _load_returns(exp_dir: str):
    path = Path(exp_dir) / "raw" / "decisions.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "realized_net_return" not in df.columns:
        return None
    return df["realized_net_return"]


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    master_df = _load_master_summary(args.master_summary, save_dir)

    comparisons = [
        ("Full Model", [{"experiment_group": "Main", "variant": "Full Model"}], [{"experiment_group": "Baseline", "variant": "Equal Weight"}]),
        ("Full Model", [{"experiment_group": "Main", "variant": "Full Model"}], [{"experiment_group": "Baseline", "variant": "Traditional MVO"}]),
        ("Full Model", [{"experiment_group": "Main", "variant": "Full Model"}], [{"experiment_group": "Baseline", "variant": "Shrinkage MVO"}]),
        ("Best Mean", [{"experiment_group": "Mean Comparison", "variant": "Attention Mean"}, {"experiment_group": "Mean Comparison", "variant": "Transformer Mean"}], [{"experiment_group": "Mean Comparison", "variant": "Hist Mean"}]),
        ("Best Vol", [{"experiment_group": "Vol Comparison", "variant": "DWT-LSTM"}, {"experiment_group": "Vol Comparison", "variant": "GARCH-LSTM"}], [{"experiment_group": "Vol Comparison", "variant": "EWMA Vol"}]),
        ("Best Corr", [{"experiment_group": "Corr Comparison", "variant": "Pearson Corr"}, {"experiment_group": "Corr Comparison", "variant": "Distance Corr"}], [{"experiment_group": "Corr Comparison", "variant": "Pearson Corr"}]),
    ]

    rows = []
    warnings = []

    for label, left_candidates, right_candidates in comparisons:
        left = _find_row(master_df, left_candidates)
        right = _find_row(master_df, right_candidates)
        if left is None or right is None:
            warnings.append(f"Missing comparison pair for {label}")
            continue

        if left["exp_dir"] == right["exp_dir"]:
            warnings.append(f"Skipped degenerate comparison for {label}")
            continue

        left_returns = _load_returns(left["exp_dir"])
        right_returns = _load_returns(right["exp_dir"])
        if left_returns is None or right_returns is None:
            warnings.append(f"Missing decision returns for {label}")
            continue

        row = compare_two_return_series(
            left_returns,
            right_returns,
            name_a=f"{left['variant']} ({left['experiment_group']})",
            name_b=f"{right['variant']} ({right['experiment_group']})",
            metric_name="realized_net_return",
        )
        row["comparison_label"] = label
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(save_dir / "paper_significance_table.csv", index=False, encoding="utf-8-sig")
    with open(save_dir / "paper_significance_warnings.json", "w", encoding="utf-8") as f:
        json.dump({"warnings": warnings}, f, ensure_ascii=False, indent=2)
    print(f"[DONE] Significance tables saved to: {save_dir}")


if __name__ == "__main__":
    main()
