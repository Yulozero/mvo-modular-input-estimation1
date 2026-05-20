import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_root", type=str, default="outputs_paper_full_matrix")
    parser.add_argument("--save_dir", type=str, default="outputs_paper_full_matrix/_aggregated/figures")
    parser.add_argument("--master_summary", type=str, default=None)
    return parser.parse_args()


def _load_master(path: str | None, outputs_root: str):
    if path:
        return pd.read_csv(path)
    return pd.read_csv(Path(outputs_root) / "_aggregated" / "master_summary.csv")


def _ensure_matplotlib():
    if plt is None:
        raise ImportError("matplotlib is required to build plots.")


def _bar_plot(df, title, save_path, metric="net_sharpe_ratio"):
    if df.empty:
        return
    df = df.sort_values(metric, ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df["variant"].astype(str), df[metric].astype(float))
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def _load_nav(exp_dir: str):
    path = Path(exp_dir) / "raw" / "decisions.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "realized_net_return" not in df.columns:
        return None
    nav = (1.0 + df["realized_net_return"].fillna(0.0).astype(float)).cumprod()
    return nav


def _plot_nav(master_df: pd.DataFrame, save_path: Path):
    picks = []
    for group, variant in [("Main", "Full Model"), ("Baseline", "Equal Weight"), ("Baseline", "Traditional MVO"), ("Baseline", "Shrinkage MVO")]:
        hit = master_df[(master_df["experiment_group"] == group) & (master_df["variant"] == variant)]
        if not hit.empty:
            picks.append(hit.iloc[0])
    if not picks:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for row in picks:
        nav = _load_nav(row["exp_dir"])
        if nav is not None:
            ax.plot(nav.values, label=f"{row['variant']} ({row['experiment_group']})")
    ax.set_title("NAV Comparison")
    ax.set_ylabel("Cumulative Net Value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def _plot_heatmap(robust_df: pd.DataFrame, save_path: Path):
    if robust_df.empty:
        return
    robust_df = robust_df.copy()
    robust_df["row"] = np.arange(len(robust_df))
    values = robust_df[["net_sharpe_ratio", "max_drawdown", "avg_turnover"]].astype(float).fillna(0.0).values
    fig, ax = plt.subplots(figsize=(8, max(4, len(robust_df) * 0.25)))
    im = ax.imshow(values, aspect="auto", cmap="RdYlGn")
    ax.set_yticks(np.arange(len(robust_df)))
    ax.set_yticklabels(robust_df["variant"].astype(str))
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["net_sharpe", "max_drawdown", "avg_turnover"])
    ax.set_title("Robustness Heatmap")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def _plot_scatter(master_df: pd.DataFrame, save_path: Path):
    needed = {"avg_concentration_hhi", "net_sharpe_ratio"}
    if not needed.issubset(master_df.columns):
        return
    df = master_df.dropna(subset=list(needed)).copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["avg_concentration_hhi"], df["net_sharpe_ratio"], alpha=0.8)
    for _, row in df.iterrows():
        ax.annotate(str(row["variant"]), (row["avg_concentration_hhi"], row["net_sharpe_ratio"]), fontsize=7, alpha=0.75)
    ax.set_xlabel("Average Concentration HHI")
    ax.set_ylabel("Net Sharpe Ratio")
    ax.set_title("Concentration vs Sharpe")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def main():
    _ensure_matplotlib()
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    master_df = _load_master(args.master_summary, args.outputs_root)

    _plot_nav(master_df, save_dir / "nav_comparison.png")
    _bar_plot(master_df[master_df["experiment_group"] == "Mean Comparison"], "Mean Comparison", save_dir / "mean_comparison_bar.png")
    _bar_plot(master_df[master_df["experiment_group"] == "Vol Comparison"], "Vol Comparison", save_dir / "vol_comparison_bar.png")
    _bar_plot(master_df[master_df["experiment_group"] == "Corr Comparison"], "Corr Comparison", save_dir / "corr_comparison_bar.png")
    _bar_plot(master_df[master_df["experiment_group"] == "Ablation"], "Ablation", save_dir / "ablation_bar.png")
    _plot_heatmap(master_df[master_df["experiment_group"] == "Robustness"], save_dir / "robustness_heatmap.png")
    _plot_scatter(master_df, save_dir / "concentration_vs_sharpe.png")
    print(f"[DONE] Figures saved to: {save_dir}")


if __name__ == "__main__":
    main()
