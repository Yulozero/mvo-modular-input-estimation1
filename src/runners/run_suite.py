import argparse
import json
from pathlib import Path

import pandas as pd

from src.runners.run_experiment import run_one
from src.utils.config import load_yaml, load_config_from_preset, deep_update


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=str, required=True, help="Path to suite yaml")
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help=(
            "Optional experiment group filter, e.g. "
            "Baseline/Mean Comparison/Vol Comparison/Corr Comparison/"
            "Ablation/Robustness/Generalization/DWT Special."
        ),
    )
    return parser.parse_args()


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _make_suite_output_dir(suite_cfg: dict) -> Path:
    output_cfg = suite_cfg.get("output", {})
    root_dir = output_cfg.get("root_dir", "outputs")
    suite_name = suite_cfg.get("suite_name", "suite")
    out_dir = Path(root_dir) / "_suite_runs" / suite_name
    _ensure_dir(out_dir)
    return out_dir


def _build_config_from_suite_item(exp: dict, default_base: str = None):
    if "config" in exp and isinstance(exp["config"], dict):
        cfg = exp["config"].copy()
        if "_meta" not in cfg:
            cfg["_meta"] = {}
        if exp.get("experiment_name") and "_meta" in cfg:
            cfg["_meta"]["experiment_name"] = exp["experiment_name"]
        return cfg

    preset = exp.get("preset")
    override = exp.get("override", {}) or {}
    base_path = exp.get("base", default_base)

    if not preset:
        raise ValueError("Each suite experiment must provide 'preset' or inline 'config'.")
    if not base_path:
        raise ValueError("Base yaml path is required either in suite.base or experiment.base.")

    cfg = load_config_from_preset(base_path, preset, override=override)

    if exp.get("experiment_name"):
        cfg.setdefault("_meta", {})
        cfg["_meta"]["experiment_name"] = exp["experiment_name"]
        cfg.setdefault("output", {})
        cfg["output"]["experiment_name"] = exp["experiment_name"]

    return cfg


def _infer_group_from_suite_item(exp: dict, default_base: str = None):
    if "config" in exp and isinstance(exp["config"], dict):
        return (exp["config"].get("experiment", {}) or {}).get("group")

    preset = exp.get("preset")
    if not preset:
        return None

    preset_cfg = load_yaml(preset)
    return (preset_cfg.get("experiment", {}) or {}).get("group")


def main():
    args = parse_args()
    suite_cfg = load_yaml(args.suite)

    suite_name = suite_cfg.get("suite_name", "suite")
    base_path = suite_cfg.get("base")
    experiments = suite_cfg.get("experiments", [])

    if not experiments:
        raise ValueError("No experiments found in suite config")

    suite_out_dir = _make_suite_output_dir(suite_cfg)

    summaries = []
    run_records = []

    if args.group:
        wanted = args.group.strip().lower()
        filtered = []
        for exp in experiments:
            group_name = _infer_group_from_suite_item(exp, default_base=base_path)
            if group_name and group_name.strip().lower() == wanted:
                filtered.append(exp)
                continue

            # Convenience alias: DWT Special matches vol comparison DWT and robustness DWT presets.
            if wanted in {"dwt", "dwt special", "dwt_special"}:
                preset = str(exp.get("preset", "")).lower()
                variant = str(((exp.get("config", {}) or {}).get("experiment", {}) or {}).get("variant", "")).lower()
                if "dwt" in preset or "dwt" in variant:
                    filtered.append(exp)

        experiments = filtered

    total = len(experiments)
    if total == 0:
        raise ValueError("No experiments matched the suite/group filter.")

    for i, exp in enumerate(experiments, start=1):
        cfg = _build_config_from_suite_item(exp, default_base=base_path)

        exp_name = (
            cfg.get("output", {}).get("experiment_name")
            or cfg.get("_meta", {}).get("experiment_name")
            or cfg.get("_meta", {}).get("preset_name")
            or f"exp_{i}"
        )

        print(f"[RUN {i}/{total}] suite={suite_name}, experiment={exp_name}")
        _, summary = run_one(cfg)

        row = dict(summary)
        row["suite_name"] = suite_name
        row["experiment_name"] = exp_name
        row["preset_name"] = cfg.get("_meta", {}).get("preset_name")
        row["experiment_group"] = cfg.get("experiment", {}).get("group")
        row["variant"] = cfg.get("experiment", {}).get("variant")
        row["paper_tag"] = cfg.get("experiment", {}).get("paper_tag")
        row["mean_model"] = cfg.get("mean_model", {}).get("name")
        row["vol_model"] = cfg.get("vol_model", {}).get("name")
        row["corr_model"] = cfg.get("corr_model", {}).get("name")
        row["optimizer"] = cfg.get("optimizer", {}).get("name")
        summaries.append(row)

        run_records.append({
            "suite_name": suite_name,
            "experiment_name": exp_name,
            "preset_name": cfg.get("_meta", {}).get("preset_name"),
            "experiment_group": cfg.get("experiment", {}).get("group"),
            "variant": cfg.get("experiment", {}).get("variant"),
            "paper_tag": cfg.get("experiment", {}).get("paper_tag"),
        })

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(suite_out_dir / "suite_master_summary.csv", index=False, encoding="utf-8-sig")

    with open(suite_out_dir / "suite_master_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    pd.DataFrame(run_records).to_csv(
        suite_out_dir / "suite_run_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"[DONE] Suite finished. Outputs saved to: {suite_out_dir}")


if __name__ == "__main__":
    main()
