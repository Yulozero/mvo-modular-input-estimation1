import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict, List

from src.utils.config import load_yaml, load_config_from_preset


MODEL_PKG_REQUIREMENTS = {
    "garch": ["arch"],
    "garch_lstm": ["arch", "torch"],
    "filter_lstm": ["torch"],
    "dwt_lstm": ["torch", "pywt"],
    "lstm_mean": ["torch"],
    "attention_mean": ["torch"],
    "transformer_mean": ["torch"],
    "transformer_attn_mean": ["torch"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Preflight validation for a suite config.")
    parser.add_argument("--suite", required=True, help="Path to suite yaml")
    parser.add_argument("--save", default=None, help="Optional output json path")
    return parser.parse_args()


def _pkg_missing(pkgs: List[str]) -> List[str]:
    return [pkg for pkg in pkgs if importlib.util.find_spec(pkg) is None]


def _required_data_files(cfg: Dict) -> Dict[str, str]:
    data_cfg = cfg.get("data", {}) or {}
    return {
        "returns_path": data_cfg.get("returns_path"),
        "prices_path": data_cfg.get("prices_path"),
        "market_cap_path": data_cfg.get("market_cap_path"),
    }


def main():
    args = parse_args()
    suite_cfg = load_yaml(args.suite)
    base_path = suite_cfg.get("base")
    if not base_path:
        raise ValueError("Suite yaml missing 'base'.")

    experiments = suite_cfg.get("experiments", [])
    if not experiments:
        raise ValueError("No experiments in suite yaml.")

    missing_counts = {"data": 0, "dependency": 0}
    rows = []

    for idx, item in enumerate(experiments, start=1):
        preset = item.get("preset")
        if not preset:
            rows.append({
                "index": idx,
                "status": "invalid",
                "reason": "missing preset field",
            })
            continue

        override = item.get("override", {}) or {}
        cfg = load_config_from_preset(base_path, preset, override=override)

        exp_name = (
            cfg.get("output", {}).get("experiment_name")
            or cfg.get("_meta", {}).get("preset_name")
            or f"exp_{idx}"
        )
        mean_name = cfg.get("mean_model", {}).get("name")
        vol_name = cfg.get("vol_model", {}).get("name")

        required_pkgs = []
        required_pkgs.extend(MODEL_PKG_REQUIREMENTS.get(mean_name, []))
        required_pkgs.extend(MODEL_PKG_REQUIREMENTS.get(vol_name, []))
        required_pkgs = sorted(set(required_pkgs))
        missing_pkgs = _pkg_missing(required_pkgs)

        data_paths = _required_data_files(cfg)
        missing_data = []
        for key, rel_path in data_paths.items():
            if not rel_path:
                missing_data.append(f"{key}:<empty>")
                continue
            if not Path(rel_path).exists():
                missing_data.append(f"{key}:{rel_path}")

        if missing_data:
            missing_counts["data"] += 1
        if missing_pkgs:
            missing_counts["dependency"] += 1

        status = "ok" if (not missing_data and not missing_pkgs) else "blocked"
        rows.append({
            "index": idx,
            "status": status,
            "experiment_name": exp_name,
            "preset": preset,
            "mean_model": mean_name,
            "vol_model": vol_name,
            "missing_data": missing_data,
            "missing_dependency": missing_pkgs,
        })

    report = {
        "suite_name": suite_cfg.get("suite_name", Path(args.suite).stem),
        "total": len(rows),
        "ok": sum(r.get("status") == "ok" for r in rows),
        "blocked": sum(r.get("status") == "blocked" for r in rows),
        "missing_data_count": missing_counts["data"],
        "missing_dependency_count": missing_counts["dependency"],
        "items": rows,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
