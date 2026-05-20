import argparse
import json
import subprocess
import sys
from pathlib import Path
import yaml
from src.utils.config import deep_update


DONE_FILES = {
    "summary.json",
    "metrics.json",
    "paper_metrics.json",
    "paper_table.csv",
    "result.json",
}


def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def infer_experiment_name(base_cfg: dict, preset_cfg: dict, preset_path: Path, override_cfg: dict = None) -> str:
    merged = deep_update(preset_cfg, override_cfg or {})
    output_cfg = merged.get("output", {}) or {}
    if output_cfg.get("experiment_name"):
        return str(output_cfg["experiment_name"])

    base_output = base_cfg.get("output", {}) or {}
    if base_output.get("experiment_name"):
        return str(base_output["experiment_name"])

    return preset_path.stem


def infer_output_root(suite_cfg: dict, base_cfg: dict, preset_cfg: dict, override_cfg: dict = None) -> Path:
    merged = deep_update(preset_cfg, override_cfg or {})
    preset_output = merged.get("output", {}) or {}
    if preset_output.get("root_dir"):
        return Path(preset_output["root_dir"])

    suite_output = suite_cfg.get("output", {}) or {}
    if suite_output.get("root_dir"):
        return Path(suite_output["root_dir"])

    base_output = base_cfg.get("output", {}) or {}
    if base_output.get("root_dir"):
        return Path(base_output["root_dir"])

    return Path("outputs")


def looks_completed(exp_dir: Path) -> bool:
    if not exp_dir.exists() or not exp_dir.is_dir():
        return False

    existing = {p.name for p in exp_dir.iterdir() if p.is_file()}
    if existing & DONE_FILES:
        return True

    for sub in exp_dir.iterdir():
        if sub.is_dir():
            sub_existing = {p.name for p in sub.iterdir() if p.is_file()}
            if sub_existing & DONE_FILES:
                return True

    return False


def main():
    parser = argparse.ArgumentParser(description="Run suite with resume/skip support.")
    parser.add_argument("--suite", required=True, help="Path to suite yaml.")
    parser.add_argument("--python", default=sys.executable, help="Python executable.")
    parser.add_argument("--dry-run", action="store_true", help="Only print actions.")
    args = parser.parse_args()

    suite_path = Path(args.suite).resolve()
    if not suite_path.exists():
        raise FileNotFoundError(f"Suite file not found: {suite_path}")

    project_root = Path.cwd()
    suite_cfg = load_yaml(suite_path)

    base_rel = suite_cfg.get("base")
    if not base_rel:
        raise ValueError("Suite yaml missing 'base' field.")

    base_path = (project_root / base_rel).resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_path}")

    base_cfg = load_yaml(base_path)

    experiments = suite_cfg.get("experiments", [])
    if not experiments:
        print("No experiments found in suite.")
        return

    total = len(experiments)
    skipped = 0
    executed = 0
    failed = 0

    print(f"[INFO] Suite: {suite_cfg.get('suite_name', suite_path.stem)}")
    print(f"[INFO] Base : {base_path}")
    print(f"[INFO] Total experiments: {total}")
    print("-" * 80)

    for idx, item in enumerate(experiments, start=1):
        preset_rel = item.get("preset")
        if not preset_rel:
            print(f"[WARN] #{idx}: missing preset, skipped.")
            skipped += 1
            continue

        preset_path = (project_root / preset_rel).resolve()
        if not preset_path.exists():
            print(f"[ERROR] #{idx}: preset not found: {preset_path}")
            failed += 1
            continue

        preset_cfg = load_yaml(preset_path)
        override_cfg = item.get("override", {}) or {}
        effective_override = json.loads(json.dumps(override_cfg, ensure_ascii=False))

        exp_name = infer_experiment_name(base_cfg, preset_cfg, preset_path, override_cfg=effective_override)
        output_root = infer_output_root(suite_cfg, base_cfg, preset_cfg, override_cfg=effective_override)
        exp_dir = (project_root / output_root / exp_name).resolve()

        if looks_completed(exp_dir):
            print(f"[SKIP] #{idx}/{total} {exp_name} -> already completed")
            skipped += 1
            continue

        cmd = [
            args.python,
            "-m",
            "src.runners.run_experiment",
            "--base",
            str(base_path),
            "--preset",
            str(preset_path),
        ]
        effective_override.setdefault("output", {})
        effective_override["output"].setdefault("root_dir", str(output_root))

        if effective_override:
            cmd.extend(["--override-json", json.dumps(effective_override, ensure_ascii=False)])

        print(f"[RUN ] #{idx}/{total} {exp_name}")
        print("       " + " ".join(cmd))

        if args.dry_run:
            executed += 1
            continue

        try:
            result = subprocess.run(cmd, check=True)
            if result.returncode == 0:
                executed += 1
        except subprocess.CalledProcessError as e:
            print(f"[FAIL] {exp_name} -> return code {e.returncode}")
            failed += 1

        print("-" * 80)

    print("\n[SUMMARY]")
    print(f"Total   : {total}")
    print(f"Skipped : {skipped}")
    print(f"Executed: {executed}")
    print(f"Failed  : {failed}")


if __name__ == "__main__":
    main()
