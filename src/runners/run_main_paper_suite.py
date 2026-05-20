import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=str, default="configs/experiments/exp_suite_paper_full_matrix.yaml")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--outputs_root", type=str, default="outputs_paper_full_matrix")
    return parser.parse_args()


def _run(cmd):
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    agg_dir = str(Path(args.outputs_root) / "_aggregated")

    _run([args.python, "-m", "src.runners.run_suite_resume", "--suite", args.suite, "--python", args.python])
    _run([args.python, "-m", "src.runners.aggregate_suite_results", "--outputs_root", args.outputs_root, "--save_dir", agg_dir])
    _run([args.python, "-m", "src.runners.build_significance_tables", "--outputs_root", args.outputs_root, "--save_dir", agg_dir])
    _run([args.python, "-m", "src.runners.build_main_plots", "--outputs_root", args.outputs_root, "--save_dir", str(Path(agg_dir) / "figures")])


if __name__ == "__main__":
    main()
