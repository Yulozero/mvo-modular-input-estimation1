import argparse
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=str, default="configs/experiments/exp_suite_paper_full_matrix.yaml")
    parser.add_argument("--python", type=str, default=sys.executable)
    return parser.parse_args()


def main():
    args = parse_args()
    cmd = [
        args.python,
        "-m",
        "src.runners.run_suite",
        "--suite",
        args.suite,
        "--group",
        "Baseline",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
