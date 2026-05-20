"""
整个实验
python -m src.runners.run_suite --suite configs/experiments/exp_suite_full.yaml


单个实验
python -m src.runners.run_experiment --base configs/base.yaml --preset configs/presets/fusion_no_kg_default.yaml

聚合结果
python aggregate_suite_results.py --outputs_root outputs --save_dir outputs/_aggregated

可跳过的运行
python -m src.runners.run_suite_resume --suite configs/experiments/exp_suite_full.yaml

atention.yaml
"""
