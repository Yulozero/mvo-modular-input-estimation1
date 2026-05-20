import argparse
import json
from pathlib import Path

from src.builders import (
    build_corr_model,
    build_executor,
    build_mean_model,
    build_optimizer,
    build_psd_projector,
    build_splitter,
    build_vol_model,
)
from src.backtest.rolling_engine import RollingBacktestEngine, CovarianceAssembler
from src.data.loader import load_kg_data, load_market_data
from src.evaluation.evaluator import ExperimentEvaluator
from src.utils.config import load_config_from_preset, load_experiment_config
from src.utils.paper_exporters import (
    export_config_and_meta,
    export_paper_tables,
    export_raw_results,
    export_summary_files,
)
from src.core.seed import set_seed
from src.data.cleaner import clean_returns, align_dataframes
from src.data.realized_vol import (
    build_backward_realized_vol_from_returns,
)


def prepare_market_data(market_data, config):
    returns = clean_returns(market_data.returns)

    rv_window = config.get("data", {}).get("rv_window", 20)
    realized_vol = market_data.realized_vol
    if realized_vol is None:
        realized_vol = build_backward_realized_vol_from_returns(returns, window=rv_window)
    aligned = align_dataframes(returns, realized_vol)
    returns, realized_vol = aligned

    market_data.returns = returns
    market_data.realized_vol = realized_vol
    # Strict protocol: forward labels are generated inside each train/valid block.
    market_data.forward_returns = None
    market_data.forward_vol = None
    return market_data


def infer_periods_per_year(config):
    rebalance_freq = (
        config.get("split", {}).get("rebalance_freq")
        or config.get("data", {}).get("rebalance_freq")
        or 20
    )

    rebalance_freq = float(rebalance_freq)
    if rebalance_freq <= 0:
        raise ValueError(f"rebalance_freq must be positive, got {rebalance_freq}.")

    # Holding-period metrics should annualize by the actual rebalance cadence
    # so robustness runs with different horizons remain comparable.
    return 252.0 / rebalance_freq


def infer_daily_periods_per_year(config):
    return float(config.get("evaluation", {}).get("daily_periods_per_year", 252))


def make_output_dir(config):
    root_dir = config.get("output", {}).get("root_dir", "outputs")
    exp_name = (
        config.get("output", {}).get("experiment_name")
        or config.get("_meta", {}).get("experiment_name")
        or config.get("_meta", {}).get("preset_name")
        or "experiment"
    )
    output_dir = Path(root_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)


def export_protocol_summary(config, split_info, market_data, output_dir: str):
    meta_dir = Path(output_dir) / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "strict_no_future_leakage": True,
        "label_construction_policy": "block_internal_only",
        "tail_purge_policy": "train_and_valid_block_tail_purged",
        "test_label_policy": "computed_after_holding_period_only",
        "price_feature_visibility": "history_window_only",
        "dwt_visibility_policy": "sample_local_history_only",
        "n_dates": int(len(market_data.returns)),
        "n_assets": int(market_data.returns.shape[1]),
        "date_start": str(market_data.returns.index.min()),
        "date_end": str(market_data.returns.index.max()),
        **split_info,
    }

    with open(meta_dir / "protocol_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def validate_weight_cap_config(config, n_assets: int):
    opt_cfg = config.get("optimizer", {}) or {}
    weight_cap = opt_cfg.get("weight_cap")
    if weight_cap is None:
        return

    weight_cap = float(weight_cap)
    if weight_cap <= 0 or weight_cap > 1:
        raise ValueError(f"optimizer.weight_cap must be in (0, 1], got {weight_cap}.")
    if weight_cap * int(n_assets) < 1.0 - 1e-12:
        raise ValueError(
            f"Infeasible weight_cap={weight_cap} for n_assets={n_assets}. "
            f"Need weight_cap * n_assets >= 1."
        )


def _get_seq_like(cfg: dict, default: int = 60) -> int:
    return int(cfg.get("seq_len", cfg.get("lookback", default)))


def validate_splitter_overview(config):
    split_cfg = config.get("split", {})
    data_cfg = config.get("data", {})
    mean_cfg = config.get("mean_model", {})
    vol_cfg = config.get("vol_model", {})

    train_window = int(split_cfg.get("train_window", data_cfg.get("train_window", 120)))
    valid_window = int(split_cfg.get("valid_window", data_cfg.get("valid_window", 40)))
    history_window = int(split_cfg.get("history_window", data_cfg.get("history_window", train_window + valid_window)))
    horizon = int(
        config.get("targets", {}).get("horizon")
        or split_cfg.get("test_horizon", data_cfg.get("test_horizon", 20))
    )
    rebalance_freq = int(split_cfg.get("rebalance_freq", data_cfg.get("rebalance_freq", horizon)))

    mean_seq_len = _get_seq_like(mean_cfg, 60)
    vol_seq_len = _get_seq_like(vol_cfg, 60)
    max_seq_len = max(mean_seq_len, vol_seq_len)

    if train_window <= 0 or valid_window <= 0 or horizon <= 0 or rebalance_freq <= 0:
        raise ValueError("train_window, valid_window, horizon, rebalance_freq must all be positive.")
    if history_window <= 0:
        raise ValueError("history_window must be positive.")
    if train_window + valid_window != history_window:
        raise ValueError(
            f"Invalid config: train_window + valid_window must equal history_window. "
            f"Got {train_window} + {valid_window} != {history_window}."
        )

    if valid_window <= horizon:
        raise ValueError(
            f"Invalid config: valid_window={valid_window} must be > horizon={horizon}."
        )

    if max_seq_len > history_window:
        raise ValueError(
            f"Invalid config: max(seq_len/lookback)={max_seq_len} exceeds history_window={history_window}."
        )

    if train_window <= horizon:
        print(
            f"[WARN] train_window={train_window} is close to horizon={horizon}. "
            f"Effective labeled train rows may be small."
        )

    if valid_window < 2 * horizon:
        print(
            f"[WARN] valid_window={valid_window} < 2*horizon={2 * horizon}. "
            f"This is legal but not ideal for stable model selection."
        )

    return {
        "train_window": train_window,
        "valid_window": valid_window,
        "history_window": history_window,
        "horizon": horizon,
        "rebalance_freq": rebalance_freq,
        "mean_seq_len_or_lookback": mean_seq_len,
        "vol_seq_len_or_lookback": vol_seq_len,
        "train_label_rows_after_purge": max(train_window - horizon, 0),
        "valid_label_rows_after_purge": max(valid_window - horizon, 0),
    }


def print_splitter_overview(info: dict):
    print("[SPLIT OVERVIEW]")
    for k, v in info.items():
        print(f"  - {k}: {v}")


def run_one(config):
    set_seed(config.get("seed", 42))

    split_info = validate_splitter_overview(config)
    print_splitter_overview(split_info)

    market_data = load_market_data(config)
    market_data = prepare_market_data(market_data, config)
    validate_weight_cap_config(config, n_assets=market_data.returns.shape[1])
    kg_data = load_kg_data(config)

    mean_model = build_mean_model(config.get("mean_model", {}))
    vol_model = build_vol_model(config.get("vol_model", {}))
    corr_model = build_corr_model(config.get("corr_model", {}), config)
    optimizer = build_optimizer(config.get("optimizer", {}))
    splitter = build_splitter(config)
    executor = build_executor(config)
    psd_projector = build_psd_projector(config.get("corr_model", {}))
    assembler = CovarianceAssembler(psd_projector=psd_projector)

    engine = RollingBacktestEngine(
    mean_model=mean_model,
    vol_model=vol_model,
    corr_model=corr_model,
    optimizer=optimizer,
    assembler=assembler,
    executor=executor,
    logger=None,
    verbose=True,
    progress_interval=1,
    stage_verbose=True,
    )


    engine_results = engine.run(
        market_data=market_data,
        splitter=splitter,
        kg_data=kg_data,
    )

    actual_periods_per_year = infer_periods_per_year(config)
    actual_daily_periods_per_year = infer_daily_periods_per_year(config)
    config.setdefault("evaluation", {})
    config["evaluation"]["periods_per_year"] = actual_periods_per_year
    config["evaluation"]["daily_periods_per_year"] = actual_daily_periods_per_year

    evaluator = ExperimentEvaluator(
        periods_per_year=actual_periods_per_year,
        daily_periods_per_year=actual_daily_periods_per_year,
    )
    summary = evaluator.summarize(engine_results)

    output_dir = make_output_dir(config)
    export_config_and_meta(config, output_dir)
    export_protocol_summary(config, split_info, market_data, output_dir)
    export_raw_results(engine_results, output_dir)
    export_summary_files(summary, output_dir)
    export_paper_tables(config, summary, output_dir)

    print(f"[DONE] Experiment finished. Output saved to: {output_dir}")
    return engine_results, summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to experiment yaml")
    parser.add_argument("--base", type=str, default="configs/base.yaml", help="Path to base yaml")
    parser.add_argument("--preset", type=str, default=None, help="Path to preset yaml")
    parser.add_argument("--override", type=str, default=None, help="Path to override yaml/json")
    parser.add_argument("--override-json", type=str, default=None, help="Inline override json string")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.config:
        config = load_experiment_config(args.config)
    elif args.preset:
        override = {}
        if args.override:
            override_path = Path(args.override)
            if not override_path.exists():
                raise FileNotFoundError(f"Override file not found: {override_path}")
            if override_path.suffix.lower() in {".json"}:
                with open(override_path, "r", encoding="utf-8") as f:
                    override = json.load(f) or {}
            else:
                from src.utils.config import load_yaml
                override = load_yaml(str(override_path)) or {}
        if args.override_json:
            parsed = json.loads(args.override_json)
            if not isinstance(parsed, dict):
                raise ValueError("--override-json must decode to a JSON object")
            override = parsed

        config = load_config_from_preset(args.base, args.preset, override=override)
    else:
        raise ValueError("Please provide either --config or --preset")

    run_one(config)


if __name__ == "__main__":
    main()
