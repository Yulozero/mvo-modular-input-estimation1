from copy import deepcopy

def make_base_config():
    horizon = 20

    return {
        "seed": 42,
        "data": {
            "returns_path": "data/processed/pool30/returns.csv",
            "prices_path": "data/processed/pool30/close.csv",
            "market_cap_path": "data/processed/pool30/market_cap.csv",
            "realized_vol_path": None,
            "history_window": 160,
            "train_window": 120,
            "valid_window": 40,
            "test_horizon": horizon,
            "rebalance_freq": horizon,
            "rv_window": 20,
        },
        "split": {
            "history_window": 160,
            "train_window": 120,
            "valid_window": 40,
            "test_horizon": horizon,
            "rebalance_freq": horizon,
            "mode": "rolling",
        },
        "targets": {
            "mean_target": "forward_cum_return",
            "vol_target": "forward_realized_vol",
            "horizon": horizon,
        },
        "kg": {
            "enabled": False,
        },
        "mean_model": {
            "name": "hist_mean",
            "lookback": 60,
        },
        "vol_model": {
            "name": "ewma_vol",
            "lookback": 60,
            "decay": 0.94,
            "target_horizon": horizon,
        },
        "corr_model": {
            "name": "pearson",
            "psd_eps": 1e-8,
        },
        "optimizer": {
            "name": "mean_variance",
            "gamma": 3.0,
            "long_only": True,
            "l2_reg": 0.001,
            "weight_cap": 0.20,
            "turnover_penalty": 0.0,
            "transaction_cost": 0.001,
            "charge_initial_cost": True,
        },
        "backtest": {
            "mode": "rolling",
        },
        "evaluation": {
            "periods_per_year": 12,
            "daily_periods_per_year": 252,
        },
        "output": {
            "root_dir": "outputs",
            "experiment_name": "base_experiment",
        },
        "experiment": {
            "group": "Main",
            "variant": "Base",
            "paper_tag": "main_base",
        },
    }


def _mean_hist():
    return {"name": "hist_mean", "lookback": 60}


def _mean_ewma():
    return {"name": "ewma_mean", "lookback": 60, "decay": 0.94}


def _mean_lstm():
    return {
        "name": "lstm_mean",
        "seq_len": 60,
        "hidden_size": 32,
        "num_layers": 1,
        "epochs": 30,
        "batch_size": 64,
        "lr": 1e-3,
        "dropout": 0.1,
        "early_stopping_patience": 5,
        "random_state": 42,
    }


def _mean_attention():
    return {
        "name": "attention_mean",
        "seq_len": 60,
        "hidden_size": 32,
        "epochs": 30,
        "batch_size": 64,
        "lr": 1e-3,
        "dropout": 0.1,
        "early_stopping_patience": 5,
        "random_state": 42,
    }


def _mean_transformer():
    return {
        "name": "transformer_mean",
        "seq_len": 60,
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "epochs": 40,
        "batch_size": 64,
        "lr": 1e-3,
        "dropout": 0.1,
        "weight_decay": 1e-4,
        "grad_clip_norm": 1.0,
        "early_stopping_patience": 7,
        "use_huber_loss": False,
        "random_state": 42,
    }


def _vol_rolling(h):
    return {"name": "rolling_std", "lookback": 60, "target_horizon": h}


def _vol_ewma(h):
    return {"name": "ewma_vol", "lookback": 60, "decay": 0.94, "target_horizon": h}


def _vol_garch(h):
    return {
        "name": "garch",
        "p": 1,
        "q": 1,
        "target_horizon": h,
        "mean": "Zero",
        "vol": "GARCH",
        "dist": "normal",
        "rescale_returns": 100.0,
        "min_history": 60,
    }


def _vol_garch_lstm(h):
    return {
        "name": "garch_lstm",
        "seq_len": 60,
        "hidden_size": 32,
        "num_layers": 1,
        "epochs": 30,
        "batch_size": 64,
        "lr": 1e-3,
        "dropout": 0.1,
        "early_stopping_patience": 5,
        "garch_p": 1,
        "garch_q": 1,
        "rescale_returns": 100.0,
        "min_history": 60,
        "target_horizon": h,
        "random_state": 42,
    }


def _vol_filter_lstm(h):
    return {
        "name": "filter_lstm",
        "seq_len": 60,
        "hidden_size": 32,
        "num_layers": 1,
        "epochs": 30,
        "batch_size": 64,
        "lr": 1e-3,
        "dropout": 0.1,
        "smooth_span": 5,
        "early_stopping_patience": 5,
        "target_horizon": h,
        "random_state": 42,
    }


def _vol_dwt_lstm(h):
    return {
        "name": "dwt_lstm",
        "seq_len": 60,
        "hidden_size": 32,
        "num_layers": 1,
        "epochs": 30,
        "batch_size": 64,
        "lr": 1e-3,
        "dropout": 0.1,
        "early_stopping_patience": 5,
        "wavelet": "db4",
        "level": 3,
        "dwt_mode": "symmetric",
        "target_horizon": h,
        "random_state": 42,
    }


def _corr(name):
    return {"name": name, "psd_eps": 1e-8}


def _set_meta(cfg, group, variant, exp_name, paper_tag=None):
    cfg["output"]["experiment_name"] = exp_name
    cfg["experiment"]["group"] = group
    cfg["experiment"]["variant"] = variant
    cfg["experiment"]["paper_tag"] = paper_tag or exp_name
    return cfg


def _set_split(cfg, train_window, valid_window, horizon, rebalance_freq=None):
    if rebalance_freq is None:
        rebalance_freq = horizon
    history_window = train_window + valid_window

    cfg["data"]["history_window"] = history_window
    cfg["data"]["train_window"] = train_window
    cfg["data"]["valid_window"] = valid_window
    cfg["data"]["test_horizon"] = horizon
    cfg["data"]["rebalance_freq"] = rebalance_freq

    cfg["split"]["history_window"] = history_window
    cfg["split"]["train_window"] = train_window
    cfg["split"]["valid_window"] = valid_window
    cfg["split"]["test_horizon"] = horizon
    cfg["split"]["rebalance_freq"] = rebalance_freq

    cfg["targets"]["horizon"] = horizon

    if "vol_model" in cfg and isinstance(cfg["vol_model"], dict):
        cfg["vol_model"]["target_horizon"] = horizon

    return cfg


def make_baseline_experiments():
    exps = []
    base = make_base_config()

    baseline_opts = [
        ("equal_weight", "Equal Weight"),
        ("min_variance", "Min Variance"),
        ("inverse_volatility", "Inverse Volatility"),
        ("risk_parity", "Risk Parity"),
        ("mean_variance", "Traditional MVO"),
        ("shrinkage_mvo", "Shrinkage MVO"),
    ]

    for opt_name, variant in baseline_opts:
        cfg = deepcopy(base)
        cfg["optimizer"]["name"] = opt_name
        cfg = _set_meta(
            cfg,
            group="Baseline",
            variant=variant,
            exp_name=f"baseline__{opt_name}",
            paper_tag=f"baseline_{opt_name}",
        )
        exps.append(cfg)

    return exps


def make_mean_comparison_experiments():
    exps = []
    base = make_base_config()
    h = base["targets"]["horizon"]

    candidates = [
        (_mean_hist(), "Hist Mean"),
        (_mean_ewma(), "EWMA Mean"),
        (_mean_lstm(), "LSTM Mean"),
        (_mean_attention(), "Attention Mean"),
        (_mean_transformer(), "Transformer Mean"),
    ]

    for mean_cfg, variant in candidates:
        cfg = deepcopy(base)
        cfg["mean_model"] = deepcopy(mean_cfg)
        cfg["vol_model"] = _vol_ewma(h)
        cfg["corr_model"] = _corr("pearson")
        cfg["optimizer"]["name"] = "mean_variance"
        cfg = _set_meta(
            cfg,
            group="Mean Comparison",
            variant=variant,
            exp_name=f"mean_cmp__{mean_cfg['name']}",
            paper_tag=f"mean_{mean_cfg['name']}",
        )
        exps.append(cfg)

    return exps


def make_vol_comparison_experiments(best_mean_name="transformer_mean"):
    exps = []
    base = make_base_config()
    h = base["targets"]["horizon"]

    if best_mean_name == "attention_mean":
        mean_cfg = _mean_attention()
    elif best_mean_name == "lstm_mean":
        mean_cfg = _mean_lstm()
    elif best_mean_name == "ewma_mean":
        mean_cfg = _mean_ewma()
    elif best_mean_name == "hist_mean":
        mean_cfg = _mean_hist()
    else:
        mean_cfg = _mean_transformer()

    candidates = [
        (_vol_rolling(h), "Rolling Std"),
        (_vol_ewma(h), "EWMA Vol"),
        (_vol_garch(h), "GARCH"),
        (_vol_garch_lstm(h), "GARCH-LSTM"),
        (_vol_filter_lstm(h), "Filter-LSTM"),
        (_vol_dwt_lstm(h), "DWT-LSTM"),
    ]

    for vol_cfg, variant in candidates:
        cfg = deepcopy(base)
        cfg["mean_model"] = deepcopy(mean_cfg)
        cfg["vol_model"] = deepcopy(vol_cfg)
        cfg["corr_model"] = _corr("pearson")
        cfg["optimizer"]["name"] = "mean_variance"
        cfg = _set_meta(
            cfg,
            group="Vol Comparison",
            variant=variant,
            exp_name=f"vol_cmp__{mean_cfg['name']}__{vol_cfg['name']}",
            paper_tag=f"vol_{vol_cfg['name']}",
        )
        exps.append(cfg)

    return exps


def make_corr_comparison_experiments(best_mean_name="transformer_mean", best_vol_name="garch_lstm"):
    exps = []
    base = make_base_config()
    h = base["targets"]["horizon"]

    mean_map = {
        "hist_mean": _mean_hist(),
        "ewma_mean": _mean_ewma(),
        "lstm_mean": _mean_lstm(),
        "attention_mean": _mean_attention(),
        "transformer_mean": _mean_transformer(),
        "transformer_attn_mean": _mean_transformer(),
    }
    vol_map = {
        "rolling_std": _vol_rolling(h),
        "ewma_vol": _vol_ewma(h),
        "garch": _vol_garch(h),
        "garch_lstm": _vol_garch_lstm(h),
        "filter_lstm": _vol_filter_lstm(h),
        "dwt_lstm": _vol_dwt_lstm(h),
    }

    mean_cfg = deepcopy(mean_map.get(best_mean_name, _mean_transformer()))
    vol_cfg = deepcopy(vol_map.get(best_vol_name, _vol_garch_lstm(h)))

    candidates = [
        ("pearson", "Pearson Corr"),
        ("spearman", "Spearman Corr"),
        ("kendall", "Kendall Corr"),
        ("distance_corr", "Distance Corr"),
    ]

    for corr_name, variant in candidates:
        cfg = deepcopy(base)
        cfg["mean_model"] = deepcopy(mean_cfg)
        cfg["vol_model"] = deepcopy(vol_cfg)
        cfg["corr_model"] = _corr(corr_name)
        cfg["optimizer"]["name"] = "mean_variance"
        cfg = _set_meta(
            cfg,
            group="Corr Comparison",
            variant=variant,
            exp_name=f"corr_cmp__{mean_cfg['name']}__{vol_cfg['name']}__{corr_name}",
            paper_tag=f"corr_{corr_name}",
        )
        exps.append(cfg)

    return exps


def make_full_model_ablation_experiments(
    best_mean_name="transformer_mean",
    best_vol_name="garch_lstm",
    best_corr_name="pearson",
):
    base = make_base_config()
    exps = []
    h = base["targets"]["horizon"]
    mean_map = {
        "hist_mean": _mean_hist(),
        "ewma_mean": _mean_ewma(),
        "lstm_mean": _mean_lstm(),
        "attention_mean": _mean_attention(),
        "transformer_mean": _mean_transformer(),
        "transformer_attn_mean": _mean_transformer(),
    }
    vol_map = {
        "rolling_std": _vol_rolling(h),
        "ewma_vol": _vol_ewma(h),
        "garch": _vol_garch(h),
        "garch_lstm": _vol_garch_lstm(h),
        "filter_lstm": _vol_filter_lstm(h),
        "dwt_lstm": _vol_dwt_lstm(h),
    }
    corr_cfg = _corr(best_corr_name)
    best_mean_cfg = deepcopy(mean_map.get(best_mean_name, _mean_transformer()))
    best_vol_cfg = deepcopy(vol_map.get(best_vol_name, _vol_garch_lstm(h)))

    baseline = deepcopy(base)
    baseline["mean_model"] = _mean_hist()
    baseline["vol_model"] = _vol_ewma(h)
    baseline["corr_model"] = _corr("pearson")
    baseline["optimizer"]["name"] = "mean_variance"
    exps.append(_set_meta(baseline, "Ablation", "Baseline", "ablation__baseline", "ablation_baseline"))

    best_mean_only = deepcopy(base)
    best_mean_only["mean_model"] = deepcopy(best_mean_cfg)
    best_mean_only["vol_model"] = _vol_ewma(h)
    best_mean_only["corr_model"] = _corr("pearson")
    exps.append(_set_meta(best_mean_only, "Ablation", "Best Mean Only", "ablation__best_mean_only", "ablation_best_mean"))

    best_vol_only = deepcopy(base)
    best_vol_only["mean_model"] = _mean_hist()
    best_vol_only["vol_model"] = deepcopy(best_vol_cfg)
    best_vol_only["corr_model"] = _corr("pearson")
    exps.append(_set_meta(best_vol_only, "Ablation", "Best Vol Only", "ablation__best_vol_only", "ablation_best_vol"))

    nonlinear_corr_only = deepcopy(base)
    nonlinear_corr_only["mean_model"] = _mean_hist()
    nonlinear_corr_only["vol_model"] = _vol_ewma(h)
    nonlinear_corr_only["corr_model"] = deepcopy(corr_cfg)
    exps.append(_set_meta(nonlinear_corr_only, "Ablation", "Nonlinear Corr Only", "ablation__nonlinear_corr_only", "ablation_nonlinear_corr"))

    full_model = deepcopy(base)
    full_model["mean_model"] = deepcopy(best_mean_cfg)
    full_model["vol_model"] = deepcopy(best_vol_cfg)
    full_model["corr_model"] = deepcopy(corr_cfg)
    exps.append(_set_meta(full_model, "Ablation", "Full Model", "ablation__full_model", "ablation_full_model"))

    return exps


def make_robustness_gamma_experiments():
    exps = []
    base = make_base_config()
    h = base["targets"]["horizon"]

    for gamma in [1.0, 3.0, 5.0, 10.0]:
        cfg = deepcopy(base)
        cfg["mean_model"] = _mean_transformer()
        cfg["vol_model"] = _vol_garch_lstm(h)
        cfg["corr_model"] = _corr("pearson")
        cfg["optimizer"]["name"] = "mean_variance"
        cfg["optimizer"]["gamma"] = gamma
        cfg = _set_meta(
            cfg,
            group="Robustness",
            variant=f"Gamma={gamma}",
            exp_name=f"robust_gamma__{gamma}",
            paper_tag=f"robust_gamma_{gamma}",
        )
        exps.append(cfg)

    return exps


def make_robustness_cost_experiments():
    exps = []
    base = make_base_config()
    h = base["targets"]["horizon"]

    for tc in [0.0, 0.001, 0.002, 0.005]:
        cfg = deepcopy(base)
        cfg["mean_model"] = _mean_transformer()
        cfg["vol_model"] = _vol_garch_lstm(h)
        cfg["corr_model"] = _corr("pearson")
        cfg["optimizer"]["name"] = "mean_variance"
        cfg["optimizer"]["transaction_cost"] = tc
        cfg = _set_meta(
            cfg,
            group="Robustness",
            variant=f"Cost={tc}",
            exp_name=f"robust_cost__{tc}",
            paper_tag=f"robust_cost_{tc}",
        )
        exps.append(cfg)

    return exps


def make_robustness_history_experiments():
    exps = []

    candidates = [
        {"train_window": 80, "valid_window": 40, "horizon": 20, "tag": "80_40_20"},
        {"train_window": 120, "valid_window": 40, "horizon": 20, "tag": "120_40_20"},
        {"train_window": 180, "valid_window": 60, "horizon": 20, "tag": "180_60_20"},
    ]

    for c in candidates:
        cfg = make_base_config()
        cfg = _set_split(
            cfg,
            train_window=c["train_window"],
            valid_window=c["valid_window"],
            horizon=c["horizon"],
            rebalance_freq=c["horizon"],
        )
        cfg["mean_model"] = _mean_transformer()
        cfg["vol_model"] = _vol_garch_lstm(c["horizon"])
        cfg["corr_model"] = _corr("pearson")
        cfg = _set_meta(
            cfg,
            group="Robustness",
            variant=f"History={c['tag']}",
            exp_name=f"robust_history__{c['tag']}",
            paper_tag=f"robust_history_{c['tag']}",
        )
        exps.append(cfg)

    return exps


def make_robustness_horizon_experiments():
    exps = []

    candidates = [
        {"train_window": 120, "valid_window": 20, "horizon": 5, "rebalance_freq": 5, "seq_len": 60, "tag": "h5"},
        {"train_window": 120, "valid_window": 40, "horizon": 20, "rebalance_freq": 20, "seq_len": 60, "tag": "h20"},
        {"train_window": 180, "valid_window": 120, "horizon": 60, "rebalance_freq": 60, "seq_len": 100, "tag": "h60"},
    ]

    for c in candidates:
        cfg = make_base_config()
        cfg = _set_split(
            cfg,
            train_window=c["train_window"],
            valid_window=c["valid_window"],
            horizon=c["horizon"],
            rebalance_freq=c["rebalance_freq"],
        )

        cfg["mean_model"] = _mean_transformer()
        cfg["mean_model"]["seq_len"] = c["seq_len"]

        cfg["vol_model"] = _vol_garch_lstm(c["horizon"])
        cfg["vol_model"]["seq_len"] = c["seq_len"]

        cfg["corr_model"] = _corr("pearson")

        cfg = _set_meta(
            cfg,
            group="Robustness",
            variant=f"Horizon={c['horizon']}",
            exp_name=f"robust_horizon__{c['tag']}",
            paper_tag=f"robust_horizon_{c['tag']}",
        )
        exps.append(cfg)

    return exps


def make_all_paper_experiments(
    best_mean_name="transformer_mean",
    best_vol_name="garch_lstm",
    best_corr_name="pearson",
):
    exps = []
    exps.extend(make_baseline_experiments())
    exps.extend(make_mean_comparison_experiments())
    exps.extend(make_vol_comparison_experiments(best_mean_name=best_mean_name))
    exps.extend(make_corr_comparison_experiments(
        best_mean_name=best_mean_name,
        best_vol_name=best_vol_name,
    ))
    exps.extend(make_full_model_ablation_experiments(
        best_mean_name=best_mean_name,
        best_vol_name=best_vol_name,
        best_corr_name=best_corr_name,
    ))
    exps.extend(make_robustness_gamma_experiments())
    exps.extend(make_robustness_cost_experiments())
    exps.extend(make_robustness_history_experiments())
    exps.extend(make_robustness_horizon_experiments())
    return exps
