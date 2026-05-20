from typing import Any, Dict

from src.models.mean.hist_mean import HistoricalMeanModel
from src.models.mean.ewma_mean import EWMAMeanModel
from src.models.mean.lstm_mean import LSTMMeanModel
from src.models.mean.attention_mean import AttentionWeightedMeanModel
from src.models.mean.asset_conditioned_attention_mean import AssetConditionedAttentionMeanModel
from src.models.mean.asset_conditioned_temp_attention_mean import AssetConditionedTempAttentionMeanModel
from src.models.mean.asset_conditioned_multihead_attention_mean import AssetConditionedMultiHeadAttentionMeanModel
from src.models.mean.transformer_attn_mean import TransformerAttentionMeanModel

from src.models.vol.rolling_std import RollingStdVolModel
from src.models.vol.ewma_vol import EWMAVolModel
from src.models.vol.garch import GARCHVolModel
from src.models.vol.garch_lstm import GARCHLSTMVolModel
from src.models.vol.filter_lstm import FilterLSTMVolModel
from src.models.vol.dwt_lstm import DWTLSTMVolModel
from src.models.vol.asset_conditioned_dwt_lstm import AssetConditionedDWTLSTMVolModel

from src.models.corr.pearson import PearsonCorrModel
from src.models.corr.spearman import SpearmanCorrModel
from src.models.corr.kendall import KendallCorrModel
from src.models.corr.distance_corr import DistanceCorrModel
from src.models.corr.signed_distance_corr import SignedDistanceCorrModel
from src.models.corr.denoised_signed_distance_corr import DenoisedSignedDistanceCorrModel
from src.models.corr.fusion_corr import FusionCorrModel
from src.models.corr.psd import PSDProjector

from src.optimization.mean_variance import MeanVarianceOptimizer
from src.optimization.equal_weight import EqualWeightOptimizer
from src.optimization.min_variance import MinimumVarianceOptimizer
from src.optimization.inverse_volatility import InverseVolatilityOptimizer
from src.optimization.risk_parity import RiskParityOptimizer
from src.optimization.shrinkage_mvo import ShrinkageMVOOptimizer

from src.backtest.splitter import RollingWindowSplitter
from src.backtest.executor import PortfolioExecutor


def _get_seq_len(cfg: Dict[str, Any], default: int = 60) -> int:
    return int(cfg.get("seq_len", cfg.get("lookback", default)))


def build_mean_model(cfg: Dict[str, Any]):
    name = cfg["name"]

    if name == "hist_mean":
        return HistoricalMeanModel(
            lookback=cfg.get("lookback", 60)
        )

    if name == "ewma_mean":
        return EWMAMeanModel(
            lookback=cfg.get("lookback", 60),
            decay=cfg.get("decay", 0.94),
        )

    if name == "lstm_mean":
        return LSTMMeanModel(
            seq_len=_get_seq_len(cfg, 60),
            hidden_size=cfg.get("hidden_size", 32),
            num_layers=cfg.get("num_layers", 1),
            epochs=cfg.get("epochs", 30),
            batch_size=cfg.get("batch_size", 64),
            lr=cfg.get("lr", 1e-3),
            dropout=cfg.get("dropout", 0.1),
            early_stopping_patience=cfg.get("early_stopping_patience", 5),
            random_state=cfg.get("random_state", 42),
        )

    if name == "attention_mean":
        return AttentionWeightedMeanModel(
            seq_len=_get_seq_len(cfg, 60),
            hidden_size=cfg.get("hidden_size", 32),
            epochs=cfg.get("epochs", 30),
            batch_size=cfg.get("batch_size", 64),
            lr=cfg.get("lr", 1e-3),
            dropout=cfg.get("dropout", 0.1),
            early_stopping_patience=cfg.get("early_stopping_patience", 5),
            random_state=cfg.get("random_state", 42),
        )

    if name in ["asset_conditioned_attention_mean", "asset_conditioned_attn_mean"]:
        return AssetConditionedAttentionMeanModel(
            seq_len=_get_seq_len(cfg, 60),
            hidden_size=cfg.get("hidden_size", 32),
            asset_emb_dim=cfg.get("asset_emb_dim", 8),
            epochs=cfg.get("epochs", 30),
            batch_size=cfg.get("batch_size", 64),
            lr=cfg.get("lr", 1e-3),
            dropout=cfg.get("dropout", 0.1),
            early_stopping_patience=cfg.get("early_stopping_patience", 5),
            random_state=cfg.get("random_state", 42),
        )

    if name in ["asset_conditioned_temp_attention_mean", "asset_conditioned_temp_attn_mean"]:
        return AssetConditionedTempAttentionMeanModel(
            seq_len=_get_seq_len(cfg, 60),
            hidden_size=cfg.get("hidden_size", 32),
            asset_emb_dim=cfg.get("asset_emb_dim", 8),
            temperature=cfg.get("temperature", 0.5),
            epochs=cfg.get("epochs", 30),
            batch_size=cfg.get("batch_size", 64),
            lr=cfg.get("lr", 1e-3),
            dropout=cfg.get("dropout", 0.1),
            early_stopping_patience=cfg.get("early_stopping_patience", 5),
            random_state=cfg.get("random_state", 42),
        )

    if name in ["asset_conditioned_multihead_attention_mean", "asset_conditioned_multihead_attn_mean"]:
        return AssetConditionedMultiHeadAttentionMeanModel(
            seq_len=_get_seq_len(cfg, 60),
            hidden_size=cfg.get("hidden_size", 32),
            asset_emb_dim=cfg.get("asset_emb_dim", 8),
            n_heads=cfg.get("n_heads", 4),
            epochs=cfg.get("epochs", 30),
            batch_size=cfg.get("batch_size", 64),
            lr=cfg.get("lr", 1e-3),
            dropout=cfg.get("dropout", 0.1),
            early_stopping_patience=cfg.get("early_stopping_patience", 5),
            random_state=cfg.get("random_state", 42),
        )

    if name in ["transformer_attn_mean", "transformer_mean"]:
        return TransformerAttentionMeanModel(
            seq_len=_get_seq_len(cfg, 60),
            d_model=cfg.get("d_model", 64),
            n_heads=cfg.get("n_heads", 4),
            n_layers=cfg.get("n_layers", 2),
            epochs=cfg.get("epochs", 40),
            batch_size=cfg.get("batch_size", 64),
            lr=cfg.get("lr", 1e-3),
            dropout=cfg.get("dropout", 0.1),
            weight_decay=cfg.get("weight_decay", 1e-4),
            grad_clip_norm=cfg.get("grad_clip_norm", 1.0),
            early_stopping_patience=cfg.get("early_stopping_patience", 7),
            use_huber_loss=cfg.get("use_huber_loss", False),
            random_state=cfg.get("random_state", 42),
        )

    raise ValueError(f"Unknown mean model: {name}")


def build_vol_model(cfg: Dict[str, Any]):
    name = cfg["name"]
    target_horizon = int(cfg.get("target_horizon", 20))

    if name == "rolling_std":
        return RollingStdVolModel(
            lookback=cfg.get("lookback", 60),
            target_horizon=target_horizon,
        )

    if name == "ewma_vol":
        return EWMAVolModel(
            lookback=cfg.get("lookback", 60),
            decay=cfg.get("decay", 0.94),
            target_horizon=target_horizon,
        )

    if name == "garch":
        return GARCHVolModel(
            p=cfg.get("p", 1),
            q=cfg.get("q", 1),
            target_horizon=target_horizon,
            mean=cfg.get("mean", "Zero"),
            vol=cfg.get("vol", "GARCH"),
            dist=cfg.get("dist", "normal"),
            rescale_returns=cfg.get("rescale_returns", 100.0),
            min_history=cfg.get("min_history", 60),
        )

    if name == "garch_lstm":
        return GARCHLSTMVolModel(
            seq_len=_get_seq_len(cfg, 60),
            hidden_size=cfg.get("hidden_size", 32),
            num_layers=cfg.get("num_layers", 1),
            epochs=cfg.get("epochs", 30),
            batch_size=cfg.get("batch_size", 64),
            lr=cfg.get("lr", 1e-3),
            dropout=cfg.get("dropout", 0.1),
            early_stopping_patience=cfg.get("early_stopping_patience", 5),
            garch_p=cfg.get("garch_p", 1),
            garch_q=cfg.get("garch_q", 1),
            rescale_returns=cfg.get("rescale_returns", 100.0),
            min_history=cfg.get("min_history", 60),
            random_state=cfg.get("random_state", 42),
        )

    if name == "filter_lstm":
        return FilterLSTMVolModel(
            seq_len=_get_seq_len(cfg, 60),
            hidden_size=cfg.get("hidden_size", 32),
            num_layers=cfg.get("num_layers", 1),
            epochs=cfg.get("epochs", 30),
            batch_size=cfg.get("batch_size", 64),
            lr=cfg.get("lr", 1e-3),
            dropout=cfg.get("dropout", 0.1),
            smooth_span=cfg.get("smooth_span", 5),
            early_stopping_patience=cfg.get("early_stopping_patience", 5),
            random_state=cfg.get("random_state", 42),
        )

    if name == "dwt_lstm":
        return DWTLSTMVolModel(
            seq_len=_get_seq_len(cfg, 60),
            hidden_size=cfg.get("hidden_size", 32),
            num_layers=cfg.get("num_layers", 1),
            epochs=cfg.get("epochs", 30),
            batch_size=cfg.get("batch_size", 64),
            lr=cfg.get("lr", 1e-3),
            dropout=cfg.get("dropout", 0.1),
            early_stopping_patience=cfg.get("early_stopping_patience", 5),
            wavelet=cfg.get("wavelet", "db4"),
            level=cfg.get("level", 3),
            dwt_mode=cfg.get("dwt_mode", "symmetric"),
            target_horizon=target_horizon,
            random_state=cfg.get("random_state", 42),
        )

    if name in ["asset_conditioned_dwt_lstm", "asset_conditioned_dwt"]:
        return AssetConditionedDWTLSTMVolModel(
            seq_len=_get_seq_len(cfg, 60),
            hidden_size=cfg.get("hidden_size", 32),
            asset_emb_dim=cfg.get("asset_emb_dim", 8),
            num_layers=cfg.get("num_layers", 1),
            epochs=cfg.get("epochs", 30),
            batch_size=cfg.get("batch_size", 64),
            lr=cfg.get("lr", 1e-3),
            dropout=cfg.get("dropout", 0.1),
            early_stopping_patience=cfg.get("early_stopping_patience", 5),
            wavelet=cfg.get("wavelet", "db4"),
            level=cfg.get("level", 3),
            dwt_mode=cfg.get("dwt_mode", "symmetric"),
            target_horizon=target_horizon,
            random_state=cfg.get("random_state", 42),
        )

    raise ValueError(f"Unknown vol model: {name}")


def build_corr_model(cfg: Dict[str, Any], global_config: Dict[str, Any] = None):
    name = cfg["name"]

    if name == "pearson":
        return PearsonCorrModel()

    if name == "spearman":
        return SpearmanCorrModel()

    if name == "kendall":
        return KendallCorrModel()

    if name == "distance_corr":
        return DistanceCorrModel()

    if name == "signed_distance_corr":
        return SignedDistanceCorrModel()

    if name == "denoised_signed_distance_corr":
        return DenoisedSignedDistanceCorrModel(
            blend_with_pearson=cfg.get("blend_with_pearson", 0.5),
            shrink_to_identity=cfg.get("shrink_to_identity", 0.1),
            clip_value=cfg.get("clip_value", 0.95),
        )

    if name == "fusion_corr":
        return FusionCorrModel(lambda_=cfg.get("lambda_", 0.7))

    raise ValueError(f"Unknown corr model: {name}")


def build_optimizer(cfg: Dict[str, Any]):
    name = cfg.get("name", "mean_variance")

    if name in ["mean_variance", "traditional_mvo"]:
        return MeanVarianceOptimizer(
            gamma=cfg.get("gamma", 3.0),
            long_only=cfg.get("long_only", True),
            l2_reg=cfg.get("l2_reg", 0.001),
            turnover_penalty=cfg.get("turnover_penalty", 0.0),
            weight_cap=cfg.get("weight_cap"),
        )

    if name == "equal_weight":
        return EqualWeightOptimizer()

    if name == "min_variance":
        return MinimumVarianceOptimizer(
            long_only=cfg.get("long_only", True),
            l2_reg=cfg.get("l2_reg", 0.001),
            turnover_penalty=cfg.get("turnover_penalty", 0.0),
            weight_cap=cfg.get("weight_cap"),
        )

    if name == "inverse_volatility":
        return InverseVolatilityOptimizer()

    if name == "risk_parity":
        return RiskParityOptimizer(
            long_only=cfg.get("long_only", True),
            l2_reg=cfg.get("l2_reg", 1e-6),
            turnover_penalty=cfg.get("turnover_penalty", 0.0),
            weight_cap=cfg.get("weight_cap"),
        )

    if name == "shrinkage_mvo":
        return ShrinkageMVOOptimizer(
            gamma=cfg.get("gamma", 3.0),
            long_only=cfg.get("long_only", True),
            l2_reg=cfg.get("l2_reg", 0.001),
            shrink_alpha=cfg.get("shrink_alpha", 0.2),
            turnover_penalty=cfg.get("turnover_penalty", 0.0),
            weight_cap=cfg.get("weight_cap"),
        )

    raise ValueError(f"Unknown optimizer: {name}")


def build_splitter(cfg: Dict[str, Any]):
    split_cfg = cfg.get("split", {})
    data_cfg = cfg.get("data", {})
    backtest_cfg = cfg.get("backtest", {})

    train_window = split_cfg.get("train_window", data_cfg.get("train_window", 120))
    valid_window = split_cfg.get("valid_window", data_cfg.get("valid_window", 40))
    history_window = split_cfg.get(
        "history_window",
        data_cfg.get("history_window", int(train_window) + int(valid_window)),
    )
    test_horizon = split_cfg.get("test_horizon", data_cfg.get("test_horizon", 20))
    rebalance_freq = split_cfg.get("rebalance_freq", data_cfg.get("rebalance_freq", test_horizon))
    mode = split_cfg.get("mode", backtest_cfg.get("mode", "rolling"))

    return RollingWindowSplitter(
        train_window=int(train_window),
        valid_window=int(valid_window),
        history_window=int(history_window),
        test_horizon=int(test_horizon),
        rebalance_freq=int(rebalance_freq),
        mode=mode,
    )


def build_executor(cfg: Dict[str, Any]):
    optimizer_cfg = cfg.get("optimizer", cfg)
    return PortfolioExecutor(
        transaction_cost=optimizer_cfg.get("transaction_cost", 0.0),
        charge_initial_cost=optimizer_cfg.get("charge_initial_cost", True),
    )


def build_psd_projector(corr_cfg: Dict[str, Any]):
    return PSDProjector(eps=corr_cfg.get("psd_eps", 1e-8))
