from dataclasses import dataclass
from typing import Optional, Dict, List
import pandas as pd
import numpy as np


@dataclass
class MarketData:
    returns: pd.DataFrame
    prices: Optional[pd.DataFrame] = None
    market_caps: Optional[pd.DataFrame] = None

    # backward-looking feature / reference volatility
    realized_vol: Optional[pd.DataFrame] = None

    # forward labels aligned to returns index
    # at date t, forward_returns.loc[t] means return over future horizon [t+1, ..., t+H]
    forward_returns: Optional[pd.DataFrame] = None
    forward_vol: Optional[pd.DataFrame] = None

    features: Optional[Dict[str, pd.DataFrame]] = None
    asset_meta: Optional[pd.DataFrame] = None


@dataclass
class WindowSlice:
    train_returns: pd.DataFrame
    valid_returns: Optional[pd.DataFrame]
    test_returns: pd.DataFrame

    # all visible history before predict_date
    history_returns: Optional[pd.DataFrame] = None
    history_realized_vol: Optional[pd.DataFrame] = None

    # block-purged labels
    train_forward_returns: Optional[pd.DataFrame] = None
    valid_forward_returns: Optional[pd.DataFrame] = None
    test_forward_returns: Optional[pd.DataFrame] = None

    train_forward_vol: Optional[pd.DataFrame] = None
    valid_forward_vol: Optional[pd.DataFrame] = None
    test_forward_vol: Optional[pd.DataFrame] = None

    train_realized_vol: Optional[pd.DataFrame] = None
    valid_realized_vol: Optional[pd.DataFrame] = None
    test_realized_vol: Optional[pd.DataFrame] = None

    rebalance_date: Optional[pd.Timestamp] = None
    predict_date: Optional[pd.Timestamp] = None
    asset_list: Optional[List[str]] = None

    train_start_idx: Optional[int] = None
    train_end_idx: Optional[int] = None
    valid_start_idx: Optional[int] = None
    valid_end_idx: Optional[int] = None
    test_start_idx: Optional[int] = None
    test_end_idx: Optional[int] = None

    n_train_label_valid: Optional[int] = None
    n_valid_label_valid: Optional[int] = None


@dataclass
class PredictionBundle:
    mu_hat: np.ndarray
    sigma_hat: np.ndarray
    D_hat: np.ndarray
    R_hat: np.ndarray
    Sigma_hat: np.ndarray
    meta: dict


@dataclass
class PortfolioDecision:
    date: pd.Timestamp
    weights: np.ndarray
    expected_return: float
    expected_risk: float
    turnover: float
    transaction_cost: float
    realized_return: float
    realized_net_return: float
    meta: dict


@dataclass
class KGraphData:
    similarity: Optional[pd.DataFrame] = None
    adjacency: Optional[pd.DataFrame] = None
    node_embeddings: Optional[pd.DataFrame] = None
