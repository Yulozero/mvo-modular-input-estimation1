import numpy as np
import pandas as pd
from .base import BaseVolModel


class RollingStdVolModel(BaseVolModel):
    def __init__(self, lookback: int = 60, target_horizon: int = 20):
        super().__init__()
        self.lookback = lookback
        self.target_horizon = target_horizon

    def fit(
        self,
        train_returns: pd.DataFrame,
        train_targets: pd.DataFrame,
        valid_returns: pd.DataFrame = None,
        valid_targets: pd.DataFrame = None,
    ):
        window = train_returns.tail(self.lookback)
        self.asset_list_ = list(window.columns)

        daily_std = window.std(axis=0, ddof=1).values.astype(float)
        self.sigma_hat_ = np.sqrt(self.target_horizon) * daily_std

        self.diagnostics = {
            "model": "RollingStdVolModel",
            "lookback": self.lookback,
            "target_horizon": self.target_horizon,
            "n_assets": len(self.asset_list_),
            "n_samples_used": len(window),
            "output_type": "forward_realized_vol_std",
        }
        return self

    def predict_vol(self, returns: pd.DataFrame) -> np.ndarray:
        window = returns[self.asset_list_].tail(self.lookback)
        daily_std = window.std(axis=0, ddof=1).values.astype(float)
        sigma_hat = np.sqrt(self.target_horizon) * daily_std
        sigma_hat = np.clip(sigma_hat, 1e-8, None)
        return np.nan_to_num(sigma_hat, nan=1e-8, posinf=1e6, neginf=1e-8)
