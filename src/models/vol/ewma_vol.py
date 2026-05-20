import numpy as np
import pandas as pd
from .base import BaseVolModel


class EWMAVolModel(BaseVolModel):
    def __init__(self, lookback: int = 60, decay: float = 0.94, target_horizon: int = 20):
        super().__init__()
        self.lookback = lookback
        self.decay = decay
        self.target_horizon = target_horizon

    def _get_weights(self, n: int) -> np.ndarray:
        raw = np.array([self.decay ** (n - 1 - i) for i in range(n)], dtype=float)
        raw /= raw.sum()
        return raw

    def _ewma_daily_std(self, window: pd.DataFrame) -> np.ndarray:
        X = window.values.astype(float)
        w = self._get_weights(len(window))
        ewma_var = (w[:, None] * (X ** 2)).sum(axis=0)
        ewma_std = np.sqrt(np.clip(ewma_var, 1e-12, None))
        return ewma_std

    def fit(
        self,
        train_returns: pd.DataFrame,
        train_targets: pd.DataFrame,
        valid_returns: pd.DataFrame = None,
        valid_targets: pd.DataFrame = None,
    ):
        window = train_returns.tail(self.lookback)
        self.asset_list_ = list(window.columns)
        daily_std = self._ewma_daily_std(window)
        self.sigma_hat_ = np.sqrt(self.target_horizon) * daily_std

        self.diagnostics = {
            "model": "EWMAVolModel",
            "lookback": self.lookback,
            "decay": self.decay,
            "target_horizon": self.target_horizon,
            "n_assets": len(self.asset_list_),
            "n_samples_used": len(window),
            "output_type": "forward_realized_vol_std",
        }
        return self

    def predict_vol(self, returns: pd.DataFrame) -> np.ndarray:
        window = returns[self.asset_list_].tail(self.lookback)
        daily_std = self._ewma_daily_std(window)
        sigma_hat = np.sqrt(self.target_horizon) * daily_std
        sigma_hat = np.clip(sigma_hat, 1e-8, None)
        return np.nan_to_num(sigma_hat, nan=1e-8, posinf=1e6, neginf=1e-8)
