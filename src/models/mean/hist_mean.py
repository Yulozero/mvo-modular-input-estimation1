import numpy as np
import pandas as pd
from .base import BaseMeanModel


class HistoricalMeanModel(BaseMeanModel):
    def __init__(self, lookback: int = 60):
        super().__init__()
        self.lookback = lookback

    def fit(
        self,
        train_returns: pd.DataFrame,
        train_targets: pd.DataFrame,
        valid_returns: pd.DataFrame = None,
        valid_targets: pd.DataFrame = None,
    ):
        window = train_targets.tail(self.lookback)
        self.asset_list_ = list(window.columns)
        self.mu_hat_ = window.mean(axis=0).values.astype(float)

        self.diagnostics = {
            "model": "HistoricalMeanModel",
            "lookback": self.lookback,
            "n_assets": len(self.asset_list_),
            "n_target_samples_used": len(window),
            "target_definition": "forward_cumulative_return",
        }
        return self

    def predict(self, returns: pd.DataFrame) -> np.ndarray:
        return np.nan_to_num(self.mu_hat_, nan=0.0, posinf=0.0, neginf=0.0)
