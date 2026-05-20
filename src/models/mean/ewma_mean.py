import numpy as np
import pandas as pd
from .base import BaseMeanModel


class EWMAMeanModel(BaseMeanModel):
    def __init__(self, lookback: int = 60, decay: float = 0.94):
        super().__init__()
        self.lookback = lookback
        self.decay = decay

    def _get_weights(self, n: int) -> np.ndarray:
        raw = np.array([self.decay ** (n - 1 - i) for i in range(n)], dtype=float)
        raw /= raw.sum()
        return raw

    def fit(
        self,
        train_returns: pd.DataFrame,
        train_targets: pd.DataFrame,
        valid_returns: pd.DataFrame = None,
        valid_targets: pd.DataFrame = None,
    ):
        window = train_targets.tail(self.lookback)
        X = window.values.astype(float)
        w = self._get_weights(len(window))
        self.mu_hat_ = (X * w[:, None]).sum(axis=0)
        self.asset_list_ = list(window.columns)

        self.diagnostics = {
            "model": "EWMAMeanModel",
            "lookback": self.lookback,
            "decay": self.decay,
            "n_assets": len(self.asset_list_),
            "n_target_samples_used": len(window),
            "target_definition": "forward_cumulative_return",
        }
        return self

    def predict(self, returns: pd.DataFrame) -> np.ndarray:
        return np.nan_to_num(self.mu_hat_, nan=0.0, posinf=0.0, neginf=0.0)
