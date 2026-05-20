import numpy as np
import pandas as pd

from src.models.corr.base import BaseCorrModel
from src.models.corr.distance_corr import _distance_corr


def _safe_corr_sign(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) <= 1:
        return 0.0

    rho = np.corrcoef(x, y)[0, 1]
    if not np.isfinite(rho):
        return 0.0
    return float(np.sign(rho))


class SignedDistanceCorrModel(BaseCorrModel):
    def fit(self, train_returns: pd.DataFrame, kg_data=None, valid_returns: pd.DataFrame = None):
        return self

    def predict_corr(self, hist_returns: pd.DataFrame, kg_data=None) -> np.ndarray:
        arr = np.asarray(hist_returns.values, dtype=float)
        n_assets = arr.shape[1]
        mat = np.eye(n_assets, dtype=float)

        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                strength = _distance_corr(arr[:, i], arr[:, j])
                sign = _safe_corr_sign(arr[:, i], arr[:, j])
                value = float(sign * strength)
                mat[i, j] = value
                mat[j, i] = value

        mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
        mat = (mat + mat.T) / 2.0
        np.fill_diagonal(mat, 1.0)
        return mat

