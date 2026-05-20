import numpy as np
import pandas as pd
from src.models.corr.base import BaseCorrModel


def _distance_corr(x, y):
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float).reshape(-1, 1)

    mask = np.isfinite(x.reshape(-1)) & np.isfinite(y.reshape(-1))
    x = x[mask]
    y = y[mask]

    if len(x) <= 1:
        return 0.0

    a = np.abs(x - x.T)
    b = np.abs(y - y.T)

    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2 = float((A * B).mean())
    dvarx = float((A * A).mean())
    dvary = float((B * B).mean())

    if dvarx <= 1e-12 or dvary <= 1e-12:
        return 0.0

    val = np.sqrt(max(dcov2, 0.0)) / np.sqrt(np.sqrt(dvarx * dvary))
    if not np.isfinite(val):
        return 0.0
    return float(np.clip(val, 0.0, 1.0))


class DistanceCorrModel(BaseCorrModel):
    def fit(self, train_returns: pd.DataFrame, kg_data=None, valid_returns: pd.DataFrame = None):
        return self

    def predict_corr(self, hist_returns: pd.DataFrame, kg_data=None) -> np.ndarray:
        arr = np.asarray(hist_returns.values, dtype=float)
        n_assets = arr.shape[1]
        mat = np.eye(n_assets, dtype=float)

        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                v = _distance_corr(arr[:, i], arr[:, j])
                mat[i, j] = v
                mat[j, i] = v

        mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
        mat = (mat + mat.T) / 2.0
        np.fill_diagonal(mat, 1.0)
        return mat
