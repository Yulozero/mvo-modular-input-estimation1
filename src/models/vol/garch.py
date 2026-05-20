import numpy as np
import pandas as pd
from .base import BaseVolModel

try:
    from arch import arch_model
except ImportError:
    arch_model = None


class GARCHVolModel(BaseVolModel):
    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        target_horizon: int = 20,
        mean: str = "Zero",
        vol: str = "GARCH",
        dist: str = "normal",
        rescale_returns: float = 100.0,
        min_history: int = 60,
    ):
        super().__init__()
        if arch_model is None:
            raise ImportError("arch package is required for GARCHVolModel.")
        self.p = p
        self.q = q
        self.target_horizon = target_horizon
        self.mean = mean
        self.vol = vol
        self.dist = dist
        self.rescale_returns = rescale_returns
        self.min_history = min_history

    def _fit_single_asset(self, series: np.ndarray):
        if len(series) < self.min_history:
            raise ValueError(f"Insufficient data for GARCH fit: got {len(series)} < {self.min_history}")

        x = np.asarray(series, dtype=float)
        x = x[np.isfinite(x)]
        if len(x) < self.min_history:
            raise ValueError("Insufficient finite observations for GARCH fit.")

        am = arch_model(
            x * self.rescale_returns,
            mean=self.mean,
            vol=self.vol,
            p=self.p,
            q=self.q,
            dist=self.dist,
            rescale=False,
        )
        res = am.fit(disp="off")
        fcast = res.forecast(horizon=1, reindex=False)
        one_step_var = float(fcast.variance.values[-1, 0]) / (self.rescale_returns ** 2)
        one_step_std = np.sqrt(max(one_step_var, 1e-12))
        horizon_std = np.sqrt(self.target_horizon) * one_step_std
        return horizon_std, {
            "aic": float(res.aic),
            "bic": float(res.bic),
            "loglikelihood": float(res.loglikelihood),
        }

    def fit(
        self,
        train_returns: pd.DataFrame,
        train_targets: pd.DataFrame,
        valid_returns: pd.DataFrame = None,
        valid_targets: pd.DataFrame = None,
    ):
        self.asset_list_ = list(train_returns.columns)
        sigma_list = []
        per_asset_diag = {}

        for col in self.asset_list_:
            sigma_hat, diag = self._fit_single_asset(train_returns[col].values.astype(float))
            sigma_list.append(float(sigma_hat))
            per_asset_diag[col] = diag

        self.sigma_hat_ = np.asarray(sigma_list, dtype=float)
        self.per_asset_diag_ = per_asset_diag

        self.diagnostics = {
            "model": "GARCHVolModel",
            "p": self.p,
            "q": self.q,
            "target_horizon": self.target_horizon,
            "mean": self.mean,
            "vol": self.vol,
            "dist": self.dist,
            "n_assets": len(self.asset_list_),
            "output_type": "forward_realized_vol_std",
            "per_asset_diag": per_asset_diag,
        }
        return self

    def predict_vol(self, returns: pd.DataFrame) -> np.ndarray:
        sigma_list = []
        for col in self.asset_list_:
            sigma_hat, _ = self._fit_single_asset(returns[col].values.astype(float))
            sigma_list.append(float(sigma_hat))

        sigma_hat = np.asarray(sigma_list, dtype=float)
        sigma_hat = np.clip(sigma_hat, 1e-8, None)
        return np.nan_to_num(sigma_hat, nan=1e-8, posinf=1e6, neginf=1e-8)
