from dataclasses import dataclass
import numpy as np


@dataclass
class InverseVolatilityOptimizer:
    eps: float = 1e-12

    def solve(self, mu_hat, Sigma_hat, w_prev=None) -> np.ndarray:
        Sigma_hat = np.asarray(Sigma_hat, dtype=float)
        diag = np.diag(Sigma_hat)
        vol = np.sqrt(np.clip(diag, self.eps, None))
        inv_vol = 1.0 / vol
        weights = inv_vol / inv_vol.sum()
        return weights
