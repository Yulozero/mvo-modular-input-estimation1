import numpy as np


class EqualWeightOptimizer:
    def solve(self, mu_hat: np.ndarray, Sigma_hat: np.ndarray, w_prev: np.ndarray = None) -> np.ndarray:
        n = len(mu_hat)
        if n == 0:
            return np.array([], dtype=float)
        return np.ones(n, dtype=float) / n
