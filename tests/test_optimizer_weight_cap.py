import unittest

import numpy as np

from src.optimization.mean_variance import MeanVarianceOptimizer
from src.optimization.min_variance import MinimumVarianceOptimizer
from src.optimization.risk_parity import RiskParityOptimizer
from src.optimization.shrinkage_mvo import ShrinkageMVOOptimizer


class TestOptimizerWeightCap(unittest.TestCase):
    def setUp(self):
        self.mu = np.array([0.08, 0.05, 0.04, 0.03], dtype=float)
        base = np.array(
            [
                [0.10, 0.02, 0.01, 0.00],
                [0.02, 0.09, 0.01, 0.01],
                [0.01, 0.01, 0.08, 0.02],
                [0.00, 0.01, 0.02, 0.07],
            ],
            dtype=float,
        )
        self.sigma = (base + base.T) / 2.0
        self.cap = 0.40

    def _assert_constraints(self, w):
        self.assertTrue(np.all(np.isfinite(w)))
        self.assertTrue(np.all(w >= -1e-8))
        self.assertAlmostEqual(float(w.sum()), 1.0, places=6)
        self.assertLessEqual(float(w.max()), self.cap + 1e-6)

    def test_mean_variance_weight_cap(self):
        opt = MeanVarianceOptimizer(gamma=3.0, long_only=True, weight_cap=self.cap)
        self._assert_constraints(opt.solve(self.mu, self.sigma))

    def test_minimum_variance_weight_cap(self):
        opt = MinimumVarianceOptimizer(long_only=True, weight_cap=self.cap)
        self._assert_constraints(opt.solve(self.mu, self.sigma))

    def test_shrinkage_mvo_weight_cap(self):
        opt = ShrinkageMVOOptimizer(gamma=3.0, long_only=True, weight_cap=self.cap)
        self._assert_constraints(opt.solve(self.mu, self.sigma))

    def test_risk_parity_weight_cap(self):
        opt = RiskParityOptimizer(long_only=True, weight_cap=self.cap)
        self._assert_constraints(opt.solve(self.mu, self.sigma))


if __name__ == "__main__":
    unittest.main()
