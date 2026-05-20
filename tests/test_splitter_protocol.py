import unittest

import numpy as np
import pandas as pd

from src.backtest.splitter import RollingWindowSplitter
from src.core.types import MarketData


class TestSplitterProtocol(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2020-01-01", periods=220, freq="B")
        cols = ["A", "B", "C"]
        values = np.linspace(-0.02, 0.02, num=len(dates) * len(cols)).reshape(len(dates), len(cols))
        self.returns = pd.DataFrame(values, index=dates, columns=cols)
        self.market_data = MarketData(returns=self.returns, realized_vol=self.returns.abs())

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            RollingWindowSplitter(train_window=120, valid_window=30, test_horizon=20, rebalance_freq=20, history_window=160)
        with self.assertRaises(ValueError):
            RollingWindowSplitter(train_window=120, valid_window=40, test_horizon=40, rebalance_freq=20, history_window=160)

    def test_boundaries_and_alignment(self):
        splitter = RollingWindowSplitter(
            train_window=120,
            valid_window=40,
            test_horizon=20,
            rebalance_freq=20,
            history_window=160,
        )
        windows = list(splitter.generate(self.market_data))
        self.assertGreater(len(windows), 0)

        first = windows[0]
        self.assertEqual(first.train_start_idx, 0)
        self.assertEqual(first.train_end_idx, 120)
        self.assertEqual(first.valid_start_idx, 120)
        self.assertEqual(first.valid_end_idx, 160)
        self.assertEqual(first.test_start_idx, 160)
        self.assertEqual(first.test_end_idx, 180)
        self.assertEqual(first.rebalance_date, self.returns.index[159])
        self.assertEqual(first.predict_date, self.returns.index[160])

    def test_tail_purge_is_correct(self):
        splitter = RollingWindowSplitter(
            train_window=120,
            valid_window=40,
            test_horizon=20,
            rebalance_freq=20,
            history_window=160,
        )
        first = next(splitter.generate(self.market_data))

        self.assertEqual(first.n_train_label_valid, 100)
        self.assertEqual(first.n_valid_label_valid, 20)
        self.assertTrue(first.train_forward_returns.iloc[-20:].isna().all().all())
        self.assertTrue(first.valid_forward_returns.iloc[-20:].isna().all().all())
        self.assertFalse(first.train_forward_returns.iloc[99].isna().all())
        self.assertFalse(first.valid_forward_returns.iloc[19].isna().all())


if __name__ == "__main__":
    unittest.main()
