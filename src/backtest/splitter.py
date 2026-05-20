from typing import Iterator
import pandas as pd
from src.core.types import WindowSlice
from src.protocol.window_labels import build_forward_labels_in_block, purge_block_tail


class RollingWindowSplitter:
    def __init__(
        self,
        train_window: int,
        valid_window: int,
        test_horizon: int,
        rebalance_freq: int,
        history_window: int = None,
        mode: str = "rolling"
    ):
        self.train_window = int(train_window)
        self.valid_window = int(valid_window)
        self.test_horizon = int(test_horizon)
        self.rebalance_freq = int(rebalance_freq)
        self.history_window = int(history_window) if history_window is not None else self.train_window + self.valid_window
        self.mode = mode

        if self.train_window <= 0:
            raise ValueError("train_window must be positive")
        if self.valid_window <= 0:
            raise ValueError("valid_window must be positive")
        if self.test_horizon <= 0:
            raise ValueError("test_horizon must be positive")
        if self.rebalance_freq <= 0:
            raise ValueError("rebalance_freq must be positive")
        if self.history_window <= 0:
            raise ValueError("history_window must be positive")
        if self.history_window < self.valid_window:
            raise ValueError("history_window must be >= valid_window")
        if self.train_window + self.valid_window != self.history_window:
            raise ValueError(
                f"Invalid split: train_window + valid_window must equal history_window. "
                f"Got {self.train_window} + {self.valid_window} != {self.history_window}."
            )

        if self.valid_window <= self.test_horizon:
            raise ValueError(
                f"Invalid split: valid_window={self.valid_window} must be > "
                f"test_horizon={self.test_horizon} to avoid validation leakage."
            )

    @staticmethod
    def _count_valid_rows(df: pd.DataFrame) -> int:
        if df is None or len(df) == 0:
            return 0
        valid_mask = ~df.isna().all(axis=1)
        return int(valid_mask.sum())

    def get_n_steps(self, market_data) -> int:
        returns = market_data.returns
        start_idx = self.history_window
        end_idx = len(returns) - self.test_horizon

        if end_idx <= start_idx:
            return 0

        return len(range(start_idx, end_idx, self.rebalance_freq))

    def generate(self, market_data) -> Iterator[WindowSlice]:
        returns = market_data.returns
        realized_vol = market_data.realized_vol

        start_idx = self.history_window
        end_idx = len(returns) - self.test_horizon

        for t in range(start_idx, end_idx, self.rebalance_freq):
            if self.mode == "rolling":
                train_start = t - self.history_window
            elif self.mode == "expanding":
                train_start = 0
            else:
                raise ValueError(f"Unsupported mode: {self.mode}")

            train_end = t - self.valid_window
            valid_start = train_end
            valid_end = t
            test_start = t
            test_end = t + self.test_horizon

            train_returns = returns.iloc[train_start:train_end]
            valid_returns = returns.iloc[valid_start:valid_end]
            test_returns = returns.iloc[test_start:test_end]

            history_returns = returns.iloc[train_start:valid_end]

            train_rv = realized_vol.iloc[train_start:train_end] if realized_vol is not None else None
            valid_rv = realized_vol.iloc[valid_start:valid_end] if realized_vol is not None else None
            test_rv = realized_vol.iloc[test_start:test_end] if realized_vol is not None else None
            history_rv = realized_vol.iloc[train_start:valid_end] if realized_vol is not None else None

            # Build labels *inside each block* so no target crosses block boundary.
            train_fr_raw, train_fv_raw = build_forward_labels_in_block(train_returns, self.test_horizon)
            valid_fr_raw, valid_fv_raw = build_forward_labels_in_block(valid_returns, self.test_horizon)

            # Keep explicit tail purge for readability and defensive safety.
            train_fr = purge_block_tail(train_fr_raw, self.test_horizon)
            valid_fr = purge_block_tail(valid_fr_raw, self.test_horizon)
            train_fv = purge_block_tail(train_fv_raw, self.test_horizon)
            valid_fv = purge_block_tail(valid_fv_raw, self.test_horizon)

            yield WindowSlice(
                train_returns=train_returns,
                valid_returns=valid_returns,
                test_returns=test_returns,

                history_returns=history_returns,
                history_realized_vol=history_rv,

                train_forward_returns=train_fr,
                valid_forward_returns=valid_fr,
                test_forward_returns=None,

                train_forward_vol=train_fv,
                valid_forward_vol=valid_fv,
                test_forward_vol=None,

                train_realized_vol=train_rv,
                valid_realized_vol=valid_rv,
                test_realized_vol=test_rv,

                rebalance_date=returns.index[t - 1],
                predict_date=returns.index[t],
                asset_list=list(returns.columns),

                train_start_idx=train_start,
                train_end_idx=train_end,
                valid_start_idx=valid_start,
                valid_end_idx=valid_end,
                test_start_idx=test_start,
                test_end_idx=test_end,

                n_train_label_valid=self._count_valid_rows(train_fr),
                n_valid_label_valid=self._count_valid_rows(valid_fr),
            )
