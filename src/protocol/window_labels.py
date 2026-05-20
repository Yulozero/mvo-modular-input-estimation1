from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def build_forward_labels_in_block(
    block_returns: Optional[pd.DataFrame],
    horizon: int,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Build forward labels strictly inside one block.

    For each timestamp t in this block:
    - forward return: prod(1 + r_{t+1:t+h}) - 1
    - forward vol: sqrt(sum(r_{t+1:t+h}^2))

    If t+h exceeds block boundary, label is NaN (purged).
    This guarantees no label crosses train/valid/test boundaries.
    """
    if block_returns is None:
        return None, None

    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError("horizon must be positive when building block labels.")

    values = np.asarray(block_returns.values, dtype=float)
    n_time, n_assets = values.shape

    fwd_ret = np.full((n_time, n_assets), np.nan, dtype=float)
    fwd_vol = np.full((n_time, n_assets), np.nan, dtype=float)

    for t in range(0, max(n_time - horizon, 0)):
        future = values[t + 1:t + 1 + horizon, :]
        fwd_ret[t, :] = np.prod(1.0 + future, axis=0) - 1.0
        fwd_vol[t, :] = np.sqrt(np.sum(future ** 2, axis=0))

    fwd_ret_df = pd.DataFrame(fwd_ret, index=block_returns.index, columns=block_returns.columns)
    fwd_vol_df = pd.DataFrame(fwd_vol, index=block_returns.index, columns=block_returns.columns)
    return fwd_ret_df, fwd_vol_df


def purge_block_tail(df: Optional[pd.DataFrame], horizon: int) -> Optional[pd.DataFrame]:
    """
    Explicitly mark the last `horizon` rows as NaN for safety and clarity.
    """
    if df is None:
        return None
    out = df.copy()
    if len(out) > 0:
        n = min(int(horizon), len(out))
        out.iloc[-n:] = float("nan")
    return out

