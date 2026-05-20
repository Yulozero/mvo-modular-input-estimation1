import numpy as np
import pandas as pd


def build_backward_realized_vol_from_returns(
    returns: pd.DataFrame,
    window: int = 20
) -> pd.DataFrame:
    """
    backward-looking rolling volatility, mainly used as a feature/reference.
    """
    return returns.rolling(window).std()


def build_forward_realized_vol_from_returns(
    returns: pd.DataFrame,
    horizon: int = 20
) -> pd.DataFrame:
    """
    At date t, label is realized vol over future horizon:
    sqrt(sum_{s=1..H} r_{t+s}^2)
    """
    values = returns.values.astype(float)
    out = np.full(values.shape, np.nan, dtype=float)

    for t in range(len(returns) - horizon):
        future = values[t + 1:t + 1 + horizon, :]
        out[t, :] = np.sqrt(np.sum(future ** 2, axis=0))

    return pd.DataFrame(out, index=returns.index, columns=returns.columns)


def build_forward_cum_returns_from_returns(
    returns: pd.DataFrame,
    horizon: int = 20
) -> pd.DataFrame:
    """
    At date t, label is cumulative compounded return over future horizon:
    prod(1 + r_{t+s}) - 1, s=1..H
    """
    values = returns.values.astype(float)
    out = np.full(values.shape, np.nan, dtype=float)

    for t in range(len(returns) - horizon):
        future = values[t + 1:t + 1 + horizon, :]
        out[t, :] = np.prod(1.0 + future, axis=0) - 1.0

    return pd.DataFrame(out, index=returns.index, columns=returns.columns)
