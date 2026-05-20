import numpy as np


def annualized_return(period_returns, periods_per_year=12):
    r = np.asarray(period_returns, dtype=float)
    if len(r) == 0:
        return 0.0
    cum = np.prod(1.0 + r)
    if cum <= 0:
        return -1.0
    return float(cum ** (periods_per_year / len(r)) - 1.0)


def annualized_vol(period_returns, periods_per_year=12):
    r = np.asarray(period_returns, dtype=float)
    if len(r) <= 1:
        return 0.0
    return float(np.std(r, ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(period_returns, rf=0.0, periods_per_year=12):
    ar = annualized_return(period_returns, periods_per_year=periods_per_year)
    av = annualized_vol(period_returns, periods_per_year=periods_per_year)
    if av <= 1e-12:
        return 0.0
    return float((ar - rf) / av)


def downside_vol(period_returns, mar=0.0, periods_per_year=12):
    r = np.asarray(period_returns, dtype=float)
    downside = r[r < mar] - mar
    if len(downside) <= 1:
        return 0.0
    return float(np.std(downside, ddof=1) * np.sqrt(periods_per_year))


def sortino_ratio(period_returns, rf=0.0, mar=0.0, periods_per_year=12):
    ar = annualized_return(period_returns, periods_per_year=periods_per_year)
    dv = downside_vol(period_returns, mar=mar, periods_per_year=periods_per_year)
    if dv <= 1e-12:
        return 0.0
    return float((ar - rf) / dv)


def max_drawdown(period_returns):
    r = np.asarray(period_returns, dtype=float)
    if len(r) == 0:
        return 0.0
    wealth = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(wealth)
    dd = wealth / peak - 1.0
    return float(dd.min())


def calmar_ratio(period_returns, periods_per_year=12):
    ar = annualized_return(period_returns, periods_per_year=periods_per_year)
    mdd = abs(max_drawdown(period_returns))
    if mdd <= 1e-12:
        return 0.0
    return float(ar / mdd)


def average_turnover(turnovers):
    t = np.asarray(turnovers, dtype=float)
    return float(np.mean(t)) if len(t) else 0.0
