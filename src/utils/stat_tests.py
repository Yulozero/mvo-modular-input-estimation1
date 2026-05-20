from typing import Dict

import numpy as np
import pandas as pd

try:
    from scipy.stats import ttest_rel, wilcoxon, norm
except ImportError:
    ttest_rel = None
    wilcoxon = None
    norm = None


def _to_numpy(x):
    if isinstance(x, pd.Series):
        return x.values.astype(float)
    return np.asarray(x, dtype=float)


def paired_t_test(x, y):
    x = _to_numpy(x)
    y = _to_numpy(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return {"t_stat": None, "p_value": None}

    if ttest_rel is None:
        d = x - y
        std = np.std(d, ddof=1)
        if std <= 1e-12:
            return {"t_stat": None, "p_value": None}
        stat = np.mean(d) / (std / np.sqrt(len(d)))
        return {"t_stat": float(stat), "p_value": None}

    stat, p = ttest_rel(x, y)
    return {"t_stat": float(stat), "p_value": float(p)}


def wilcoxon_signed_rank_test(x, y):
    x = _to_numpy(x)
    y = _to_numpy(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2 or wilcoxon is None:
        return {"wilcoxon_stat": None, "p_value": None}

    try:
        stat, p = wilcoxon(x, y)
        return {"wilcoxon_stat": float(stat), "p_value": float(p)}
    except Exception:
        return {"wilcoxon_stat": None, "p_value": None}


def bootstrap_mean_ci(x, n_boot: int = 2000, alpha: float = 0.05, random_state: int = 42):
    x = _to_numpy(x)
    x = x[np.isfinite(x)]

    if len(x) < 2:
        return {"mean": None, "ci_low": None, "ci_high": None}

    rng = np.random.default_rng(random_state)
    means = []
    n = len(x)

    for _ in range(n_boot):
        sample = rng.choice(x, size=n, replace=True)
        means.append(np.mean(sample))

    means = np.asarray(means)
    return {
        "mean": float(np.mean(x)),
        "ci_low": float(np.quantile(means, alpha / 2)),
        "ci_high": float(np.quantile(means, 1 - alpha / 2)),
    }


def bootstrap_difference_ci(x, y, n_boot: int = 2000, alpha: float = 0.05, random_state: int = 42):
    x = _to_numpy(x)
    y = _to_numpy(y)
    mask = np.isfinite(x) & np.isfinite(y)
    d = x[mask] - y[mask]

    if len(d) < 2:
        return {"mean_diff": None, "ci_low": None, "ci_high": None}

    rng = np.random.default_rng(random_state)
    diffs = []
    n = len(d)

    for _ in range(n_boot):
        sample = rng.choice(d, size=n, replace=True)
        diffs.append(np.mean(sample))

    diffs = np.asarray(diffs)
    return {
        "mean_diff": float(np.mean(d)),
        "ci_low": float(np.quantile(diffs, alpha / 2)),
        "ci_high": float(np.quantile(diffs, 1 - alpha / 2)),
    }


def diebold_mariano_test(e1, e2, power: int = 2) -> Dict:
    e1 = _to_numpy(e1)
    e2 = _to_numpy(e2)

    mask = np.isfinite(e1) & np.isfinite(e2)
    e1 = e1[mask]
    e2 = e2[mask]

    if len(e1) < 5:
        return {"dm_stat": None, "p_value": None}

    d = np.abs(e1) ** power - np.abs(e2) ** power
    d_bar = np.mean(d)
    var_d = np.var(d, ddof=1)

    if var_d <= 1e-12:
        return {"dm_stat": None, "p_value": None}

    dm_stat = d_bar / np.sqrt(var_d / len(d))

    if norm is None:
        return {"dm_stat": float(dm_stat), "p_value": None}

    p_value = 2 * (1 - norm.cdf(abs(dm_stat)))
    return {"dm_stat": float(dm_stat), "p_value": float(p_value)}


def compare_two_return_series(
    returns_a: pd.Series,
    returns_b: pd.Series,
    name_a: str,
    name_b: str,
    metric_name: str = "realized_net_return",
) -> Dict:
    aligned = pd.concat([pd.Series(returns_a).rename("a"), pd.Series(returns_b).rename("b")], axis=1).dropna()

    if aligned.empty:
        return {
            "compare_a": name_a,
            "compare_b": name_b,
            "metric": metric_name,
            "mean_diff": None,
            "t_stat": None,
            "t_pvalue": None,
            "wilcoxon_stat": None,
            "wilcoxon_pvalue": None,
            "boot_ci_low": None,
            "boot_ci_high": None,
            "better_model": None,
        }

    x = aligned["a"].values
    y = aligned["b"].values

    t_res = paired_t_test(x, y)
    w_res = wilcoxon_signed_rank_test(x, y)
    b_res = bootstrap_difference_ci(x, y)

    mean_diff = float(np.mean(x - y))
    if abs(mean_diff) <= 1e-15:
        better = "Tie"
    else:
        better = name_a if mean_diff > 0 else name_b

    return {
        "compare_a": name_a,
        "compare_b": name_b,
        "metric": metric_name,
        "mean_diff": mean_diff,
        "t_stat": t_res["t_stat"],
        "t_pvalue": t_res["p_value"],
        "wilcoxon_stat": w_res["wilcoxon_stat"],
        "wilcoxon_pvalue": w_res["p_value"],
        "boot_ci_low": b_res["ci_low"],
        "boot_ci_high": b_res["ci_high"],
        "better_model": better,
    }
