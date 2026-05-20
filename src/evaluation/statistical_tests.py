import numpy as np

try:
    from scipy import stats
except ImportError:
    stats = None


def paired_t_test(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = x - y
    if len(d) <= 1:
        return {"t_stat": 0.0, "p_value": 1.0}
    if stats is None:
        return {"t_stat": float(np.mean(d) / (np.std(d, ddof=1) / np.sqrt(len(d) + 1e-12))), "p_value": None}
    t_stat, p_value = stats.ttest_rel(x, y, nan_policy="omit")
    return {"t_stat": float(t_stat), "p_value": float(p_value)}


def wilcoxon_signed_rank_test(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if stats is None or len(x) <= 1:
        return {"stat": None, "p_value": None}
    stat, p_value = stats.wilcoxon(x, y)
    return {"stat": float(stat), "p_value": float(p_value)}
