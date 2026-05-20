import numpy as np


def mse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def hit_ratio(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def information_coefficient(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size <= 1:
        return 0.0
    if np.std(y_true) <= 1e-12 or np.std(y_pred) <= 1e-12:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _rankdata(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return ranks


def rank_information_coefficient(y_true, y_pred):
    ry = _rankdata(y_true)
    rp = _rankdata(y_pred)
    if ry.size <= 1:
        return 0.0
    if np.std(ry) <= 1e-12 or np.std(rp) <= 1e-12:
        return 0.0
    return float(np.corrcoef(ry, rp)[0, 1])
