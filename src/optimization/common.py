import numpy as np


def validate_weight_cap(weight_cap, n_assets: int):
    if weight_cap is None:
        return None

    weight_cap = float(weight_cap)
    if weight_cap <= 0:
        raise ValueError("weight_cap must be positive when provided.")
    if weight_cap > 1.0:
        raise ValueError("weight_cap cannot exceed 1.0.")
    if n_assets > 0 and weight_cap * n_assets < 1.0 - 1e-12:
        raise ValueError(
            f"Infeasible weight_cap={weight_cap} for n_assets={n_assets}. "
            "Need weight_cap * n_assets >= 1."
        )
    return weight_cap


def project_long_only_full_investment_cap(raw_weights, weight_cap=None):
    raw_weights = np.asarray(raw_weights, dtype=float).reshape(-1)
    n_assets = len(raw_weights)
    if n_assets == 0:
        return np.array([], dtype=float)

    weight_cap = validate_weight_cap(weight_cap, n_assets)

    weights = np.nan_to_num(raw_weights, nan=0.0, posinf=0.0, neginf=0.0)
    weights = np.maximum(weights, 0.0)

    if np.sum(weights) <= 1e-12:
        weights = np.ones(n_assets, dtype=float)

    remaining = 1.0
    active = np.ones(n_assets, dtype=bool)
    solution = np.zeros(n_assets, dtype=float)
    base = weights.copy()

    while True:
        if not np.any(active):
            break

        active_sum = np.sum(base[active])
        if active_sum <= 1e-12:
            base[active] = 1.0
            active_sum = np.sum(base[active])

        candidate = remaining * base[active] / active_sum
        if weight_cap is None:
            solution[active] = candidate
            remaining = 0.0
            break

        over_mask = candidate > weight_cap + 1e-12
        if not np.any(over_mask):
            solution[active] = candidate
            remaining = 0.0
            break

        active_idx = np.where(active)[0]
        capped_idx = active_idx[over_mask]
        solution[capped_idx] = weight_cap
        remaining = 1.0 - np.sum(solution)
        active[capped_idx] = False
        base[capped_idx] = 0.0

    if remaining > 1e-10:
        free_idx = np.where(solution < (weight_cap if weight_cap is not None else 1.0) - 1e-12)[0]
        if len(free_idx) == 0:
            free_idx = np.arange(n_assets)
        solution[free_idx] += remaining / len(free_idx)

    solution = np.maximum(solution, 0.0)
    if weight_cap is not None:
        solution = np.minimum(solution, weight_cap)
        slack = 1.0 - np.sum(solution)
        if slack > 1e-8:
            for idx in np.argsort(solution):
                room = weight_cap - solution[idx]
                if room <= 0:
                    continue
                add = min(room, slack)
                solution[idx] += add
                slack -= add
                if slack <= 1e-10:
                    break

    total = np.sum(solution)
    if total <= 1e-12:
        solution = np.ones(n_assets, dtype=float) / n_assets
        if weight_cap is not None:
            solution = project_long_only_full_investment_cap(solution, weight_cap=weight_cap)
        return solution

    solution = solution / total
    return solution

