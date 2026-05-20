from dataclasses import dataclass
import numpy as np
from src.optimization.common import project_long_only_full_investment_cap, validate_weight_cap

try:
    import cvxpy as cp
except ImportError:
    cp = None


def linear_shrink_cov(sample_cov: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    sample_cov = np.asarray(sample_cov, dtype=float)
    diag = np.diag(np.diag(sample_cov))
    return (1.0 - alpha) * sample_cov + alpha * diag


@dataclass
class ShrinkageMVOOptimizer:
    gamma: float = 3.0
    long_only: bool = True
    l2_reg: float = 1e-3
    shrink_alpha: float = 0.2
    turnover_penalty: float = 0.0
    weight_cap: float = None

    def __post_init__(self):
        self.solve_count = 0
        self.solver_success_count = 0
        self.fallback_count = 0
        self.last_diagnostics = {}

    def solve(self, mu_hat, Sigma_hat, w_prev=None) -> np.ndarray:
        mu_hat = np.asarray(mu_hat, dtype=float).reshape(-1)
        Sigma_hat = linear_shrink_cov(Sigma_hat, alpha=self.shrink_alpha)
        n = len(mu_hat)
        self.solve_count += 1
        weight_cap = validate_weight_cap(self.weight_cap, n) if self.long_only else None

        if cp is None:
            diag = np.clip(np.diag(Sigma_hat), 1e-8, None)
            score = np.maximum(mu_hat / diag, 0.0)
            sol = project_long_only_full_investment_cap(score, weight_cap=weight_cap)
            self.fallback_count += 1
            self.last_diagnostics = {
                "solver_name": None,
                "solver_success": False,
                "used_fallback": True,
                "weight_cap": weight_cap,
            }
            return sol

        w = cp.Variable(n)
        penalty = 0.0
        if self.turnover_penalty > 0 and w_prev is not None:
            w_prev = np.asarray(w_prev, dtype=float).reshape(-1)
            if len(w_prev) == n:
                penalty = self.turnover_penalty * cp.sum_squares(w - w_prev)
        objective = cp.Maximize(
            mu_hat @ w
            - 0.5 * self.gamma * cp.quad_form(w, Sigma_hat)
            - self.l2_reg * cp.sum_squares(w)
            - penalty
        )
        constraints = [cp.sum(w) == 1]
        if self.long_only:
            constraints.append(w >= 0)
            if weight_cap is not None:
                constraints.append(w <= weight_cap)

        prob = cp.Problem(objective, constraints)

        for solver in [cp.OSQP, cp.ECOS, cp.SCS]:
            try:
                prob.solve(solver=solver, warm_start=True)
                if w.value is not None:
                    ww = np.asarray(w.value).reshape(-1)
                    if self.long_only:
                        ww = project_long_only_full_investment_cap(ww, weight_cap=weight_cap)
                    else:
                        ww = np.asarray(ww, dtype=float)
                        ww = ww / ww.sum()
                    self.solver_success_count += 1
                    self.last_diagnostics = {
                        "solver_name": str(solver),
                        "solver_success": True,
                        "used_fallback": False,
                        "weight_cap": weight_cap,
                    }
                    return ww
            except Exception:
                continue

        self.fallback_count += 1
        self.last_diagnostics = {
            "solver_name": None,
            "solver_success": False,
            "used_fallback": True,
            "weight_cap": weight_cap,
        }
        return project_long_only_full_investment_cap(np.maximum(mu_hat, 0.0), weight_cap=weight_cap)

    def get_diagnostics(self):
        success_rate = self.solver_success_count / self.solve_count if self.solve_count else None
        out = dict(self.last_diagnostics)
        out.update({
            "solve_count": self.solve_count,
            "solver_success_count": self.solver_success_count,
            "fallback_count": self.fallback_count,
            "solver_success_rate": success_rate,
        })
        return out
