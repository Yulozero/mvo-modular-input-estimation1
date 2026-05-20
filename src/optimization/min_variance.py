import numpy as np
from src.optimization.common import project_long_only_full_investment_cap, validate_weight_cap

try:
    import cvxpy as cp
except ImportError:
    cp = None


class MinimumVarianceOptimizer:
    def __init__(self, long_only=True, l2_reg=0.0, turnover_penalty=0.0, weight_cap=None):
        self.long_only = long_only
        self.l2_reg = l2_reg
        self.turnover_penalty = float(turnover_penalty)
        self.weight_cap = weight_cap
        self.solve_count = 0
        self.solver_success_count = 0
        self.fallback_count = 0
        self.last_diagnostics = {}

    def solve(self, mu_hat: np.ndarray, Sigma_hat: np.ndarray, w_prev: np.ndarray = None) -> np.ndarray:
        Sigma_hat = np.asarray(Sigma_hat, dtype=float)
        n = Sigma_hat.shape[0]
        self.solve_count += 1

        if n == 0:
            return np.array([], dtype=float)

        weight_cap = validate_weight_cap(self.weight_cap, n) if self.long_only else None
        Sigma_hat = (Sigma_hat + Sigma_hat.T) / 2.0
        Sigma_hat = Sigma_hat + np.eye(n) * 1e-8

        if cp is None:
            diag = np.clip(np.diag(Sigma_hat), 1e-8, None)
            inv_diag = 1.0 / diag
            inv_diag = np.maximum(inv_diag, 0.0)
            sol = project_long_only_full_investment_cap(inv_diag, weight_cap=weight_cap)
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

        objective = cp.Minimize(
            cp.quad_form(w, Sigma_hat) + self.l2_reg * cp.sum_squares(w) + penalty
        )

        constraints = [cp.sum(w) == 1]
        if self.long_only:
            constraints.append(w >= 0)
            if weight_cap is not None:
                constraints.append(w <= weight_cap)

        prob = cp.Problem(objective, constraints)

        solved = False
        for solver in [cp.SCS, cp.OSQP, cp.ECOS]:
            try:
                prob.solve(solver=solver, verbose=False)
                if w.value is not None:
                    solved = True
                    break
            except Exception:
                continue

        if (not solved) or (w.value is None):
            self.fallback_count += 1
            self.last_diagnostics = {
                "solver_name": None,
                "solver_success": False,
                "used_fallback": True,
                "weight_cap": weight_cap,
            }
            return project_long_only_full_investment_cap(1.0 / np.clip(np.diag(Sigma_hat), 1e-8, None), weight_cap=weight_cap)

        sol = np.asarray(w.value, dtype=float).reshape(-1)
        sol = np.nan_to_num(sol, nan=0.0, posinf=0.0, neginf=0.0)

        if self.long_only:
            sol = project_long_only_full_investment_cap(sol, weight_cap=weight_cap)
        else:
            s = sol.sum()
            if np.abs(s) <= 1e-12:
                return np.ones(n) / n
            sol = sol / s

        self.solver_success_count += 1
        self.last_diagnostics = {
            "solver_name": str(solver),
            "solver_success": True,
            "used_fallback": False,
            "weight_cap": weight_cap,
        }
        return sol

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
