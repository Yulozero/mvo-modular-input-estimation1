import numpy as np
from src.optimization.common import project_long_only_full_investment_cap, validate_weight_cap

try:
    import cvxpy as cp
except ImportError:
    cp = None


class MeanVarianceOptimizer:
    def __init__(self, gamma=3.0, long_only=True, l2_reg=0.0, turnover_penalty=0.0, weight_cap=None):
        self.gamma = gamma
        self.long_only = long_only
        self.l2_reg = l2_reg
        self.turnover_penalty = float(turnover_penalty)
        self.weight_cap = weight_cap
        self.solve_count = 0
        self.solver_success_count = 0
        self.fallback_count = 0
        self.last_diagnostics = {}

    def solve(self, mu_hat: np.ndarray, Sigma_hat: np.ndarray, w_prev: np.ndarray = None) -> np.ndarray:
        mu_hat = np.asarray(mu_hat, dtype=float).reshape(-1)
        Sigma_hat = np.asarray(Sigma_hat, dtype=float)
        n = len(mu_hat)
        self.solve_count += 1

        if n == 0:
            return np.array([], dtype=float)

        weight_cap = validate_weight_cap(self.weight_cap, n) if self.long_only else None
        Sigma_hat = (Sigma_hat + Sigma_hat.T) / 2.0
        diag_eps = 1e-8
        Sigma_hat = Sigma_hat + np.eye(n) * diag_eps

        if cp is None:
            diag = np.clip(np.diag(Sigma_hat), 1e-8, None)
            score = mu_hat / diag

            if self.long_only:
                score = np.maximum(score, 0.0)

            if np.abs(score).sum() <= 1e-12:
                score = np.ones(n, dtype=float)

            if self.long_only:
                w = project_long_only_full_investment_cap(score, weight_cap=weight_cap)
            else:
                if np.abs(score.sum()) <= 1e-12:
                    w = np.ones(n, dtype=float) / n
                else:
                    w = score / score.sum()

            self.fallback_count += 1
            self.last_diagnostics = {
                "solver_name": None,
                "solver_success": False,
                "used_fallback": True,
                "weight_cap": weight_cap,
            }
            return np.asarray(w, dtype=float)

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
            return project_long_only_full_investment_cap(np.maximum(mu_hat, 0.0), weight_cap=weight_cap)

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
