from dataclasses import dataclass
import numpy as np
from src.optimization.common import project_long_only_full_investment_cap, validate_weight_cap

try:
    import cvxpy as cp
except ImportError:
    cp = None


@dataclass
class RiskParityOptimizer:
    long_only: bool = True
    max_iter: int = 2000
    tol: float = 1e-8
    l2_reg: float = 1e-6
    turnover_penalty: float = 0.0
    weight_cap: float = None

    def __post_init__(self):
        self.solve_count = 0
        self.solver_success_count = 0
        self.fallback_count = 0
        self.last_diagnostics = {}

    def solve(self, mu_hat, Sigma_hat, w_prev=None) -> np.ndarray:
        Sigma_hat = np.asarray(Sigma_hat, dtype=float)
        n = Sigma_hat.shape[0]
        self.solve_count += 1
        weight_cap = validate_weight_cap(self.weight_cap, n) if self.long_only else None

        if cp is None:
            diag = np.clip(np.diag(Sigma_hat), 1e-8, None)
            raw = 1.0 / np.sqrt(diag)
            sol = project_long_only_full_investment_cap(raw, weight_cap=weight_cap)
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

        objective = cp.Minimize(cp.quad_form(w, Sigma_hat) + self.l2_reg * cp.sum_squares(w) + penalty)
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
                    w0 = np.asarray(w.value).reshape(-1)
                    w0 = np.clip(w0, 0, None)
                    if w0.sum() > 0:
                        w0 = w0 / w0.sum()
                        sol = self._project_to_risk_parity(Sigma_hat, w0, weight_cap=weight_cap)
                        self.solver_success_count += 1
                        self.last_diagnostics = {
                            "solver_name": str(solver),
                            "solver_success": True,
                            "used_fallback": False,
                            "weight_cap": weight_cap,
                        }
                        return sol
            except Exception:
                continue

        self.fallback_count += 1
        self.last_diagnostics = {
            "solver_name": None,
            "solver_success": False,
            "used_fallback": True,
            "weight_cap": weight_cap,
        }
        diag = np.clip(np.diag(Sigma_hat), 1e-8, None)
        return project_long_only_full_investment_cap(1.0 / np.sqrt(diag), weight_cap=weight_cap)

    def _project_to_risk_parity(self, Sigma_hat: np.ndarray, w0: np.ndarray, weight_cap=None) -> np.ndarray:
        w = project_long_only_full_investment_cap(w0, weight_cap=weight_cap)
        n = len(w)

        for _ in range(self.max_iter):
            port_var = float(w.T @ Sigma_hat @ w)
            if port_var <= 0:
                break

            mrc = Sigma_hat @ w
            rc = w * mrc
            target = port_var / n

            grad = rc - target
            step = 0.05
            w_new = w - step * grad

            w_new = project_long_only_full_investment_cap(w_new, weight_cap=weight_cap)

            if np.linalg.norm(w_new - w, ord=1) < self.tol:
                return w_new
            w = w_new

        return w

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
