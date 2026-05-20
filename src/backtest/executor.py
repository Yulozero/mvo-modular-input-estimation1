import numpy as np
from src.core.types import PortfolioDecision


class PortfolioExecutor:
    def __init__(self, transaction_cost: float = 0.0, charge_initial_cost: bool = True):
        self.transaction_cost = float(transaction_cost)
        self.charge_initial_cost = bool(charge_initial_cost)

    def execute(
        self,
        date,
        weights,
        next_returns,
        prev_weights=None,
        expected_return=None,
        expected_risk=None,
    ):
        weights = np.asarray(weights, dtype=float).reshape(-1)
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

        if prev_weights is None:
            initial_cost_charged = bool(self.charge_initial_cost)
            turnover = float(np.abs(weights).sum()) if self.charge_initial_cost else 0.0
        else:
            initial_cost_charged = False
            prev_weights = np.asarray(prev_weights, dtype=float).reshape(-1)
            prev_weights = np.nan_to_num(prev_weights, nan=0.0, posinf=0.0, neginf=0.0)

            if len(prev_weights) != len(weights):
                raise ValueError(
                    f"weights length mismatch: current={len(weights)}, prev={len(prev_weights)}"
                )
            turnover = float(np.abs(weights - prev_weights).sum())

        tcost = float(self.transaction_cost * turnover)

        if next_returns is None or len(next_returns) == 0:
            realized_return = 0.0
            daily_portfolio_returns = np.array([], dtype=float)
            gross_cumulative_path = np.array([], dtype=float)
            net_cumulative_path = np.array([], dtype=float)
        else:
            asset_ret_mat = np.asarray(next_returns.values, dtype=float)

            if asset_ret_mat.shape[1] != len(weights):
                raise ValueError(
                    f"next_returns asset dim mismatch: next_returns has {asset_ret_mat.shape[1]} cols, "
                    f"weights has len={len(weights)}"
                )

            # Missing asset returns during the holding window are treated conservatively
            # as capital moving to cash with zero return from that point onward.
            # This avoids silently imputing tradability or future prices after delisting/suspension.
            asset_ret_mat = np.where(np.isfinite(asset_ret_mat), asset_ret_mat, np.nan)
            asset_wealth = weights.astype(float).copy()
            total_wealth = 1.0
            daily_portfolio_returns_list = []
            missing_mask = np.isnan(asset_ret_mat)
            for t in range(asset_ret_mat.shape[0]):
                prev_total = total_wealth
                day_ret = asset_ret_mat[t].copy()
                day_ret = np.where(np.isnan(day_ret), 0.0, day_ret)
                asset_wealth = asset_wealth * (1.0 + day_ret)
                total_wealth = float(np.sum(asset_wealth))
                if prev_total <= 0:
                    port_ret = 0.0
                else:
                    port_ret = total_wealth / prev_total - 1.0
                daily_portfolio_returns_list.append(float(port_ret))

            daily_portfolio_returns = np.asarray(daily_portfolio_returns_list, dtype=float)
            daily_portfolio_returns = np.nan_to_num(daily_portfolio_returns, nan=0.0, posinf=0.0, neginf=0.0)
            realized_return = float(total_wealth - 1.0)
            gross_cumulative_path = np.cumprod(1.0 + daily_portfolio_returns) - 1.0
            daily_net_returns = daily_portfolio_returns.copy()
            if len(daily_net_returns) > 0:
                daily_net_returns[0] = daily_net_returns[0] - tcost
            net_cumulative_path = np.cumprod(1.0 + daily_net_returns) - 1.0

        realized_net_return = float(realized_return - tcost)
        cost_drag = float(realized_return - realized_net_return)

        if expected_return is None:
            expected_return = float("nan")
        if expected_risk is None:
            expected_risk = float("nan")

        concentration_hhi = float(np.sum(np.square(weights)))
        effective_n = float(1.0 / concentration_hhi) if concentration_hhi > 1e-12 else 0.0

        return PortfolioDecision(
            date=date,
            weights=weights,
            expected_return=float(expected_return),
            expected_risk=float(expected_risk),
            turnover=float(turnover),
            transaction_cost=float(tcost),
            realized_return=float(realized_return),
            realized_net_return=float(realized_net_return),
            meta={
                "n_test_days": int(len(next_returns)) if next_returns is not None else 0,
                "daily_portfolio_returns": daily_portfolio_returns.tolist(),
                "gross_cumulative_path": gross_cumulative_path.tolist(),
                "net_cumulative_path": net_cumulative_path.tolist(),
                "cost_drag": cost_drag,
                "initial_cost_charged": initial_cost_charged,
                "concentration_hhi": concentration_hhi,
                "effective_n_assets": effective_n,
                "max_weight": float(np.max(weights)) if len(weights) > 0 else 0.0,
                "missing_asset_return_days": int(np.isnan(asset_ret_mat).sum()) if next_returns is not None and len(next_returns) > 0 else 0,
                "assets_with_missing_during_holding": int(np.isnan(asset_ret_mat).any(axis=0).sum()) if next_returns is not None and len(next_returns) > 0 else 0,
            },
        )
