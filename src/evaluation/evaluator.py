import numpy as np
from src.evaluation.portfolio_metrics import (
    annualized_return,
    annualized_vol,
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    max_drawdown,
    average_turnover,
)
from src.evaluation.module_evaluator import ModuleEvaluator


class ExperimentEvaluator:
    def __init__(self, periods_per_year: float = 12.0, daily_periods_per_year: float = 252.0):
        self.periods_per_year = periods_per_year
        self.daily_periods_per_year = daily_periods_per_year
        self.module_evaluator = ModuleEvaluator()

    def _collect_holding_period_returns(self, results):
        gross = [r["decision"].realized_return for r in results]
        net = [r["decision"].realized_net_return for r in results]
        turnovers = [r["decision"].turnover for r in results]
        transaction_costs = [r["decision"].transaction_cost for r in results]
        ex_ante_risk = [r["decision"].expected_risk for r in results]

        concentration_hhi = []
        effective_n_assets = []
        cost_drag = []
        avg_max_weight = []

        for r in results:
            meta = r["decision"].meta or {}
            concentration_hhi.append(meta.get("concentration_hhi", np.nan))
            effective_n_assets.append(meta.get("effective_n_assets", np.nan))
            cost_drag.append(meta.get("cost_drag", np.nan))
            avg_max_weight.append(meta.get("max_weight", np.nan))

        return gross, net, turnovers, transaction_costs, ex_ante_risk, concentration_hhi, effective_n_assets, cost_drag, avg_max_weight

    def _collect_daily_returns(self, results):
        daily_gross = []
        daily_net = []

        for r in results:
            d = r["decision"]
            arr = np.asarray(d.meta.get("daily_portfolio_returns", []), dtype=float)
            if len(arr) == 0:
                continue

            daily_gross.extend(arr.tolist())

            net_arr = arr.copy()
            if len(net_arr) > 0:
                net_arr[0] = net_arr[0] - d.transaction_cost
            daily_net.extend(net_arr.tolist())

        return np.asarray(daily_gross, dtype=float), np.asarray(daily_net, dtype=float)

    @staticmethod
    def _nanmean(x):
        arr = np.asarray(x, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(arr.mean()) if len(arr) > 0 else 0.0

    def summarize(self, results):
        gross_hp, net_hp, turnovers, transaction_costs, ex_ante_risk, concentration_hhi, effective_n_assets, cost_drag, avg_max_weight = \
            self._collect_holding_period_returns(results)
        daily_gross, daily_net = self._collect_daily_returns(results)

        summary = {
            "gross_annual_return_holding": annualized_return(gross_hp, self.periods_per_year),
            "gross_annual_vol_holding": annualized_vol(gross_hp, self.periods_per_year),
            "gross_sharpe_holding": sharpe_ratio(gross_hp, periods_per_year=self.periods_per_year),
            "gross_sortino_holding": sortino_ratio(gross_hp, periods_per_year=self.periods_per_year),
            "gross_calmar_holding": calmar_ratio(gross_hp, periods_per_year=self.periods_per_year),
            "gross_max_drawdown_holding": max_drawdown(gross_hp),

            "net_annual_return_holding": annualized_return(net_hp, self.periods_per_year),
            "net_of_cost_annual_return_holding": annualized_return(net_hp, self.periods_per_year),
            "net_annual_vol_holding": annualized_vol(net_hp, self.periods_per_year),
            "net_sharpe_holding": sharpe_ratio(net_hp, periods_per_year=self.periods_per_year),
            "net_sortino_holding": sortino_ratio(net_hp, periods_per_year=self.periods_per_year),
            "net_calmar_holding": calmar_ratio(net_hp, periods_per_year=self.periods_per_year),
            "net_max_drawdown_holding": max_drawdown(net_hp),

            "avg_turnover": average_turnover(turnovers),
            "avg_transaction_cost": self._nanmean(transaction_costs),
            "avg_cost_drag": self._nanmean(cost_drag),
            "avg_expected_risk": self._nanmean(ex_ante_risk),
            "avg_concentration_hhi": self._nanmean(concentration_hhi),
            "avg_effective_n_assets": self._nanmean(effective_n_assets),
            "avg_max_weight": self._nanmean(avg_max_weight),
            "gross_minus_net_annual_return_gap": annualized_return(gross_hp, self.periods_per_year) - annualized_return(net_hp, self.periods_per_year),
            "gross_minus_net_sharpe_gap": sharpe_ratio(gross_hp, periods_per_year=self.periods_per_year) - sharpe_ratio(net_hp, periods_per_year=self.periods_per_year),
            "n_rebalances": len(results),
        }

        if len(daily_gross) > 0:
            summary.update({
                "gross_annual_return_daily": annualized_return(daily_gross, self.daily_periods_per_year),
                "gross_annual_vol_daily": annualized_vol(daily_gross, self.daily_periods_per_year),
                "gross_sharpe_daily": sharpe_ratio(daily_gross, periods_per_year=self.daily_periods_per_year),
                "gross_sortino_daily": sortino_ratio(daily_gross, periods_per_year=self.daily_periods_per_year),
                "gross_calmar_daily": calmar_ratio(daily_gross, periods_per_year=self.daily_periods_per_year),
                "gross_max_drawdown_daily": max_drawdown(daily_gross),

                "net_annual_return_daily": annualized_return(daily_net, self.daily_periods_per_year),
                "net_annual_vol_daily": annualized_vol(daily_net, self.daily_periods_per_year),
                "net_sharpe_daily": sharpe_ratio(daily_net, periods_per_year=self.daily_periods_per_year),
                "net_sortino_daily": sortino_ratio(daily_net, periods_per_year=self.daily_periods_per_year),
                "net_calmar_daily": calmar_ratio(daily_net, periods_per_year=self.daily_periods_per_year),
                "net_max_drawdown_daily": max_drawdown(daily_net),

                "n_daily_obs": int(len(daily_gross)),
            })

        summary["mean_module_metrics"] = self.module_evaluator.summarize_mean(results)
        summary["vol_module_metrics"] = self.module_evaluator.summarize_vol(results)
        summary["cov_module_metrics"] = self.module_evaluator.summarize_covariance(results)

        return summary
