import numpy as np
from src.evaluation.mean_metrics import mse as mean_mse, mae as mean_mae, hit_ratio, information_coefficient, rank_information_coefficient
from src.evaluation.vol_metrics import mse as vol_mse, rmse as vol_rmse, mae as vol_mae, qlike
from src.evaluation.corr_metrics import frobenius_distance


class ModuleEvaluator:
    def summarize_mean(self, results):
        y_true_all = []
        y_pred_all = []

        cross_sectional_ics = []
        cross_sectional_rank_ics = []

        for r in results:
            y_true = r.get("true_forward_return")
            y_pred = r.get("mu_hat")
            if y_true is None or y_pred is None:
                continue

            y_true = np.asarray(y_true, dtype=float).reshape(-1)
            y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            if mask.sum() == 0:
                continue

            yt = y_true[mask]
            yp = y_pred[mask]

            y_true_all.extend(yt.tolist())
            y_pred_all.extend(yp.tolist())

            cross_sectional_ics.append(information_coefficient(yt, yp))
            cross_sectional_rank_ics.append(rank_information_coefficient(yt, yp))

        if len(y_true_all) == 0:
            return {}

        y_true_all = np.asarray(y_true_all, dtype=float)
        y_pred_all = np.asarray(y_pred_all, dtype=float)

        return {
            "mse": mean_mse(y_true_all, y_pred_all),
            "mae": mean_mae(y_true_all, y_pred_all),
            "hit_ratio": hit_ratio(y_true_all, y_pred_all),
            "global_ic": information_coefficient(y_true_all, y_pred_all),
            "global_rank_ic": rank_information_coefficient(y_true_all, y_pred_all),
            "avg_cross_sectional_ic": float(np.mean(cross_sectional_ics)) if cross_sectional_ics else 0.0,
            "avg_cross_sectional_rank_ic": float(np.mean(cross_sectional_rank_ics)) if cross_sectional_rank_ics else 0.0,
            "median_cross_sectional_ic": float(np.median(cross_sectional_ics)) if cross_sectional_ics else 0.0,
            "median_cross_sectional_rank_ic": float(np.median(cross_sectional_rank_ics)) if cross_sectional_rank_ics else 0.0,
            "n_obs": int(len(y_true_all)),
            "n_rebalances_used": int(len(cross_sectional_ics)),
        }

    def summarize_vol(self, results):
        y_true_all = []
        y_pred_all = []

        for r in results:
            y_true = r.get("true_forward_vol")
            y_pred = r.get("sigma_hat")
            if y_true is None or y_pred is None:
                continue

            y_true = np.asarray(y_true, dtype=float).reshape(-1)
            y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            if mask.sum() == 0:
                continue

            y_true_all.extend(y_true[mask].tolist())
            y_pred_all.extend(y_pred[mask].tolist())

        if len(y_true_all) == 0:
            return {}

        y_true_all = np.asarray(y_true_all, dtype=float)
        y_pred_all = np.asarray(y_pred_all, dtype=float)

        return {
            "mse": vol_mse(y_true_all, y_pred_all),
            "rmse": vol_rmse(y_true_all, y_pred_all),
            "mae": vol_mae(y_true_all, y_pred_all),
            "median_abs_error": float(np.median(np.abs(y_true_all - y_pred_all))),
            "qlike": qlike(y_true_all, y_pred_all),
            "n_obs": int(len(y_true_all)),
        }

    def summarize_covariance(self, results):
        fro_list = []
        ex_ante_risk = []
        ex_post_risk = []

        for r in results:
            Sigma_hat = r.get("Sigma_hat")
            Sigma_real = r.get("realized_cov")
            w = r.get("weights")

            if Sigma_hat is None or Sigma_real is None or w is None:
                continue

            Sigma_hat = np.asarray(Sigma_hat, dtype=float)
            Sigma_real = np.asarray(Sigma_real, dtype=float)
            w = np.asarray(w, dtype=float).reshape(-1)

            if Sigma_hat.shape != Sigma_real.shape:
                continue

            fro_list.append(frobenius_distance(Sigma_hat, Sigma_real))

            ex_ante = float(w.reshape(1, -1) @ Sigma_hat @ w.reshape(-1, 1))
            ex_post = float(w.reshape(1, -1) @ Sigma_real @ w.reshape(-1, 1))
            ex_ante_risk.append(max(ex_ante, 0.0))
            ex_post_risk.append(max(ex_post, 0.0))

        if len(fro_list) == 0:
            return {}

        ex_ante_risk = np.asarray(ex_ante_risk, dtype=float)
        ex_post_risk = np.asarray(ex_post_risk, dtype=float)

        return {
            "avg_frobenius_distance": float(np.mean(fro_list)),
            "median_frobenius_distance": float(np.median(fro_list)),
            "avg_ex_ante_variance": float(np.mean(ex_ante_risk)),
            "avg_ex_post_variance": float(np.mean(ex_post_risk)),
            "avg_abs_risk_bias": float(np.mean(np.abs(ex_ante_risk - ex_post_risk))),
            "n_rebalances_used": int(len(fro_list)),
        }
