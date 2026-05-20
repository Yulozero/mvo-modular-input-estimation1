import time
import numpy as np


class CovarianceAssembler:
    def __init__(self, psd_projector=None):
        self.psd_projector = psd_projector

    def build(self, sigma_hat: np.ndarray, R_hat: np.ndarray) -> np.ndarray:
        sigma_hat = np.asarray(sigma_hat, dtype=float).reshape(-1)
        R_hat = np.asarray(R_hat, dtype=float)

        n = len(sigma_hat)
        if R_hat.shape != (n, n):
            raise ValueError(f"Shape mismatch: sigma_hat len={n}, R_hat shape={R_hat.shape}")

        sigma_hat = np.nan_to_num(sigma_hat, nan=1e-8, posinf=1e6, neginf=1e-8)
        sigma_hat = np.clip(sigma_hat, 1e-8, None)

        R_hat = np.nan_to_num(R_hat, nan=0.0, posinf=0.0, neginf=0.0)
        R_hat = (R_hat + R_hat.T) / 2.0
        np.fill_diagonal(R_hat, 1.0)

        D_hat = np.diag(sigma_hat)
        Sigma_hat = D_hat @ R_hat @ D_hat
        Sigma_hat = (Sigma_hat + Sigma_hat.T) / 2.0

        if self.psd_projector is not None:
            Sigma_hat = self.psd_projector.project(Sigma_hat)

        return Sigma_hat


class RollingBacktestEngine:
    def __init__(
        self,
        mean_model,
        vol_model,
        corr_model,
        optimizer,
        assembler,
        executor,
        logger=None,
        verbose: bool = True,
        progress_interval: int = 1,
        stage_verbose: bool = True,
    ):
        self.mean_model = mean_model
        self.vol_model = vol_model
        self.corr_model = corr_model
        self.optimizer = optimizer
        self.assembler = assembler
        self.executor = executor
        self.logger = logger
        self.verbose = bool(verbose)
        self.progress_interval = max(int(progress_interval), 1)
        self.stage_verbose = bool(stage_verbose)
        self._continuous_window_models = {
            "LSTMMeanModel",
            "AttentionWeightedMeanModel",
            "TransformerAttentionMeanModel",
            "GARCHLSTMVolModel",
            "FilterLSTMVolModel",
            "DWTLSTMVolModel",
        }

    def _supports_continuous_sliding_labels(self, model) -> bool:
        return model.__class__.__name__ in self._continuous_window_models

    @staticmethod
    def _build_continuous_label_views(
        history_returns,
        train_targets_block,
        valid_targets_block,
    ):
        if history_returns is None:
            return train_targets_block, valid_targets_block

        idx = history_returns.index
        cols = history_returns.columns

        train_view = np.full((len(idx), len(cols)), np.nan, dtype=float)
        valid_view = np.full((len(idx), len(cols)), np.nan, dtype=float)

        train_df = np.asarray(train_targets_block.values, dtype=float)
        valid_df = np.asarray(valid_targets_block.values, dtype=float)

        train_pos = idx.get_indexer(train_targets_block.index)
        valid_pos = idx.get_indexer(valid_targets_block.index)

        for src_i, dst_i in enumerate(train_pos):
            if dst_i >= 0:
                train_view[dst_i, :] = train_df[src_i, :]

        for src_i, dst_i in enumerate(valid_pos):
            if dst_i >= 0:
                valid_view[dst_i, :] = valid_df[src_i, :]

        train_out = history_returns.copy().astype(float)
        valid_out = history_returns.copy().astype(float)
        train_out.iloc[:, :] = train_view
        valid_out.iloc[:, :] = valid_view
        return train_out, valid_out

    @staticmethod
    def _compute_realized_cov(test_returns):
        if test_returns is None or len(test_returns) <= 1:
            return None

        mat = np.asarray(test_returns.values, dtype=float)
        mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)

        daily_cov = np.cov(mat, rowvar=False, ddof=1)
        horizon = mat.shape[0]
        cov = daily_cov * horizon
        cov = np.asarray(cov, dtype=float)
        cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
        cov = (cov + cov.T) / 2.0
        return cov

    @staticmethod
    def _compute_true_forward_return(test_returns):
        if test_returns is None or len(test_returns) == 0:
            return None
        mat = np.asarray(test_returns.values, dtype=float)
        mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
        return np.prod(1.0 + mat, axis=0) - 1.0

    @staticmethod
    def _compute_true_forward_vol(test_returns):
        if test_returns is None or len(test_returns) == 0:
            return None
        mat = np.asarray(test_returns.values, dtype=float)
        mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
        return np.sqrt(np.sum(mat ** 2, axis=0))

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        seconds = max(float(seconds), 0.0)
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes = int(seconds // 60)
        remain = seconds % 60
        if minutes < 60:
            return f"{minutes}m {remain:.1f}s"
        hours = minutes // 60
        minutes = minutes % 60
        return f"{hours}h {minutes}m {remain:.1f}s"

    def _print_progress_header(self, total_steps: int):
        if self.verbose:
            print(f"[INFO] Rolling backtest started. Total rebalance steps: {total_steps}")

    def _print_progress(self, step_idx, total_steps, window, start_time):
        if not self.verbose:
            return

        elapsed = time.time() - start_time
        avg_time = elapsed / max(step_idx, 1)
        remain_steps = max(total_steps - step_idx, 0)
        eta = avg_time * remain_steps

        print(
            f"[PROGRESS] {step_idx}/{total_steps} "
            f"({100.0 * step_idx / max(total_steps, 1):.1f}%) | "
            f"rebalance_date={window.rebalance_date} | "
            f"predict_date={window.predict_date} | "
            f"elapsed={self._format_seconds(elapsed)} | "
            f"eta={self._format_seconds(eta)}"
        )

    def _print_stage(self, step_idx, total_steps, stage_name, window):
        if not self.stage_verbose:
            return

        print(
            f"    [STEP {step_idx}/{total_steps}] {stage_name} | "
            f"predict_date={window.predict_date}"
        )

    def run(self, market_data, splitter, kg_data=None):
        results = []
        w_prev = None

        total_steps = splitter.get_n_steps(market_data)
        start_time = time.time()

        self._print_progress_header(total_steps)

        for i, window in enumerate(splitter.generate(market_data), start=1):
            if self.verbose and (i == 1 or i % self.progress_interval == 0 or i == total_steps):
                self._print_progress(i, total_steps, window, start_time)

            if self.logger:
                self.logger.info(
                    f"Running rebalance step {i} at {window.rebalance_date} | "
                    f"predict_date={window.predict_date} | "
                    f"valid_labels(train={window.n_train_label_valid}, valid={window.n_valid_label_valid})"
                )

            self._print_stage(i, total_steps, "mean_model.fit", window)
            mean_train_returns = window.train_returns
            mean_train_targets = window.train_forward_returns
            mean_valid_returns = window.valid_returns
            mean_valid_targets = window.valid_forward_returns
            if (
                self._supports_continuous_sliding_labels(self.mean_model)
                and window.history_returns is not None
                and window.train_forward_returns is not None
                and window.valid_forward_returns is not None
            ):
                mean_train_returns = window.history_returns
                mean_valid_returns = window.history_returns
                mean_train_targets, mean_valid_targets = self._build_continuous_label_views(
                    history_returns=window.history_returns,
                    train_targets_block=window.train_forward_returns,
                    valid_targets_block=window.valid_forward_returns,
                )

            self.mean_model.fit(
                train_returns=mean_train_returns,
                train_targets=mean_train_targets,
                valid_returns=mean_valid_returns,
                valid_targets=mean_valid_targets,
            )

            self._print_stage(i, total_steps, "vol_model.fit", window)
            vol_train_returns = window.train_returns
            vol_train_targets = window.train_forward_vol
            vol_valid_returns = window.valid_returns
            vol_valid_targets = window.valid_forward_vol
            if (
                self._supports_continuous_sliding_labels(self.vol_model)
                and window.history_returns is not None
                and window.train_forward_vol is not None
                and window.valid_forward_vol is not None
            ):
                vol_train_returns = window.history_returns
                vol_valid_returns = window.history_returns
                vol_train_targets, vol_valid_targets = self._build_continuous_label_views(
                    history_returns=window.history_returns,
                    train_targets_block=window.train_forward_vol,
                    valid_targets_block=window.valid_forward_vol,
                )

            self.vol_model.fit(
                train_returns=vol_train_returns,
                train_targets=vol_train_targets,
                valid_returns=vol_valid_returns,
                valid_targets=vol_valid_targets,
            )

            self._print_stage(i, total_steps, "corr_model.fit", window)
            self.corr_model.fit(
                window.train_returns,
                kg_data,
                window.valid_returns,
            )

            hist_returns = window.history_returns

            self._print_stage(i, total_steps, "predict", window)
            mu_hat = self.mean_model.predict(hist_returns)
            mu_hat = np.asarray(mu_hat, dtype=float).reshape(-1)
            mu_hat = np.nan_to_num(mu_hat, nan=0.0, posinf=0.0, neginf=0.0)

            sigma_hat = self.vol_model.predict_vol(hist_returns)
            sigma_hat = np.asarray(sigma_hat, dtype=float).reshape(-1)
            sigma_hat = np.nan_to_num(sigma_hat, nan=1e-8, posinf=1e6, neginf=1e-8)
            sigma_hat = np.clip(sigma_hat, 1e-8, None)
            D_hat = np.diag(sigma_hat)

            R_hat = self.corr_model.predict_corr(hist_returns, kg_data)
            R_hat = np.asarray(R_hat, dtype=float)

            Sigma_hat = self.assembler.build(sigma_hat, R_hat)

            self._print_stage(i, total_steps, "optimize", window)
            w_hat = self.optimizer.solve(mu_hat, Sigma_hat, w_prev=w_prev)
            w_hat = np.asarray(w_hat, dtype=float).reshape(-1)
            w_hat = np.nan_to_num(w_hat, nan=0.0, posinf=0.0, neginf=0.0)

            # If an asset is not observable/tradable at the decision date, do not allow a new target weight.
            tradable_mask = np.ones(len(w_hat), dtype=bool)
            if hist_returns is not None and len(hist_returns) > 0:
                last_visible = hist_returns.iloc[-1]
                tradable_mask = np.asarray(last_visible.notna().values, dtype=bool)
                if tradable_mask.shape[0] == len(w_hat):
                    w_hat = np.where(tradable_mask, w_hat, 0.0)

            if len(w_hat) > 0:
                if np.abs(w_hat.sum()) <= 1e-12:
                    w_hat = np.ones(len(w_hat), dtype=float) / len(w_hat)
                else:
                    w_hat = w_hat / w_hat.sum()

            expected_return = float(w_hat @ mu_hat)
            expected_var = float(w_hat.reshape(1, -1) @ Sigma_hat @ w_hat.reshape(-1, 1))
            expected_var = max(expected_var, 0.0)
            expected_risk = float(np.sqrt(expected_var))

            self._print_stage(i, total_steps, "execute", window)
            decision = self.executor.execute(
                date=window.predict_date,
                weights=w_hat,
                next_returns=window.test_returns,
                prev_weights=w_prev,
                expected_return=expected_return,
                expected_risk=expected_risk,
            )

            true_forward_return = self._compute_true_forward_return(window.test_returns)
            true_forward_vol = self._compute_true_forward_vol(window.test_returns)
            realized_cov = self._compute_realized_cov(window.test_returns)

            results.append({
                "rebalance_date": window.rebalance_date,
                "predict_date": window.predict_date,
                "asset_list": window.asset_list,

                "mu_hat": mu_hat,
                "sigma_hat": sigma_hat,
                "D_hat": D_hat,
                "R_hat": R_hat,
                "Sigma_hat": Sigma_hat,
                "weights": w_hat,

                "true_forward_return": true_forward_return,
                "true_forward_vol": true_forward_vol,
                "realized_cov": realized_cov,

                "decision": decision,

                "mean_diag": self.mean_model.get_diagnostics(),
                "vol_diag": self.vol_model.get_diagnostics(),
                "corr_diag": self.corr_model.get_diagnostics(),
                "optimizer_diag": self.optimizer.get_diagnostics() if hasattr(self.optimizer, "get_diagnostics") else {},

                "test_returns": window.test_returns,
                "test_realized_vol": window.test_realized_vol,

                "window_meta": {
                    "train_start_idx": window.train_start_idx,
                    "train_end_idx": window.train_end_idx,
                    "valid_start_idx": window.valid_start_idx,
                    "valid_end_idx": window.valid_end_idx,
                    "test_start_idx": window.test_start_idx,
                    "test_end_idx": window.test_end_idx,
                    "n_train_label_valid": window.n_train_label_valid,
                    "n_valid_label_valid": window.n_valid_label_valid,
                    "n_non_tradable_assets_at_decision": int((~tradable_mask).sum()) if len(w_hat) > 0 else 0,
                },
            })

            w_prev = w_hat.copy()

        if self.verbose:
            total_elapsed = time.time() - start_time
            print(f"[INFO] Rolling backtest finished in {self._format_seconds(total_elapsed)}")

        return results
