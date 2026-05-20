import copy
import numpy as np
import pandas as pd
from .base import BaseVolModel

try:
    import pywt
except ImportError:
    pywt = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


class _AssetConditionedDWTLSTMRegressor(nn.Module):
    def __init__(
        self,
        input_size=4,
        n_assets=1,
        hidden_size=32,
        asset_emb_dim=8,
        num_layers=1,
        dropout=0.1,
    ):
        super().__init__()
        self.asset_embedding = nn.Embedding(n_assets, asset_emb_dim)
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + asset_emb_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Softplus(),
        )

    def forward(self, x, asset_ids):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        h = self.dropout(h)
        emb = self.asset_embedding(asset_ids)
        features = torch.cat([h, emb], dim=1)
        y = self.head(features).squeeze(-1)
        return y


class AssetConditionedDWTLSTMVolModel(BaseVolModel):
    def __init__(
        self,
        seq_len: int = 60,
        hidden_size: int = 32,
        asset_emb_dim: int = 8,
        num_layers: int = 1,
        epochs: int = 30,
        batch_size: int = 64,
        lr: float = 1e-3,
        dropout: float = 0.1,
        early_stopping_patience: int = 5,
        wavelet: str = "db4",
        level: int = 3,
        dwt_mode: str = "symmetric",
        target_horizon: int = 20,
        device: str = None,
        random_state: int = 42,
    ):
        super().__init__()
        if pywt is None:
            raise ImportError("PyWavelets is required for AssetConditionedDWTLSTMVolModel.")
        if torch is None:
            raise ImportError("Torch is required for AssetConditionedDWTLSTMVolModel.")

        self.seq_len = int(seq_len)
        self.hidden_size = int(hidden_size)
        self.asset_emb_dim = int(asset_emb_dim)
        self.num_layers = int(num_layers)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.dropout = float(dropout)
        self.early_stopping_patience = int(early_stopping_patience)
        self.wavelet = str(wavelet)
        self.level = int(level)
        self.dwt_mode = str(dwt_mode)
        self.target_horizon = int(target_horizon)
        self.random_state = int(random_state)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _safe_wavelet_level(self, seq_len: int) -> int:
        wavelet_obj = pywt.Wavelet(self.wavelet)
        max_level = pywt.dwt_max_level(seq_len, wavelet_obj.dec_len)
        return max(1, min(self.level, max_level))

    def _dwt_multiscale_features(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if len(x) != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {len(x)}")

        level = self._safe_wavelet_level(len(x))
        coeffs = pywt.wavedec(x, wavelet=self.wavelet, mode=self.dwt_mode, level=level)

        rec_features = []

        approx_only = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
        rec_a = pywt.waverec(approx_only, wavelet=self.wavelet, mode=self.dwt_mode)[: self.seq_len]
        rec_features.append(rec_a)

        for i in range(1, len(coeffs)):
            detail_only = [np.zeros_like(coeffs[0])]
            for j in range(1, len(coeffs)):
                detail_only.append(coeffs[j] if j == i else np.zeros_like(coeffs[j]))
            rec_d = pywt.waverec(detail_only, wavelet=self.wavelet, mode=self.dwt_mode)[: self.seq_len]
            rec_features.append(rec_d)

        rec_features.insert(0, x.copy())
        return np.column_stack(rec_features).astype(np.float32)

    def _build_dataset(self, returns: pd.DataFrame, targets: pd.DataFrame):
        X_list, y_list, asset_id_list = [], [], []
        arr_r = returns.values.astype(np.float32)
        arr_y = targets.values.astype(np.float32)
        n_time, n_assets = arr_r.shape

        for j in range(n_assets):
            series_r = arr_r[:, j].astype(float)
            series_y = arr_y[:, j].astype(float)

            if np.isfinite(series_r).sum() < self.seq_len + 5:
                continue

            for t in range(self.seq_len - 1, n_time):
                y = series_y[t]
                if not np.isfinite(y):
                    continue

                x = series_r[t - self.seq_len + 1:t + 1]
                if not np.isfinite(x).all():
                    continue

                feat = self._dwt_multiscale_features(x)
                X_list.append(feat)
                y_list.append(float(y))
                asset_id_list.append(j)

        if len(X_list) == 0:
            return None, None, None

        return (
            np.stack(X_list).astype(np.float32),
            np.array(y_list, dtype=np.float32),
            np.array(asset_id_list, dtype=np.int64),
        )

    def fit(
        self,
        train_returns: pd.DataFrame,
        train_targets: pd.DataFrame,
        valid_returns: pd.DataFrame = None,
        valid_targets: pd.DataFrame = None,
    ):
        self.asset_list_ = list(train_returns.columns)
        self.asset_to_id_ = {asset: idx for idx, asset in enumerate(self.asset_list_)}

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_train, y_train, asset_ids_train = self._build_dataset(train_returns, train_targets)
        if X_train is None:
            raise ValueError("Insufficient training data for AssetConditionedDWTLSTMVolModel.")

        X_valid, y_valid, asset_ids_valid = None, None, None
        if valid_returns is not None and valid_targets is not None:
            X_valid, y_valid, asset_ids_valid = self._build_dataset(valid_returns, valid_targets)

        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(asset_ids_train, dtype=torch.long),
        )
        g = torch.Generator()
        g.manual_seed(self.random_state)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=g,
        )

        input_size = X_train.shape[-1]
        model = _AssetConditionedDWTLSTMRegressor(
            input_size=input_size,
            n_assets=len(self.asset_list_),
            hidden_size=self.hidden_size,
            asset_emb_dim=self.asset_emb_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        best_state = None
        best_valid_loss = float("inf")
        best_epoch = -1
        no_improve = 0
        train_losses = []
        valid_losses = []

        for epoch in range(self.epochs):
            model.train()
            losses = []

            for xb, yb, ab in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                ab = ab.to(self.device)

                optimizer.zero_grad()
                pred = model(xb, ab)
                loss = loss_fn(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(loss.item())

            train_loss = float(np.mean(losses)) if losses else np.nan
            train_losses.append(train_loss)

            if X_valid is not None:
                model.eval()
                with torch.no_grad():
                    xb = torch.tensor(X_valid, dtype=torch.float32).to(self.device)
                    yb = torch.tensor(y_valid, dtype=torch.float32).to(self.device)
                    ab = torch.tensor(asset_ids_valid, dtype=torch.long).to(self.device)
                    pred = model(xb, ab)
                    valid_loss = float(loss_fn(pred, yb).item())
            else:
                valid_loss = train_loss

            valid_losses.append(valid_loss)

            if valid_loss < best_valid_loss - 1e-8:
                best_valid_loss = valid_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.early_stopping_patience:
                break

        if best_state is None:
            raise RuntimeError("AssetConditionedDWTLSTMVolModel failed to train.")

        model.load_state_dict(best_state)
        model.eval()

        self.model_ = model
        self.input_size_ = input_size
        self.actual_wavelet_level_ = self._safe_wavelet_level(self.seq_len)
        self.last_asset_embedding_norm_ = None
        self.diagnostics = {
            "model": "AssetConditionedDWTLSTMVolModel",
            "seq_len": self.seq_len,
            "hidden_size": self.hidden_size,
            "asset_emb_dim": self.asset_emb_dim,
            "num_layers": self.num_layers,
            "wavelet": self.wavelet,
            "level_requested": self.level,
            "level_used": self.actual_wavelet_level_,
            "dwt_mode": self.dwt_mode,
            "target_horizon": self.target_horizon,
            "input_size": self.input_size_,
            "epochs_requested": self.epochs,
            "best_epoch": best_epoch,
            "best_valid_loss": best_valid_loss,
            "n_assets": len(self.asset_list_),
            "n_train_samples": int(len(X_train)),
            "n_valid_samples": int(len(X_valid)) if X_valid is not None else 0,
            "feature_definition": "raw_return + wavelet_reconstructed_multiscale_components + asset_embedding",
            "output_type": "forward_realized_vol_std",
            "train_loss_history": train_losses,
            "valid_loss_history": valid_losses,
        }
        return self

    def predict_vol(self, returns: pd.DataFrame) -> np.ndarray:
        returns = returns[self.asset_list_]
        preds = []
        asset_embedding_norms = []

        self.model_.eval()
        with torch.no_grad():
            for col in returns.columns:
                series_r = returns[col].values.astype(float)
                if len(series_r) < self.seq_len:
                    raise ValueError(f"Insufficient data for asset {col} in AssetConditionedDWTLSTMVolModel.")

                x = series_r[-self.seq_len:]
                feat = self._dwt_multiscale_features(x)
                asset_id = self.asset_to_id_[col]
                xb = torch.tensor(
                    feat.reshape(1, self.seq_len, self.input_size_),
                    dtype=torch.float32,
                ).to(self.device)
                ab = torch.tensor([asset_id], dtype=torch.long).to(self.device)

                pred = self.model_(xb, ab).cpu().item()
                emb_norm = torch.norm(self.model_.asset_embedding(ab), dim=1).cpu().item()
                preds.append(float(pred))
                asset_embedding_norms.append(float(emb_norm))

        self.last_asset_embedding_norm_ = asset_embedding_norms
        preds = np.asarray(preds, dtype=float)
        preds = np.clip(preds, 1e-8, None)
        return np.nan_to_num(preds, nan=1e-8, posinf=1e6, neginf=1e-8)

    def get_diagnostics(self):
        diag = dict(self.diagnostics)
        if self.last_asset_embedding_norm_ is not None:
            diag["last_asset_embedding_norm"] = self.last_asset_embedding_norm_
        return diag
