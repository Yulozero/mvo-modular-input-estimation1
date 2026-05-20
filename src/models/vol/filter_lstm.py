import copy
import numpy as np
import pandas as pd
from .base import BaseVolModel

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


class _FilterLSTMRegressor(nn.Module):
    def __init__(self, input_size=3, hidden_size=32, num_layers=1, dropout=0.1):
        super().__init__()
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
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        h = self.dropout(h)
        y = self.head(h).squeeze(-1)
        return y


class FilterLSTMVolModel(BaseVolModel):
    def __init__(
        self,
        seq_len: int = 20,
        hidden_size: int = 32,
        num_layers: int = 1,
        epochs: int = 30,
        batch_size: int = 64,
        lr: float = 1e-3,
        dropout: float = 0.1,
        smooth_span: int = 5,
        early_stopping_patience: int = 5,
        device: str = None,
        random_state: int = 42,
    ):
        super().__init__()
        if torch is None:
            raise ImportError("Torch is required for FilterLSTMVolModel.")

        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout
        self.smooth_span = smooth_span
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _smooth_series(self, x: np.ndarray) -> np.ndarray:
        # EMA smoothing
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x, dtype=float)
        alpha = 2.0 / (self.smooth_span + 1.0)
        out[0] = x[0]
        for i in range(1, len(x)):
            out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
        return out

    def _build_dataset(self, returns: pd.DataFrame, targets: pd.DataFrame):
        X_list, y_list = [], []
        arr_r = returns.values.astype(np.float32)
        arr_y = targets.values.astype(np.float32)
        n_time, n_assets = arr_r.shape

        for j in range(n_assets):
            series_r = arr_r[:, j].astype(float)
            series_y = arr_y[:, j].astype(float)

            if np.isfinite(series_r).sum() < self.seq_len + 5:
                continue

            smooth_r = self._smooth_series(series_r)
            smooth_abs_r = self._smooth_series(np.abs(series_r))

            for t in range(self.seq_len - 1, n_time):
                y = series_y[t]
                if not np.isfinite(y):
                    continue

                x1 = series_r[t - self.seq_len + 1:t + 1]
                x2 = smooth_r[t - self.seq_len + 1:t + 1]
                x3 = smooth_abs_r[t - self.seq_len + 1:t + 1]

                if (not np.isfinite(x1).all()) or (not np.isfinite(x2).all()) or (not np.isfinite(x3).all()):
                    continue

                x = np.column_stack([x1, x2, x3]).astype(np.float32)
                X_list.append(x)
                y_list.append(float(y))

        if len(X_list) == 0:
            return None, None

        X = np.stack(X_list).astype(np.float32)
        y = np.array(y_list, dtype=np.float32)
        return X, y

    def fit(
        self,
        train_returns: pd.DataFrame,
        train_targets: pd.DataFrame,
        valid_returns: pd.DataFrame = None,
        valid_targets: pd.DataFrame = None,
    ):
        self.asset_list_ = list(train_returns.columns)

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_train, y_train = self._build_dataset(train_returns, train_targets)
        if X_train is None:
            raise ValueError("Insufficient training data for FilterLSTMVolModel.")

        X_valid, y_valid = None, None
        if valid_returns is not None and valid_targets is not None:
            X_valid, y_valid = self._build_dataset(valid_returns, valid_targets)

        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        g = torch.Generator()
        g.manual_seed(self.random_state)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=g,
        )

        model = _FilterLSTMRegressor(
            input_size=3,
            hidden_size=self.hidden_size,
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

            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()
                pred = model(xb)
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
                    pred = model(xb)
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
            raise RuntimeError("FilterLSTMVolModel failed to train.")

        model.load_state_dict(best_state)
        model.eval()

        self.model_ = model
        self.diagnostics = {
            "model": "FilterLSTMVolModel",
            "seq_len": self.seq_len,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "smooth_span": self.smooth_span,
            "epochs_requested": self.epochs,
            "best_epoch": best_epoch,
            "best_valid_loss": best_valid_loss,
            "n_assets": len(self.asset_list_),
            "n_train_samples": int(len(X_train)),
            "n_valid_samples": int(len(X_valid)) if X_valid is not None else 0,
            "feature_definition": "[returns, smoothed_returns, smoothed_abs_returns]",
            "output_type": "forward_realized_vol_std",
            "train_loss_history": train_losses,
            "valid_loss_history": valid_losses,
        }
        return self

    def predict_vol(self, returns: pd.DataFrame) -> np.ndarray:
        returns = returns[self.asset_list_]
        preds = []

        self.model_.eval()
        with torch.no_grad():
            for col in returns.columns:
                series_r = returns[col].values.astype(float)
                if len(series_r) < self.seq_len:
                    raise ValueError(f"Insufficient data for asset {col} in FilterLSTMVolModel.")

                smooth_r = self._smooth_series(series_r)
                smooth_abs_r = self._smooth_series(np.abs(series_r))

                x = np.column_stack([
                    series_r[-self.seq_len:],
                    smooth_r[-self.seq_len:],
                    smooth_abs_r[-self.seq_len:],
                ]).astype(np.float32)

                x = torch.tensor(x.reshape(1, self.seq_len, 3), dtype=torch.float32).to(self.device)
                pred = self.model_(x).cpu().item()
                preds.append(float(pred))

        preds = np.asarray(preds, dtype=float)
        preds = np.clip(preds, 1e-8, None)
        return np.nan_to_num(preds, nan=1e-8, posinf=1e6, neginf=1e-8)
