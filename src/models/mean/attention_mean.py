import copy
import numpy as np
import pandas as pd
from .base import BaseMeanModel

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


class _AttentionWeightedMeanNet(nn.Module):
    def __init__(self, seq_len: int, hidden_size: int = 32, dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.scorer = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        # x: [B, L, 1]
        raw = x.squeeze(-1)  # [B, L]
        scores = self.scorer(x).squeeze(-1)      # [B, L]
        alpha = torch.softmax(scores, dim=1)     # [B, L]
        weighted_mean = torch.sum(alpha * raw, dim=1, keepdim=True)  # [B,1]
        weighted_abs = torch.sum(alpha * torch.abs(raw), dim=1, keepdim=True)
        pred = self.head(torch.cat([weighted_mean, weighted_abs], dim=1)).squeeze(-1)
        return pred, alpha


class AttentionWeightedMeanModel(BaseMeanModel):
    def __init__(
        self,
        seq_len: int = 20,
        hidden_size: int = 32,
        epochs: int = 30,
        batch_size: int = 64,
        lr: float = 1e-3,
        dropout: float = 0.1,
        early_stopping_patience: int = 5,
        device: str = None,
        random_state: int = 42,
    ):
        super().__init__()
        if torch is None:
            raise ImportError("Torch is required for AttentionWeightedMeanModel.")
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _build_dataset(self, returns: pd.DataFrame, targets: pd.DataFrame):
        X_list, y_list = [], []
        arr_x = returns.values.astype(np.float32)
        arr_y = targets.values.astype(np.float32)

        for j in range(arr_x.shape[1]):
            series_x = arr_x[:, j]
            series_y = arr_y[:, j]
            for t in range(self.seq_len - 1, len(series_x)):
                y = series_y[t]
                if np.isnan(y):
                    continue
                x = series_x[t - self.seq_len + 1:t + 1]
                if np.isnan(x).any():
                    continue
                X_list.append(x.reshape(self.seq_len, 1))
                y_list.append(y)

        if len(X_list) == 0:
            return None, None

        return np.stack(X_list), np.array(y_list, dtype=np.float32)

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
            raise ValueError("Insufficient training data for AttentionWeightedMeanModel.")

        X_valid, y_valid = None, None
        if valid_returns is not None and valid_targets is not None:
            X_valid, y_valid = self._build_dataset(valid_returns, valid_targets)

        dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        g = torch.Generator()
        g.manual_seed(self.random_state)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, generator=g)

        model = _AttentionWeightedMeanNet(
            seq_len=self.seq_len,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        best_state = None
        best_valid_loss = float("inf")
        best_epoch = -1
        no_improve = 0

        for epoch in range(self.epochs):
            model.train()
            losses = []

            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()
                pred, _ = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            train_loss = float(np.mean(losses)) if losses else np.nan

            if X_valid is not None:
                model.eval()
                with torch.no_grad():
                    xb = torch.tensor(X_valid, dtype=torch.float32).to(self.device)
                    yb = torch.tensor(y_valid, dtype=torch.float32).to(self.device)
                    pred, _ = model(xb)
                    valid_loss = float(loss_fn(pred, yb).item())
            else:
                valid_loss = train_loss

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
            raise RuntimeError("AttentionWeightedMeanModel failed to train.")

        model.load_state_dict(best_state)
        model.eval()

        self.model_ = model
        self.last_attention_ = None
        self.diagnostics = {
            "model": "AttentionWeightedMeanModel",
            "seq_len": self.seq_len,
            "hidden_size": self.hidden_size,
            "epochs_requested": self.epochs,
            "best_epoch": best_epoch,
            "best_valid_loss": best_valid_loss,
            "n_assets": len(self.asset_list_),
            "n_train_samples": int(len(X_train)),
            "n_valid_samples": int(len(X_valid)) if X_valid is not None else 0,
            "target_definition": "forward_cumulative_return",
        }
        return self

    def predict(self, returns: pd.DataFrame) -> np.ndarray:
        returns = returns[self.asset_list_]
        preds = []
        attentions = []

        self.model_.eval()
        with torch.no_grad():
            for col in returns.columns:
                series = returns[col].values.astype(np.float32)
                if len(series) < self.seq_len:
                    raise ValueError(f"Insufficient sequence length for asset {col}.")
                x = series[-self.seq_len:].reshape(1, self.seq_len, 1)
                x = torch.tensor(x, dtype=torch.float32).to(self.device)
                pred, alpha = self.model_(x)
                preds.append(float(pred.cpu().item()))
                attentions.append(alpha.cpu().numpy().reshape(-1).tolist())

        self.last_attention_ = attentions
        return np.nan_to_num(np.array(preds, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    def get_diagnostics(self):
        diag = dict(self.diagnostics)
        if self.last_attention_ is not None:
            diag["last_attention"] = self.last_attention_
        return diag
