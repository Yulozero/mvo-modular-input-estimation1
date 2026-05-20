import copy
import math
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


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class _TransformerAttentionMeanNet(nn.Module):
    """
    严谨版 Transformer 时间加权均值/收益预测网络

    输入:
      x: [B, L, 1]

    流程:
      1) raw returns -> projection -> positional encoding
      2) Transformer encoder
      3) temporal attention scores over all time steps
      4) weighted raw-return summary m = sum(alpha * r)
      5) weighted hidden-state context c = sum(alpha * h)
      6) concat([m, c]) -> MLP head -> forward-horizon return prediction

    这样兼顾:
      - 注意力时间加权解释性
      - 非线性时序特征建模能力
    """
    def __init__(
        self,
        seq_len: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        self.input_proj = nn.Linear(1, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.pos_encoder = _PositionalEncoding(d_model=d_model, max_len=max(512, seq_len + 8))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.post_norm = nn.LayerNorm(d_model)

        # 时间注意力打分网络
        self.attn_scorer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        # 预测头：融合 "加权历史收益摘要" + "加权时序上下文"
        self.head = nn.Sequential(
            nn.Linear(d_model + 1, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        # x: [B, L, 1]
        raw = x.squeeze(-1)                         # [B, L]
        h = self.input_proj(x)                     # [B, L, d_model]
        h = self.input_norm(h)
        h = self.pos_encoder(h)
        h = self.encoder(h)                        # [B, L, d_model]
        h = self.post_norm(h)

        scores = self.attn_scorer(h).squeeze(-1)   # [B, L]
        alpha = torch.softmax(scores, dim=1)       # [B, L]

        weighted_raw_mean = torch.sum(alpha * raw, dim=1, keepdim=True)  # [B,1]
        weighted_context = torch.sum(alpha.unsqueeze(-1) * h, dim=1)     # [B,d_model]

        features = torch.cat([weighted_raw_mean, weighted_context], dim=1)
        pred = self.head(features).squeeze(-1)     # [B]

        return pred, alpha, weighted_raw_mean.squeeze(-1), weighted_context


class TransformerAttentionMeanModel(BaseMeanModel):
    def __init__(
        self,
        seq_len: int = 20,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        epochs: int = 40,
        batch_size: int = 64,
        lr: float = 1e-3,
        dropout: float = 0.1,
        weight_decay: float = 1e-4,
        grad_clip_norm: float = 1.0,
        early_stopping_patience: int = 7,
        use_huber_loss: bool = False,
        device: str = None,
        random_state: int = 42,
    ):
        super().__init__()
        if torch is None:
            raise ImportError("Torch is required for TransformerAttentionMeanModel.")

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.grad_clip_norm = grad_clip_norm
        self.early_stopping_patience = early_stopping_patience
        self.use_huber_loss = use_huber_loss
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _build_dataset(self, returns: pd.DataFrame, targets: pd.DataFrame):
        X_list, y_list, meta_list = [], [], []
        arr_x = returns.values.astype(np.float32)
        arr_y = targets.values.astype(np.float32)
        columns = list(returns.columns)

        n_time, n_assets = arr_x.shape

        for j in range(n_assets):
            series_x = arr_x[:, j]
            series_y = arr_y[:, j]

            for t in range(self.seq_len - 1, n_time):
                y = series_y[t]
                if np.isnan(y):
                    continue

                x = series_x[t - self.seq_len + 1:t + 1]
                if np.isnan(x).any():
                    continue

                X_list.append(x.reshape(self.seq_len, 1))
                y_list.append(y)
                meta_list.append({
                    "asset": columns[j],
                    "time_idx": t,
                })

        if len(X_list) == 0:
            return None, None, None

        X = np.stack(X_list).astype(np.float32)
        y = np.array(y_list, dtype=np.float32)
        return X, y, meta_list

    def _build_loader(self, X, y, shuffle: bool):
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        g = torch.Generator()
        g.manual_seed(self.random_state)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            generator=g if shuffle else None,
        )

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

        X_train, y_train, _ = self._build_dataset(train_returns, train_targets)
        if X_train is None:
            raise ValueError("Insufficient training data for TransformerAttentionMeanModel.")

        X_valid, y_valid, _ = None, None, None
        if valid_returns is not None and valid_targets is not None:
            X_valid, y_valid, _ = self._build_dataset(valid_returns, valid_targets)

        train_loader = self._build_loader(X_train, y_train, shuffle=True)

        model = _TransformerAttentionMeanNet(
            seq_len=self.seq_len,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        loss_fn = nn.HuberLoss(delta=1.0) if self.use_huber_loss else nn.MSELoss()

        best_state = None
        best_valid_loss = float("inf")
        best_epoch = -1
        no_improve = 0
        train_loss_history = []
        valid_loss_history = []

        for epoch in range(self.epochs):
            model.train()
            batch_losses = []

            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()
                pred, _, _, _ = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()

                if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip_norm)

                optimizer.step()
                batch_losses.append(loss.item())

            train_loss = float(np.mean(batch_losses)) if batch_losses else np.nan
            train_loss_history.append(train_loss)

            if X_valid is not None:
                model.eval()
                with torch.no_grad():
                    xb = torch.tensor(X_valid, dtype=torch.float32).to(self.device)
                    yb = torch.tensor(y_valid, dtype=torch.float32).to(self.device)
                    pred, _, _, _ = model(xb)
                    valid_loss = float(loss_fn(pred, yb).item())
            else:
                valid_loss = train_loss

            valid_loss_history.append(valid_loss)

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
            raise RuntimeError("TransformerAttentionMeanModel failed to train.")

        model.load_state_dict(best_state)
        model.eval()

        self.model_ = model
        self.last_attention_ = None
        self.last_weighted_raw_mean_ = None
        self.last_weighted_context_norm_ = None

        self.diagnostics = {
            "model": "TransformerAttentionMeanModel",
            "seq_len": self.seq_len,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "epochs_requested": self.epochs,
            "best_epoch": best_epoch,
            "best_valid_loss": best_valid_loss,
            "train_loss_history": train_loss_history,
            "valid_loss_history": valid_loss_history,
            "n_assets": len(self.asset_list_),
            "n_train_samples": int(len(X_train)),
            "n_valid_samples": int(len(X_valid)) if X_valid is not None else 0,
            "optimizer": "AdamW",
            "loss": "Huber" if self.use_huber_loss else "MSE",
            "target_definition": "forward_cumulative_return",
        }
        return self

    def predict(self, returns: pd.DataFrame) -> np.ndarray:
        returns = returns[self.asset_list_]
        preds = []
        attentions = []
        weighted_raw_means = []
        context_norms = []

        self.model_.eval()
        with torch.no_grad():
            for col in returns.columns:
                series = returns[col].values.astype(np.float32)
                if len(series) < self.seq_len:
                    raise ValueError(f"Insufficient sequence length for asset {col}.")

                x = series[-self.seq_len:].reshape(1, self.seq_len, 1)
                x = torch.tensor(x, dtype=torch.float32).to(self.device)

                pred, alpha, weighted_raw_mean, weighted_context = self.model_(x)

                preds.append(float(pred.cpu().item()))
                attentions.append(alpha.cpu().numpy().reshape(-1).tolist())
                weighted_raw_means.append(float(weighted_raw_mean.cpu().item()))
                context_norms.append(float(torch.norm(weighted_context, dim=1).cpu().item()))

        self.last_attention_ = attentions
        self.last_weighted_raw_mean_ = weighted_raw_means
        self.last_weighted_context_norm_ = context_norms

        return np.nan_to_num(np.array(preds, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    def get_diagnostics(self):
        diag = dict(self.diagnostics)
        if self.last_attention_ is not None:
            diag["last_attention"] = self.last_attention_
        if self.last_weighted_raw_mean_ is not None:
            diag["last_weighted_raw_mean"] = self.last_weighted_raw_mean_
        if self.last_weighted_context_norm_ is not None:
            diag["last_weighted_context_norm"] = self.last_weighted_context_norm_
        return diag
