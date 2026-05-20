from pathlib import Path
from typing import Optional

import pandas as pd

from src.core.types import MarketData, KGraphData


def _read_csv(path: Optional[str], index_col: int = 0) -> Optional[pd.DataFrame]:
    if path is None:
        return None

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(p, index_col=index_col)

    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass

    return df


def load_market_data(config: dict) -> MarketData:
    data_cfg = config.get("data", {})

    returns = _read_csv(data_cfg.get("returns_path"))
    if returns is None:
        raise ValueError("data.returns_path is required")

    prices = _read_csv(data_cfg.get("prices_path"))
    if prices is None:
        prices = _read_csv(data_cfg.get("close_path"))

    realized_vol = _read_csv(data_cfg.get("realized_vol_path"))
    market_caps = _read_csv(data_cfg.get("market_caps_path"))
    if market_caps is None:
        market_caps = _read_csv(data_cfg.get("market_cap_path"))

    start_date = data_cfg.get("start_date")
    end_date = data_cfg.get("end_date")

    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        returns = returns.loc[returns.index >= start_date]
        if prices is not None:
            prices = prices.loc[prices.index >= start_date]
        if realized_vol is not None:
            realized_vol = realized_vol.loc[realized_vol.index >= start_date]
        if market_caps is not None:
            market_caps = market_caps.loc[market_caps.index >= start_date]

    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        returns = returns.loc[returns.index <= end_date]
        if prices is not None:
            prices = prices.loc[prices.index <= end_date]
        if realized_vol is not None:
            realized_vol = realized_vol.loc[realized_vol.index <= end_date]
        if market_caps is not None:
            market_caps = market_caps.loc[market_caps.index <= end_date]

    return MarketData(
        returns=returns,
        prices=prices,
        market_caps=market_caps,
        realized_vol=realized_vol,
        forward_returns=None,
        forward_vol=None,
        features=None,
        asset_meta=None,
    )


def load_kg_data(config: dict):
    kg_cfg = config.get("kg", config.get("kg_data", {}))
    if not kg_cfg.get("enabled", False):
        return None

    similarity = _read_csv(kg_cfg.get("similarity_path"))
    adjacency = _read_csv(kg_cfg.get("adjacency_path"))
    node_embeddings = _read_csv(kg_cfg.get("node_embeddings_path"))

    return KGraphData(
        similarity=similarity,
        adjacency=adjacency,
        node_embeddings=node_embeddings,
    )
