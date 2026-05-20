import numpy as np
import pandas as pd
from src.models.corr.base import BaseCorrModel


class PearsonCorrModel(BaseCorrModel):
    def fit(self, train_returns: pd.DataFrame, kg_data=None, valid_returns: pd.DataFrame = None):
        return self

    def predict_corr(self, hist_returns: pd.DataFrame, kg_data=None) -> np.ndarray:
        corr = hist_returns.corr(method="pearson").values.astype(float)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        corr = (corr + corr.T) / 2.0
        np.fill_diagonal(corr, 1.0)
        return corr
