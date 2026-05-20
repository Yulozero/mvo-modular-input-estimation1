import numpy as np
import pandas as pd
from src.models.corr.base import BaseCorrModel

class KGCorrModel(BaseCorrModel):
    def fit(self, train_returns: pd.DataFrame, kg_data=None, valid_returns: pd.DataFrame = None):
        self.asset_list = list(train_returns.columns)
        return self

    def predict_corr(self, hist_returns: pd.DataFrame, kg_data=None) -> np.ndarray:
        if kg_data is None or kg_data.similarity is None:
            n = hist_returns.shape[1]
            return np.eye(n)

        sim = kg_data.similarity.copy()
        sim = sim.loc[self.asset_list, self.asset_list]
        mat = sim.values.astype(float)

        diag = np.sqrt(np.clip(np.diag(mat), 1e-12, None))
        mat = mat / (diag[:, None] * diag[None, :])
        np.fill_diagonal(mat, 1.0)
        return mat
