import numpy as np
import pandas as pd
from src.models.corr.base import BaseCorrModel
from src.models.corr.pearson import PearsonCorrModel
from src.models.corr.kg_corr import KGCorrModel

class FusionCorrModel(BaseCorrModel):
    def __init__(self, stat_model=None, kg_model=None, lambda_=0.7):
        self.stat_model = stat_model or PearsonCorrModel()
        self.kg_model = kg_model or KGCorrModel()
        self.lambda_ = lambda_

    def fit(self, train_returns: pd.DataFrame, kg_data=None, valid_returns: pd.DataFrame = None):
        self.stat_model.fit(train_returns, kg_data, valid_returns)
        self.kg_model.fit(train_returns, kg_data, valid_returns)
        return self

    def predict_corr(self, hist_returns: pd.DataFrame, kg_data=None) -> np.ndarray:
        r_stat = self.stat_model.predict_corr(hist_returns, kg_data)
        r_kg = self.kg_model.predict_corr(hist_returns, kg_data)
        mat = self.lambda_ * r_stat + (1 - self.lambda_) * r_kg
        mat = (mat + mat.T) / 2.0
        np.fill_diagonal(mat, 1.0)
        return mat
