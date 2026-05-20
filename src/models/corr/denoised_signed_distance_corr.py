import numpy as np
import pandas as pd

from src.models.corr.base import BaseCorrModel
from src.models.corr.signed_distance_corr import SignedDistanceCorrModel
from src.models.corr.pearson import PearsonCorrModel


class DenoisedSignedDistanceCorrModel(BaseCorrModel):
    def __init__(
        self,
        blend_with_pearson: float = 0.5,
        shrink_to_identity: float = 0.1,
        clip_value: float = 0.95,
    ):
        self.blend_with_pearson = float(blend_with_pearson)
        self.shrink_to_identity = float(shrink_to_identity)
        self.clip_value = float(clip_value)
        self._signed = SignedDistanceCorrModel()
        self._pearson = PearsonCorrModel()

    def fit(self, train_returns: pd.DataFrame, kg_data=None, valid_returns: pd.DataFrame = None):
        return self

    def predict_corr(self, hist_returns: pd.DataFrame, kg_data=None) -> np.ndarray:
        r_signed = self._signed.predict_corr(hist_returns, kg_data)
        r_pearson = self._pearson.predict_corr(hist_returns, kg_data)

        lam = float(np.clip(self.blend_with_pearson, 0.0, 1.0))
        alpha = float(np.clip(self.shrink_to_identity, 0.0, 1.0))

        mat = (1.0 - lam) * r_signed + lam * r_pearson
        if alpha > 0.0:
            n = mat.shape[0]
            mat = (1.0 - alpha) * mat + alpha * np.eye(n, dtype=float)

        mat = np.clip(mat, -self.clip_value, self.clip_value)
        mat = (mat + mat.T) / 2.0
        np.fill_diagonal(mat, 1.0)
        return mat

