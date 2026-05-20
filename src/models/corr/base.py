from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BaseCorrModel(ABC):
    @abstractmethod
    def fit(
        self,
        train_returns: pd.DataFrame,
        kg_data=None,
        valid_returns: pd.DataFrame = None
    ):
        pass

    @abstractmethod
    def predict_corr(self, hist_returns: pd.DataFrame, kg_data=None) -> np.ndarray:
        pass

    def get_diagnostics(self) -> dict:
        return {}
