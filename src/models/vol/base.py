from abc import ABC, abstractmethod
from typing import Optional, Dict
import pandas as pd
import numpy as np


class BaseVolModel(ABC):
    def __init__(self):
        self.diagnostics: Dict = {}

    @abstractmethod
    def fit(
        self,
        train_returns: pd.DataFrame,
        train_targets: pd.DataFrame,
        valid_returns: Optional[pd.DataFrame] = None,
        valid_targets: Optional[pd.DataFrame] = None,
    ):
        pass

    @abstractmethod
    def predict_vol(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Return predicted forward-horizon volatility as standard deviation vector.
        shape: (n_assets,)
        """
        pass

    def get_diagnostics(self) -> Dict:
        return self.diagnostics
