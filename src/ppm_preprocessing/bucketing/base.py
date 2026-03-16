from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd

class Bucketer(ABC):
    name: str = "bucketer"

    def fit(self, prefixes_train: pd.DataFrame) -> "Bucketer":
        """
        Optional: learn parameters from train prefixes (stateful bucketers).
        Default is no-op for stateless bucketers.
        """
        return self

    @abstractmethod
    def assign(self, prefixes: pd.DataFrame) -> pd.Series:
        """
        Return a pd.Series of bucket_id aligned with prefixes.index.
        """
        raise NotImplementedError
