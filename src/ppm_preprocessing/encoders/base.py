from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

@dataclass
class EncodedDataset:
    X: np.ndarray
    y: np.ndarray
    meta: Dict[str, Any]

class Encoder(ABC):
    name: str = "encoder"

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "Encoder":
        """Fit encoder state (e.g., vocab)."""
        raise NotImplementedError

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> EncodedDataset:
        """Transform prefixes into fixed-shape X and y."""
        raise NotImplementedError
