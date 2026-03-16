from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Literal
import numpy as np

TaskKind = Literal["multiclass", "binary", "regression"]


@dataclass
class AutoMLConfig:
    time_budget_s: int = 120
    metric: str = "macro_f1"   # FLAML metric name (macro_f1, roc_auc, mae, ...)
    seed: int = 42
    n_jobs: int = -1
    estimator_list: Optional[list[str]] = None  # e.g. ["lgbm","xgboost","rf"]


class AutoMLAdapter:
    name: str = "base"

    def fit_predict(
        self,
        task: TaskKind,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: AutoMLConfig,
    ) -> Dict[str, Any]:
        raise NotImplementedError
