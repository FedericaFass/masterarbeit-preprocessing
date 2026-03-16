from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    median_absolute_error,
)


TaskType = Literal["multiclass_classification", "binary_classification", "regression"]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    task_type: TaskType
    label_col: str

    # model params
    max_iter: int = 300
    solver: str = "saga"
    class_weight: Optional[str] = None
    ridge_alpha: float = 1.0

    # metric
    primary_metric: str = "f1_macro"

    def build_model(self) -> BaseEstimator:
        if self.task_type in ("multiclass_classification", "binary_classification"):
            return make_pipeline(
                StandardScaler(with_mean=False),
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=100,
                    class_weight=self.class_weight,
                ),
            )
        if self.task_type == "regression":
            return make_pipeline(
                StandardScaler(with_mean=False),
                Ridge(alpha=self.ridge_alpha),
            )
        raise ValueError(f"Unknown task_type: {self.task_type}")

    def build_probe_model(self) -> BaseEstimator:
        """
        Faster, tree-based probe model for strategy search.
        LightGBM with a small number of trees — much better proxy for the
        final AutoML model than Ridge/LogisticRegression, without taking long.
        """
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
        except ImportError:
            # Graceful fallback to default model if LightGBM not available
            return self.build_model()

        if self.task_type in ("multiclass_classification", "binary_classification"):
            return LGBMClassifier(
                n_estimators=100,
                num_leaves=31,
                learning_rate=0.1,
                class_weight=self.class_weight,
                random_state=42,
                verbose=-1,
            )
        if self.task_type == "regression":
            return LGBMRegressor(
                n_estimators=100,
                num_leaves=31,
                learning_rate=0.1,
                random_state=42,
                verbose=-1,
            )
        raise ValueError(f"Unknown task_type: {self.task_type}")

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        out: Dict[str, Any] = {}

        if self.task_type == "multiclass_classification":
            out["f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))
            out["f1_micro"] = float(f1_score(y_true, y_pred, average="micro"))
            out["primary"] = out[self.primary_metric] if self.primary_metric in out else out["f1_macro"]

        elif self.task_type == "binary_classification":
            classes = np.unique(y_true)
            if len(classes) <= 2:
                out["f1"] = float(f1_score(y_true, y_pred, average="binary"))
                if y_proba is not None:
                    out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
                out["primary"] = out[self.primary_metric] if self.primary_metric in out else out["f1"]
            else:
                out["f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))
                out["f1_micro"] = float(f1_score(y_true, y_pred, average="micro"))
                out["primary"] = out["f1_macro"]

        elif self.task_type == "regression":
            out["mae_sec"] = float(mean_absolute_error(y_true, y_pred))
            out["mae_days"] = float(out["mae_sec"] / 86400.0)
            out["median_ae_sec"] = float(median_absolute_error(y_true, y_pred))
            out["median_ae_days"] = float(out["median_ae_sec"] / 86400.0)

            out["primary"] = out[self.primary_metric] if self.primary_metric in out else out["mae_sec"]

        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

        return out


def _safe_attr_name(col: str) -> str:
    """Convert any column name to a safe identifier for use in label column names."""
    return col.replace(":", "_").replace(" ", "_").replace("-", "_").replace(".", "_")


def make_next_attr_task_spec(attr_col: str) -> TaskSpec:
    """
    Create a TaskSpec for predicting the next value of any event attribute.
    For attr_col='activity' this produces the same spec as the built-in 'next_activity' task.
    For attr_col='org:resource' it produces label_col='label_next_org_resource', etc.
    """
    safe = _safe_attr_name(attr_col)
    return TaskSpec(
        name=f"next_{safe}",
        task_type="multiclass_classification",
        label_col=f"label_next_{safe}",
        primary_metric="f1_macro",
        max_iter=300,
        class_weight="balanced",
    )


def default_task_specs() -> Dict[str, TaskSpec]:
    return {
        "next_activity": TaskSpec(
            name="next_activity",
            task_type="multiclass_classification",
            label_col="label_next_activity",
            primary_metric="f1_macro",
            max_iter=300,
            class_weight="balanced",
        ),
        "outcome": TaskSpec(
            name="outcome",
            task_type="multiclass_classification",
            label_col="label_outcome",
            primary_metric="f1_macro",
            max_iter=300,
            class_weight="balanced",
        ),
        "remaining_time": TaskSpec(
            name="remaining_time",
            task_type="regression",
            label_col="label_remaining_time_sec",
            primary_metric="mae_sec",
            ridge_alpha=1.0,
        ),
    }

