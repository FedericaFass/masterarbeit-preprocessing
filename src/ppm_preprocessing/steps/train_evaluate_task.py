from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
from sklearn.base import BaseEstimator

from ppm_preprocessing.steps.base import Step
from ppm_preprocessing.domain.context import PipelineContext
from ppm_preprocessing.tasks.specs import TaskSpec


@dataclass
class TrainEvaluateTaskConfig:
    encoded_key: str = "encoded_data"       # where X, y, case_id are stored
    case_splits_key: str = "case_splits"    # train/val/test case sets
    task_spec_key: str = "task_spec"        # current TaskSpec in artifacts
    use_probe_model: bool = False           # use LightGBM probe instead of default model


class TrainEvaluateTaskStep(Step):
    """
    Trains a fixed model defined by TaskSpec and evaluates on validation set.
    When use_probe_model=True, uses TaskSpec.build_probe_model() (LightGBM)
    instead of the default Ridge/LogisticRegression — better proxy for AutoML.
    """
    name = "train_evaluate_task"

    def __init__(self, config: TrainEvaluateTaskConfig | None = None):
        self.config = config or TrainEvaluateTaskConfig()

    def run(self, ctx: PipelineContext) -> PipelineContext:
        data = ctx.artifacts[self.config.encoded_key]
        splits = ctx.artifacts[self.config.case_splits_key]
        task: TaskSpec = ctx.artifacts[self.config.task_spec_key]

        X = data["X"]
        y = data["y"]
        case_ids = data["case_id"]

        train_cases = splits["train"]
        val_cases = splits["val"]

        train_idx = np.array([c in train_cases for c in case_ids])
        val_idx = np.array([c in val_cases for c in case_ids])

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        use_probe = self.config.use_probe_model or ctx.artifacts.get("_use_probe_model", False)
        model: BaseEstimator = task.build_probe_model() if use_probe else task.build_model()
        model.fit(X_train, y_train)

        # predictions
        y_pred = model.predict(X_val)

        # optional proba for binary
        y_proba: Optional[np.ndarray] = None
        if task.task_type == "binary_classification":
            # pipeline -> last step is classifier
            try:
                proba = model.predict_proba(X_val)
                # positive class probability = column 1 typically
                if proba.shape[1] == 2:
                    y_proba = proba[:, 1]
            except Exception:
                y_proba = None

        metrics = task.evaluate(y_true=y_val, y_pred=y_pred, y_proba=y_proba)

        ctx.artifacts["evaluation"] = {
            "task": task.name,
            "task_type": task.task_type,
            "label_col": task.label_col,
            "metrics": metrics,
            "primary_metric": task.primary_metric,
            "primary_score": metrics.get("primary"),
            "num_train": int(len(y_train)),
            "num_val": int(len(y_val)),
        }
        return ctx
