from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..base import Step
from ppm_preprocessing.domain.context import PipelineContext


@dataclass
class TrainEvalConfig:
    encoded_key: str = "encoded_data"
    case_splits_key: str = "case_splits"
    metric: str = "f1_macro"


class TrainEvaluateStep(Step):
    """
    Trains ONE Logistic Regression model and evaluates it.
    """
    name = "train_evaluate"

    def __init__(self, config: TrainEvalConfig | None = None):
        self.config = config or TrainEvalConfig()

    def run(self, ctx: PipelineContext) -> PipelineContext:
        data = ctx.artifacts[self.config.encoded_key]
        splits = ctx.artifacts[self.config.case_splits_key]

        X = data["X"]
        y = data["y"]
        case_ids = data["case_id"]

        train_cases = splits["train"]
        val_cases = splits["val"]

        train_idx = np.array([c in train_cases for c in case_ids])
        val_idx = np.array([c in val_cases for c in case_ids])

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        clf = make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(
                solver="saga",
                max_iter=2000,
            )
        )

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)

        score = f1_score(y_val, y_pred, average="macro")

        ctx.artifacts["evaluation"] = {
            "metric": self.config.metric,
            "score": float(score),
            "num_train": int(len(y_train)),
            "num_val": int(len(y_val)),
        }
        return ctx
