from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from ..base import Step
from ppm_preprocessing.domain.context import PipelineContext


@dataclass
class ModelSelectionConfig:
    input_key: str = "encoded_buckets"
    output_key: str = "model_selection"
    val_ratio: float = 0.2
    random_state: int = 42
    min_samples_per_bucket: int = 500  # skip tiny buckets


class ModelSelectionStep(Step):
    name = "model_selection"

    def __init__(self, config: ModelSelectionConfig | None = None):
        self.config = config or ModelSelectionConfig()

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if self.config.input_key not in ctx.artifacts:
            raise RuntimeError("Encoded buckets missing. Run EncodingStep first.")

        encoded = ctx.artifacts[self.config.input_key]
        results: Dict[int, Any] = {}

        rng = np.random.default_rng(self.config.random_state)

        for bucket_id, bucket_data in encoded.items():
            best_score = -1.0
            best_conf = None

            for enc_name, data in bucket_data["datasets"].items():
                X = data["X"]
                y = data["y"]
                case_ids = data["case_id"]

                if len(y) < self.config.min_samples_per_bucket:
                    continue

                # --- Case-level split ---
                unique_cases = np.unique(case_ids)
                rng.shuffle(unique_cases)

                split = int(len(unique_cases) * (1.0 - self.config.val_ratio))
                train_cases = set(unique_cases[:split])
                val_cases = set(unique_cases[split:])

                train_idx = np.array([c in train_cases for c in case_ids])
                val_idx = np.array([c in val_cases for c in case_ids])

                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]

                if len(y_val) == 0 or len(y_train) == 0:
                    continue

                # --- Train model ---
                clf = LogisticRegression(
                    max_iter=500,
                    solver="lbfgs",
                    n_jobs=1,
                )

                clf.fit(X_train, y_train)

                y_pred = clf.predict(X_val)
                score = f1_score(y_val, y_pred, average="macro")

                if score > best_score:
                    best_score = score
                    best_conf = {
                        "bucket_id": bucket_id,
                        "encoding": enc_name,
                        "model": "LogisticRegression",
                        "f1_macro": float(score),
                        "num_train": int(len(y_train)),
                        "num_val": int(len(y_val)),
                    }

            if best_conf is not None:
                results[bucket_id] = best_conf

        ctx.artifacts[self.config.output_key] = results
        return ctx
