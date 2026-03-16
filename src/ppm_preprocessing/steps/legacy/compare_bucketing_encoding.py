

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np

from ..base import Step
from ppm_preprocessing.domain.context import PipelineContext

from ppm_preprocessing.steps.bucketing import BucketingStep
from ppm_preprocessing.steps.encoding import EncodingStep
from ppm_preprocessing.steps.legacy.train_evaluate import TrainEvaluateStep


@dataclass
class CompareConfig:
    bucketers: Dict[str, Any]
    encodings: List[str]  # ["last_n_5","last_n_10","aggregation"]


class CompareBucketingEncodingStep(Step):
    """
    Compares (bucketing × encoding) combinations
    using ONE fixed model (LogisticRegression).
    """
    name = "compare_bucketing_encoding"

    def __init__(self, config: CompareConfig):
        self.config = config

    def run(self, ctx: PipelineContext) -> PipelineContext:
        results: List[Dict[str, Any]] = []

        for bucketing_name, bucketer in self.config.bucketers.items():
            # --- Bucketing ---
            BucketingStep(bucketer=bucketer).run(ctx)

            for encoding_name in self.config.encodings:
                # --- Encoding ---
                EncodingStep().run(ctx)

                encoded = ctx.artifacts["encoded_buckets"]

                # merge all buckets into ONE dataset
                Xs, ys, cases = [], [], []

                for b in encoded.values():
                    ds = b["datasets"][encoding_name]
                    Xs.append(ds["X"])
                    ys.append(ds["y"])
                    cases.append(ds["case_id"])

                ctx.artifacts["encoded_data"] = {
                    "X": np.vstack(Xs),
                    "y": np.concatenate(ys),
                    "case_id": np.concatenate(cases),
                }

                # --- Train & evaluate ---
                TrainEvaluateStep().run(ctx)

                eval_res = ctx.artifacts["evaluation"]

                results.append({
                    "bucketing": bucketing_name,
                    "encoding": encoding_name,
                    "metric": eval_res["metric"],
                    "score": eval_res["score"],
                    "num_train": eval_res["num_train"],
                    "num_val": eval_res["num_val"],
                })

        ctx.artifacts["comparison_results"] = results
        return ctx
