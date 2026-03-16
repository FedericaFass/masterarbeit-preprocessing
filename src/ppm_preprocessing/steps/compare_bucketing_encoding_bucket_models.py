import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from .base import Step
from ppm_preprocessing.domain.context import PipelineContext
from ppm_preprocessing.steps.bucketing import BucketingStep
from ppm_preprocessing.steps.encoding import EncodingStep


@dataclass
class CompareConfig:
    bucketers: Dict[str, Any]                 # {"no_bucket": NoBucketer(), "prefix_length": PrefixLengthBucketer()}
    encodings: List[str]                      # ["last_n_5","last_n_10","aggregation"]
    metric: str = "f1_macro"
    min_bucket_samples: int = 200             # skip tiny buckets
    max_iter: int = 2000


class CompareBucketingEncodingBucketModelsStep(Step):
    """
    Correct PPM-style bucketing comparison:
    - no_bucket => one global model
    - prefix_length => one model per bucket_id (e.g., per prefix length)
    Aggregates results via weighted average across buckets (by validation sample count).
    """
    name = "compare_bucketing_encoding_bucket_models"

    def __init__(self, config: CompareConfig):
        self.config = config

    def _make_clf(self):
        return make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(solver="saga", max_iter=self.config.max_iter),
        )

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if "prefix_samples" not in ctx.artifacts:
            raise RuntimeError("prefix_samples missing. Run PrefixExtractionStep first.")
        if "case_splits" not in ctx.artifacts:
            raise RuntimeError("case_splits missing. Run CaseSplitStep first.")

        splits = ctx.artifacts["case_splits"]
        train_cases = splits["train"]
        val_cases = splits["val"]

        results: List[Dict[str, Any]] = []

        for bucketing_name, bucketer in self.config.bucketers.items():
            # 1) Bucketing on prefix_samples
            BucketingStep(bucketer=bucketer).run(ctx)
            bucketed_df = ctx.artifacts["bucketed_prefixes"]

            # 2) Encode once (produces datasets per bucket)
            EncodingStep().run(ctx)
            encoded = ctx.artifacts["encoded_buckets"]

            for encoding_name in self.config.encodings:
                if bucketing_name == "no_bucket":
                    # --- One global model ---
                    Xs, ys, cases = [], [], []
                    for b in encoded.values():
                        ds = b["datasets"][encoding_name]
                        Xs.append(ds["X"])
                        ys.append(ds["y"])
                        cases.append(ds["case_id"])

                    X = np.vstack(Xs)
                    y = np.concatenate(ys)
                    case_ids = np.concatenate(cases)

                    train_idx = np.array([c in train_cases for c in case_ids])
                    val_idx = np.array([c in val_cases for c in case_ids])

                    clf = self._make_clf()
                    clf.fit(X[train_idx], y[train_idx])
                    y_pred = clf.predict(X[val_idx])

                    score = f1_score(y[val_idx], y_pred, average="macro")

                    results.append({
                        "bucketing": bucketing_name,
                        "encoding": encoding_name,
                        "metric": self.config.metric,
                        "score": float(score),
                        "mode": "global_model",
                        "num_train": int(train_idx.sum()),
                        "num_val": int(val_idx.sum()),
                    })

                else:
                    # --- One model per bucket ---
                    bucket_scores = []
                    bucket_weights = []

                    for bucket_id, b in encoded.items():
                        ds = b["datasets"][encoding_name]
                        X = ds["X"]
                        y = ds["y"]
                        case_ids = ds["case_id"]

                        train_idx = np.array([c in train_cases for c in case_ids])
                        val_idx = np.array([c in val_cases for c in case_ids])

                        n_train = int(train_idx.sum())
                        n_val = int(val_idx.sum())

                        # skip very small buckets (stability)
                        if n_train < self.config.min_bucket_samples or n_val < self.config.min_bucket_samples:
                            continue

                        clf = self._make_clf()
                        clf.fit(X[train_idx], y[train_idx])
                        y_pred = clf.predict(X[val_idx])

                        s = f1_score(y[val_idx], y_pred, average="macro")

                        bucket_scores.append(float(s))
                        bucket_weights.append(n_val)

                    if not bucket_scores:
                        results.append({
                            "bucketing": bucketing_name,
                            "encoding": encoding_name,
                            "metric": self.config.metric,
                            "score": None,
                            "mode": "per_bucket_models",
                            "note": "no buckets met min_bucket_samples",
                        })
                    else:
                        # weighted avg by validation sample size
                        score = float(np.average(bucket_scores, weights=bucket_weights))
                        results.append({
                            "bucketing": bucketing_name,
                            "encoding": encoding_name,
                            "metric": self.config.metric,
                            "score": score,
                            "mode": "per_bucket_models",
                            "num_buckets_used": int(len(bucket_scores)),
                            "min_bucket_samples": int(self.config.min_bucket_samples),
                        })

        ctx.artifacts["comparison_results"] = results
        return ctx
