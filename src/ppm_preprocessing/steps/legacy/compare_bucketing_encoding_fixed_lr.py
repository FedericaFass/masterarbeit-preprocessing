import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from ppm_preprocessing.steps.base import Step
from ppm_preprocessing.domain.context import PipelineContext
from ppm_preprocessing.steps.bucketing import BucketingStep
from ppm_preprocessing.steps.encoding import EncodingStep


@dataclass
class CompareFixedLRConfig:
    # { "no_bucket": NoBucketer(), "prefix_len_bins": PrefixLenBinsBucketer(...) }
    bucketers: Dict[str, Any]
    # must match keys produced by EncodingStep: ["last_n_5","last_n_10","aggregation"]
    encodings: List[str]

    metric: str = "f1_macro"

    # training hyperparams (keep modest for speed)
    solver: str = "saga"
    max_iter: int = 300

    # skip tiny buckets for stability + speed (only relevant for per-bucket mode)
    min_bucket_samples: int = 1000


class CompareBucketingEncodingFixedLRStep(Step):
    """
    Compare bucketing×encoding using ONE fixed model family: LogisticRegression.
    - no_bucket => 1 global model per encoding
    - other bucketers => 1 model per bucket per encoding
    """
    name = "compare_bucketing_encoding_fixed_lr"

    def __init__(self, config: CompareFixedLRConfig):
        self.config = config

    def _make_clf(self):
        # StandardScaler helps LR; with_mean=False is safe for sparse/future use
        return make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(
                solver=self.config.solver,
                max_iter=self.config.max_iter,
            )
        )

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if "prefix_samples" not in ctx.artifacts:
            raise RuntimeError("Need ctx.artifacts['prefix_samples'] (run PrefixExtractionStep first).")
        if "case_splits" not in ctx.artifacts:
            raise RuntimeError("Need ctx.artifacts['case_splits'] (run CaseSplitStep first).")

        splits = ctx.artifacts["case_splits"]
        train_cases = splits["train"]
        val_cases = splits["val"]

        results: List[Dict[str, Any]] = []

        for bucketing_name, bucketer in self.config.bucketers.items():
            # 1) Bucketing
            BucketingStep(bucketer=bucketer).run(ctx)

            # 2) Encoding (produces encoded_buckets)
            EncodingStep().run(ctx)
            encoded = ctx.artifacts["encoded_buckets"]

            for encoding_name in self.config.encodings:
                if bucketing_name == "no_bucket":
                    # ---- One global dataset ----
                    Xs, ys, cs = [], [], []
                    for b in encoded.values():
                        ds = b["datasets"][encoding_name]
                        Xs.append(ds["X"])
                        ys.append(ds["y"])
                        cs.append(ds["case_id"])

                    X = np.vstack(Xs)
                    y = np.concatenate(ys)
                    case_ids = np.concatenate(cs)

                    train_idx = np.array([c in train_cases for c in case_ids])
                    val_idx = np.array([c in val_cases for c in case_ids])

                    clf = self._make_clf()
                    clf.fit(X[train_idx], y[train_idx])
                    y_pred = clf.predict(X[val_idx])

                    score = f1_score(y[val_idx], y_pred, average="macro")

                    results.append({
                        "bucketing": bucketing_name,
                        "encoding": encoding_name,
                        "mode": "global_model",
                        "metric": self.config.metric,
                        "score": float(score),
                        "num_train": int(train_idx.sum()),
                        "num_val": int(val_idx.sum()),
                    })

                else:
                    # ---- One model per bucket ----
                    bucket_scores = []
                    bucket_weights = []
                    used_buckets = 0

                    for bucket_id, b in encoded.items():
                        ds = b["datasets"][encoding_name]
                        X = ds["X"]
                        y = ds["y"]
                        case_ids = ds["case_id"]

                        train_idx = np.array([c in train_cases for c in case_ids])
                        val_idx = np.array([c in val_cases for c in case_ids])

                        n_train = int(train_idx.sum())
                        n_val = int(val_idx.sum())

                        if n_train < self.config.min_bucket_samples or n_val < self.config.min_bucket_samples:
                            continue

                        clf = self._make_clf()
                        clf.fit(X[train_idx], y[train_idx])
                        y_pred = clf.predict(X[val_idx])

                        s = f1_score(y[val_idx], y_pred, average="macro")

                        bucket_scores.append(float(s))
                        bucket_weights.append(n_val)
                        used_buckets += 1

                    if used_buckets == 0:
                        results.append({
                            "bucketing": bucketing_name,
                            "encoding": encoding_name,
                            "mode": "per_bucket_models",
                            "metric": self.config.metric,
                            "score": None,
                            "note": f"no buckets met min_bucket_samples={self.config.min_bucket_samples}",
                        })
                    else:
                        weighted = float(np.average(bucket_scores, weights=bucket_weights))
                        results.append({
                            "bucketing": bucketing_name,
                            "encoding": encoding_name,
                            "mode": "per_bucket_models",
                            "metric": self.config.metric,
                            "score": weighted,
                            "num_buckets_used": int(used_buckets),
                            "min_bucket_samples": int(self.config.min_bucket_samples),
                        })

        ctx.artifacts["comparison_results"] = results
        return ctx
