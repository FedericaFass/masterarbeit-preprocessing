from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np

from ppm_preprocessing.steps.base import Step
from ppm_preprocessing.domain.context import PipelineContext
from ppm_preprocessing.steps.bucketing import BucketingStep
from ppm_preprocessing.steps.encoding import EncodingStep
from ppm_preprocessing.steps.train_evaluate_task import TrainEvaluateTaskStep
from ppm_preprocessing.tasks.specs import TaskSpec


@dataclass
class CompareTasksConfig:
    tasks: Dict[str, TaskSpec]              # e.g. default_task_specs()
    bucketers: Dict[str, Any]               # {"no_bucket": NoBucketer(), "prefix_len_bins": PrefixLenBinsBucketer(...) }
    encodings: List[str]                    # ["last_n_5","last_n_10","aggregation"]
    min_bucket_samples: int = 1000          # per-bucket: skip tiny buckets
    skip_single_class: bool = True          # classification: skip train with <2 classes


class CompareTasksBucketingEncodingStep(Step):
    """
    Compare tasks × bucketing × encoding with FIXED (non-AutoML) models via TrainEvaluateTaskStep.

    Requirements:
      - ctx.artifacts["prefix_samples"] exists and contains label columns required by TaskSpec.label_col
      - ctx.artifacts["case_splits"] exists (train/val/test case id lists)
      - EncodingStep stores row_idx for each encoded row so we can map task-specific labels:
           encoded_buckets[bucket_id]["datasets"][encoding]["row_idx"]  -> indices into prefix_samples rows
    """
    name = "compare_tasks_bucketing_encoding"

    def __init__(self, config: CompareTasksConfig):
        self.config = config

    def _ensure_labels_present(self, ctx: PipelineContext, task: TaskSpec) -> None:
        ps = ctx.artifacts.get("prefix_samples")
        if ps is None:
            raise RuntimeError("prefix_samples missing. Run PrefixExtractionStep first.")
        if task.label_col not in ps.columns:
            raise RuntimeError(
                f"Label column '{task.label_col}' missing in prefix_samples. "
                f"Add it in PrefixExtractionStep (or add a dedicated labeling step)."
            )

    def _set_task_spec(self, ctx: PipelineContext, task: TaskSpec) -> None:
        # TrainEvaluateTaskStep reads this
        ctx.artifacts["task_spec"] = task

    @staticmethod
    def _as_str_array(x: np.ndarray) -> np.ndarray:
        return x.astype(str)

    def _is_classification(self, task: TaskSpec) -> bool:
        # adjust if your TaskSpec uses different naming
        t = getattr(task, "task_type", "")
        return "classification" in str(t).lower()

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if "case_splits" not in ctx.artifacts:
            raise RuntimeError("case_splits missing. Run CaseSplitStep first.")
        if "prefix_samples" not in ctx.artifacts:
            raise RuntimeError("prefix_samples missing. Run PrefixExtractionStep first.")

        ps = ctx.artifacts["prefix_samples"]

        splits = ctx.artifacts["case_splits"]
        train_cases = set(map(str, splits.get("train", [])))
        val_cases = set(map(str, splits.get("val", [])))

        results: List[Dict[str, Any]] = []

        for task_name, task in self.config.tasks.items():
            # 1) labels exist?
            self._ensure_labels_present(ctx, task)
            y_all = ps[task.label_col].to_numpy()

            # 2) make TaskSpec available to TrainEvaluateTaskStep
            self._set_task_spec(ctx, task)

            for bucketing_name, bucketer in self.config.bucketers.items():
                # 3) bucketing
                BucketingStep(bucketer=bucketer).run(ctx)

                # 4) encoding
                EncodingStep().run(ctx)
                encoded = ctx.artifacts["encoded_buckets"]

                for encoding_name in self.config.encodings:
                    if bucketing_name == "no_bucket":
                        # ---------- GLOBAL MODEL: merge all buckets ----------
                        Xs, ys, cs = [], [], []

                        for b in encoded.values():
                            ds = b["datasets"].get(encoding_name)
                            if ds is None:
                                continue

                            if "row_idx" not in ds:
                                raise RuntimeError(
                                    "Encoded dataset missing 'row_idx'. "
                                    "Please store row indices in EncodingStep so labels can be mapped for each task."
                                )

                            Xs.append(ds["X"])
                            cs.append(self._as_str_array(ds["case_id"]))
                            ys.append(y_all[ds["row_idx"]])

                        if not Xs:
                            results.append({
                                "task": task_name,
                                "bucketing": bucketing_name,
                                "encoding": encoding_name,
                                "mode": "global_model",
                                "primary_metric": getattr(task, "primary_metric", None),
                                "primary_score": None,
                                "metrics": None,
                                "details": {"note": "no data produced"},
                            })
                            continue

                        X = np.vstack(Xs)
                        y = np.concatenate(ys)
                        case_ids = np.concatenate(cs)

                        # guardrail: classification needs >=2 train classes
                        if self.config.skip_single_class and self._is_classification(task):
                            uniq_train = np.unique(y[[c in train_cases for c in case_ids]])
                            if len(uniq_train) < 2:
                                results.append({
                                    "task": task_name,
                                    "bucketing": bucketing_name,
                                    "encoding": encoding_name,
                                    "mode": "global_model",
                                    "primary_metric": getattr(task, "primary_metric", None),
                                    "primary_score": None,
                                    "metrics": None,
                                    "details": {"note": "skipped: train has <2 classes"},
                                })
                                continue

                        ctx.artifacts["encoded_data"] = {"X": X, "y": y, "case_id": case_ids}
                        TrainEvaluateTaskStep().run(ctx)
                        ev = ctx.artifacts["evaluation"]

                        results.append({
                            "task": task_name,
                            "bucketing": bucketing_name,
                            "encoding": encoding_name,
                            "mode": "global_model",
                            "primary_metric": ev.get("primary_metric"),
                            "primary_score": ev.get("primary_score"),
                            "metrics": ev.get("metrics"),
                            "details": {"train": ev.get("num_train"), "val": ev.get("num_val")},
                        })

                    else:
                        # ---------- PER-BUCKET MODELS ----------
                        bucket_scores = []
                        bucket_weights = []
                        used_buckets = 0

                        for bucket_id, b in encoded.items():
                            ds = b["datasets"].get(encoding_name)
                            if ds is None:
                                continue

                            if "row_idx" not in ds:
                                raise RuntimeError(
                                    "Encoded dataset missing 'row_idx'. "
                                    "Please store row indices in EncodingStep so labels can be mapped for each task."
                                )

                            X = ds["X"]
                            case_ids = self._as_str_array(ds["case_id"])
                            y = y_all[ds["row_idx"]]

                            # train/val indices by case_id
                            train_idx = np.array([c in train_cases for c in case_ids], dtype=bool)
                            val_idx = np.array([c in val_cases for c in case_ids], dtype=bool)

                            n_tr = int(train_idx.sum())
                            n_va = int(val_idx.sum())

                            if n_tr < self.config.min_bucket_samples or n_va < self.config.min_bucket_samples:
                                continue

                            # guardrail: classification needs >=2 train classes within this bucket
                            if self.config.skip_single_class and self._is_classification(task):
                                uniq_train = np.unique(y[train_idx])
                                if len(uniq_train) < 2:
                                    continue

                            ctx.artifacts["encoded_data"] = {"X": X, "y": y, "case_id": case_ids}
                            TrainEvaluateTaskStep().run(ctx)
                            ev = ctx.artifacts["evaluation"]

                            score = ev.get("primary_score")
                            if score is None:
                                continue

                            bucket_scores.append(float(score))
                            bucket_weights.append(n_va)  # weight by val size
                            used_buckets += 1

                        if used_buckets == 0:
                            results.append({
                                "task": task_name,
                                "bucketing": bucketing_name,
                                "encoding": encoding_name,
                                "mode": "per_bucket_models",
                                "primary_metric": getattr(task, "primary_metric", None),
                                "primary_score": None,
                                "metrics": None,
                                "details": {"note": f"no buckets met constraints (min_bucket_samples={self.config.min_bucket_samples})"},
                            })
                        else:
                            weighted = float(np.average(bucket_scores, weights=bucket_weights))
                            results.append({
                                "task": task_name,
                                "bucketing": bucketing_name,
                                "encoding": encoding_name,
                                "mode": "per_bucket_models",
                                "primary_metric": getattr(task, "primary_metric", None),
                                "primary_score": weighted,
                                "metrics": {"weighted_primary": weighted},
                                "details": {"buckets": used_buckets, "min_bucket_samples": self.config.min_bucket_samples},
                            })

        ctx.artifacts["comparison_results"] = results
        return ctx
