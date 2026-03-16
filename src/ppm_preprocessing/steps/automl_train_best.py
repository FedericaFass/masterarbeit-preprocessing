from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from .base import Step
from ppm_preprocessing.domain.context import PipelineContext
from ppm_preprocessing.automl.base import AutoMLConfig
from ppm_preprocessing.automl.flaml_adapter import FLAMLAutoMLAdapter


@dataclass
class AutoMLTrainBestConfig:
    comparison_key: str = "comparison_results"
    encoded_key: str = "encoded_buckets"
    prefix_key: str = "prefix_samples"
    splits_key: str = "case_splits"

    # AutoML budget
    time_budget_s: int = 180
    seed: int = 42
   

    # Optional: enforce "global_model only" to avoid bucket edge-cases initially
    global_only: bool = False

    # Restrict estimators for stability/speed
    estimator_list: Optional[List[str]] = None  # e.g. ["lgbm", "xgboost", "rf", "extra_tree"]


class AutoMLTrainBestStep(Step):
    name = "automl_train_best"

    def __init__(self, config: AutoMLTrainBestConfig | None = None):
        self.config = config or AutoMLTrainBestConfig()

    def _pick_best_per_task(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Choose best strategy per task from comparison_results.
        Assumes rows contain:
          - task, bucketing, encoding, mode, primary_score
        For remaining_time lower is better; for classification higher is better.
        """
        best: Dict[str, Dict[str, Any]] = {}
        for r in results:
            task = r.get("task")
            score = r.get("primary_score")
            if task is None or score is None:
                continue

            lower_better = (task == "remaining_time")

            if task not in best:
                best[task] = r
                continue

            prev = best[task].get("primary_score")
            if prev is None:
                best[task] = r
                continue

            if lower_better:
                if score < prev:
                    best[task] = r
            else:
                if score > prev:
                    best[task] = r

        return best

    def _infer_label_col(self, task_name: str) -> str:
        if task_name == "next_activity":
            return "label_next_activity"
        if task_name == "outcome":
            return "label_outcome"
        if task_name == "remaining_time":
            return "label_remaining_time_sec"
        raise ValueError(f"Unknown task: {task_name}")

    def _infer_task_kind_metric(self, task_name: str) -> Tuple[str, str]:
        """
        returns: (task_kind, flaml_metric)
        """
        if task_name == "remaining_time":
            return "regression", "mae"
        if task_name == "outcome":
            return "multiclass", "macro_f1"
        return "multiclass", "macro_f1"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        results: List[Dict[str, Any]] = ctx.artifacts.get(self.config.comparison_key, [])
        if not results:
            raise RuntimeError(f"No comparison results found in '{self.config.comparison_key}'.")

        encoded = ctx.artifacts.get(self.config.encoded_key)
        if encoded is None:
            raise RuntimeError(f"Missing '{self.config.encoded_key}'.")

        prefix_samples: pd.DataFrame = ctx.artifacts.get(self.config.prefix_key)
        if prefix_samples is None or len(prefix_samples) == 0:
            raise RuntimeError(f"Missing/empty '{self.config.prefix_key}'.")

        splits = ctx.artifacts.get(self.config.splits_key, {})
        train_cases = set(map(str, splits.get("train", [])))
        val_cases = set(map(str, splits.get("val", [])))

        best_per_task = self._pick_best_per_task(results)

        adapter = FLAMLAutoMLAdapter()

        report: Dict[str, Any] = {
            "adapter": adapter.name,
            "best_per_task": {},
            "runs": {},
        }

        for task_name, best in best_per_task.items():
            bucketing_name = best["bucketing"]
            encoding_name = best["encoding"]
            mode = best.get("mode", "global_model")

            if self.config.global_only and mode != "global_model":
                report["runs"][task_name] = {
                    "skipped": True,
                    "reason": "global_only_enabled",
                    "strategy": {"bucketing": bucketing_name, "encoding": encoding_name, "mode": mode},
                }
                continue

            label_col = self._infer_label_col(task_name)
            task_kind, flaml_metric = self._infer_task_kind_metric(task_name)

            y_all = prefix_samples[label_col].to_numpy()

            # Determine which buckets to use
            bucket_ids = sorted(encoded.keys())
            if not bucket_ids:
                report["runs"][task_name] = {"skipped": True, "reason": "no_encoded_buckets"}
                continue

            # Global model convention: use first bucket only (no_bucket should produce 1 bucket)
            if mode == "global_model":
                bucket_ids = [bucket_ids[0]]

            X_tr_list, y_tr_list = [], []
            X_va_list, y_va_list = [], []

            # collect train/val data from selected buckets
            for b in bucket_ids:
                datasets = encoded[b].get("datasets", {})
                if encoding_name not in datasets:
                    continue

                ds = datasets[encoding_name]
                X = ds["X"]
                case_ids = ds["case_id"]
                row_idx = ds.get("row_idx")

                if row_idx is None:
                    raise RuntimeError(
                        "encoded dataset missing 'row_idx'. "
                        "Ensure your EncodingStep stores row_idx (prefix_samples row index) in every dataset."
                    )

                y_task = y_all[row_idx]

                mask_tr = np.array([str(c) in train_cases for c in case_ids], dtype=bool)
                mask_va = np.array([str(c) in val_cases for c in case_ids], dtype=bool)

                if mask_tr.any():
                    X_tr_list.append(X[mask_tr])
                    y_tr_list.append(y_task[mask_tr])

                if mask_va.any():
                    X_va_list.append(X[mask_va])
                    y_va_list.append(y_task[mask_va])

            if not X_tr_list or not X_va_list:
                report["runs"][task_name] = {
                    "skipped": True,
                    "reason": "empty_train_or_val_after_split",
                    "strategy": {"bucketing": bucketing_name, "encoding": encoding_name, "mode": mode},
                }
                continue

            X_train = np.vstack(X_tr_list)
            y_train = np.concatenate(y_tr_list)
            X_val = np.vstack(X_va_list)
            y_val = np.concatenate(y_va_list)

            # --- Guardrails: classification needs >=2 classes ---
            if task_kind in ("binary", "multiclass"):
                uniq_train = np.unique(y_train)
                uniq_val = np.unique(y_val)

                if len(uniq_train) < 2:
                    report["runs"][task_name] = {
                        "skipped": True,
                        "reason": "train_has_single_class",
                        "unique_train_labels": uniq_train.tolist(),
                        "shapes": {"X_train": list(X_train.shape), "X_val": list(X_val.shape)},
                        "strategy": {"bucketing": bucketing_name, "encoding": encoding_name, "mode": mode},
                    }
                    continue

                # Optional: for binary tasks, if val has only one class metrics can be undefined
                if task_kind == "binary" and len(uniq_val) < 2:
                    report["runs"][task_name] = {
                        "skipped": True,
                        "reason": "val_has_single_class",
                        "unique_val_labels": uniq_val.tolist(),
                        "shapes": {"X_train": list(X_train.shape), "X_val": list(X_val.shape)},
                        "strategy": {"bucketing": bucketing_name, "encoding": encoding_name, "mode": mode},
                    }
                    continue

            cfg = AutoMLConfig(
                time_budget_s=self.config.time_budget_s,
                metric=flaml_metric,
                seed=self.config.seed,
                n_jobs=self.config.n_jobs,
                estimator_list=self.config.estimator_list,
            )

            run_info = adapter.fit_predict(
                task=task_kind, X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                config=cfg,
            )

            best_est = run_info.get("best_estimator")
            best_loss = run_info.get("best_loss")
            time_best = (run_info.get("training_log", {}) or {}).get("time_to_find_best_s")
            print(
                f"[AutoML][{task_name}] best_estimator={best_est} best_loss={best_loss} "
                f"metric={run_info.get('metric')} time_to_best_s={time_best}"
            )

            report["best_per_task"][task_name] = {
                "bucketing": bucketing_name,
                "encoding": encoding_name,
                "mode": mode,
                "primary_score": best.get("primary_score"),
            }
            report["runs"][task_name] = {
                "skipped": False,
                "automl": run_info,
                "data": {
                    "X_train": list(X_train.shape),
                    "X_val": list(X_val.shape),
                    "y_train_unique": int(len(np.unique(y_train))) if task_kind != "regression" else None,
                },
                "strategy": {"bucketing": bucketing_name, "encoding": encoding_name, "mode": mode},
            }

        ctx.artifacts["automl_report"] = report
        return ctx
