"""
Test evaluation step for single-task models.

Evaluates the trained model on the held-out test set to get unbiased performance metrics.
Supports both regression (MAE, Median AE) and classification (F1, accuracy).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

from .base import Step
from ppm_preprocessing.domain.context import PipelineContext
from ppm_preprocessing.tasks.specs import TaskSpec


@dataclass
class SingleTaskTestEvaluationConfig:
    """Configuration for test evaluation."""
    automl_artifact_key: str = "single_task_automl"
    models_artifact_key: str = "single_task_models"
    output_key: str = "single_task_test_evaluation"
    save_json: bool = True
    n_example_predictions: int = 20


class SingleTaskTestEvaluationStep(Step):
    """
    Evaluate the trained model on the held-out test set.

    This step provides an unbiased estimate of model performance by evaluating
    on data that was never used for:
    - Strategy selection
    - Hyperparameter tuning
    - Model selection

    The test set is truly held-out and only evaluated once at the very end.

    Outputs:
    - Global metrics (MAE for regression, F1 for classification)
    - Per-bucket metrics
    - Example predictions
    """

    name = "test_evaluation"

    def __init__(self, config: SingleTaskTestEvaluationConfig, task: TaskSpec):
        self.config = config
        self.task = task

    @staticmethod
    def _postprocess_predictions(
        y_pred: np.ndarray,
        target_log1p: bool,
        clamp_nonnegative: bool,
    ) -> np.ndarray:
        """Apply inverse target transform to raw predictions (regression only)."""
        y_pred = np.asarray(y_pred, dtype=np.float64)
        if target_log1p:
            y_pred = np.expm1(y_pred)
        if clamp_nonnegative:
            y_pred = np.maximum(y_pred, 0.0)
        return y_pred

    def _is_classification(self) -> bool:
        return "classification" in self.task.task_type

    def run(self, ctx: PipelineContext) -> PipelineContext:
        c = self.config
        is_clf = self._is_classification()

        print(f"\n{'='*60}")
        print(f"TEST SET EVALUATION - Unbiased Performance Estimate")
        print(f"{'='*60}")

        # Check if we have the required artifacts
        if c.automl_artifact_key not in ctx.artifacts:
            print(f"Warning: {c.automl_artifact_key} not found. Skipping test evaluation.")
            ctx.artifacts[c.output_key] = {"error": "automl_artifact_missing", "note": "Test evaluation requires training to complete first"}
            return ctx

        if c.models_artifact_key not in ctx.artifacts:
            print(f"Warning: {c.models_artifact_key} not found. Skipping test evaluation.")
            ctx.artifacts[c.output_key] = {"error": "models_artifact_missing", "note": "Test evaluation requires trained models"}
            return ctx

        automl_result = ctx.artifacts[c.automl_artifact_key]
        models = ctx.artifacts[c.models_artifact_key]

        # Extract strategy information
        strategy = automl_result.get("strategy", {})
        best_bucketing = strategy.get("bucketing")
        best_encoding = strategy.get("encoding")
        mode = strategy.get("mode")

        # Extract target transform settings
        target_log1p = bool(automl_result.get("target_log1p", False))
        clamp_nonnegative = bool(automl_result.get("clamp_nonnegative", True))

        print(f"Best strategy: {best_bucketing} + {best_encoding} ({mode})")
        if not is_clf:
            print(f"Target transform: log1p={target_log1p}, clamp_nonneg={clamp_nonnegative}")

        # Get case splits
        if "case_splits" not in ctx.artifacts:
            print("Warning: case_splits not found. Skipping test evaluation.")
            ctx.artifacts[c.output_key] = {"error": "case_splits_missing"}
            return ctx

        splits = ctx.artifacts["case_splits"]
        test_cases = set(map(str, splits.get("test", [])))

        if not test_cases:
            print("Warning: No test cases found.")
            ctx.artifacts[c.output_key] = {"error": "no_test_cases"}
            return ctx

        print(f"Test cases: {len(test_cases)}")

        # Get encoded buckets (these should exist from AutoML training)
        if "encoded_buckets" not in ctx.artifacts:
            print("Warning: encoded_buckets not found. Skipping test evaluation.")
            ctx.artifacts[c.output_key] = {"error": "encoded_buckets_missing"}
            return ctx

        encoded = ctx.artifacts["encoded_buckets"]

        # Get prefix samples for labels
        if "prefix_samples" not in ctx.artifacts:
            print("Warning: prefix_samples not found. Skipping test evaluation.")
            ctx.artifacts[c.output_key] = {"error": "prefix_samples_missing"}
            return ctx

        ps = ctx.artifacts["prefix_samples"]
        y_all = ps[self.task.label_col].to_numpy()

        # Collect test data from encoded buckets
        X_test_list = []
        y_test_list = []
        case_ids_test_list = []
        bucket_ids_test_list = []
        row_idx_test_list = []

        for bucket_id, bucket_data in encoded.items():
            ds = bucket_data.get("datasets", {}).get(best_encoding)
            if ds is None:
                continue

            X = ds["X"]
            case_ids = np.asarray(ds["case_id"]).astype(str)
            row_idx = np.asarray(ds["row_idx"], dtype=np.int64)

            # Filter for test cases
            test_mask = np.array([cid in test_cases for cid in case_ids])
            if test_mask.sum() == 0:
                continue

            X_test_list.append(X[test_mask])
            y_test_list.append(y_all[row_idx[test_mask]])
            case_ids_test_list.append(case_ids[test_mask])
            bucket_ids_test_list.append(np.full(test_mask.sum(), bucket_id))
            row_idx_test_list.append(row_idx[test_mask])

        if not X_test_list:
            print("Warning: No test data found in encoded buckets.")
            ctx.artifacts[c.output_key] = {"error": "no_test_data"}
            return ctx

        X_test = np.vstack(X_test_list)
        y_test = np.concatenate(y_test_list)
        bucket_ids_test = np.concatenate(bucket_ids_test_list)
        case_ids_test = np.concatenate(case_ids_test_list)
        row_idx_test = np.concatenate(row_idx_test_list)

        # For classification, coerce labels to string (avoid None values)
        if is_clf:
            y_test = np.array([str(v) if v is not None else "unknown" for v in y_test])

        print(f"Test samples: {len(y_test)}")
        print(f"Test features shape: {X_test.shape}")

        # -----------------------------------------------
        # Make predictions
        # -----------------------------------------------
        # For classification, predictions are strings; for regression, floats
        if is_clf:
            y_pred_raw = np.empty(len(y_test), dtype=object)
        else:
            y_pred_raw = np.zeros(len(y_test), dtype=np.float64)

        if mode == "per_bucket_models":
            available_bids = sorted(models.keys())
            for bid in np.unique(bucket_ids_test):
                mask = bucket_ids_test == bid
                model_data = models.get(str(bid))
                if model_data is None or model_data.get("automl") is None:
                    # Fallback: use closest available bucket model
                    fallback_bid = min(available_bids, key=lambda k: abs(int(k) - int(bid)), default=None)
                    if fallback_bid is not None:
                        model_data = models[fallback_bid]
                        print(f"Warning: No model for bucket {bid}, falling back to bucket {fallback_bid}")
                    else:
                        print(f"Warning: No model found for bucket {bid} and no fallback available")
                        continue

                automl_obj = model_data.get("automl")
                if automl_obj is None:
                    continue

                y_pred_raw[mask] = automl_obj.predict(X_test[mask])
        else:
            # Global model
            model_data = models.get("global")
            if model_data is None:
                print("Warning: No global model found.")
                ctx.artifacts[c.output_key] = {"error": "no_global_model"}
                return ctx

            automl_obj = model_data.get("automl")
            if automl_obj is None:
                print("Warning: AutoML object not found in global model.")
                ctx.artifacts[c.output_key] = {"error": "no_automl_object"}
                return ctx

            y_pred_raw = automl_obj.predict(X_test)

        # Post-process predictions (regression only)
        if is_clf:
            # Fill any None entries from skipped buckets
            y_pred_raw = np.array([str(v) if v is not None else "unknown" for v in y_pred_raw])
            y_pred = y_pred_raw
        else:
            y_pred = self._postprocess_predictions(y_pred_raw, target_log1p, clamp_nonnegative)

        # -----------------------------------------------
        # Global metrics
        # -----------------------------------------------
        test_metrics_dict = self.task.evaluate(y_true=y_test, y_pred=y_pred)

        # -----------------------------------------------
        # Per-bucket metrics
        # -----------------------------------------------
        per_bucket_metrics: Dict[str, Any] = {}
        unique_buckets = sorted(np.unique(bucket_ids_test))

        for bid in unique_buckets:
            mask = bucket_ids_test == bid
            bucket_y_true = y_test[mask]
            bucket_y_pred = y_pred[mask]
            n_samples = int(mask.sum())

            if is_clf:
                bucket_acc = float(accuracy_score(bucket_y_true, bucket_y_pred))
                bucket_f1 = float(f1_score(bucket_y_true, bucket_y_pred, average="macro", zero_division=0))
                per_bucket_metrics[str(bid)] = {
                    "accuracy": bucket_acc,
                    "f1_macro": bucket_f1,
                    "n_samples": n_samples,
                }
                print(f"  Bucket {bid}: {n_samples} samples, "
                      f"Accuracy = {bucket_acc:.3f}, F1 macro = {bucket_f1:.3f}")
            else:
                bucket_y_true_f = bucket_y_true.astype(float)
                bucket_y_pred_f = bucket_y_pred.astype(float)
                abs_errors = np.abs(bucket_y_true_f - bucket_y_pred_f)
                bucket_mae = float(np.mean(abs_errors))
                bucket_median_ae = float(np.median(abs_errors))
                per_bucket_metrics[str(bid)] = {
                    "mae_sec": bucket_mae,
                    "mae_days": bucket_mae / 86400.0,
                    "median_ae_sec": bucket_median_ae,
                    "median_ae_days": bucket_median_ae / 86400.0,
                    "n_samples": n_samples,
                }
                print(f"  Bucket {bid}: {n_samples} samples, "
                      f"MAE = {bucket_mae:.2f}s ({bucket_mae / 86400:.2f} days), "
                      f"MedAE = {bucket_median_ae:.2f}s")

        # -----------------------------------------------
        # Example predictions
        # -----------------------------------------------
        if is_clf:
            example_predictions = self._select_example_predictions_classification(
                y_test=y_test, y_pred=y_pred,
                bucket_ids=bucket_ids_test, case_ids=case_ids_test,
                row_idx=row_idx_test, ps=ps,
                n_examples=c.n_example_predictions,
            )
        else:
            example_predictions = self._select_example_predictions_regression(
                y_test=y_test, y_pred=y_pred,
                bucket_ids=bucket_ids_test, case_ids=case_ids_test,
                row_idx=row_idx_test, ps=ps,
                n_examples=c.n_example_predictions,
            )

        # Build result
        result = {
            "task": self.task.name,
            "task_type": self.task.task_type,
            "best_bucketing": best_bucketing,
            "best_encoding": best_encoding,
            "mode": mode,
            "target_log1p": target_log1p,
            "clamp_nonnegative": clamp_nonnegative,
            "test_samples": len(y_test),
            "test_metrics": test_metrics_dict,
            "per_bucket_metrics": per_bucket_metrics,
            "example_predictions": example_predictions,
            "note": (
                "Test set was held-out and never used for encoder fitting, "
                "strategy selection, or hyperparameter tuning. "
                "Encoders were fit on train data only (no leakage)."
            ),
        }

        ctx.artifacts[c.output_key] = result

        # Print results
        print(f"\n{'='*60}")
        print(f"TEST RESULTS (GLOBAL)")
        print(f"{'='*60}")

        for metric_name, metric_val in test_metrics_dict.items():
            if isinstance(metric_val, (int, float)):
                print(f"  {metric_name:20s}: {metric_val:,.4f}")
            else:
                print(f"  {metric_name:20s}: {metric_val}")

        if example_predictions:
            print(f"\n{'='*60}")
            print(f"EXAMPLE PREDICTIONS (test set)")
            print(f"{'='*60}")
            if is_clf:
                print(f"  {'Case ID':<16} {'Bucket':>6} {'PrefLen':>7} "
                      f"{'y_true':<25} {'y_pred':<25} {'Correct':>7}")
                print(f"  {'-'*90}")
                for ex in example_predictions[:10]:
                    correct = "Y" if ex["correct"] else "N"
                    print(f"  {str(ex['case_id']):<16} {ex['bucket_id']:>6} {ex['prefix_len']:>7} "
                          f"{str(ex['y_true']):<25} {str(ex['y_pred']):<25} {correct:>7}")
            else:
                print(f"  {'Case ID':<16} {'Bucket':>6} {'PrefLen':>7} "
                      f"{'y_true(s)':>12} {'y_pred(s)':>12} {'AbsErr(s)':>12}")
                print(f"  {'-'*67}")
                for ex in example_predictions[:10]:
                    print(f"  {str(ex['case_id']):<16} {ex['bucket_id']:>6} {ex['prefix_len']:>7} "
                          f"{ex['y_true_sec']:>12.1f} {ex['y_pred_sec']:>12.1f} {ex['abs_error_sec']:>12.1f}")

        # Save to JSON if requested
        if c.save_json:
            out_dir = Path(ctx.artifacts.get("out_dir", "outputs/single_task"))
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "test_evaluation.json"
            out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
            print(f"\nTest evaluation saved to {out_path}")

        print(f"{'='*60}\n")

        return ctx

    def _select_example_predictions_regression(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        bucket_ids: np.ndarray,
        case_ids: np.ndarray,
        row_idx: np.ndarray,
        ps: pd.DataFrame,
        n_examples: int,
    ) -> List[Dict[str, Any]]:
        """Pick representative example predictions at various quantiles of y_true."""
        if n_examples <= 0 or len(y_test) == 0:
            return []

        y_true_f = y_test.astype(float)
        y_pred_f = y_pred.astype(float)

        # Pick examples at quantiles for representativeness
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        picked_indices: List[int] = []
        used: set = set()

        for q in quantiles:
            target_val = float(np.quantile(y_true_f, q))
            closest_idx = int(np.argmin(np.abs(y_true_f - target_val)))
            if closest_idx not in used:
                picked_indices.append(closest_idx)
                used.add(closest_idx)

        # Fill remaining with random diverse samples
        if len(picked_indices) < n_examples:
            rng = np.random.RandomState(42)
            remaining = [i for i in range(len(y_test)) if i not in used]
            if remaining:
                extra = rng.choice(
                    remaining,
                    size=min(n_examples - len(picked_indices), len(remaining)),
                    replace=False,
                )
                picked_indices.extend(extra.tolist())

        examples: List[Dict[str, Any]] = []
        for idx in picked_indices[:n_examples]:
            ridx = int(row_idx[idx])
            prefix_len = int(ps.loc[ridx, "prefix_len"]) if "prefix_len" in ps.columns else 0
            examples.append({
                "case_id": str(case_ids[idx]),
                "bucket_id": int(bucket_ids[idx]),
                "prefix_len": prefix_len,
                "y_true_sec": float(y_true_f[idx]),
                "y_pred_sec": float(y_pred_f[idx]),
                "abs_error_sec": float(abs(y_true_f[idx] - y_pred_f[idx])),
                "y_true_days": float(y_true_f[idx]) / 86400.0,
                "y_pred_days": float(y_pred_f[idx]) / 86400.0,
            })

        return examples

    def _select_example_predictions_classification(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        bucket_ids: np.ndarray,
        case_ids: np.ndarray,
        row_idx: np.ndarray,
        ps: pd.DataFrame,
        n_examples: int,
    ) -> List[Dict[str, Any]]:
        """Pick example predictions, prioritizing misclassified samples."""
        if n_examples <= 0 or len(y_test) == 0:
            return []

        wrong_mask = y_test != y_pred
        wrong_indices = np.where(wrong_mask)[0]
        correct_indices = np.where(~wrong_mask)[0]

        rng = np.random.RandomState(42)
        picked_indices: List[int] = []

        # Pick up to half from wrong predictions
        n_wrong = min(n_examples // 2, len(wrong_indices))
        if n_wrong > 0:
            picked_indices.extend(rng.choice(wrong_indices, size=n_wrong, replace=False).tolist())

        # Fill rest from correct predictions
        n_correct = min(n_examples - len(picked_indices), len(correct_indices))
        if n_correct > 0:
            picked_indices.extend(rng.choice(correct_indices, size=n_correct, replace=False).tolist())

        examples: List[Dict[str, Any]] = []
        for idx in picked_indices[:n_examples]:
            ridx = int(row_idx[idx])
            prefix_len = int(ps.loc[ridx, "prefix_len"]) if "prefix_len" in ps.columns else 0
            examples.append({
                "case_id": str(case_ids[idx]),
                "bucket_id": int(bucket_ids[idx]),
                "prefix_len": prefix_len,
                "y_true": str(y_test[idx]),
                "y_pred": str(y_pred[idx]),
                "correct": bool(y_test[idx] == y_pred[idx]),
            })

        return examples
