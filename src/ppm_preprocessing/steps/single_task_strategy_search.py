from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from ppm_preprocessing.steps.base import Step
from ppm_preprocessing.domain.context import PipelineContext
from ppm_preprocessing.steps.bucketing import BucketingStep
from ppm_preprocessing.steps.encoding import EncodingStep, EncodingConfig
from ppm_preprocessing.steps.train_evaluate_task import TrainEvaluateTaskStep
from ppm_preprocessing.tasks.specs import TaskSpec


@dataclass
class SingleTaskStrategySearchConfig:
    task: TaskSpec
    bucketers: Dict[str, Any]          # {"no_bucket": NoBucketer(), ...}
    encodings: List[str]              # ["last_state", "aggregation"]
    min_bucket_samples: int = 1000
    skip_single_class: bool = True
    use_probe_model: bool = True       # use LightGBM probe instead of Ridge/LogReg

    # Report settings
    save_report: bool = True
    save_csv: bool = True
    report_basename: str = "single_task_strategy_search"


class SingleTaskStrategySearchStep(Step):
    """
    Strategy search for ONE task:
      For each bucketer x encoding:
        - run bucketing
        - run encoding
        - evaluate

      If bucketer is 'no_bucket' -> evaluate as GLOBAL MODEL (merge all buckets)
      Else -> evaluate PER-BUCKET MODELS (train/eval per bucket, weighted by val-size)

    Output:
      ctx.artifacts["single_task_comparison"] : list[rows]
      ctx.artifacts["best_strategy"] : dict winner row
    """
    name = "single_task_strategy_search"

    def __init__(self, config: SingleTaskStrategySearchConfig):
        self.config = config

    @staticmethod
    def _as_str_array(x: Any) -> np.ndarray:
        return np.asarray(x).astype(str)

    def _is_classification(self) -> bool:
        return "class" in str(self.config.task.task_type).lower()

    @staticmethod
    def _maybe_fit_bucketer_on_train(bucketer: Any, ps_train: pd.DataFrame) -> None:
        """
        Fit bucketer if it exposes .fit(). No-op for stateless bucketers.
        """
        if hasattr(bucketer, "fit") and callable(getattr(bucketer, "fit")):
            bucketer.fit(ps_train)

    def _get_reports_dir(self, ctx: PipelineContext) -> Path:
        """Get the reports directory from context or default."""
        out_dir = ctx.artifacts.get("out_dir") or getattr(ctx, "output_dir", None)
        if out_dir:
            return Path(out_dir) / "reports"
        return Path("outputs") / "reports"

    def _save_json_report(self, ctx: PipelineContext, results: List[Dict[str, Any]], best_row: Optional[Dict[str, Any]], total_time: float) -> None:
        """Save comprehensive JSON report."""
        if not self.config.save_report:
            return

        reports_dir = self._get_reports_dir(ctx)
        reports_dir.mkdir(parents=True, exist_ok=True)

        task_name = self.config.task.name
        report_path = reports_dir / f"{self.config.report_basename}__{task_name}.json"

        report = {
            "step": self.name,
            "task": task_name,
            "task_type": str(self.config.task.task_type),
            "primary_metric": self.config.task.primary_metric,
            "total_time_s": round(total_time, 2),
            "num_strategies_tested": len(results),
            "config": {
                "min_bucket_samples": self.config.min_bucket_samples,
                "skip_single_class": self.config.skip_single_class,
                "bucketers": list(self.config.bucketers.keys()),
                "encodings": self.config.encodings,
            },
            "best_strategy": best_row,
            "all_strategies": results,
            "note": "Comparison of all bucketing x encoding strategies. Best strategy chosen by primary_metric.",
        }

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        ctx.artifacts[f"{self.config.report_basename}_report_path"] = str(report_path)
        print(f"\nStrategy search report saved: {report_path}")

    def _save_csv_report(self, ctx: PipelineContext, results: List[Dict[str, Any]]) -> None:
        """Save comparison CSV for easy viewing."""
        if not self.config.save_csv or not results:
            return

        reports_dir = self._get_reports_dir(ctx)
        reports_dir.mkdir(parents=True, exist_ok=True)

        task_name = self.config.task.name
        csv_path = reports_dir / f"{self.config.report_basename}__{task_name}.csv"

        # Flatten results for CSV
        rows = []
        for r in results:
            row = {
                "task": r.get("task"),
                "bucketing": r.get("bucketing"),
                "encoding": r.get("encoding"),
                "mode": r.get("mode"),
                "primary_metric": r.get("primary_metric"),
                "primary_score": r.get("primary_score"),
                "time_s": r.get("time_s"),
                "feature_dim": r.get("feature_dim"),
            }

            # Add all metrics
            metrics = r.get("metrics", {})
            for k, v in metrics.items():
                row[f"metric_{k}"] = v

            # Add details
            details = r.get("details", {})
            for k, v in details.items():
                if isinstance(v, (int, float, str)):
                    row[f"detail_{k}"] = v

            rows.append(row)

        df = pd.DataFrame(rows)

        # Sort by primary_score (best first)
        is_reg = str(self.config.task.task_type).lower() == "regression"
        if "primary_score" in df.columns:
            df = df.sort_values("primary_score", ascending=is_reg)  # regression: lower is better

        df.to_csv(csv_path, index=False, encoding="utf-8")
        ctx.artifacts[f"{self.config.report_basename}_csv_path"] = str(csv_path)
        print(f"Strategy comparison CSV saved: {csv_path}")

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if "case_splits" not in ctx.artifacts:
            raise RuntimeError("case_splits missing. Run CaseSplitStep first.")
        if "prefix_samples" not in ctx.artifacts:
            raise RuntimeError("prefix_samples missing. Run PrefixExtractionStep first.")

        ps: pd.DataFrame = ctx.artifacts["prefix_samples"]
        task = self.config.task

        if "case_id" not in ps.columns:
            raise RuntimeError("prefix_samples missing 'case_id' column.")
        if task.label_col not in ps.columns:
            raise RuntimeError(f"Label col '{task.label_col}' missing in prefix_samples.")

        # make TaskSpec visible to TrainEvaluateTaskStep (your existing contract)
        ctx.artifacts["task_spec"] = task
        # signal whether to use LightGBM probe model for scoring
        ctx.artifacts["_use_probe_model"] = self.config.use_probe_model

        splits = ctx.artifacts["case_splits"]
        train_cases = set(map(str, splits.get("train", [])))
        val_cases = set(map(str, splits.get("val", [])))

        # train-only subset for fitting stateful bucketers
        ps_train = ps[ps["case_id"].astype(str).isin(train_cases)].copy()

        # full y array for stable indexing with row_idx
        y_all = ps[task.label_col].to_numpy()

        results: List[Dict[str, Any]] = []
        total_start_time = time.time()

        print(f"\n{'='*60}")
        print(f"STRATEGY SEARCH: {task.name}")
        print(f"{'='*60}")
        print(f"Testing {len(self.config.bucketers)} bucketers x {len(self.config.encodings)} encodings = {len(self.config.bucketers) * len(self.config.encodings)} strategies")

        for bucketing_name, bucketer in self.config.bucketers.items():
            print(f"\n{'-'*60}")
            print(f"Bucketer: {bucketing_name}")
            print(f"{'-'*60}")

            # Fit bucketer on TRAIN only (no leakage). No-op for stateless bucketers.
            self._maybe_fit_bucketer_on_train(bucketer, ps_train)

            # BucketingStep uses ctx.artifacts["prefix_samples"] and produces bucketed_prefixes
            BucketingStep(bucketer=bucketer).run(ctx)

            # EncodingStep should read bucketed_prefixes and produce encoded_buckets
            # IMPORTANT: EncodingStep must build datasets for the encodings we want.
            is_clf = "classification" in str(task.task_type)
            enc_config = EncodingConfig(
                label_col_override=task.label_col,
                label_is_numeric=not is_clf,
                use_log1p_label=not is_clf,
            )
            EncodingStep(config=enc_config).run(ctx)

            encoded = ctx.artifacts.get("encoded_buckets", {})
            if not encoded:
                raise RuntimeError("EncodingStep produced empty encoded_buckets.")

            # Get encoding QC for feature dimensions
            encoding_qc = ctx.artifacts.get("encoding_qc", {})

            # evaluate each encoding name
            for encoding_name in self.config.encodings:
                strategy_start_time = time.time()
                print(f"  Encoding: {encoding_name}...", end=" ", flush=True)
                # -----------------------------
                # GLOBAL MODEL (no_bucket)
                # -----------------------------
                if bucketing_name == "no_bucket":
                    Xs, ys, cs = [], [], []

                    for b in encoded.values():
                        ds = (b.get("datasets") or {}).get(encoding_name)
                        if ds is None:
                            continue

                        if "row_idx" not in ds or "X" not in ds or "case_id" not in ds:
                            raise RuntimeError(
                                f"Encoding dataset for '{encoding_name}' must contain keys: row_idx, X, case_id."
                            )

                        Xs.append(ds["X"])
                        cs.append(self._as_str_array(ds["case_id"]))
                        ys.append(y_all[np.asarray(ds["row_idx"])])

                    if not Xs:
                        strategy_time = time.time() - strategy_start_time
                        print(f"SKIPPED (no data) [{strategy_time:.1f}s]")
                        results.append({
                            "task": task.name,
                            "bucketing": bucketing_name,
                            "encoding": encoding_name,
                            "mode": "global_model",
                            "primary_metric": task.primary_metric,
                            "primary_score": None,
                            "time_s": round(strategy_time, 2),
                            "feature_dim": None,
                            "metrics": {},
                            "details": {"note": "no data for encoding"},
                        })
                        continue

                    X = np.vstack(Xs)
                    y = np.concatenate(ys)
                    case_ids = np.concatenate(cs)

                    # Guardrail: classification must have >= 2 classes in train
                    if self.config.skip_single_class and self._is_classification():
                        train_mask = np.array([c in train_cases for c in case_ids], dtype=bool)
                        y_train = y[train_mask]
                        if len(np.unique(y_train)) < 2:
                            strategy_time = time.time() - strategy_start_time
                            print(f"SKIPPED (<2 classes) [{strategy_time:.1f}s]")
                            results.append({
                                "task": task.name,
                                "bucketing": bucketing_name,
                                "encoding": encoding_name,
                                "mode": "global_model",
                                "primary_metric": task.primary_metric,
                                "primary_score": None,
                                "time_s": round(strategy_time, 2),
                                "feature_dim": int(X.shape[1]) if X.ndim == 2 else None,
                                "metrics": {},
                                "details": {"note": "skipped: train has <2 classes"},
                            })
                            continue

                    ctx.artifacts["encoded_data"] = {"X": X, "y": y, "case_id": case_ids}
                    TrainEvaluateTaskStep().run(ctx)
                    ev = ctx.artifacts.get("evaluation", {})
                    strategy_time = time.time() - strategy_start_time

                    # Extract all metrics
                    all_metrics = {k: v for k, v in ev.items() if k not in ["primary_metric", "primary_score", "num_train", "num_val", "model"]}

                    # Get feature dimension from encoding candidates
                    feature_dim = None
                    if encoding_qc and "candidates" in encoding_qc:
                        enc_info = encoding_qc["candidates"].get(encoding_name, {})
                        feature_dim = enc_info.get("feature_dim")

                    primary_score = ev.get("primary_score")
                    score_str = f"{primary_score:.4f}" if primary_score is not None else "None"
                    print(f"{score_str} [{strategy_time:.1f}s]")

                    results.append({
                        "task": task.name,
                        "bucketing": bucketing_name,
                        "encoding": encoding_name,
                        "mode": "global_model",
                        "primary_metric": ev.get("primary_metric"),
                        "primary_score": primary_score,
                        "time_s": round(strategy_time, 2),
                        "feature_dim": feature_dim,
                        "metrics": all_metrics,
                        "details": {
                            "train": ev.get("num_train"),
                            "val": ev.get("num_val"),
                            "model_type": str(type(ev.get("model", "")).__name__) if ev.get("model") else None,
                        },
                    })
                    continue

                # -----------------------------
                # PER-BUCKET MODELS
                # -----------------------------
                bucket_scores: List[float] = []
                bucket_weights: List[int] = []
                used_buckets = 0

                for bucket_id, b in encoded.items():
                    ds = (b.get("datasets") or {}).get(encoding_name)
                    if ds is None:
                        continue

                    if "row_idx" not in ds or "X" not in ds or "case_id" not in ds:
                        raise RuntimeError(
                            f"Encoding dataset for '{encoding_name}' must contain keys: row_idx, X, case_id."
                        )

                    X = ds["X"]
                    case_ids = self._as_str_array(ds["case_id"])
                    row_idx = np.asarray(ds["row_idx"])
                    y = y_all[row_idx]

                    train_mask = np.array([c in train_cases for c in case_ids], dtype=bool)
                    val_mask = np.array([c in val_cases for c in case_ids], dtype=bool)

                    n_tr = int(train_mask.sum())
                    n_va = int(val_mask.sum())

                    if n_tr < self.config.min_bucket_samples or n_va < self.config.min_bucket_samples:
                        continue

                    if self.config.skip_single_class and self._is_classification():
                        if len(np.unique(y[train_mask])) < 2:
                            continue

                    ctx.artifacts["encoded_data"] = {"X": X, "y": y, "case_id": case_ids}
                    TrainEvaluateTaskStep().run(ctx)
                    ev = ctx.artifacts.get("evaluation", {})

                    score = ev.get("primary_score")
                    if score is None:
                        continue

                    bucket_scores.append(float(score))
                    bucket_weights.append(n_va)
                    used_buckets += 1

                strategy_time = time.time() - strategy_start_time

                # Get feature dimension
                feature_dim = None
                if encoding_qc and "candidates" in encoding_qc:
                    enc_info = encoding_qc["candidates"].get(encoding_name, {})
                    feature_dim = enc_info.get("feature_dim")

                if used_buckets == 0:
                    print(f"SKIPPED (no buckets met constraints) [{strategy_time:.1f}s]")
                    results.append({
                        "task": task.name,
                        "bucketing": bucketing_name,
                        "encoding": encoding_name,
                        "mode": "per_bucket_models",
                        "primary_metric": task.primary_metric,
                        "primary_score": None,
                        "time_s": round(strategy_time, 2),
                        "feature_dim": feature_dim,
                        "metrics": {},
                        "details": {"note": f"no buckets met constraints (min_bucket_samples={self.config.min_bucket_samples})"},
                    })
                else:
                    weighted = float(np.average(bucket_scores, weights=bucket_weights))
                    score_str = f"{weighted:.4f}"
                    print(f"{score_str} ({used_buckets} buckets) [{strategy_time:.1f}s]")

                    results.append({
                        "task": task.name,
                        "bucketing": bucketing_name,
                        "encoding": encoding_name,
                        "mode": "per_bucket_models",
                        "primary_metric": task.primary_metric,
                        "primary_score": weighted,
                        "time_s": round(strategy_time, 2),
                        "feature_dim": feature_dim,
                        "metrics": {"weighted_score": weighted},
                        "details": {
                            "buckets": used_buckets,
                            "min_bucket_samples": self.config.min_bucket_samples,
                            "total_val_samples": sum(bucket_weights),
                        },
                    })

        # Calculate total time
        total_time = time.time() - total_start_time

        # pick best row
        is_reg = str(task.task_type).lower() == "regression"
        best_row: Optional[Dict[str, Any]] = None

        for r in results:
            s = r.get("primary_score")
            if s is None:
                continue
            if best_row is None:
                best_row = r
                continue
            prev = best_row.get("primary_score")
            if prev is None:
                best_row = r
                continue

            if is_reg:
                if s < prev:  # MAE etc.: lower is better
                    best_row = r
            else:
                if s > prev:  # F1/AUC: higher is better
                    best_row = r

        # Print summary
        print(f"\n{'='*60}")
        print(f"STRATEGY SEARCH COMPLETE")
        print(f"{'='*60}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Strategies tested: {len(results)}")

        if best_row:
            print(f"\n*** BEST STRATEGY ***")
            print(f"  Bucketing: {best_row.get('bucketing')}")
            print(f"  Encoding: {best_row.get('encoding')}")
            print(f"  Mode: {best_row.get('mode')}")
            print(f"  {best_row.get('primary_metric')}: {best_row.get('primary_score'):.4f}")
            print(f"  Time: {best_row.get('time_s'):.1f}s")
        else:
            print("\nWARNING: No valid strategy found!")

        ctx.artifacts["single_task_comparison"] = results
        ctx.artifacts["best_strategy"] = best_row

        # Save reports
        self._save_json_report(ctx, results, best_row, total_time)
        self._save_csv_report(ctx, results)

        return ctx
