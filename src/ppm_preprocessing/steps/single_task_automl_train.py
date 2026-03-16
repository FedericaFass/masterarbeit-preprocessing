# ppm_preprocessing/steps/single_task_automl_train.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ppm_preprocessing.domain.context import PipelineContext
from ppm_preprocessing.steps.base import Step
from ppm_preprocessing.steps.bucketing import BucketingStep
from ppm_preprocessing.steps.encoding import EncodingStep, EncodingConfig
from ppm_preprocessing.tasks.specs import TaskSpec

from ppm_preprocessing.automl.base import AutoMLConfig
from ppm_preprocessing.automl.flaml_adapter import FLAMLAutoMLAdapter


@dataclass
class SingleTaskAutoMLTrainConfig:
    task_name: str = "remaining_time"

    bucketers: Dict[str, Any] | None = None
    encodings: List[str] | None = None  # must match EncodingStep keys: ["last_state","aggregation"]

    min_bucket_samples: int = 1000
    skip_single_class: bool = True

    time_budget_s: int = 300
    seed: int = 42
    n_jobs: int = -1
    estimator_list: Optional[List[str]] = None

    n_example_rows: int = 3
    examples_policy: str = "quantiles"

    # Remaining-time domain constraint:
    clamp_nonnegative: bool = True

    # Train on log1p(y), predict expm1(y_hat), evaluate in original seconds.
    target_log1p: bool = False


class SingleTaskAutoMLTrainStep(Step):
    """
    Train AutoML for the best strategy found by SingleTaskStrategySearchStep.

    Requires:
      - prefix_samples
      - case_splits
      - best_strategy  (bucketing, encoding, mode)

    Produces:
      - single_task_automl   (JSON-friendly)
      - single_task_models
      - encoded_buckets
    """
    name = "single_task_automl_train"

    def __init__(self, config: SingleTaskAutoMLTrainConfig, tasks: Dict[str, TaskSpec]):
        self.config = config
        self.tasks = tasks

        if self.config.bucketers is None:
            raise ValueError("config.bucketers must be provided")
        if self.config.encodings is None:
            raise ValueError("config.encodings must be provided")

    @staticmethod
    def _as_str_array(x: Any) -> np.ndarray:
        return np.asarray(x).astype(str)

    @staticmethod
    def _is_classification(task: TaskSpec) -> bool:
        t = getattr(task, "task_type", "")
        return "classification" in str(t).lower()

    @staticmethod
    def _infer_flaml_kind_metric(task_name: str) -> Tuple[str, str]:
        # Keep your convention; adjust if you store different task_name strings
        if task_name == "remaining_time":
            return "regression", "mae"
        if task_name == "outcome":
            return "multiclass", "macro_f1"
        return "multiclass", "macro_f1"

    def _build_split_masks(
        self,
        case_ids: np.ndarray,
        train_cases: set[str],
        val_cases: set[str],
        test_cases: set[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ci = self._as_str_array(case_ids)
        m_tr = np.array([c in train_cases for c in ci], dtype=bool)
        m_va = np.array([c in val_cases for c in ci], dtype=bool)
        m_te = np.array([c in test_cases for c in ci], dtype=bool)
        return m_tr, m_va, m_te

    @staticmethod
    def _safe_float(x: Any) -> float:
        try:
            return float(x)
        except Exception:
            return float("nan")

    def _label_qc(self, ps: pd.DataFrame, label_col: str, is_classification: bool = False) -> Dict[str, Any]:
        qc: Dict[str, Any] = {
            "label_col": label_col,
            "num_rows": int(len(ps)),
        }
        if is_classification:
            labels = ps[label_col].astype(str)
            qc["num_na"] = int(labels.isin(["", "nan", "None", "<NA>"]).sum())
            vc = labels.value_counts()
            qc["num_classes"] = int(vc.shape[0])
            qc["class_distribution"] = {str(k): int(v) for k, v in vc.head(20).items()}
            qc["most_common"] = str(vc.index[0]) if len(vc) > 0 else None
            qc["least_common"] = str(vc.index[-1]) if len(vc) > 0 else None
        else:
            y = pd.to_numeric(ps[label_col], errors="coerce")
            qc["num_na"] = int(y.isna().sum())
            qc["min"] = self._safe_float(y.min())
            qc["max"] = self._safe_float(y.max())
            if len(y.dropna()):
                qs = y.quantile([0.5, 0.9, 0.95, 0.99, 0.999]).to_dict()
                qc["quantiles"] = {str(k): self._safe_float(v) for k, v in qs.items()}
                qc["num_negative"] = int((y < 0).sum())
        return qc

    def _maybe_fit_bucketer(self, bucketer: Any, ps_train: pd.DataFrame) -> Any:
        """Fit bucketer on TRAIN only if it exposes .fit(df)."""
        if hasattr(bucketer, "fit") and callable(getattr(bucketer, "fit")):
            bucketer.fit(ps_train)
        return bucketer

    def _postprocess_regression_preds(self, y_pred: np.ndarray) -> np.ndarray:
        y_pred = np.asarray(y_pred, dtype=float)
        if self.config.target_log1p:
            y_pred = np.expm1(y_pred)
        if self.config.clamp_nonnegative:
            y_pred = np.maximum(y_pred, 0.0)
        return y_pred

    def _preprocess_regression_targets(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if self.config.target_log1p:
            y = np.maximum(y, 0.0)
            y = np.log1p(y)
        return y

    def _select_examples_quantiles(
        self,
        ps: pd.DataFrame,
        label_col: str,
        test_row_idx: np.ndarray,
        n_examples: int,
        seed: int,
    ) -> List[int]:
        if n_examples <= 0 or len(test_row_idx) == 0:
            return []

        sub = ps.loc[test_row_idx].copy()
        sub["case_id_str"] = sub["case_id"].astype(str)
        sub["prefix_len_num"] = pd.to_numeric(sub.get("prefix_len"), errors="coerce")

        sub = sub.sort_values(["case_id_str", "prefix_len_num"], kind="mergesort")
        reps = sub.groupby("case_id_str", as_index=False).tail(1)

        y = pd.to_numeric(reps[label_col], errors="coerce")
        reps = reps.loc[~y.isna()].copy()
        if len(reps) == 0:
            return []

        target_qs = [0.5]
        if n_examples >= 2:
            target_qs.append(0.9)
        if n_examples >= 3:
            target_qs.append(0.99)

        qvals = y.quantile(target_qs).to_dict()

        picked: List[int] = []
        used_cases: set[str] = set()

        for q in target_qs:
            target = float(qvals[q])
            reps["__dist"] = (pd.to_numeric(reps[label_col], errors="coerce") - target).abs()
            reps_sorted = reps.sort_values("__dist", kind="mergesort")
            chosen = None
            for _, row in reps_sorted.iterrows():
                cid = str(row["case_id"])
                if cid in used_cases:
                    continue
                chosen = int(row.name)
                used_cases.add(cid)
                break
            if chosen is not None:
                picked.append(chosen)

        if len(picked) < n_examples:
            rng = np.random.RandomState(seed)
            remaining = reps[~reps["case_id"].astype(str).isin(used_cases)]
            if len(remaining):
                fill = remaining.sample(
                    n=min(n_examples - len(picked), len(remaining)),
                    random_state=rng,
                )
                picked.extend([int(i) for i in fill.index.tolist()])

        return picked[:n_examples]

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if ctx is None:
            raise RuntimeError("single_task_automl_train received ctx=None.")

        if "case_splits" not in ctx.artifacts:
            raise RuntimeError("case_splits missing. Run CaseSplitStep first.")
        if "prefix_samples" not in ctx.artifacts:
            raise RuntimeError("prefix_samples missing. Run PrefixExtractionStep first.")
        if "best_strategy" not in ctx.artifacts:
            raise RuntimeError("best_strategy missing. Run SingleTaskStrategySearchStep first.")

        task_name = self.config.task_name
        if task_name not in self.tasks:
            raise ValueError(f"Unknown task_name={task_name}. Available: {list(self.tasks.keys())}")

        task = self.tasks[task_name]
        ps: pd.DataFrame = ctx.artifacts["prefix_samples"]

        if task.label_col not in ps.columns:
            raise RuntimeError(f"Label column '{task.label_col}' missing in prefix_samples.")
        if "case_id" not in ps.columns:
            raise RuntimeError("prefix_samples missing 'case_id' column.")

        splits = ctx.artifacts["case_splits"]
        train_cases = set(map(str, splits.get("train", [])))
        val_cases = set(map(str, splits.get("val", [])))
        test_cases = set(map(str, splits.get("test", [])))

        ps_train = ps[ps["case_id"].astype(str).isin(train_cases)]

        best = ctx.artifacts["best_strategy"]
        bucketing_name = best.get("bucketing")
        encoding_name = best.get("encoding")
        mode = best.get("mode")

        if bucketing_name not in self.config.bucketers:
            raise ValueError(f"Winner bucketing '{bucketing_name}' not in config.bucketers.")
        if encoding_name not in set(self.config.encodings):
            raise ValueError(f"Winner encoding '{encoding_name}' not in config.encodings.")
        if mode not in {"global_model", "per_bucket_models"}:
            raise ValueError(f"Unknown winner mode: {mode}")

        # --- fit bucketer on TRAIN only (no leakage) ---
        winner_bucketer = self._maybe_fit_bucketer(self.config.bucketers[bucketing_name], ps_train)

        # Bucketing -> ctx.artifacts["bucketed_prefixes"]
        ctx = BucketingStep(bucketer=winner_bucketer).run(ctx)
        if ctx is None:
            raise RuntimeError("BucketingStep returned None; each step must return ctx.")

        # Store fitted bucketer for persistence
        ctx.artifacts["fitted_bucketer"] = winner_bucketer

        # Encoding -> ctx.artifacts["encoded_buckets"]
        is_clf = self._is_classification(task)
        enc_cfg = EncodingConfig(
            bucket_col="bucket_id",
            input_key="bucketed_prefixes",
            output_key="encoded_buckets",
            use_last_state=True,
            use_aggregation=True,
            row_id_col="prefix_row_id",
            label_col_override=task.label_col,
            label_is_numeric=not is_clf,
            use_log1p_label=not is_clf,
        )
        ctx = EncodingStep(config=enc_cfg).run(ctx)
        if ctx is None:
            raise RuntimeError("EncodingStep returned None; each step must return ctx.")

        # Store the winning fitted encoder for persistence
        fitted_encoders = ctx.artifacts.get("fitted_encoders", {})
        if encoding_name in fitted_encoders:
            ctx.artifacts["fitted_encoder"] = fitted_encoders[encoding_name]

        encoded: Dict[Any, Any] = ctx.artifacts.get("encoded_buckets", {})
        if not encoded:
            raise RuntimeError("encoded_buckets empty after EncodingStep; cannot train AutoML.")

        # labels come from prefix_samples via row_idx mapping
        y_all = ps[task.label_col].to_numpy()

        flaml_task_kind, flaml_metric = self._infer_flaml_kind_metric(task_name)
        adapter = FLAMLAutoMLAdapter()

        models: Dict[str, Any] = {}
        out_json: Dict[str, Any] = {
            "task_name": task_name,
            "task_type": task.task_type,
            "strategy": {"bucketing": bucketing_name, "encoding": encoding_name, "mode": mode},
            "target_log1p": bool(self.config.target_log1p),
            "clamp_nonnegative": bool(self.config.clamp_nonnegative),
            "label_qc": self._label_qc(ps, task.label_col, is_classification="classification" in str(task.task_type)),
            "mode_results": {},
            "examples_policy": self.config.examples_policy,
            "examples": [],
        }

        # --------------------------
        # GLOBAL MODEL
        # --------------------------
        if mode == "global_model":
            Xs, ys, cs, ridxs = [], [], [], []

            for b in encoded.values():
                ds = b.get("datasets", {}).get(encoding_name)
                if ds is None:
                    continue
                if "row_idx" not in ds:
                    raise RuntimeError("Encoded dataset missing 'row_idx'.")
                Xs.append(ds["X"])
                cs.append(self._as_str_array(ds["case_id"]))
                ys.append(y_all[np.asarray(ds["row_idx"], dtype=np.int64)])
                ridxs.append(np.asarray(ds["row_idx"], dtype=np.int64))

            if not Xs:
                raise RuntimeError("No data produced for winning encoding in global_model.")

            X_all = np.vstack(Xs)
            y_rows = np.concatenate(ys)
            case_all = np.concatenate(cs)
            row_all = np.concatenate(ridxs)

            m_tr, m_va, m_te = self._build_split_masks(case_all, train_cases, val_cases, test_cases)

            X_train, y_train = X_all[m_tr], y_rows[m_tr]
            X_val, y_val = X_all[m_va], y_rows[m_va]
            X_test, y_test = X_all[m_te], y_rows[m_te]
            row_test = row_all[m_te]

            if flaml_task_kind == "regression":
                y_train = self._preprocess_regression_targets(y_train)
                y_val = self._preprocess_regression_targets(y_val)

            if self._is_classification(task) and self.config.skip_single_class and len(np.unique(y_train)) < 2:
                raise RuntimeError("Training split has single class; cannot train classification AutoML.")

            cfg = AutoMLConfig(
                time_budget_s=self.config.time_budget_s,
                metric=flaml_metric,
                seed=self.config.seed,
                n_jobs=self.config.n_jobs,
                estimator_list=self.config.estimator_list,
            )

            automl_info = adapter.fit_predict(
                task=flaml_task_kind,
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                config=cfg,
            )

            automl_obj = automl_info.get("_automl_object")
            models["global"] = {"automl": automl_obj, "info": automl_info}

            test_block: Dict[str, Any] = {"has_test": bool(len(X_test) > 0), "num_rows": int(len(X_test))}
            y_pred = None

            if len(X_test) > 0 and automl_obj is not None:
                raw_pred = np.asarray(automl_obj.predict(X_test))
                if flaml_task_kind == "regression":
                    y_pred = self._postprocess_regression_preds(raw_pred)
                    test_block["mae"] = float(np.mean(np.abs(y_test.astype(float) - y_pred.astype(float))))
                else:
                    y_pred = raw_pred
                    test_block["accuracy"] = float(np.mean(y_test == y_pred))

            # examples
            if len(X_test) > 0 and y_pred is not None and (self.config.n_example_rows > 0 or self.config.examples_policy == "all"):
                picked_rows: List[int] = []

                # NEW: If examples_policy is "all", save ALL test predictions
                if self.config.examples_policy == "all":
                    picked_rows = [int(r) for r in row_test]
                elif self.config.examples_policy == "quantiles" and flaml_task_kind == "regression":
                    picked_rows = self._select_examples_quantiles(
                        ps=ps,
                        label_col=task.label_col,
                        test_row_idx=row_test,
                        n_examples=self.config.n_example_rows,
                        seed=self.config.seed,
                    )

                if not picked_rows:
                    picked_rows = [int(r) for r in row_test[: self.config.n_example_rows]]

                pos = {int(r): i for i, r in enumerate(row_test)}
                for ridx in picked_rows:
                    if int(ridx) not in pos:
                        continue
                    i = pos[int(ridx)]
                    ex = {
                        "split": "test",
                        "case_id": str(ps.loc[int(ridx), "case_id"]),
                        "prefix_len": int(ps.loc[int(ridx), "prefix_len"]),
                        "prefix_activities": ps.loc[int(ridx), "prefix_activities"],
                        "y_true": float(y_test[i]) if flaml_task_kind == "regression" else str(y_test[i]),
                        "y_pred": float(y_pred[i]) if flaml_task_kind == "regression" else str(y_pred[i]),
                    }
                    if flaml_task_kind == "regression":
                        ex["abs_error"] = float(abs(ex["y_true"] - ex["y_pred"]))
                    out_json["examples"].append(ex)

            out_json["mode_results"]["global_model"] = {
                "flaml": {
                    "task_kind": automl_info.get("task"),
                    "metric": automl_info.get("metric"),
                    "time_budget_s": automl_info.get("time_budget_s"),
                    "best_estimator": automl_info.get("best_estimator"),
                    "best_loss": automl_info.get("best_loss"),
                    "best_config": automl_info.get("best_config"),
                    "best_iteration": automl_info.get("best_iteration"),
                    "time_to_find_best_s": automl_info.get("training_log", {}).get("time_to_find_best_s"),
                    "best_model_repr": automl_info.get("best_model_repr"),
                },
                "data_shapes": {"X_train": list(X_train.shape), "X_val": list(X_val.shape), "X_test": list(X_test.shape)},
                "test": test_block,
            }

            # ✅ ensure artifacts + return ctx
            ctx.artifacts["single_task_models"] = models
            ctx.artifacts["single_task_automl"] = out_json
            return ctx

        # -------------------------------------------------
        # PER_BUCKET_MODELS
        # -------------------------------------------------
        bucket_ids = sorted(encoded.keys())
        bucket_ids = [bid for bid in bucket_ids if encoding_name in encoded[bid].get("datasets", {})]
        if not bucket_ids:
            raise RuntimeError("No buckets contain the winning encoding dataset.")

        eligible: List[Any] = []
        bucket_meta: Dict[str, Dict[str, Any]] = {}

        for bid in bucket_ids:
            ds = encoded[bid]["datasets"][encoding_name]
            if "row_idx" not in ds:
                raise RuntimeError("Encoded dataset missing 'row_idx'.")

            X = ds["X"]
            case_ids = self._as_str_array(ds["case_id"])
            row_idx = np.asarray(ds["row_idx"], dtype=np.int64)
            y = y_all[row_idx]

            m_tr, m_va, m_te = self._build_split_masks(case_ids, train_cases, val_cases, test_cases)
            n_tr, n_va, n_te = int(m_tr.sum()), int(m_va.sum()), int(m_te.sum())

            if n_tr < self.config.min_bucket_samples or n_va < self.config.min_bucket_samples:
                continue

            if self._is_classification(task) and self.config.skip_single_class and len(np.unique(y[m_tr])) < 2:
                continue

            eligible.append(bid)
            bucket_meta[str(bid)] = {"n_train": n_tr, "n_val": n_va, "n_test": n_te, "n_total": int(len(X))}

        if not eligible:
            raise RuntimeError("Winner per_bucket_models: no bucket met constraints.")

        per_bucket_budget = max(30, int(self.config.time_budget_s / max(1, len(eligible))))

        per_bucket_results: Dict[str, Any] = {}
        pool_test_rows: List[Dict[str, Any]] = []
        sum_abs_err = 0.0
        sum_n = 0

        for bid in eligible:
            ds = encoded[bid]["datasets"][encoding_name]
            X = ds["X"]
            case_ids = self._as_str_array(ds["case_id"])
            row_idx = np.asarray(ds["row_idx"], dtype=np.int64)
            y = y_all[row_idx]

            m_tr, m_va, m_te = self._build_split_masks(case_ids, train_cases, val_cases, test_cases)

            X_train, y_train = X[m_tr], y[m_tr]
            X_val, y_val = X[m_va], y[m_va]
            X_test, y_test = X[m_te], y[m_te]
            row_test = row_idx[m_te]

            if flaml_task_kind == "regression":
                y_train_t = self._preprocess_regression_targets(y_train)
                y_val_t = self._preprocess_regression_targets(y_val)
            else:
                y_train_t, y_val_t = y_train, y_val

            cfg = AutoMLConfig(
                time_budget_s=per_bucket_budget,
                metric=flaml_metric,
                seed=self.config.seed,
                n_jobs=self.config.n_jobs,
                estimator_list=self.config.estimator_list,
            )

            automl_info = adapter.fit_predict(
                task=flaml_task_kind,
                X_train=X_train, y_train=y_train_t,
                X_val=X_val, y_val=y_val_t,
                config=cfg,
            )
            automl_obj = automl_info.get("_automl_object")
            models[str(bid)] = {"automl": automl_obj, "info": automl_info}

            test_block: Dict[str, Any] = {"has_test": bool(len(X_test) > 0), "num_rows": int(len(X_test))}
            if len(X_test) > 0 and automl_obj is not None:
                raw_pred = np.asarray(automl_obj.predict(X_test))
                if flaml_task_kind == "regression":
                    y_pred = self._postprocess_regression_preds(raw_pred)
                    abs_err = np.abs(y_test.astype(float) - y_pred.astype(float))
                    test_block["mae"] = float(np.mean(abs_err))

                    sum_abs_err += float(abs_err.sum())
                    sum_n += int(len(abs_err))

                    for i in range(len(X_test)):
                        pool_test_rows.append({
                            "bucket_id": str(bid),
                            "row_idx": int(row_test[i]),
                            "y_true": float(y_test[i]),
                            "y_pred": float(y_pred[i]),
                        })
                else:
                    y_pred = raw_pred
                    test_block["accuracy"] = float(np.mean(y_test == y_pred))
                    for i in range(len(X_test)):
                        pool_test_rows.append({
                            "bucket_id": str(bid),
                            "row_idx": int(row_test[i]),
                            "y_true": str(y_test[i]),
                            "y_pred": str(y_pred[i]),
                        })

            per_bucket_results[str(bid)] = {
                "bucket_id": str(bid),
                "budget_s": per_bucket_budget,
                "data_shapes": {"X_train": list(X_train.shape), "X_val": list(X_val.shape), "X_test": list(X_test.shape)},
                "flaml": {
                    "task_kind": automl_info.get("task"),
                    "metric": automl_info.get("metric"),
                    "best_estimator": automl_info.get("best_estimator"),
                    "best_loss": automl_info.get("best_loss"),
                    "best_config": automl_info.get("best_config"),
                    "best_iteration": automl_info.get("best_iteration"),
                    "time_to_find_best_s": automl_info.get("training_log", {}).get("time_to_find_best_s"),
                    "best_model_repr": automl_info.get("best_model_repr"),
                },
                "test": test_block,
                "split_counts": bucket_meta.get(str(bid), {}),
            }

        agg_test: Dict[str, Any] = {"has_test": any(r["test"]["has_test"] for r in per_bucket_results.values())}
        if agg_test["has_test"] and flaml_task_kind == "regression" and sum_n > 0:
            agg_test["weighted_mae"] = float(sum_abs_err / sum_n)

        out_json["mode_results"]["per_bucket_models"] = {
            "eligible_buckets": [str(b) for b in eligible],
            "per_bucket": per_bucket_results,
            "aggregate_test": agg_test,
        }

        # -------------------------------------------------
        # Examples: pick at least one example PER BUCKET (or ALL if policy="all")
        # -------------------------------------------------
        if pool_test_rows and (self.config.n_example_rows > 0 or self.config.examples_policy == "all"):
            dfp = pd.DataFrame(pool_test_rows)

            dfp["case_id"] = ps.loc[dfp["row_idx"].values, "case_id"].astype(str).values
            dfp["prefix_len"] = pd.to_numeric(
                ps.loc[dfp["row_idx"].values, "prefix_len"],
                errors="coerce"
            ).values

            picked_examples: List[Dict[str, Any]] = []

            # NEW: If examples_policy is "all", save ALL test predictions
            if self.config.examples_policy == "all":
                for _, row in dfp.iterrows():
                    ridx = int(row["row_idx"])
                    if flaml_task_kind == "regression":
                        ex = {
                            "split": "test",
                            "bucket_id": str(row.get("bucket_id")),
                            "case_id": str(row["case_id"]),
                            "prefix_len": int(row["prefix_len"]),
                            "prefix_activities": ps.loc[ridx, "prefix_activities"],
                            "y_true": float(row["y_true"]),
                            "y_pred": float(row["y_pred"]),
                            "abs_error": float(abs(float(row["y_true"]) - float(row["y_pred"]))),
                        }
                    else:
                        ex = {
                            "split": "test",
                            "bucket_id": str(row.get("bucket_id")),
                            "case_id": str(row["case_id"]),
                            "prefix_len": int(row["prefix_len"]),
                            "prefix_activities": ps.loc[ridx, "prefix_activities"],
                            "y_true": str(row["y_true"]),
                            "y_pred": str(row["y_pred"]),
                            "wrong": int(row["y_true"] != row["y_pred"]),
                        }
                    picked_examples.append(ex)
            else:
                # Original logic: pick one example per bucket
                max_buckets = int(self.config.n_example_rows)

                for i_bucket, (bid, g) in enumerate(dfp.groupby("bucket_id", sort=True)):
                    if max_buckets > 0 and i_bucket >= max_buckets:
                        break

                    g = g.copy()
                    g = g.sort_values(["case_id", "prefix_len"], kind="mergesort")
                    reps = g.groupby("case_id", as_index=False).tail(1)
                    if len(reps) == 0:
                        continue

                    if flaml_task_kind == "regression":
                        y = pd.to_numeric(reps["y_true"], errors="coerce")
                        reps = reps.loc[~y.isna()].copy()
                        if len(reps) == 0:
                            continue

                        if self.config.examples_policy == "quantiles":
                            target = float(y.quantile(0.5))
                            reps["__dist"] = (pd.to_numeric(reps["y_true"], errors="coerce") - target).abs()
                            chosen = reps.sort_values("__dist", kind="mergesort").head(1)
                        else:
                            reps["abs_error"] = (reps["y_true"].astype(float) - reps["y_pred"].astype(float)).abs()
                            chosen = reps.sort_values("abs_error", ascending=False, kind="mergesort").head(1)

                        row = chosen.iloc[0]
                        ridx = int(row["row_idx"])
                        ex = {
                            "split": "test",
                            "bucket_id": str(row.get("bucket_id")),
                            "case_id": str(ps.loc[ridx, "case_id"]),
                            "prefix_len": int(ps.loc[ridx, "prefix_len"]),
                            "prefix_activities": ps.loc[ridx, "prefix_activities"],
                            "y_true": float(row["y_true"]),
                            "y_pred": float(row["y_pred"]),
                            "abs_error": float(abs(float(row["y_true"]) - float(row["y_pred"]))),
                        }
                        picked_examples.append(ex)
                    else:
                        reps["wrong"] = (reps["y_true"] != reps["y_pred"]).astype(int)
                        chosen = reps.sort_values("wrong", ascending=False, kind="mergesort").head(1)

                        row = chosen.iloc[0]
                        ridx = int(row["row_idx"])
                        ex = {
                            "split": "test",
                            "bucket_id": str(row.get("bucket_id")),
                            "case_id": str(ps.loc[ridx, "case_id"]),
                            "prefix_len": int(ps.loc[ridx, "prefix_len"]),
                            "prefix_activities": ps.loc[ridx, "prefix_activities"],
                            "y_true": str(row["y_true"]),
                            "y_pred": str(row["y_pred"]),
                            "wrong": int(row["wrong"]),
                        }
                        picked_examples.append(ex)

            out_json["examples"] = picked_examples

        # ✅ CRITICAL FIX: write artifacts + return ctx (this was missing in your file)
        ctx.artifacts["single_task_models"] = models
        ctx.artifacts["single_task_automl"] = out_json
        return ctx
