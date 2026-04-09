# ppm_preprocessing/steps/encoding.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import json
import numpy as np
import pandas as pd

from .base import Step
from ppm_preprocessing.domain.context import PipelineContext

from ppm_preprocessing.encoders.last_state import LastStateEncoder, LastStateConfig
from ppm_preprocessing.encoders.aggregation import AggregationEncoder, AggregationConfig
from ppm_preprocessing.encoders.index_latest_payload import IndexLatestPayloadEncoder, IndexLatestPayloadConfig
from ppm_preprocessing.encoders.embedding import EmbeddingEncoder, EmbeddingConfig


@dataclass
class EncodingConfig:
    bucket_col: str = "bucket_id"
    input_key: str = "bucketed_prefixes"
    output_key: str = "encoded_buckets"

    use_last_state: bool = True
    use_aggregation: bool = True
    use_index_latest_payload: bool = True  # Index Latest-Payload encoding
    use_embedding: bool = True  # Sentence-transformer embedding encoding

    row_id_col: str = "prefix_row_id"

    # --- Task-aware label selection ---
    # If label_col_override is set, it takes priority over the remaining_time defaults.
    label_col_override: Optional[str] = None
    label_is_numeric: bool = True  # False for classification tasks (next_activity, outcome)

    # For remaining_time, choose one of these. If you train on log1p, set use_log1p_label=True.
    remaining_time_sec_label_col: str = "label_remaining_time_sec"
    remaining_time_log1p_label_col: str = "label_remaining_time_log1p"
    use_log1p_label: bool = True

    report_basename: str = "07_encoding_qc"
    n_report_examples: int = 3

    # Leakage prevention: fit encoders on train split only
    fit_on_train_only: bool = True

    save_encoded_preview_csv: bool = True
    preview_rows_per_bucket: int = 50
    preview_topk_features: int = 25
    preview_bucket_ids: Optional[List[int]] = None


class EncodingStep(Step):
    name = "encoding"

    def __init__(self, config: EncodingConfig | None = None):
        self.config = config or EncodingConfig()
        if not (self.config.use_last_state or self.config.use_aggregation or self.config.use_index_latest_payload):
            raise ValueError("At least one encoding must be enabled (use_last_state/use_aggregation/use_index_latest_payload).")

    def _get_train_df(self, ctx: PipelineContext, df: pd.DataFrame) -> pd.DataFrame:
        """Filter DataFrame to train cases only for leakage-safe encoder fitting."""
        if not self.config.fit_on_train_only:
            return df

        splits = ctx.artifacts.get("case_splits")
        if not splits or "train" not in splits:
            raise RuntimeError(
                "EncodingStep requires ctx.artifacts['case_splits']['train'] "
                "when fit_on_train_only=True. Run CaseSplitStep before encoding."
            )

        train_cases = set(map(str, splits["train"]))
        df_train = df[df["case_id"].astype(str).isin(train_cases)]

        if len(df_train) == 0:
            raise RuntimeError(
                "No train cases found in bucketed_prefixes. "
                "Check that case_splits and bucketed_prefixes share case_id values."
            )

        return df_train

    def _get_reports_dir(self, ctx: PipelineContext) -> Path:
        out_dir = getattr(ctx, "output_dir", None)
        if out_dir:
            return Path(out_dir) / "reports"
        return Path("outputs") / "reports"

    def _save_report(self, ctx: PipelineContext, payload: Dict[str, Any]) -> None:
        reports_dir = self._get_reports_dir(ctx)
        reports_dir.mkdir(parents=True, exist_ok=True)
        path = reports_dir / f"{self.config.report_basename}.json"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        ctx.artifacts[f"{self.config.report_basename}_report_path"] = str(path)

    @staticmethod
    def _nonzero_preview(
        X: np.ndarray,
        feature_names: Optional[List[str]],
        max_items: int = 20,
    ) -> Dict[str, Any]:
        x0 = np.asarray(X[0]).ravel()
        nz = np.where(x0 != 0)[0].tolist()
        nz = nz[:max_items]
        names = [feature_names[i] for i in nz] if feature_names else [f"f{i}" for i in nz]
        vals = [float(x0[i]) for i in nz]
        return {"idx": nz, "names": names, "vals": vals}

    @staticmethod
    def _topk_features_row(
        x: np.ndarray,
        feature_names: Optional[List[str]],
        topk: int,
    ) -> List[Tuple[str, float]]:
        x = np.asarray(x).ravel()
        nz = np.where(x != 0)[0]
        if nz.size == 0:
            return []
        vals = x[nz]
        order = np.argsort(-np.abs(vals))
        nz = nz[order][:topk]
        out: List[Tuple[str, float]] = []
        for j in nz:
            name = feature_names[j] if feature_names and j < len(feature_names) else f"f{int(j)}"
            out.append((str(name), float(x[j])))
        return out

    def _save_encoded_preview_csv(
        self,
        ctx: PipelineContext,
        bucket_id: int,
        encoding_name: str,
        X: np.ndarray,
        y: np.ndarray,
        case_id: np.ndarray,
        row_idx: np.ndarray,
        feature_names: Optional[List[str]],
    ) -> None:
        reports_dir = self._get_reports_dir(ctx)
        reports_dir.mkdir(parents=True, exist_ok=True)

        n_rows = min(int(self.config.preview_rows_per_bucket), int(X.shape[0]))
        topk = int(self.config.preview_topk_features)

        rows: List[Dict[str, Any]] = []
        for i in range(n_rows):
            pairs = self._topk_features_row(X[i], feature_names, topk=topk)

            try:
                y_val: Any = float(y[i])
            except (ValueError, TypeError):
                y_val = str(y[i])

            r: Dict[str, Any] = {
                "bucket_id": int(bucket_id),
                "encoding": str(encoding_name),
                "case_id": str(case_id[i]),
                "row_idx": int(row_idx[i]),
                "y": y_val,
                "num_nonzero": int(np.count_nonzero(X[i])),
                "feature_dim": int(X.shape[1]) if X.ndim == 2 else None,
            }
            for k, (fname, fval) in enumerate(pairs, start=1):
                r[f"feat_{k}_name"] = fname
                r[f"feat_{k}_val"] = fval
            rows.append(r)

        out_df = pd.DataFrame(rows)
        out_path = reports_dir / f"encoded_preview__{encoding_name}__bucket_{int(bucket_id)}.csv"
        out_df.to_csv(out_path, index=False, encoding="utf-8")
        ctx.artifacts[f"encoded_preview_csv__{encoding_name}__bucket_{int(bucket_id)}"] = str(out_path)

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if self.config.input_key not in ctx.artifacts:
            raise RuntimeError(f"Missing ctx.artifacts['{self.config.input_key}']. Run BucketingStep first.")

        df: pd.DataFrame = ctx.artifacts[self.config.input_key]
        if len(df) == 0:
            ctx.artifacts[self.config.output_key] = {}
            ctx.artifacts["encoding_qc"] = {"step": self.name, "num_buckets": 0, "candidates": {}}
            self._save_report(ctx, {"step": self.name, "num_rows": 0, "note": "Empty input."})
            return ctx

        if self.config.bucket_col not in df.columns:
            raise ValueError(f"Expected bucket column '{self.config.bucket_col}' in bucketed prefixes.")
        if self.config.row_id_col not in df.columns:
            raise ValueError(
                f"Expected '{self.config.row_id_col}' in bucketed prefixes. "
                f"Add it in PrefixExtractionStep and ensure BucketingStep keeps it."
            )
        if "case_id" not in df.columns:
            raise ValueError("Expected 'case_id' in bucketed prefixes.")

        # ---------------------------
        # Label column selection (task-aware)
        # ---------------------------
        if self.config.label_col_override:
            label_col = self.config.label_col_override
        else:
            label_col = (
                self.config.remaining_time_log1p_label_col
                if bool(self.config.use_log1p_label)
                else self.config.remaining_time_sec_label_col
            )
        label_is_numeric = self.config.label_is_numeric

        if label_col not in df.columns:
            raise ValueError(
                f"Expected label column '{label_col}' in input. "
                f"Available label cols: {[c for c in df.columns if str(c).startswith('label_')]}"
            )

        # ---------------------------
        # Fit encoders on TRAIN data only (leakage-safe)
        # ---------------------------
        df_train = self._get_train_df(ctx, df)
        print(f"[EncodingStep] Fitting encoders on {len(df_train)} train rows "
              f"(out of {len(df)} total rows, fit_on_train_only={self.config.fit_on_train_only})")

        encoders: Dict[str, Any] = {}
        qc_candidates: Dict[str, Any] = {}

        if self.config.use_last_state:
            key = "last_state"
            enc = LastStateEncoder(
                LastStateConfig(
                    label_col=label_col,
                    label_is_numeric=label_is_numeric,
                    snapshot_prefixes=("case__", "event_last__", "feat_"),
                    max_categories_per_col=20,
                    min_freq_per_category=50,
                    max_categorical_cols=30,
                    # You can ignore noisy IDs here too if needed
                    ignore_cols=[
                        "event_last__EventID",
                        "event_last___event_index",
                    ],
                )
            )
            enc.fit(df_train)
            encoders[key] = enc
            qc_candidates[key] = {
                "type": "last_state",
                "feature_dim": int(len(enc.feature_names_)),
                "num_numeric_cols": int(len(enc.numeric_cols_)),
                "num_categorical_cols": int(len(enc.categorical_cols_)),
                "label_col": label_col,
                "label_is_numeric": label_is_numeric,
            }

        if self.config.use_aggregation:
            key = "aggregation"
            enc = AggregationEncoder(
                AggregationConfig(
                    label_col=label_col,
                    label_is_numeric=label_is_numeric,
                    include_prefix_len=True,
                    include_extra_features=True,
                    max_categories_per_col=20,
                    min_freq_per_category=50,
                    max_categorical_cols=30,
                    ignore_cols=[
                        "event_last__EventID",
                        "event_last___event_index",
                    ],
                )
            )
            enc.fit(df_train)
            encoders[key] = enc
            qc_candidates[key] = {
                "type": "aggregation",
                "feature_dim": int(len(enc.feature_names_)),
                "num_numeric_cols": int(len(enc.numeric_cols_)),
                "num_categorical_cols": int(len(enc.categorical_cols_)),
                "label_col": label_col,
                "label_is_numeric": label_is_numeric,
            }

        if self.config.use_index_latest_payload:
            key = "index_latest_payload"
            enc = IndexLatestPayloadEncoder(
                IndexLatestPayloadConfig(
                    label_col=label_col,
                    label_is_numeric=label_is_numeric,
                    include_case_attributes=True,
                    include_activity_sequence=True,
                    include_latest_payload=True,
                )
            )
            enc.fit(df_train)
            encoders[key] = enc
            qc_candidates[key] = {
                "type": "index_latest_payload",
                "feature_dim": int(len(enc.feature_names_)),
                "num_case_numeric_cols": int(len(enc.case_numeric_cols_)),
                "num_case_categorical_cols": int(len(enc.case_categorical_cols_)),
                "num_temporal_numeric_cols": int(len(enc.temporal_numeric_cols_)),
                "num_temporal_categorical_cols": int(len(enc.temporal_categorical_cols_)),
                "num_last_event_numeric_cols": int(len(enc.last_event_numeric_cols_)),
                "num_last_event_categorical_cols": int(len(enc.last_event_categorical_cols_)),
                "max_prefix_len": int(enc.max_prefix_len_),
                "label_col": label_col,
                "label_is_numeric": label_is_numeric,
                "note": "Index Latest-Payload: activity sequence + temporal features + latest event attributes",
            }

        if self.config.use_embedding:
            key = "embedding"
            enc = EmbeddingEncoder(
                EmbeddingConfig(
                    label_col=label_col,
                    label_is_numeric=label_is_numeric,
                    include_numeric_features=True,
                    include_prefix_len=True,
                )
            )
            enc.fit(df_train)
            encoders[key] = enc
            qc_candidates[key] = {
                "type": "embedding",
                "feature_dim": int(len(enc.feature_names_)),
                "emb_dim": enc.emb_dim_,
                "model_name": enc.config.model_name,
                "num_activities": len(enc.activity_embeddings_),
                "num_numeric_cols": len(enc.numeric_cols_),
                "label_col": label_col,
                "label_is_numeric": label_is_numeric,
            }

        # Store fitted encoders for downstream persistence
        ctx.artifacts["fitted_encoders"] = encoders

        qc: Dict[str, Any] = {
            "step": self.name,
            "num_rows": int(len(df)),
            "num_train_rows": int(len(df_train)),
            "num_buckets": int(df[self.config.bucket_col].nunique()),
            "fit_on_train_only": bool(self.config.fit_on_train_only),
            "label_col": label_col,
            "label_is_numeric": label_is_numeric,
            "candidates": qc_candidates,
        }

        # ---------------------------
        # Transform per bucket
        # ---------------------------
        encoded_buckets: Dict[int, Dict[str, Any]] = {}

        for bucket_id, g in df.groupby(self.config.bucket_col, sort=True):
            bid = int(bucket_id)
            encoded_buckets[bid] = {"datasets": {}, "num_rows": int(len(g))}

            case_ids = g["case_id"].astype(str).to_numpy()
            row_idx = g[self.config.row_id_col].to_numpy(dtype=np.int64)
            y_original = g[label_col].astype(str).to_numpy() if not label_is_numeric else None

            for enc_key, enc in encoders.items():
                ds = enc.transform(g)

                encoded_buckets[bid]["datasets"][enc_key] = {
                    "X": ds.X,
                    "y": ds.y,
                    "y_original": y_original,
                    "case_id": case_ids,
                    "row_idx": row_idx,
                    "meta": ds.meta,
                }

        ctx.artifacts[self.config.output_key] = encoded_buckets
        ctx.artifacts["encoding_qc"] = qc

        # ---------------------------
        # OPTIONAL: Save encoded preview CSVs
        # ---------------------------
        all_buckets = sorted(encoded_buckets.keys())
        if all_buckets:
            if self.config.preview_bucket_ids is None:
                preview_buckets = [all_buckets[0]]
            else:
                preview_buckets = [b for b in self.config.preview_bucket_ids if b in encoded_buckets]

            for bid in preview_buckets:
                dsets = encoded_buckets[int(bid)]["datasets"]
                for enc_key, payload in dsets.items():
                    meta = payload.get("meta", {}) or {}
                    feat_names = meta.get("feature_names", None)

                    if self.config.save_encoded_preview_csv:
                        self._save_encoded_preview_csv(
                            ctx=ctx,
                            bucket_id=int(bid),
                            encoding_name=str(enc_key),
                            X=payload["X"],
                            y=payload.get("y_original") if payload.get("y_original") is not None else payload["y"],
                            case_id=payload["case_id"],
                            row_idx=payload["row_idx"],
                            feature_names=feat_names,
                        )

        # ---------------------------
        # REPORT
        # ---------------------------
        examples: List[Dict[str, Any]] = []
        try:
            first_bid = sorted(encoded_buckets.keys())[0]
            dsets = encoded_buckets[first_bid]["datasets"]

            for enc_key in dsets.keys():
                X = dsets[enc_key]["X"]
                meta = dsets[enc_key]["meta"]
                feat_names = meta.get("feature_names", None)

                ex: Dict[str, Any] = {
                    "bucket_id": int(first_bid),
                    "encoding": enc_key,
                    "X_shape": list(X.shape),
                    "feature_dim": int(X.shape[1]) if X.ndim == 2 else None,
                    "label_col": meta.get("label_col"),
                    "label_is_numeric": meta.get("label_is_numeric"),
                    "nonzero_preview_row0": self._nonzero_preview(X, feat_names, max_items=20),
                }
                if self.config.save_encoded_preview_csv:
                    ex["preview_csv_path_artifact_key"] = f"encoded_preview_csv__{enc_key}__bucket_{int(first_bid)}"
                examples.append(ex)

                if len(examples) >= int(self.config.n_report_examples):
                    break

        except Exception as e:
            examples.append({"note": f"example generation failed: {type(e).__name__}: {e}"})

        report = {
            "step": self.name,
            "num_rows": int(len(df)),
            "num_buckets": int(df[self.config.bucket_col].nunique()),
            "label_col": label_col,
            "label_is_numeric": label_is_numeric,
            "candidates": qc_candidates,
            "examples": examples,
            "note": (
                "X/y live in ctx.artifacts['encoded_buckets'][bucket_id]['datasets'][encoding]. "
                "If enabled, encoded previews are saved as CSV in outputs/reports/."
            ),
        }
        self._save_report(ctx, report)

        return ctx
