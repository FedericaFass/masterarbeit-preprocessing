# ppm_preprocessing/steps/prefix_extraction.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pathlib import Path
import json

import numpy as np
import pandas as pd

from .base import Step
from ppm_preprocessing.domain.context import PipelineContext


@dataclass
class PrefixExtractionConfig:
    max_prefix_len: int = 30
    min_prefix_len: int = 1
    sample_frac: Optional[float] = None
    random_state: int = 42

    # --- Remaining-time features / targets ---
    add_time_features: bool = True
    add_calendar_features: bool = True
    add_log_features: bool = True
    add_log_target: bool = True

    feat_prefix: str = "feat_"

    # --- Option A: include case/event attributes ---
    include_case_attributes: bool = True
    include_event_attributes: bool = True

    case_attr_mode: str = "case_prefix"          # "case_prefix" or "explicit"
    case_attribute_cols: Optional[List[str]] = None

    event_attribute_cols: Optional[List[str]] = None
    event_agg_policy: str = "last_non_null"      # "last_non_null" or "last"

    case_out_prefix: str = "case__"
    event_out_prefix: str = "event_last__"

    # --- extra "next event" label columns for custom prediction targets ---
    # e.g. ["org:resource", "org:role"] — will create label_next_org_resource, etc.
    next_event_attr_cols: Optional[List[str]] = None

    # --- reporting ---
    report_basename: str = "06_prefix_extraction_qc"
    n_report_examples: int = 3
    preview_rows_csv: int = 200
    save_preview_csv: bool = True
    save_preview_parquet: bool = False  # parquet preserves list column nicer


class PrefixExtractionStep(Step):
    name = "prefix_extraction"

    def __init__(self, config: PrefixExtractionConfig | None = None):
        self.config = config or PrefixExtractionConfig()

    @staticmethod
    def _safe_total_seconds(td) -> Optional[float]:
        try:
            return float(td.total_seconds())
        except Exception:
            return None

    @staticmethod
    def _log1p_nonneg(x: Optional[float]) -> float:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return 0.0
        return float(np.log1p(max(0.0, float(x))))

    def _get_reports_dir(self, ctx: PipelineContext) -> Path:
        # robust: prefer ctx.reports_dir if present, else default "reports"
        out_dir = getattr(ctx, "output_dir", None)
        if out_dir:
            return Path(out_dir) / "reports"
        return Path("outputs") / "reports"

    def _save_report(self, ctx: PipelineContext, report: Dict[str, Any]) -> None:
        reports_dir = self._get_reports_dir(ctx)
        reports_dir.mkdir(parents=True, exist_ok=True)

        path = reports_dir / f"{self.config.report_basename}.json"
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

        # Also store in artifacts so you can show it in console if you want
        ctx.artifacts[f"{self.config.report_basename}_report_path"] = str(path)

    def _detect_case_cols(self, df: pd.DataFrame) -> List[str]:
        c = self.config
        if not c.include_case_attributes:
            return []
        if c.case_attr_mode == "explicit" and c.case_attribute_cols:
            return [col for col in c.case_attribute_cols if col in df.columns]
        return [col for col in df.columns if str(col).startswith("case:")]

    def _detect_event_cols(self, df: pd.DataFrame, case_cols: List[str]) -> List[str]:
        c = self.config
        if not c.include_event_attributes:
            return []

        if c.event_attribute_cols:
            return [col for col in c.event_attribute_cols if col in df.columns]

        exclude = set([
            "case_id", "activity", "timestamp",
            "prefix_row_id", "prefix_len", "prefix_activities", "prefix_end_time",
            "label_next_activity", "label_outcome",
            "label_remaining_time_sec", "label_remaining_time_log1p",
            "bucket_id",
        ])
        exclude |= set(case_cols)
        return [col for col in df.columns if col not in exclude]

    def _aggregate_event_attrs(
        self,
        g: pd.DataFrame,
        upto_idx: int,
        event_cols: List[str],
        policy: str,
    ) -> Dict[str, Any]:
        if not event_cols:
            return {}

        sub = g.iloc[: upto_idx + 1]

        out: Dict[str, Any] = {}
        if policy == "last_non_null":
            for col in event_cols:
                nn = sub[col].dropna()
                out[col] = nn.iloc[-1] if len(nn) else None
            return out

        if policy == "last":
            for col in event_cols:
                out[col] = sub[col].iloc[-1]
            return out

        raise ValueError(f"Unknown event_agg_policy={policy!r}. Use 'last_non_null' or 'last'.")

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.log is None:
            raise RuntimeError("ctx.log is None. Run schema normalization + cleaning first.")
        if "case_table" not in ctx.artifacts:
            raise RuntimeError("case_table missing. Run CaseLabelsStep before PrefixExtractionStep.")

        df = ctx.log.df.copy()
        case_table: pd.DataFrame = ctx.artifacts["case_table"]

        required = {"case_id", "activity", "timestamp"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Canonical log missing required columns: {missing}")

        max_len = int(self.config.max_prefix_len)
        min_len = int(self.config.min_prefix_len)
        if min_len < 1 or max_len < min_len:
            raise ValueError(f"Invalid prefix length bounds: min={min_len}, max={max_len}")

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

        case_table_idx = case_table.set_index("case_id").copy()
        if "case_start_time" in case_table_idx.columns:
            case_table_idx["case_start_time"] = pd.to_datetime(case_table_idx["case_start_time"], utc=True, errors="coerce")
        if "case_end_time" not in case_table_idx.columns:
            raise RuntimeError("case_table missing 'case_end_time' (required for remaining_time).")
        case_table_idx["case_end_time"] = pd.to_datetime(case_table_idx["case_end_time"], utc=True, errors="coerce")

        case_start = case_table_idx["case_start_time"].to_dict() if "case_start_time" in case_table_idx.columns else {}
        case_end = case_table_idx["case_end_time"].to_dict()
        case_out = case_table_idx["label_outcome"].to_dict() if "label_outcome" in case_table_idx.columns else {}

        fp = str(self.config.feat_prefix or "feat_")

        case_cols = self._detect_case_cols(df)
        event_cols = self._detect_event_cols(df, case_cols)

        samples: List[Dict[str, Any]] = []
        row_id = 0

        for case_id, g in df.groupby("case_id", sort=False):
            acts = g["activity"].astype(str).tolist()
            times = g["timestamp"].tolist()
            n = len(acts)
            if n <= 1:
                continue

            upper = min(max_len, n - 1)

            start_time = case_start.get(case_id)
            end_time = case_end.get(case_id)
            out_label = case_out.get(case_id, 0)

            # case attrs: first non-null per case:* col
            case_attr_vals: Dict[str, Any] = {}
            if self.config.include_case_attributes and case_cols:
                for col in case_cols:
                    s = g[col].dropna()
                    case_attr_vals[col] = s.iloc[0] if len(s) else None

            for L in range(min_len, upper + 1):
                prefix_end_time = times[L - 1]

                remaining_sec: Optional[float] = None
                if end_time is not None and pd.notna(end_time) and pd.notna(prefix_end_time):
                    remaining_sec = self._safe_total_seconds(end_time - prefix_end_time)

                row: Dict[str, Any] = {
                    "case_id": case_id,
                    "prefix_row_id": row_id,
                    "prefix_len": L,
                    "prefix_activities": acts[:L],
                    "prefix_end_time": prefix_end_time,
                    "label_next_activity": acts[L],
                    "label_outcome": str(out_label) if out_label is not None else "unknown",
                    "label_remaining_time_sec": remaining_sec,
                }

                # Extra "next event" labels for user-specified attributes
                if self.config.next_event_attr_cols:
                    next_event = g.iloc[L]
                    for col in self.config.next_event_attr_cols:
                        if col == "activity":
                            continue  # already covered by label_next_activity
                        safe = col.replace(":", "_").replace(" ", "_").replace("-", "_").replace(".", "_")
                        label_key = f"label_next_{safe}"
                        try:
                            val = next_event[col]
                            row[label_key] = str(val) if pd.notna(val) else "unknown"
                        except (KeyError, TypeError):
                            row[label_key] = "unknown"

                if self.config.add_time_features:
                    elapsed_sec = 0.0
                    if start_time is not None and pd.notna(start_time) and pd.notna(prefix_end_time):
                        es = self._safe_total_seconds(prefix_end_time - start_time)
                        elapsed_sec = float(es) if es is not None else 0.0

                    tsl_sec = 0.0
                    if L > 1 and pd.notna(times[L - 2]) and pd.notna(prefix_end_time):
                        ts = self._safe_total_seconds(prefix_end_time - times[L - 2])
                        tsl_sec = float(ts) if ts is not None else 0.0

                    row[f"{fp}elapsed_time_sec"] = elapsed_sec
                    row[f"{fp}time_since_last_event_sec"] = tsl_sec

                    if self.config.add_log_features:
                        row[f"{fp}elapsed_time_log1p"] = self._log1p_nonneg(elapsed_sec)
                        row[f"{fp}time_since_last_log1p"] = self._log1p_nonneg(tsl_sec)

                    if self.config.add_calendar_features:
                        if pd.notna(prefix_end_time):
                            row[f"{fp}prefix_end_hour"] = int(prefix_end_time.hour)
                            weekday = int(prefix_end_time.weekday())  # 0=Mon .. 6=Sun
                            row[f"{fp}prefix_end_weekday"] = weekday
                            row[f"{fp}prefix_end_is_weekend"] = int(weekday >= 5)
                            row[f"{fp}prefix_end_month"] = int(prefix_end_time.month)
                        else:
                            row[f"{fp}prefix_end_hour"] = 0
                            row[f"{fp}prefix_end_weekday"] = 0
                            row[f"{fp}prefix_end_is_weekend"] = 0
                            row[f"{fp}prefix_end_month"] = 0

                if self.config.include_case_attributes and case_attr_vals:
                    for col, v in case_attr_vals.items():
                        safe_name = str(col).replace("case:", "")
                        row[f"{self.config.case_out_prefix}{safe_name}"] = v

                if self.config.include_event_attributes and event_cols:
                    ev = self._aggregate_event_attrs(
                        g=g,
                        upto_idx=L - 1,
                        event_cols=event_cols,
                        policy=self.config.event_agg_policy,
                    )
                    for col, v in ev.items():
                        row[f"{self.config.event_out_prefix}{col}"] = v

                if self.config.add_log_target:
                    if remaining_sec is None or (isinstance(remaining_sec, float) and np.isnan(remaining_sec)):
                        row["label_remaining_time_log1p"] = None
                    else:
                        row["label_remaining_time_log1p"] = float(np.log1p(max(0.0, float(remaining_sec))))

                samples.append(row)
                row_id += 1

        out = pd.DataFrame(samples)

        if self.config.sample_frac is not None and len(out):
            out = out.sample(frac=float(self.config.sample_frac), random_state=self.config.random_state)

        # store artifacts
        ctx.artifacts["prefix_samples"] = out
        qc: Dict[str, Any] = {
            "step": self.name,
            "max_prefix_len": max_len,
            "min_prefix_len": min_len,
            "num_samples": int(len(out)),
            "num_cases_covered": int(out["case_id"].nunique()) if len(out) else 0,
            "num_columns": int(out.shape[1]),
            "columns": list(out.columns),
            "case_attr_cols_detected": case_cols[:20],
            "event_attr_cols_detected": event_cols[:20],
            "event_agg_policy": self.config.event_agg_policy,
        }
        ctx.artifacts["prefix_qc"] = qc

        # REPORTS: save JSON with examples + preview file ---
       
        examples = []
        if len(out):
            ex = out.head(int(self.config.n_report_examples)).copy()
            ex = ex.where(pd.notna(ex), None)
            examples = ex.to_dict(orient="records")

        report = {
            "step": self.name,
            "num_rows": int(len(out)),
            "num_columns": int(out.shape[1]),
            "columns": list(out.columns),
            "qc": qc,
            "examples": examples,
            "note": "prefix_samples is stored in ctx.artifacts['prefix_samples']. This report shows only a small preview.",
        }
        self._save_report(ctx, report)

       
        reports_dir = self._get_reports_dir(ctx)
        reports_dir.mkdir(parents=True, exist_ok=True)

        if self.config.save_preview_csv and len(out):
            # CSV cannot store lists nicely -> we stringify prefix_activities
            preview = out.head(int(self.config.preview_rows_csv)).copy()
            if "prefix_activities" in preview.columns:
                preview["prefix_activities"] = preview["prefix_activities"].apply(lambda x: json.dumps(x) if isinstance(x, list) else str(x))
            preview_path = reports_dir / "prefix_samples_preview.csv"
            preview.to_csv(preview_path, index=False, encoding="utf-8")
            ctx.artifacts["prefix_samples_preview_csv"] = str(preview_path)

        if self.config.save_preview_parquet and len(out):
            # Parquet preserves list objects better (depending on engine)
            pq_path = reports_dir / "prefix_samples_preview.parquet"
            out.head(int(self.config.preview_rows_csv)).to_parquet(pq_path, index=False)
            ctx.artifacts["prefix_samples_preview_parquet"] = str(pq_path)

        return ctx
