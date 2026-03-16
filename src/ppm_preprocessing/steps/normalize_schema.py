from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List

import pandas as pd

from .base import Step
from ppm_preprocessing.domain.context import PipelineContext
from ppm_preprocessing.domain.canonical_log import CanonicalLog


def _pick_best_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    existing = [c for c in candidates if c in df.columns]
    if not existing:
        return None
    # choose the one with fewest missing values
    return min(existing, key=lambda c: int(df[c].isna().sum()))


def _jsonable(v: Any) -> Any:
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if pd.isna(v):
        return None
    return v


@dataclass
class NormalizeSchemaConfig:
    # Where to write the report (THIS matches your project structure)
    report_dir: str = "outputs/reports"
    report_filename: str = "02_normalize_schema_qc.json"
    write_report: bool = True
    n_sample_rows: int = 3

    # Timestamp parsing
    parse_timestamp_utc: bool = True

    # Candidate columns
    case_candidates: Optional[List[str]] = None
    act_candidates: Optional[List[str]] = None
    ts_candidates: Optional[List[str]] = None


class NormalizeSchemaStep(Step):
    """
    Normalizes required columns to canonical schema:
      - case_id
      - activity
      - timestamp

    IMPORTANT:
      - Keeps ALL other columns unchanged (no dropping).
      - Only renames the detected case/activity/timestamp columns.

    Input:
      ctx.raw_df (pd.DataFrame)

    Output:
      ctx.log = CanonicalLog(df=..., meta=...)
      ctx.artifacts["normalize_schema_qc"] (dict)
      ctx.artifacts["normalize_schema_qc_path"] (str) if written
    """
    name = "normalize_schema"

    def __init__(self, config: NormalizeSchemaConfig | None = None):
        self.config = config or NormalizeSchemaConfig()

        if self.config.case_candidates is None:
            self.config.case_candidates = [
                "case:concept:name", "case_id", "CaseID", "case", "trace_id",
                "Case", "CaseId", "caseid"
            ]
        if self.config.act_candidates is None:
            self.config.act_candidates = [
                "concept:name", "activity", "Activity", "task", "event",
                "Action", "EventName", "activity_name" , "activityNameEN"
            ]
        if self.config.ts_candidates is None:
            self.config.ts_candidates = [
                "time:timestamp", "timestamp", "time", "datetime", "event_time",
                "Timestamp", "start_time", "end_time"
            ]

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.raw_df is None:
            raise RuntimeError("raw_df is None. Load step must run before schema normalization.")

        df = ctx.raw_df

        case_col = _pick_best_column(df, self.config.case_candidates)
        act_col = _pick_best_column(df, self.config.act_candidates)
        ts_col = _pick_best_column(df, self.config.ts_candidates)

        if not case_col or not act_col or not ts_col:
            raise ValueError(
                "Could not infer required columns. "
                f"case_col={case_col}, act_col={act_col}, ts_col={ts_col}. "
                f"Available columns: {list(df.columns)}"
            )

        # Rename only canonical columns, keep all other columns untouched
        out = df.rename(columns={case_col: "case_id", act_col: "activity", ts_col: "timestamp"}).copy()

        # Parse timestamp to datetime
        ts = pd.to_datetime(out["timestamp"], errors="coerce", utc=self.config.parse_timestamp_utc)
        ts_na = int(ts.isna().sum())
        out["timestamp"] = ts

        # semantic columns present (not renamed)
        extra = {
            "lifecycle": "lifecycle:transition" if "lifecycle:transition" in out.columns else None,
            "event_id": "EventID" if "EventID" in out.columns else None,
            "resource": "org:resource" if "org:resource" in out.columns else None,
        }

        num_rows = int(len(out))
        num_cols = int(out.shape[1])
        num_cases = int(out["case_id"].astype(str).nunique()) if num_rows else 0

        tmin = out["timestamp"].min()
        tmax = out["timestamp"].max()

        sample_rows: List[Dict[str, Any]] = []
        if num_rows:
            sample = out.head(int(self.config.n_sample_rows)).copy()
            for _, r in sample.iterrows():
                sample_rows.append({k: _jsonable(v) for k, v in r.to_dict().items()})

        qc: Dict[str, Any] = {
            "step": self.name,
            "detected": {
                "case_col": case_col,
                "activity_col": act_col,
                "timestamp_col": ts_col,
            },
            "num_rows": num_rows,
            "num_columns": num_cols,
            "columns": list(out.columns),
            "num_cases": num_cases,
            "timestamp_parse_na": ts_na,
            "time_min": None if pd.isna(tmin) else tmin.isoformat(),
            "time_max": None if pd.isna(tmax) else tmax.isoformat(),
            "available_semantic_columns": extra,
            "sample_rows": sample_rows,
            "note": "Only case/activity/timestamp were renamed. All other columns are preserved unchanged.",
        }

        # Put canonical log into ctx
        ctx.log = CanonicalLog(
            df=out,
            meta={
                "inferred_columns": {"case": case_col, "activity": act_col, "timestamp": ts_col},
                "available_semantic_columns": extra,
            },
        )

        # Keep qc in artifacts
        ctx.artifacts["normalize_schema_qc"] = qc

        # Write report to outputs/reports
        if self.config.write_report:
            report_dir = Path(self.config.report_dir)
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / self.config.report_filename
            report_path.write_text(
                pd.Series([qc]).to_json(orient="records", indent=2, force_ascii=False),
                encoding="utf-8",
            )
            ctx.artifacts["normalize_schema_qc_path"] = str(report_path)

        return ctx
