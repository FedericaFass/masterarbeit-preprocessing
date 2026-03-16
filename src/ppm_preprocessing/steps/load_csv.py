from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List

import pandas as pd

from .base import Step
from ppm_preprocessing.domain.context import PipelineContext


def _json_safe(obj: Any) -> Any:
    """Best-effort conversion to JSON-serializable types."""
    if obj is None:
        return None
    if is_dataclass(obj):
        return _json_safe(asdict(obj))
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    # fallback
    return str(obj)


class LoadCsvStep(Step):
    """
    Loads a CSV file, converts to a DataFrame, stores it in ctx.raw_df,
    and writes a QC report to outputs/reports.

    Unlike XES files (which use pm4py's standardized column names), CSV files
    can have arbitrary column names. This step:
    1. Loads the CSV file
    2. Optionally parses timestamp columns
    3. Stores raw_df for NormalizeSchemaStep to handle column mapping

    Produces:
      - ctx.raw_df (pd.DataFrame)
      - ctx.artifacts["loaded_columns"]
      - ctx.artifacts["load_csv_qc"]
      - ctx.artifacts["load_csv_report_path"]
    """
    name = "load_csv"

    def __init__(
        self,
        report_filename: str = "01_load_csv_qc.json",
        sample_rows: int = 3,
        # CSV-specific options
        delimiter: str = ",",
        encoding: str = "utf-8",
        parse_dates: Optional[List[str]] = None,
        date_format: Optional[str] = None,
    ):
        self.report_filename = report_filename
        self.sample_rows = int(sample_rows)
        self.delimiter = delimiter
        self.encoding = encoding
        self.parse_dates = parse_dates  # optional timestamp column names
        self.date_format = date_format

    def run(self, ctx: PipelineContext) -> PipelineContext:
        # --- Resolve input path ---
        if not getattr(ctx, "input_path", None):
            raise RuntimeError("ctx.input_path is missing. Provide a CSV path in the context.")
        path = Path(ctx.input_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        # --- Load CSV ---
        df = pd.read_csv(
            path,
            sep=self.delimiter,
            encoding=self.encoding,
            parse_dates=self.parse_dates or False,
        )
        if not isinstance(df, pd.DataFrame):
            raise RuntimeError(f"CSV load did not return a DataFrame, got {type(df)}")

        ctx.raw_df = df
        ctx.artifacts["loaded_columns"] = list(df.columns)

        # --- QC summary (kept small & stable) ---
        # best-effort pick canonical columns if present
        case_candidates = ["case_id", "CaseID", "case", "case:concept:name", "trace_id"]
        act_candidates = ["activity", "Activity", "task", "concept:name", "event"]
        ts_candidates = ["timestamp", "time", "datetime", "time:timestamp", "event_time"]

        case_col = next((c for c in case_candidates if c in df.columns), None)
        act_col = next((c for c in act_candidates if c in df.columns), None)
        ts_col = next((c for c in ts_candidates if c in df.columns), None)

        qc: Dict[str, Any] = {
            "step": self.name,
            "input_path": str(path),
            "format": "csv",
            "num_rows": int(len(df)),
            "num_columns": int(df.shape[1]),
            "columns": list(df.columns),
            "detected": {
                "case_col": case_col,
                "activity_col": act_col,
                "timestamp_col": ts_col,
            },
        }

        # add case/event stats if columns exist
        if case_col is not None:
            qc["num_cases"] = int(df[case_col].astype(str).nunique())
        if ts_col is not None:
            ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
            qc["timestamp_parse_na"] = int(ts.isna().sum())
            if ts.notna().any():
                qc["time_min"] = ts.min().isoformat()
                qc["time_max"] = ts.max().isoformat()

        # small preview
        try:
            qc["sample_rows"] = df.head(self.sample_rows).to_dict(orient="records")
        except Exception:
            qc["sample_rows"] = []

        ctx.artifacts["load_csv_qc"] = qc

        # --- Persist report to outputs/reports ---
        # Prefer ctx.output_dir if present; otherwise default to ./outputs
        output_dir = Path(getattr(ctx, "output_dir", "outputs"))
        reports_dir = output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        report_path = reports_dir / self.report_filename
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(qc), f, indent=2)

        ctx.artifacts["load_csv_report_path"] = str(report_path)

        return ctx
