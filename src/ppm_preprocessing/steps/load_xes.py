from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.importer.xes import importer as xes_importer

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


class LoadXesStep(Step):
    """
    Loads an XES (.xes or .xes.gz) file via pm4py, converts to a DataFrame,
    stores it in ctx.raw_df, and writes a QC report to outputs/reports.

    Produces:
      - ctx.raw_df (pd.DataFrame)
      - ctx.artifacts["loaded_columns"]
      - ctx.artifacts["load_xes_qc"]
      - ctx.artifacts["load_xes_report_path"]
    """
    name = "load_xes"

    def __init__(
        self,
        report_filename: str = "01_load_xes_qc.json",
        sample_rows: int = 3,
    ):
        self.report_filename = report_filename
        self.sample_rows = int(sample_rows)

    def run(self, ctx: PipelineContext) -> PipelineContext:
        # --- Resolve input path ---
        if not getattr(ctx, "input_path", None):
            raise RuntimeError("ctx.input_path is missing. Provide an XES path in the context.")
        path = Path(ctx.input_path)
        if not path.exists():
            raise FileNotFoundError(f"XES file not found: {path}")

        # --- Load + convert ---
        log = xes_importer.apply(str(path))
        df = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
        if not isinstance(df, pd.DataFrame):
            raise RuntimeError(f"pm4py conversion did not return a DataFrame, got {type(df)}")

        ctx.raw_df = df
        ctx.artifacts["loaded_columns"] = list(df.columns)

        # --- QC summary (kept small & stable) ---
        # best-effort pick canonical columns if present (pm4py often uses these names)
        case_col = "case:concept:name" if "case:concept:name" in df.columns else ("case_id" if "case_id" in df.columns else None)
        act_col = "concept:name" if "concept:name" in df.columns else ("activity" if "activity" in df.columns else None)
        ts_col = "time:timestamp" if "time:timestamp" in df.columns else ("timestamp" if "timestamp" in df.columns else None)

        qc: Dict[str, Any] = {
            "step": self.name,
            "input_path": str(path),
            "format": "xes",
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

        ctx.artifacts["load_xes_qc"] = qc

        # --- Persist report to outputs/reports ---
        # Prefer ctx.output_dir if present; otherwise default to ./outputs
        output_dir = Path(getattr(ctx, "output_dir", "outputs"))
        reports_dir = output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        report_path = reports_dir / self.report_filename
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(qc), f, indent=2)

        ctx.artifacts["load_xes_report_path"] = str(report_path)

        return ctx
