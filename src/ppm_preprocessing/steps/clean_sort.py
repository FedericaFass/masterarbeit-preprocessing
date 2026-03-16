from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd

from .base import Step
from ppm_preprocessing.domain.context import PipelineContext


@dataclass
class CleanAndSortConfig:
    case_col: str = "case_id"
    act_col: str = "activity"
    ts_col: str = "timestamp"

    # create stable tie-breaker
    event_index_col: str = "_event_index"

    # reporting
    report_filename: str = "04_clean_sort_qc.json"
    n_sample_rows: int = 3


class CleanAndSortStep(Step):
    """
    Clean types, parse timestamp, drop invalid rows, and apply a stable order within cases.

    Side effects:
      - updates ctx.log.df
      - writes QC to ctx.artifacts["clean_sort_qc"]
      - persists QC JSON to outputs/reports/<report_filename>
    """
    name = "clean_sort"

    def __init__(self, config: CleanAndSortConfig | None = None):
        self.config = config or CleanAndSortConfig()

    def _report_dir(self, ctx: PipelineContext) -> Path:
        base = getattr(ctx, "output_dir", None)
        if base:
            return Path(base) / "reports"
        return Path("outputs") / "reports"

    @staticmethod
    def _safe_preview(df: pd.DataFrame, cols: List[str], n: int) -> List[Dict[str, Any]]:
        if df is None or len(df) == 0 or n <= 0:
            return []
        use_cols = [c for c in cols if c in df.columns]
        if use_cols:
            return df[use_cols].head(n).to_dict(orient="records")
        return df.head(n).to_dict(orient="records")

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.log is None:
            raise RuntimeError("ctx.log is None. NormalizeSchemaStep must run before cleaning.")

        c = self.config
        df = ctx.log.df.copy()

        required = {c.case_col, c.act_col, c.ts_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CleanAndSortStep missing required columns: {missing}")

        # --- QC before ---
        before_rows = int(len(df))
        preview_cols = [c.case_col, c.act_col, c.ts_col, "lifecycle:transition", "org:resource", "EventID"]
        head_before = self._safe_preview(df, preview_cols, int(c.n_sample_rows))

        # types: keep all other columns unchanged
        df[c.case_col] = df[c.case_col].astype(str)
        df[c.act_col] = df[c.act_col].astype(str)

        ts_parsed = pd.to_datetime(df[c.ts_col], errors="coerce", utc=True)
        ts_na = int(ts_parsed.isna().sum())
        df[c.ts_col] = ts_parsed

        # drop invalid rows
        before_dropna = int(len(df))
        df = df.dropna(subset=[c.case_col, c.act_col, c.ts_col]).copy()
        after_dropna = int(len(df))
        dropped = int(before_dropna - after_dropna)
        drop_rate = float(dropped / before_dropna) if before_dropna > 0 else 0.0

        # stable ordering within same timestamp:
        # use original row order as tie-breaker (stable!)
        if c.event_index_col not in df.columns:
            df = df.reset_index(drop=False).rename(columns={"index": c.event_index_col})
        else:
            # ensure numeric-ish
            df[c.event_index_col] = pd.to_numeric(df[c.event_index_col], errors="coerce").fillna(0).astype(int)

        # stable sort
        df2 = df.sort_values([c.case_col, c.ts_col, c.event_index_col], kind="mergesort").reset_index(drop=True)

        # --- QC after ---
        head_after = self._safe_preview(df2, preview_cols + [c.event_index_col], int(c.n_sample_rows))

        num_cases = int(df2[c.case_col].nunique()) if len(df2) else 0
        time_min = df2[c.ts_col].min()
        time_max = df2[c.ts_col].max()

        qc: Dict[str, Any] = {
            "step": self.name,
            "case_col": c.case_col,
            "activity_col": c.act_col,
            "timestamp_col": c.ts_col,
            "event_index_col": c.event_index_col,
            "before_rows": before_rows,
            "timestamp_parse_na": ts_na,
            "dropped_rows_dropna": dropped,
            "drop_rate": drop_rate,
            "after_rows": int(len(df2)),
            "num_cases": num_cases,
            "time_min": None if pd.isna(time_min) else str(time_min),
            "time_max": None if pd.isna(time_max) else str(time_max),
            "head_before": head_before,
            "head_after": head_after,
            "note": "Only case_id/activity cast to str, timestamp parsed to UTC. Other columns preserved.",
        }

        # apply
        ctx.log.df = df2
        ctx.artifacts["clean_sort_qc"] = qc

        # persist
        out_dir = self._report_dir(ctx)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / c.report_filename
        out_path.write_text(pd.Series(qc).to_json(force_ascii=False), encoding="utf-8")
        ctx.artifacts["last_report_path"] = str(out_path)

        # also update meta (if you want)
        if hasattr(ctx.log, "meta") and isinstance(ctx.log.meta, dict):
            ctx.log.meta.update(
                {
                    "rows_raw_after_schema": before_rows,
                    "rows_after_dropna": int(len(df2)),
                    "timestamp_parse_na": ts_na,
                }
            )

        return ctx
