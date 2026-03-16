from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Any, List

import pandas as pd

from .base import Step
from ppm_preprocessing.domain.context import PipelineContext


@dataclass
class StableSortConfig:
    case_col: str = "case_id"
    time_col: str = "timestamp"

    # tie-breaker candidates, in priority order
    tie_breakers: Tuple[str, ...] = ("_event_index", "EventID")

    # reporting
    qc_key: str = "stable_sort_qc"
    report_filename: str = "05_stable_sort_qc.json"
    n_sample_rows: int = 3


class StableSortStep(Step):
    """
    Stable sort by case_id, timestamp, and available tie-breakers.
    Persists a QC report to outputs/reports.
    """
    name = "stable_sort"

    def __init__(self, config: StableSortConfig | None = None):
        self.config = config or StableSortConfig()

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
            raise RuntimeError("ctx.log is None. Run Load/Normalize first.")

        c = self.config
        df = ctx.log.df

        required = {c.case_col, c.time_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns for sorting: {missing}")

        sort_cols = [c.case_col, c.time_col]
        used_ties = []
        for t in c.tie_breakers:
            if t in df.columns:
                sort_cols.append(t)
                used_ties.append(t)

        before_head = self._safe_preview(df, sort_cols + ["activity"], int(c.n_sample_rows))

        # stable sort
        df2 = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

        after_head = self._safe_preview(df2, sort_cols + ["activity"], int(c.n_sample_rows))

        qc: Dict[str, Any] = {
            "step": self.name,
            "sort_cols": sort_cols,
            "used_tie_breakers": used_ties,
            "num_rows": int(len(df2)),
            "num_cases": int(df2[c.case_col].nunique()) if len(df2) else 0,
            "head_before": before_head,
            "head_after": after_head,
            "note": "Stable mergesort used. No rows changed/dropped; only ordering.",
        }

        ctx.log.df = df2
        ctx.artifacts[c.qc_key] = qc

        # persist
        out_dir = self._report_dir(ctx)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / c.report_filename
        out_path.write_text(pd.Series(qc).to_json(force_ascii=False), encoding="utf-8")
        ctx.artifacts["last_report_path"] = str(out_path)

        return ctx
