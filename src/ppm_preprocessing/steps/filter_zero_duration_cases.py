from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .base import Step
from ppm_preprocessing.domain.context import PipelineContext


@dataclass
class FilterZeroDurationCasesConfig:
    case_col: str = "case_id"
    ts_col: str = "timestamp"
    report_filename: str = "05d_filter_zero_duration_cases_qc.json"


class FilterZeroDurationCasesStep(Step):
    """
    Removes cases whose total duration is zero or negative
    (i.e. min timestamp == max timestamp or reversed).

    Rationale (Dakic et al. 2023): zero-duration cases are data quality
    artifacts that produce zero remaining time / zero cycle time labels,
    which distort regression targets and inflate accuracy for classifiers.

    Side effects:
      - updates ctx.log.df
      - writes QC to ctx.artifacts["filter_zero_duration_cases_qc"]
    """
    name = "filter_zero_duration_cases"

    def __init__(self, config: FilterZeroDurationCasesConfig | None = None):
        self.config = config or FilterZeroDurationCasesConfig()

    def _report_dir(self, ctx: PipelineContext) -> Path:
        base = ctx.artifacts.get("out_dir") or getattr(ctx, "output_dir", None)
        if base:
            return Path(base) / "reports"
        return Path("outputs") / "reports"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.log is None:
            raise RuntimeError("ctx.log is None — run NormalizeSchemaStep first.")

        c = self.config
        df = ctx.log.df

        before_cases = int(df[c.case_col].nunique())
        before_rows = int(len(df))

        ts = pd.to_datetime(df[c.ts_col], errors="coerce", utc=True)
        df2 = df.copy()
        df2["_ts_parsed"] = ts

        case_duration = (
            df2.groupby(c.case_col)["_ts_parsed"]
            .agg(lambda x: (x.max() - x.min()).total_seconds())
        )
        zero_cases = case_duration[case_duration <= 0].index

        n_zero = int(len(zero_cases))
        if n_zero > 0:
            df = df[~df[c.case_col].isin(zero_cases)].reset_index(drop=True)

        after_cases = int(df[c.case_col].nunique())
        after_rows = int(len(df))

        qc: Dict[str, Any] = {
            "step": self.name,
            "before_cases": before_cases,
            "before_rows": before_rows,
            "zero_duration_cases_removed": n_zero,
            "zero_duration_pct": round(100.0 * n_zero / before_cases, 2) if before_cases else 0.0,
            "after_cases": after_cases,
            "after_rows": after_rows,
        }

        ctx.log.df = df
        ctx.artifacts["filter_zero_duration_cases_qc"] = qc

        out_dir = self._report_dir(ctx)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / c.report_filename).write_text(
            pd.Series(qc).to_json(force_ascii=False), encoding="utf-8"
        )

        return ctx
