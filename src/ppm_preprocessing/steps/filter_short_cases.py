from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .base import Step
from ppm_preprocessing.domain.context import PipelineContext


@dataclass
class FilterShortCasesConfig:
    min_events: int = 2
    case_col: str = "case_id"
    report_filename: str = "05a_filter_short_cases_qc.json"


class FilterShortCasesStep(Step):
    """
    Removes cases with fewer than `min_events` events.

    Rationale (Marin-Castro & Tello-Leal 2021): cases with only 1 event
    carry no sequence information and degrade PPM model quality.

    Side effects:
      - updates ctx.log.df
      - writes QC to ctx.artifacts["filter_short_cases_qc"]
    """
    name = "filter_short_cases"

    def __init__(self, config: FilterShortCasesConfig | None = None):
        self.config = config or FilterShortCasesConfig()

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

        case_sizes = df.groupby(c.case_col).size()
        short_cases = case_sizes[case_sizes < c.min_events].index
        n_short = int(len(short_cases))

        before_cases = int(df[c.case_col].nunique())
        before_rows = int(len(df))

        if n_short > 0:
            df = df[~df[c.case_col].isin(short_cases)].reset_index(drop=True)

        after_cases = int(df[c.case_col].nunique())
        after_rows = int(len(df))

        qc: Dict[str, Any] = {
            "step": self.name,
            "min_events": c.min_events,
            "before_cases": before_cases,
            "before_rows": before_rows,
            "short_cases_removed": n_short,
            "short_cases_pct": round(100.0 * n_short / before_cases, 2) if before_cases else 0.0,
            "after_cases": after_cases,
            "after_rows": after_rows,
        }

        ctx.log.df = df
        ctx.artifacts["filter_short_cases_qc"] = qc

        out_dir = self._report_dir(ctx)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / c.report_filename).write_text(
            pd.Series(qc).to_json(force_ascii=False), encoding="utf-8"
        )

        return ctx
