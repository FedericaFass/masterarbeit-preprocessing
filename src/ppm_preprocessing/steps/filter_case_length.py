from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .base import Step
from ppm_preprocessing.domain.context import PipelineContext


@dataclass
class FilterCaseLengthConfig:
    # Remove cases longer than this percentile of case lengths
    max_length_percentile: float = 99.0
    # Hard cap — if set, overrides percentile
    hard_max_length: Optional[int] = None
    case_col: str = "case_id"
    report_filename: str = "05e_filter_case_length_qc.json"


class FilterCaseLengthStep(Step):
    """
    Removes cases whose event count exceeds a configurable percentile
    threshold (default: 99th percentile of case lengths).

    Rationale (Fani Sani et al. 2020, Dakic et al. 2023): extremely long
    cases are structural outliers that skew prefix extraction and make
    feature matrices unnecessarily sparse. Capping at p99 retains 99% of
    cases unchanged while removing the heaviest tail.

    Side effects:
      - updates ctx.log.df
      - writes QC to ctx.artifacts["filter_case_length_qc"]
    """
    name = "filter_case_length"

    def __init__(self, config: FilterCaseLengthConfig | None = None):
        self.config = config or FilterCaseLengthConfig()

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

        case_lengths = df.groupby(c.case_col).size()

        if c.hard_max_length is not None:
            threshold = int(c.hard_max_length)
        else:
            threshold = int(case_lengths.quantile(c.max_length_percentile / 100.0))

        long_cases = case_lengths[case_lengths > threshold].index
        n_long = int(len(long_cases))

        if n_long > 0:
            df = df[~df[c.case_col].isin(long_cases)].reset_index(drop=True)

        after_cases = int(df[c.case_col].nunique())
        after_rows = int(len(df))

        qc: Dict[str, Any] = {
            "step": self.name,
            "max_length_percentile": c.max_length_percentile,
            "hard_max_length": c.hard_max_length,
            "computed_threshold": threshold,
            "before_cases": before_cases,
            "before_rows": before_rows,
            "long_cases_removed": n_long,
            "long_cases_pct": round(100.0 * n_long / before_cases, 2) if before_cases else 0.0,
            "after_cases": after_cases,
            "after_rows": after_rows,
            "median_case_length": int(case_lengths.median()),
            "p99_case_length": int(case_lengths.quantile(0.99)),
        }

        ctx.log.df = df
        ctx.artifacts["filter_case_length_qc"] = qc

        out_dir = self._report_dir(ctx)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / c.report_filename).write_text(
            pd.Series(qc).to_json(force_ascii=False), encoding="utf-8"
        )

        return ctx
