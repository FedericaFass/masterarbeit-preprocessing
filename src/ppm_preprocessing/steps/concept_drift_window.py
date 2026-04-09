from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ppm_preprocessing.steps.base import Step
from ppm_preprocessing.domain.context import PipelineContext


@dataclass
class ConceptDriftWindowConfig:
    enabled: bool = False
    # Keep only cases that started on or after this date (ISO string, e.g. "2022-01-01")
    since_date: Optional[str] = None
    # Legacy: keep only the most recent X% of cases (used when since_date is None)
    recent_pct: float = 80.0


class ConceptDriftWindowStep(Step):
    """
    Time window filter: discard cases that started before a user-defined cutoff date.

    When since_date is set, only cases whose first event is >= since_date are kept.
    Falls back to recent_pct (keep last X% by rank) if since_date is not set.

    Rationale: older cases may reflect outdated process versions. Focusing training
    on recent data improves generalisation to the current process.
    Must run before CaseSplitStep.
    """
    name = "concept_drift_window"

    def __init__(self, config: ConceptDriftWindowConfig | None = None):
        self.config = config or ConceptDriftWindowConfig()

    def run(self, ctx: PipelineContext) -> PipelineContext:
        qc_key = "concept_drift_window_qc"
        if not self.config.enabled:
            ctx.artifacts[qc_key] = {"enabled": False}
            return ctx

        if ctx.log is None:
            ctx.artifacts[qc_key] = {"enabled": False, "skipped": "no log"}
            return ctx

        df = ctx.log.df
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

        # Case start time = earliest event per case
        case_start = df.groupby("case_id")["timestamp"].min().sort_values()
        n_cases = len(case_start)

        if self.config.since_date:
            cutoff = pd.Timestamp(self.config.since_date, tz="UTC")
            keep_cases = set(case_start[case_start >= cutoff].index.astype(str))
            cutoff_time = cutoff
        else:
            skip_n = int(n_cases * (1.0 - self.config.recent_pct / 100.0))
            keep_cases = set(case_start.index[skip_n:].astype(str))
            cutoff_time = case_start.iloc[skip_n] if skip_n < n_cases else None

        ctx.log.df = df[df["case_id"].astype(str).isin(keep_cases)].reset_index(drop=True)
        cases_after = int(ctx.log.df["case_id"].nunique())

        ctx.artifacts[qc_key] = {
            "enabled": True,
            "since_date": self.config.since_date,
            "recent_pct": self.config.recent_pct if not self.config.since_date else None,
            "cases_before": n_cases,
            "cases_after": cases_after,
            "cases_removed": n_cases - cases_after,
            "cases_removed_pct": round(100.0 * (n_cases - cases_after) / n_cases, 1) if n_cases else 0,
            "cutoff_time": str(cutoff_time) if cutoff_time is not None else None,
        }
        return ctx
