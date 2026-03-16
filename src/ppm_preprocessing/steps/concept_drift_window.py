from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ppm_preprocessing.steps.base import Step
from ppm_preprocessing.domain.context import PipelineContext


@dataclass
class ConceptDriftWindowConfig:
    enabled: bool = False
    recent_pct: float = 80.0  # keep only the most recent X% of cases (by start time)


class ConceptDriftWindowStep(Step):
    """
    Concept-drift mitigation: keep only the most recent `recent_pct`% of cases
    (ranked by their first event timestamp).

    Old cases may reflect an outdated process version. Discarding them focuses
    training on recent behaviour — the key insight that Nirdizati explicitly
    lacks. Must run before CaseSplitStep.
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

        skip_n = int(n_cases * (1.0 - self.config.recent_pct / 100.0))
        keep_cases = set(case_start.index[skip_n:].astype(str))

        ctx.log.df = df[df["case_id"].astype(str).isin(keep_cases)].reset_index(drop=True)
        cases_after = int(ctx.log.df["case_id"].nunique())

        # Report the time range of kept vs discarded cases
        cutoff_time = case_start.iloc[skip_n] if skip_n < n_cases else None

        ctx.artifacts[qc_key] = {
            "enabled": True,
            "recent_pct": self.config.recent_pct,
            "cases_before": n_cases,
            "cases_after": cases_after,
            "cases_removed": n_cases - cases_after,
            "cases_removed_pct": round(100.0 * (n_cases - cases_after) / n_cases, 1) if n_cases else 0,
            "cutoff_time": str(cutoff_time) if cutoff_time is not None else None,
        }
        return ctx
