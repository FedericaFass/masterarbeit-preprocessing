from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Set, Dict, Any

import pandas as pd

from .base import Step
from ppm_preprocessing.domain.context import PipelineContext


@dataclass
class LifecycleFilterConfig:
    # Keep only these lifecycle values (case-insensitive).
    # Common: {"complete"}
    keep: Optional[Set[str]] = None

    # Column name in canonical log if you normalized it; otherwise it's often literally "lifecycle:transition".
    lifecycle_col_candidates: tuple[str, ...] = ("lifecycle:transition", "lifecycle", "transition")

    # QC artifact key
    qc_key: str = "lifecycle_filter_qc"


class LifecycleFilterStep(Step):
    """
    Filters lifecycle transitions, e.g. keep only "complete".
    Safe no-op if no lifecycle column exists or keep is None.
    """
    name = "lifecycle_filter"

    def __init__(self, config: LifecycleFilterConfig | None = None):
        self.config = config or LifecycleFilterConfig()

    def _find_lifecycle_col(self, df: pd.DataFrame) -> Optional[str]:
        for c in self.config.lifecycle_col_candidates:
            if c in df.columns:
                return c
        return None

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.log is None:
            raise RuntimeError("ctx.log is None. Run Load/Normalize first.")

        df = ctx.log.df
        col = self._find_lifecycle_col(df)

        qc: Dict[str, Any] = {
            "enabled": self.config.keep is not None,
            "lifecycle_col": col,
            "before_events": int(len(df)),
        }

        if col is None or self.config.keep is None:
            qc["after_events"] = int(len(df))
            qc["note"] = "no-op (no lifecycle column or keep=None)"
            ctx.artifacts[self.config.qc_key] = qc
            return ctx

        keep_norm = {k.lower().strip() for k in self.config.keep}

        # normalize lifecycle values
        s = df[col].astype(str).str.lower().str.strip()
        mask = s.isin(keep_norm)

        df2 = df.loc[mask].copy()

        qc["keep"] = sorted(list(keep_norm))
        qc["after_events"] = int(len(df2))
        qc["dropped_events"] = int(len(df) - len(df2))
        qc["kept_lifecycle_values_top10"] = (
            df2[col].astype(str).str.lower().str.strip().value_counts().head(10).to_dict()
        )

        ctx.log.df = df2
        ctx.artifacts[self.config.qc_key] = qc
        return ctx
