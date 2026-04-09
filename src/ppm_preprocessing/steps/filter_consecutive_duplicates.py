from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .base import Step
from ppm_preprocessing.domain.context import PipelineContext


@dataclass
class FilterConsecutiveDuplicatesConfig:
    case_col: str = "case_id"
    activity_col: str = "activity"
    ts_col: str = "timestamp"
    report_filename: str = "05f_filter_consecutive_duplicates_qc.json"


class FilterConsecutiveDuplicatesStep(Step):
    """
    Removes consecutive repeated activity events within each case.

    E.g. [A, A, B, B, B, C] → [A, B, C]

    Only removes when the same activity appears back-to-back in chronological
    order. Non-consecutive repetitions (e.g. [A, B, A]) are preserved.

    Rationale (Dakic et al. 2023): repeated consecutive activities are often
    logging artifacts (e.g. duplicate system events) rather than meaningful
    process steps. They inflate prefix lengths and add no discriminative
    information for sequence prediction.

    This step is OPTIONAL — enabled by user choice.

    Side effects:
      - updates ctx.log.df
      - writes QC to ctx.artifacts["filter_consecutive_duplicates_qc"]
    """
    name = "filter_consecutive_duplicates"

    def __init__(self, config: FilterConsecutiveDuplicatesConfig | None = None):
        self.config = config or FilterConsecutiveDuplicatesConfig()

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

        before_rows = int(len(df))

        # Sort within cases by timestamp to ensure correct consecutive detection
        df_sorted = df.sort_values([c.case_col, c.ts_col], kind="mergesort").reset_index(drop=True)

        # Within each case, detect if activity == previous activity in same case
        prev_case = df_sorted[c.case_col].shift(1)
        prev_act = df_sorted[c.activity_col].shift(1)

        is_consec_dup = (
            (df_sorted[c.case_col] == prev_case) &
            (df_sorted[c.activity_col] == prev_act)
        )

        df_filtered = df_sorted[~is_consec_dup].reset_index(drop=True)

        after_rows = int(len(df_filtered))
        dropped = before_rows - after_rows

        qc: Dict[str, Any] = {
            "step": self.name,
            "before_rows": before_rows,
            "after_rows": after_rows,
            "consecutive_duplicate_events_removed": dropped,
            "drop_rate": round(dropped / before_rows, 4) if before_rows else 0.0,
        }

        ctx.log.df = df_filtered
        ctx.artifacts["filter_consecutive_duplicates_qc"] = qc

        out_dir = self._report_dir(ctx)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / c.report_filename).write_text(
            pd.Series(qc).to_json(force_ascii=False), encoding="utf-8"
        )

        return ctx
