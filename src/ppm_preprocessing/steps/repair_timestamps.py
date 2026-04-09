from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .base import Step
from ppm_preprocessing.domain.context import PipelineContext


@dataclass
class RepairTimestampsConfig:
    case_col: str = "case_id"
    ts_col: str = "timestamp"
    # Strategy: "forward_fill" (raise backwards ts to previous ts) or "drop" (remove the offending event)
    strategy: str = "forward_fill"
    report_filename: str = "04b_repair_timestamps_qc.json"


class RepairTimestampsStep(Step):
    """
    Repairs out-of-order (backwards) timestamps within cases.

    Within each case the events are expected to be non-decreasing in time.
    When an event's timestamp is strictly earlier than the previous event's
    timestamp it is a data quality issue (e.g. logging artifact, clock skew).

    Strategies:
      - "forward_fill": clamp the offending timestamp up to the preceding
        event's timestamp (preserves the event, makes sequence non-decreasing)
      - "drop": remove the offending event entirely

    Rationale (Dakic et al. 2023): timestamp inconsistencies corrupt remaining-
    time labels and elapsed-time features, leading to negative values that
    silently degrade model quality.

    Side effects:
      - updates ctx.log.df
      - writes QC to ctx.artifacts["repair_timestamps_qc"]
    """
    name = "repair_timestamps"

    def __init__(self, config: RepairTimestampsConfig | None = None):
        self.config = config or RepairTimestampsConfig()

    def _report_dir(self, ctx: PipelineContext) -> Path:
        base = ctx.artifacts.get("out_dir") or getattr(ctx, "output_dir", None)
        if base:
            return Path(base) / "reports"
        return Path("outputs") / "reports"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.log is None:
            raise RuntimeError("ctx.log is None — run NormalizeSchemaStep first.")

        c = self.config
        df = ctx.log.df.copy()

        ts = pd.to_datetime(df[c.ts_col], errors="coerce", utc=True)
        df[c.ts_col] = ts

        before_rows = int(len(df))
        repaired = 0
        dropped = 0
        example_repairs = []  # [{case_id, activity, before, after}]

        if c.strategy == "forward_fill":
            # Within each case, if ts[i] < ts[i-1], set ts[i] = ts[i-1]
            def _repair_group(g: pd.DataFrame) -> pd.DataFrame:
                nonlocal repaired
                ts_vals = g[c.ts_col].copy()
                for i in range(1, len(ts_vals)):
                    if pd.notna(ts_vals.iloc[i]) and pd.notna(ts_vals.iloc[i - 1]):
                        if ts_vals.iloc[i] < ts_vals.iloc[i - 1]:
                            if len(example_repairs) < 3:
                                act_col = "activity" if "activity" in g.columns else c.case_col
                                example_repairs.append({
                                    "case_id": str(g[c.case_col].iloc[i]),
                                    "activity": str(g[act_col].iloc[i]) if "activity" in g.columns else "",
                                    "before": str(ts_vals.iloc[i]),
                                    "after": str(ts_vals.iloc[i - 1]),
                                })
                            ts_vals.iloc[i] = ts_vals.iloc[i - 1]
                            repaired += 1
                g = g.copy()
                g[c.ts_col] = ts_vals
                return g

            df = df.groupby(c.case_col, group_keys=False).apply(_repair_group).reset_index(drop=True)

        elif c.strategy == "drop":
            # Within each case, drop events where ts[i] < ts[i-1]
            prev_ts = df.groupby(c.case_col)[c.ts_col].shift(1)
            is_backwards = df[c.ts_col] < prev_ts
            # first event per case is always fine (prev_ts is NaT)
            is_backwards = is_backwards.fillna(False)
            dropped = int(is_backwards.sum())
            df = df[~is_backwards].reset_index(drop=True)
        else:
            raise ValueError(f"Unknown strategy: {c.strategy!r}. Use 'forward_fill' or 'drop'.")

        after_rows = int(len(df))

        qc: Dict[str, Any] = {
            "step": self.name,
            "strategy": c.strategy,
            "before_rows": before_rows,
            "after_rows": after_rows,
            "backwards_timestamps_repaired": repaired if c.strategy == "forward_fill" else dropped,
            "example_repairs": example_repairs,
        }

        ctx.log.df = df
        ctx.artifacts["repair_timestamps_qc"] = qc

        out_dir = self._report_dir(ctx)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / c.report_filename).write_text(
            pd.Series(qc).to_json(force_ascii=False), encoding="utf-8"
        )

        return ctx
