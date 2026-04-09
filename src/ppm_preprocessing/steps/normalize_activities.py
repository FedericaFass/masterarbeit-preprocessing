from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .base import Step
from ppm_preprocessing.domain.context import PipelineContext


@dataclass
class NormalizeActivitiesConfig:
    activity_col: str = "activity"
    null_placeholder: str = "UNKNOWN"
    report_filename: str = "05b_normalize_activities_qc.json"


class NormalizeActivitiesStep(Step):
    """
    Normalizes activity labels:
      1. Strip leading/trailing whitespace
      2. Replace empty strings and NaN with null_placeholder ("UNKNOWN")

    Rationale (Marin-Castro & Tello-Leal 2021): inconsistent activity names
    (whitespace variants, missing labels) create spurious vocabulary entries
    that inflate feature spaces and reduce model quality.

    Side effects:
      - updates ctx.log.df[activity_col] in-place
      - writes QC to ctx.artifacts["normalize_activities_qc"]
    """
    name = "normalize_activities"

    def __init__(self, config: NormalizeActivitiesConfig | None = None):
        self.config = config or NormalizeActivitiesConfig()

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

        col = c.activity_col
        if col not in df.columns:
            ctx.artifacts["normalize_activities_qc"] = {
                "step": self.name, "note": f"Column '{col}' not found, skipped."
            }
            return ctx

        before_unique = int(df[col].nunique(dropna=False))

        # Strip whitespace
        original_str = df[col].astype(str)
        stripped = original_str.str.strip()
        changed_mask = stripped != original_str
        whitespace_fixed = int(changed_mask.sum())
        activity_examples_changed = [
            {"before": b, "after": a}
            for b, a in zip(original_str[changed_mask].head(3).tolist(), stripped[changed_mask].head(3).tolist())
        ]

        # Replace empty strings and "nan" strings with placeholder
        stripped = stripped.replace({"": c.null_placeholder, "nan": c.null_placeholder, "None": c.null_placeholder})

        # Replace original NaNs too
        null_filled = int(df[col].isna().sum())

        df[col] = stripped

        after_unique = int(df[col].nunique(dropna=False))

        qc: Dict[str, Any] = {
            "step": self.name,
            "activity_col": col,
            "before_unique_activities": before_unique,
            "after_unique_activities": after_unique,
            "whitespace_events_fixed": whitespace_fixed,
            "null_events_filled": null_filled,
            "null_placeholder": c.null_placeholder,
            "activity_examples_changed": activity_examples_changed,
        }

        ctx.log.df = df
        ctx.artifacts["normalize_activities_qc"] = qc

        out_dir = self._report_dir(ctx)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / c.report_filename).write_text(
            pd.Series(qc).to_json(force_ascii=False), encoding="utf-8"
        )

        return ctx
