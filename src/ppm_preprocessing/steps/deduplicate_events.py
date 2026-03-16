from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd

from .base import Step
from ppm_preprocessing.domain.context import PipelineContext


@dataclass
class DeduplicateEventsConfig:
    # If None: auto-pick a reasonable set from available columns.
    keys: Optional[List[str]] = None
    keep: str = "first"  # pandas drop_duplicates keep=
    qc_key: str = "deduplicate_qc"

    # reporting
    report_filename: str = "03_deduplicate_events_qc.json"
    n_sample_rows: int = 3


class DeduplicateEventsStep(Step):
    """
    Drops duplicate events based on a key set.
    Safe if some optional columns don't exist.

    Side effects:
      - updates ctx.log.df
      - writes qc to ctx.artifacts[qc_key]
      - writes qc JSON to outputs/reports/<report_filename>
    """
    name = "deduplicate_events"

    def __init__(self, config: DeduplicateEventsConfig | None = None):
        self.config = config or DeduplicateEventsConfig()

    def _default_keys(self, df: pd.DataFrame) -> List[str]:
        base = ["case_id", "activity", "timestamp"]
        optional = ["lifecycle:transition", "org:resource", "EventID", "EventOrigin"]
        keys = [c for c in base if c in df.columns]
        keys += [c for c in optional if c in df.columns]

        # fallback: if timestamp is missing (shouldn't happen after normalization), at least case+activity
        if not keys:
            keys = [c for c in ["case_id", "activity"] if c in df.columns]
        return keys

    def _report_dir(self, ctx: PipelineContext) -> Path:
        # Prefer ctx.output_dir if you have it in your context; otherwise fallback
        # to the repo-local "outputs/reports".
        base = getattr(ctx, "output_dir", None)
        if base:
            return Path(base) / "reports"
        return Path("outputs") / "reports"

    @staticmethod
    def _safe_preview(df: pd.DataFrame, n: int) -> List[Dict[str, Any]]:
        if df is None or len(df) == 0 or n <= 0:
            return []
        # avoid dumping huge objects/lists
        return df.head(n).to_dict(orient="records")

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.log is None:
            raise RuntimeError("ctx.log is None. Run Load/Normalize first.")

        df = ctx.log.df
        if df is None or len(df) == 0:
            qc = {
                "step": self.name,
                "note": "empty input",
                "keys": [],
                "keep": self.config.keep,
                "before_events": 0,
                "after_events": 0,
                "dropped_events": 0,
            }
            ctx.artifacts[self.config.qc_key] = qc
            # still persist qc
            out_dir = self._report_dir(ctx)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / self.config.report_filename).write_text(
                pd.Series(qc).to_json(force_ascii=False), encoding="utf-8"
            )
            return ctx

        keys = self.config.keys or self._default_keys(df)

        before = int(len(df))

        # --- compute dupe stats BEFORE dropping ---
        # duplicated mask (excluding the first occurrence because keep="first")
        dupe_mask = df.duplicated(subset=keys, keep=self.config.keep)
        dropped_events = int(dupe_mask.sum())

        # key-group stats (how many key groups have >1 row)
        # (this is "scientific": tells you structure of duplicates)
        group_sizes = df.groupby(keys, dropna=False).size()
        num_duplicate_key_groups = int((group_sizes > 1).sum()) if len(group_sizes) else 0
        max_dupe_group_size = int(group_sizes.max()) if len(group_sizes) else 0

        # actual dedupe
        df2 = df.drop_duplicates(subset=keys, keep=self.config.keep).copy()
        after = int(len(df2))

        # sanity
        # dropped_events should equal before-after unless keep=None weirdness; here keep is fixed.
        dropped_events = int(before - after)

        # previews
        n = int(self.config.n_sample_rows)
        preview_cols = [c for c in ["case_id", "activity", "timestamp", "lifecycle:transition", "org:resource", "EventID"] if c in df.columns]
        head_before = self._safe_preview(df[preview_cols] if preview_cols else df, n)
        head_after = self._safe_preview(df2[preview_cols] if preview_cols else df2, n)

        qc: Dict[str, Any] = {
            "step": self.name,
            "keys": keys,
            "keep": self.config.keep,
            "before_events": before,
            "after_events": after,
            "dropped_events": dropped_events,
            "drop_rate": float(dropped_events / before) if before > 0 else 0.0,
            "num_duplicate_key_groups": num_duplicate_key_groups,
            "max_dupe_group_size": max_dupe_group_size,
            "preview_cols": preview_cols,
            "head_before": head_before,
            "head_after": head_after,
        }

        # update ctx
        ctx.log.df = df2
        ctx.artifacts[self.config.qc_key] = qc

        # persist qc JSON
        out_dir = self._report_dir(ctx)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / self.config.report_filename
        out_path.write_text(
            pd.Series(qc).to_json(force_ascii=False), encoding="utf-8"
        )

        # helpful pointer
        ctx.artifacts["last_report_path"] = str(out_path)

        return ctx
