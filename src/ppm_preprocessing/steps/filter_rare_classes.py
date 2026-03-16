from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
import json

import numpy as np
import pandas as pd

from .base import Step
from ppm_preprocessing.domain.context import PipelineContext


@dataclass
class FilterRareClassesConfig:
    enabled: bool = False
    min_class_samples: int = 10
    label_col: str = "label_next_activity"
    report_basename: str = "09_filter_rare_classes_qc"


class FilterRareClassesStep(Step):
    """
    Removes prefixes whose label class has fewer than `min_class_samples`
    training samples. Only useful for classification tasks.

    Class frequencies are computed on TRAIN split only (no leakage),
    but rows are removed from ALL splits so val/test don't contain
    classes the model has never seen.

    Toggle with ``config.enabled = False`` or ``min_class_samples = 0``
    to skip entirely.
    """
    name = "filter_rare_classes"

    def __init__(self, config: FilterRareClassesConfig | None = None):
        self.config = config or FilterRareClassesConfig()

    def _get_reports_dir(self, ctx: PipelineContext) -> Path:
        out_dir = getattr(ctx, "output_dir", None)
        if out_dir:
            return Path(out_dir) / "reports"
        return Path("outputs") / "reports"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if not self.config.enabled or self.config.min_class_samples <= 0:
            print("[FilterRareClasses] Disabled -- skipping.")
            ctx.artifacts["filter_rare_classes_qc"] = {
                "step": self.name,
                "enabled": False,
            }
            return ctx

        if "prefix_samples" not in ctx.artifacts:
            raise RuntimeError("prefix_samples missing. Run PrefixExtractionStep first.")
        if "case_splits" not in ctx.artifacts:
            raise RuntimeError("case_splits missing. Run CaseSplitStep first.")

        ps: pd.DataFrame = ctx.artifacts["prefix_samples"]
        splits = ctx.artifacts["case_splits"]
        label_col = self.config.label_col
        min_samples = self.config.min_class_samples

        if label_col not in ps.columns:
            raise RuntimeError(
                f"Label column '{label_col}' not found in prefix_samples. "
                f"Available label cols: {[c for c in ps.columns if str(c).startswith('label_')]}"
            )

        train_cases = set(map(str, splits["train"]))
        n_rows_before = len(ps)

        # Compute class frequencies on TRAIN only
        train_mask = ps["case_id"].astype(str).isin(train_cases)
        train_labels = ps.loc[train_mask, label_col].astype(str)
        train_vc = train_labels.value_counts()

        classes_before = int(train_vc.shape[0])
        rare_classes = set(train_vc[train_vc < min_samples].index)
        kept_classes = set(train_vc[train_vc >= min_samples].index)

        if not rare_classes:
            print(f"[FilterRareClasses] No rare classes found (all >= {min_samples} train samples).")
            ctx.artifacts["filter_rare_classes_qc"] = {
                "step": self.name,
                "enabled": True,
                "min_class_samples": min_samples,
                "classes_before": classes_before,
                "rare_classes_removed": 0,
                "rows_removed": 0,
            }
            return ctx

        # Remove ALL prefixes (train + val + test) with rare labels
        keep_mask = ps[label_col].astype(str).isin(kept_classes)
        ps_filtered = ps[keep_mask].copy()

        # Re-index prefix_row_id
        ps_filtered = ps_filtered.reset_index(drop=True)
        ps_filtered["prefix_row_id"] = np.arange(len(ps_filtered))

        ctx.artifacts["prefix_samples"] = ps_filtered

        # Update case_splits: remove cases that lost ALL prefixes
        for split_name in ["train", "val", "test"]:
            if split_name not in splits:
                continue
            old_cases = set(map(str, splits[split_name]))
            remaining = set(
                ps_filtered.loc[
                    ps_filtered["case_id"].astype(str).isin(old_cases), "case_id"
                ].astype(str).unique()
            )
            splits[split_name] = remaining

        n_rows_after = len(ps_filtered)
        n_removed = n_rows_before - n_rows_after
        classes_after = int(ps_filtered[label_col].astype(str).nunique())

        # QC report
        rare_class_details = {
            str(cls): int(train_vc[cls])
            for cls in sorted(rare_classes)
        }

        qc: Dict[str, Any] = {
            "step": self.name,
            "enabled": True,
            "label_col": label_col,
            "min_class_samples": min_samples,
            "classes_before": classes_before,
            "classes_after": classes_after,
            "rare_classes_removed": len(rare_classes),
            "rows_before": n_rows_before,
            "rows_after": n_rows_after,
            "rows_removed": n_removed,
            "rows_removed_pct": round(100.0 * n_removed / n_rows_before, 2) if n_rows_before else 0.0,
            "rare_classes": rare_class_details,
        }
        ctx.artifacts["filter_rare_classes_qc"] = qc

        # Persist report
        reports_dir = self._get_reports_dir(ctx)
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / f"{self.config.report_basename}.json"
        report_path.write_text(json.dumps(qc, indent=2, default=str), encoding="utf-8")

        print(
            f"[FilterRareClasses] Removed {len(rare_classes)} rare classes "
            f"({n_removed} rows, {qc['rows_removed_pct']:.1f}%)  |  "
            f"Classes: {classes_before} → {classes_after}  |  "
            f"Threshold: {min_samples} train samples"
        )

        return ctx
