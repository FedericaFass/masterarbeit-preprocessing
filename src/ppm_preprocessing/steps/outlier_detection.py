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
class OutlierDetectionConfig:
    enabled: bool = True
    label_col: str = "label_remaining_time_sec"
    iqr_multiplier: float = 1.5
    fit_on_train_only: bool = True
    report_basename: str = "08_outlier_detection_qc"


class OutlierDetectionStep(Step):
    """
    IQR-based outlier detection on the label column.

    Computes bounds on TRAIN cases only (no leakage), then removes
    outlier prefix rows from the training set.  Val/test rows are
    never removed so evaluation reflects the real-world distribution.

    Toggle with ``config.enabled = False`` to skip entirely.
    """
    name = "outlier_detection"

    def __init__(self, config: OutlierDetectionConfig | None = None):
        self.config = config or OutlierDetectionConfig()

    def _get_reports_dir(self, ctx: PipelineContext) -> Path:
        out_dir = getattr(ctx, "output_dir", None)
        if out_dir:
            return Path(out_dir) / "reports"
        return Path("outputs") / "reports"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if not self.config.enabled:
            print("[OutlierDetection] Disabled -- skipping.")
            ctx.artifacts["outlier_detection_qc"] = {
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

        if label_col not in ps.columns:
            raise RuntimeError(
                f"Label column '{label_col}' not found in prefix_samples. "
                f"Available label cols: {[c for c in ps.columns if c.startswith('label_')]}"
            )

        train_cases = set(map(str, splits["train"]))

        # ----- compute bounds on train rows only -----
        train_mask = ps["case_id"].astype(str).isin(train_cases)
        train_labels = pd.to_numeric(ps.loc[train_mask, label_col], errors="coerce")
        train_labels = train_labels.dropna()

        if len(train_labels) == 0:
            print("[OutlierDetection] No valid train labels -- skipping.")
            ctx.artifacts["outlier_detection_qc"] = {
                "step": self.name,
                "enabled": True,
                "skipped": True,
                "reason": "no valid train labels",
            }
            return ctx

        q1 = float(train_labels.quantile(0.25))
        q3 = float(train_labels.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - self.config.iqr_multiplier * iqr
        upper = q3 + self.config.iqr_multiplier * iqr

        # ----- identify outlier rows (train only) -----
        labels_numeric = pd.to_numeric(ps[label_col], errors="coerce")
        is_train = ps["case_id"].astype(str).isin(train_cases)
        is_outlier = is_train & ((labels_numeric < lower) | (labels_numeric > upper))

        n_before = int(train_mask.sum())
        n_outliers = int(is_outlier.sum())

        # remove outlier rows from prefix_samples
        ps_clean = ps[~is_outlier].copy()

        # Reset index and prefix_row_id so downstream steps (encoding,
        # strategy search) can use row_idx as a valid positional index
        # into y_all = ps[label_col].to_numpy().
        ps_clean = ps_clean.reset_index(drop=True)
        ps_clean["prefix_row_id"] = np.arange(len(ps_clean))

        ctx.artifacts["prefix_samples"] = ps_clean

        # update train case set: drop cases that lost ALL their prefixes
        remaining_train_cases = set(
            ps_clean.loc[ps_clean["case_id"].astype(str).isin(train_cases), "case_id"]
            .astype(str)
            .unique()
        )
        removed_cases = train_cases - remaining_train_cases
        if removed_cases:
            splits["train"] = remaining_train_cases

        n_after = int(ps_clean["case_id"].astype(str).isin(remaining_train_cases).sum())

        # ----- QC report -----
        qc: Dict[str, Any] = {
            "step": self.name,
            "enabled": True,
            "label_col": label_col,
            "iqr_multiplier": self.config.iqr_multiplier,
            "train_label_stats": {
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower,
                "upper_bound": upper,
                "lower_bound_days": lower / 86400.0,
                "upper_bound_days": upper / 86400.0,
            },
            "train_rows_before": n_before,
            "train_rows_after": n_after,
            "outlier_rows_removed": n_outliers,
            "outlier_pct": round(100.0 * n_outliers / n_before, 2) if n_before else 0.0,
            "train_cases_removed": len(removed_cases),
            "total_rows_after": len(ps_clean),
        }
        ctx.artifacts["outlier_detection_qc"] = qc

        # persist report JSON
        reports_dir = self._get_reports_dir(ctx)
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / f"{self.config.report_basename}.json"
        report_path.write_text(json.dumps(qc, indent=2, default=str), encoding="utf-8")

        print(
            f"[OutlierDetection] IQR bounds: "
            f"[{lower / 86400:.1f}, {upper / 86400:.1f}] days  |  "
            f"Removed {n_outliers} train rows ({qc['outlier_pct']:.1f}%), "
            f"{len(removed_cases)} cases fully removed"
        )

        return ctx
