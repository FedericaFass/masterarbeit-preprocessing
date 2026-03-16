from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import json
import pandas as pd

from ppm_preprocessing.steps.base import Step
from ppm_preprocessing.domain.context import PipelineContext
from ppm_preprocessing.bucketing.base import Bucketer


@dataclass
class BucketingConfig:
    bucket_col: str = "bucket_id"

    # Optional: filter out extremely small buckets (None = keep all)
    min_bucket_size: Optional[int] = None

    # reporting
    report_basename: str = "07_bucketing_qc"
    n_report_examples: int = 3
    save_preview_csv: bool = True
    preview_rows_csv: int = 200


class BucketingStep(Step):
    name = "bucketing"

    def __init__(self, bucketer: Bucketer, config: BucketingConfig | None = None):
        self.bucketer = bucketer
        self.config = config or BucketingConfig()

    def _get_reports_dir(self, ctx: PipelineContext) -> Path:
        out_dir = getattr(ctx, "output_dir", None)
        if out_dir:
            return Path(out_dir) / "reports"
        return Path("outputs") / "reports"

    def _save_report(self, ctx: PipelineContext, payload: Dict[str, Any]) -> str:
        reports_dir = self._get_reports_dir(ctx)
        reports_dir.mkdir(parents=True, exist_ok=True)

        path = reports_dir / f"{self.config.report_basename}.json"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

        ctx.artifacts[f"{self.config.report_basename}_report_path"] = str(path)
        return str(path)

    def _maybe_fit_on_train(self, ctx: PipelineContext, prefixes: pd.DataFrame) -> None:
        """
        If bucketer exposes fit(), fit it on TRAIN cases only (leakage-safe).
        Requires ctx.artifacts['case_splits']['train'].
        """
        if not hasattr(self.bucketer, "fit") or not callable(getattr(self.bucketer, "fit")):
            return

        splits = ctx.artifacts.get("case_splits")
        if not splits or "train" not in splits:
            raise RuntimeError(
                f"{self.bucketer.name} requires fitting, but ctx.artifacts['case_splits']['train'] is missing. "
                f"Run CaseSplitStep before BucketingStep."
            )

        train_cases = set(map(str, splits["train"]))
        ps_train = prefixes[prefixes["case_id"].astype(str).isin(train_cases)]
        self.bucketer.fit(ps_train)

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if "prefix_samples" not in ctx.artifacts:
            raise RuntimeError("Missing ctx.artifacts['prefix_samples']. Run PrefixExtractionStep first.")

        prefixes: pd.DataFrame = ctx.artifacts["prefix_samples"]

        if len(prefixes) == 0:
            ctx.artifacts["bucketed_prefixes"] = prefixes.copy()
            qc = {
                "step": self.name,
                "bucketer": self.bucketer.name,
                "num_rows": 0,
                "num_buckets": 0,
            }
            ctx.artifacts["bucketing_qc"] = qc
            self._save_report(ctx, {"qc": qc, "examples": []})
            return ctx

        # fit on TRAIN if needed
        self._maybe_fit_on_train(ctx, prefixes)

        bucket_ids = self.bucketer.assign(prefixes)
        if not isinstance(bucket_ids, pd.Series) or not bucket_ids.index.equals(prefixes.index):
            raise RuntimeError("Bucketer.assign must return a pd.Series aligned with prefixes.index")

        out = prefixes.copy()
        out[self.config.bucket_col] = bucket_ids.astype("int32")

        dropped = 0
        if self.config.min_bucket_size is not None:
            counts = out[self.config.bucket_col].value_counts()
            keep = counts[counts >= int(self.config.min_bucket_size)].index
            before = len(out)
            out = out[out[self.config.bucket_col].isin(keep)].copy()
            dropped = before - len(out)

        bucket_counts = out[self.config.bucket_col].value_counts().sort_index()

        qc: Dict[str, Any] = {
            "step": self.name,
            "bucketer": self.bucketer.name,
            "num_rows": int(len(out)),
            "num_buckets": int(bucket_counts.shape[0]),
            "min_bucket_size": int(bucket_counts.min()) if len(bucket_counts) else 0,
            "median_bucket_size": float(bucket_counts.median()) if len(bucket_counts) else 0.0,
            "max_bucket_size": int(bucket_counts.max()) if len(bucket_counts) else 0,
            "dropped_rows_small_buckets": int(dropped),
            "bucket_sizes_preview": bucket_counts.head(50).to_dict(),
        }

        # include learned state (adaptive bucketers)
        for attr in ["edges_", "median_", "max_"]:
            if hasattr(self.bucketer, attr):
                qc[attr] = getattr(self.bucketer, attr)

        ctx.artifacts["bucketed_prefixes"] = out
        ctx.artifacts["bucketing_qc"] = qc

        # examples for report (JSON-safe)
        ex_df = out.head(int(self.config.n_report_examples)).copy()
        ex_df = ex_df.where(pd.notna(ex_df), None)
        examples = ex_df.to_dict(orient="records")

        report = {
            "qc": qc,
            "examples": examples,
            "note": "bucketed_prefixes is stored in ctx.artifacts['bucketed_prefixes'].",
        }
        self._save_report(ctx, report)

        # preview CSV
        if self.config.save_preview_csv:
            reports_dir = self._get_reports_dir(ctx)
            reports_dir.mkdir(parents=True, exist_ok=True)

            preview = out.head(int(self.config.preview_rows_csv)).copy()
            if "prefix_activities" in preview.columns:
                preview["prefix_activities"] = preview["prefix_activities"].astype(str)

            preview_path = reports_dir / "bucketed_prefixes_preview.csv"
            preview.to_csv(preview_path, index=False, encoding="utf-8")
            ctx.artifacts["bucketed_prefixes_preview_csv"] = str(preview_path)

        return ctx
