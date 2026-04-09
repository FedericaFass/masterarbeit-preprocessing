from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .base import Step
from ppm_preprocessing.domain.context import PipelineContext


@dataclass
class ImputeMissingAttributesConfig:
    # Columns to skip entirely (canonical + structural cols)
    skip_cols: List[str] = field(default_factory=lambda: [
        "case_id", "activity", "timestamp", "_event_index",
        "lifecycle:transition", "org:resource",
    ])
    # Strategy: "mode" for categorical, "median" for numeric
    numeric_strategy: str = "median"
    categorical_strategy: str = "mode"
    # Only impute columns with at most this fraction of missing values
    max_missing_frac: float = 0.5
    report_filename: str = "05g_impute_missing_attributes_qc.json"


class ImputeMissingAttributesStep(Step):
    """
    Imputes missing values in case-level attribute columns.

    Strategy:
      - Numeric columns → fill with column median
      - Categorical / object columns → fill with column mode (most frequent)
      - Columns with > max_missing_frac missing are skipped (too sparse to impute reliably)

    Rationale (Marin-Castro & Tello-Leal 2021): missing attribute values
    cause NaN-propagation through feature encoders and degrade model quality.
    Simple statistical imputation is a robust baseline that avoids data loss
    while ensuring complete feature vectors.

    This step is OPTIONAL — enabled by user choice.

    Side effects:
      - updates ctx.log.df (missing values filled)
      - writes QC to ctx.artifacts["impute_missing_attributes_qc"]
    """
    name = "impute_missing_attributes"

    def __init__(self, config: ImputeMissingAttributesConfig | None = None):
        self.config = config or ImputeMissingAttributesConfig()

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

        skip = set(c.skip_cols)
        cols_to_impute = [col for col in df.columns if col not in skip]

        imputed: Dict[str, Any] = {}
        skipped_sparse: List[str] = []

        for col in cols_to_impute:
            n_missing = int(df[col].isna().sum())
            if n_missing == 0:
                continue

            missing_frac = n_missing / len(df)
            if missing_frac > c.max_missing_frac:
                skipped_sparse.append(col)
                continue

            if pd.api.types.is_numeric_dtype(df[col]):
                fill_val = df[col].median()
                strategy = c.numeric_strategy
            else:
                mode_vals = df[col].mode()
                fill_val = mode_vals.iloc[0] if len(mode_vals) else None
                strategy = c.categorical_strategy

            if fill_val is None:
                skipped_sparse.append(col)
                continue

            df[col] = df[col].fillna(fill_val)
            imputed[col] = {
                "strategy": strategy,
                "fill_value": str(fill_val),
                "n_filled": n_missing,
                "missing_frac": round(missing_frac, 4),
            }

        qc: Dict[str, Any] = {
            "step": self.name,
            "columns_imputed": len(imputed),
            "columns_skipped_sparse": skipped_sparse,
            "details": imputed,
        }

        ctx.log.df = df
        ctx.artifacts["impute_missing_attributes_qc"] = qc

        out_dir = self._report_dir(ctx)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / c.report_filename).write_text(
            pd.Series(qc).to_json(force_ascii=False), encoding="utf-8"
        )

        return ctx
