from __future__ import annotations

from dataclasses import dataclass

from ppm_preprocessing.steps.base import Step
from ppm_preprocessing.domain.context import PipelineContext


@dataclass
class FilterRareVariantsConfig:
    enabled: bool = False
    min_variant_count: int = 5  # remove cases whose trace pattern appears fewer than N times


class FilterRareVariantsStep(Step):
    """
    Remove cases whose activity sequence (variant) occurs fewer than
    min_variant_count times in the log.

    Rare variants are hard to learn from — they represent edge-cases or
    data entry errors that would hurt generalisation without contributing
    signal. Must run on ctx.log before PrefixExtractionStep.
    """
    name = "filter_rare_variants"

    def __init__(self, config: FilterRareVariantsConfig | None = None):
        self.config = config or FilterRareVariantsConfig()

    def run(self, ctx: PipelineContext) -> PipelineContext:
        qc_key = "filter_rare_variants_qc"
        if not self.config.enabled:
            ctx.artifacts[qc_key] = {"enabled": False}
            return ctx

        if ctx.log is None:
            ctx.artifacts[qc_key] = {"enabled": False, "skipped": "no log"}
            return ctx

        df = ctx.log.df

        # Compute variant per case = ordered tuple of activities
        variant_per_case = (
            df.sort_values(["case_id", "timestamp"])
            .groupby("case_id", sort=False)["activity"]
            .apply(lambda x: tuple(x.astype(str).tolist()))
        )
        variant_counts = variant_per_case.value_counts()

        rare_variants = set(variant_counts[variant_counts < self.config.min_variant_count].index)
        rare_cases = set(
            variant_per_case[variant_per_case.isin(rare_variants)].index.astype(str)
        )

        cases_before = int(df["case_id"].nunique())
        ctx.log.df = df[~df["case_id"].astype(str).isin(rare_cases)].reset_index(drop=True)
        cases_after = int(ctx.log.df["case_id"].nunique())

        ctx.artifacts[qc_key] = {
            "enabled": True,
            "min_variant_count": self.config.min_variant_count,
            "cases_before": cases_before,
            "cases_after": cases_after,
            "cases_removed": cases_before - cases_after,
            "cases_removed_pct": round(100.0 * (cases_before - cases_after) / cases_before, 1) if cases_before else 0,
            "variants_total": int(len(variant_counts)),
            "variants_rare": int(len(rare_variants)),
        }
        return ctx
