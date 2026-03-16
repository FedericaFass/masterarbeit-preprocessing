from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np

from ..base import Step
from ppm_preprocessing.domain.context import PipelineContext
from ppm_preprocessing.steps.bucketing import BucketingStep, BucketingConfig
from ppm_preprocessing.steps.encoding import EncodingStep, EncodingConfig
from ppm_preprocessing.steps.legacy.model_selection import ModelSelectionStep, ModelSelectionConfig

from ppm_preprocessing.bucketing.no_bucket import NoBucketer
from ppm_preprocessing.bucketing.prefix_length import PrefixLengthBucketer


@dataclass
class CompareConfig:
    bucketers: List[str] = None  # ["no_bucket","prefix_length"]
    # candidates already in EncodingStep: last_n_5,last_n_10,aggregation
    selection_metric: str = "f1_macro"

class CompareBucketingEncodingStep(Step):
    name = "compare_strategies"

    def __init__(self, config: CompareConfig | None = None):
        self.config = config or CompareConfig(bucketers=["no_bucket", "prefix_length"])

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if "prefix_samples" not in ctx.artifacts:
            raise RuntimeError("Need prefix_samples before comparing strategies.")
        if "case_splits" not in ctx.artifacts:
            raise RuntimeError("Need case_splits before comparing strategies (CaseSplitStep).")

        experiments: Dict[str, Any] = {}

        bucketer_objs = {
            "no_bucket": NoBucketer(),
            "prefix_length": PrefixLengthBucketer(),
        }

        for bucketer_name in self.config.bucketers:
            bucketer = bucketer_objs[bucketer_name]

            # 1) Bucketing
            tmp_ctx = ctx  # shallow; we will overwrite artifacts keys used below
            tmp_ctx.artifacts["bucketed_prefixes"] = None

            BucketingStep(bucketer=bucketer, config=BucketingConfig()).run(tmp_ctx)

            # 2) Encoding candidates
            EncodingStep(EncodingConfig(last_n_values=[5, 10], use_aggregation=True)).run(tmp_ctx)

            # 3) Model selection (within this bucketing setup)
            # Important: ModelSelectionStep should use ctx.artifacts["case_splits"] for splitting.
            ModelSelectionStep(ModelSelectionConfig()).run(tmp_ctx)

            selection = tmp_ctx.artifacts["model_selection"]

            # Aggregate a global score for this bucketing strategy:
            # weighted average by bucket sample size
            encoded = tmp_ctx.artifacts["encoded_buckets"]
            bucket_sizes = {b: encoded[b]["num_rows"] for b in encoded.keys()}
            total = sum(bucket_sizes.values()) or 1

            weighted = 0.0
            for b, conf in selection.items():
                weighted += conf[self.config.selection_metric] * (bucket_sizes.get(b, 0) / total)

            exp_key = f"{bucketer_name}"
            experiments[exp_key] = {
                "bucketer": bucketer_name,
                "weighted_score": float(weighted),
                "per_bucket_best": selection,
                "bucket_sizes": bucket_sizes,
                "encoding_candidates": tmp_ctx.artifacts["encoding_qc"]["candidates"],
            }

        # choose best bucketing strategy overall
        best = max(experiments.items(), key=lambda kv: kv[1]["weighted_score"])

        ctx.artifacts["strategy_comparison"] = {
            "experiments": experiments,
            "best_overall": {"key": best[0], **best[1]},
        }
        return ctx
