from __future__ import annotations
from dataclasses import dataclass
from typing import List

from ppm_preprocessing.domain.context import PipelineContext
from ppm_preprocessing.steps.base import Step

@dataclass
class PreprocessingPipeline:
    steps: List[Step]

    def run(self, ctx: PipelineContext) -> PipelineContext:
        for step in self.steps:
            ctx = step.run(ctx)
        return ctx
