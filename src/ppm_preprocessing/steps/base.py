from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
from ppm_preprocessing.domain.context import PipelineContext

class Step(ABC):
    name: str = "unnamed_step"

    @abstractmethod
    def run(self, ctx: PipelineContext) -> PipelineContext:
        """Execute the step and return updated context."""
        raise NotImplementedError
