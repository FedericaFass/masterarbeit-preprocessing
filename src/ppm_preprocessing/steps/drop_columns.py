from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .base import Step
from ppm_preprocessing.domain.context import PipelineContext


@dataclass
class DropColumnsConfig:
    columns: List[str] = field(default_factory=list)


class DropColumnsStep(Step):
    """
    Drops user-specified columns from the event log DataFrame.
    Columns that don't exist are silently skipped.
    """
    name = "drop_columns"

    def __init__(self, config: DropColumnsConfig | None = None):
        self.config = config or DropColumnsConfig()

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if not self.config.columns or ctx.log is None:
            return ctx

        to_drop = [c for c in self.config.columns if c in ctx.log.df.columns]
        if to_drop:
            ctx.log.df = ctx.log.df.drop(columns=to_drop)
            print(f"  [DropColumns] Dropped {len(to_drop)} column(s): {to_drop}", flush=True)
        else:
            print(f"  [DropColumns] No matching columns found to drop.", flush=True)

        return ctx
