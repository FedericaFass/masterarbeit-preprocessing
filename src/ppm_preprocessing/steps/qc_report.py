from __future__ import annotations

from .base import Step
from ppm_preprocessing.domain.context import PipelineContext

class QcReportStep(Step):
    name = "qc_report"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.log is None:
            raise RuntimeError("log is None. Previous steps must create canonical log.")

        df = ctx.log.df
        grp_sizes = df.groupby("case_id").size()

        report = {
            "num_events": int(len(df)),
            "num_cases": int(df["case_id"].nunique()),
            "trace_len_min": int(grp_sizes.min()),
            "trace_len_median": float(grp_sizes.median()),
            "trace_len_max": int(grp_sizes.max()),
            "time_min": str(df["timestamp"].min()),
            "time_max": str(df["timestamp"].max()),
            "columns": list(df.columns),
        }

        ctx.artifacts["qc_report"] = report
        return ctx
