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

        # --- Case length stats ---
        report = {
            "num_events": int(len(df)),
            "num_cases": int(df["case_id"].nunique()),
            "trace_len_min":    int(grp_sizes.min()),
            "trace_len_median": float(grp_sizes.median()),
            "trace_len_mean":   float(grp_sizes.mean()),
            "trace_len_p95":    float(grp_sizes.quantile(0.95)),
            "trace_len_max":    int(grp_sizes.max()),
            "time_min": str(df["timestamp"].min()),
            "time_max": str(df["timestamp"].max()),
            "columns": list(df.columns),
        }

        # --- Case duration stats (days) ---
        try:
            import pandas as _pd
            ts = _pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df_ts = df[["case_id"]].copy()
            df_ts["ts"] = ts
            grp_ts = df_ts.groupby("case_id")["ts"]
            durations_days = (grp_ts.max() - grp_ts.min()).dt.total_seconds() / 86400
            report["case_duration_days"] = {
                "min":    round(float(durations_days.min()), 2),
                "median": round(float(durations_days.median()), 2),
                "mean":   round(float(durations_days.mean()), 2),
                "p95":    round(float(durations_days.quantile(0.95)), 2),
                "max":    round(float(durations_days.max()), 2),
                "zero_duration_cases": int((durations_days == 0).sum()),
            }
        except Exception:
            pass

        # --- Activity frequency ---
        try:
            act_counts = df["activity"].value_counts()
            report["num_unique_activities"] = int(len(act_counts))
            report["activities_top10"] = [
                {"activity": str(k), "count": int(v)}
                for k, v in act_counts.head(10).items()
            ]
            report["activities_bottom5"] = [
                {"activity": str(k), "count": int(v)}
                for k, v in act_counts.tail(5).items()
            ]
        except Exception:
            pass

        # --- Consecutive duplicate events ---
        try:
            shifted = df.groupby("case_id")["activity"].shift(1)
            consec_dups = int((df["activity"] == shifted).sum())
            report["consecutive_duplicate_events"] = consec_dups
        except Exception:
            pass

        # --- Missing values per column (excluding core cols) ---
        try:
            core_cols = {"case_id", "activity", "timestamp"}
            attr_cols = [c for c in df.columns if c not in core_cols and not c.startswith("_")]
            missing = {}
            for col in attr_cols:
                n_missing = int(df[col].isna().sum())
                if n_missing > 0:
                    missing[col] = {
                        "missing": n_missing,
                        "pct": round(n_missing / len(df) * 100, 1),
                    }
            report["missing_values"] = missing
            report["columns_with_missing"] = len(missing)
        except Exception:
            pass

        # --- Trace variants ---
        try:
            variants = (
                df.sort_values(["case_id", "timestamp"])
                .groupby("case_id")["activity"]
                .apply(lambda x: tuple(x))
            )
            variant_counts = variants.value_counts()
            report["num_unique_variants"] = int(len(variant_counts))
            report["variants_top5"] = [
                {"variant": " → ".join(v), "count": int(c)}
                for v, c in variant_counts.head(5).items()
            ]
            singleton_variants = int((variant_counts == 1).sum())
            report["singleton_variants"] = singleton_variants
        except Exception:
            pass

        ctx.artifacts["qc_report"] = report
        return ctx
