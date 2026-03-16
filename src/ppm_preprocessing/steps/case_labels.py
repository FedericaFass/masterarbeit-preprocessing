from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd

from .base import Step
from ppm_preprocessing.domain.context import PipelineContext


@dataclass
class CaseLabelsConfig:
    # User-specified outcome column (from the last event of each case)
    # e.g. "concept:name", "case:status", "lifecycle:transition"
    outcome_col: str = ""

    # Comma-separated valid outcome values, or empty = use all unique values
    # e.g. "Approved,Rejected,Cancelled"
    outcome_values: str = ""

    # output column name
    label_col: str = "label_outcome"

    # optional: write report artifact JSON
    save_report: bool = True
    report_dir: str = "reports"
    report_filename: str = "case_labels.json"


class CaseLabelsStep(Step):
    """
    Creates case-level table with:
      - case_id
      - case_start_time
      - case_end_time
      - label_outcome (string label from last event attribute)

    The user specifies which column of the last event represents the outcome.
    If outcome_values is provided, only those values are kept; others become "Other".
    If outcome_col is empty, label_outcome defaults to "unknown" for all cases.
    """
    name = "case_labels"

    def __init__(self, config: CaseLabelsConfig | None = None):
        self.config = config or CaseLabelsConfig()

    def _write_report(self, ctx: PipelineContext, payload: Dict[str, Any]) -> None:
        if not self.config.save_report:
            return
        base = Path(".")
        if getattr(ctx, "input_path", None):
            base = Path(ctx.input_path).resolve().parent.parent
        out_dir = (base / self.config.report_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / self.config.report_filename
        out_path.write_text(pd.Series(payload).to_json(orient="index", indent=2), encoding="utf-8")
        ctx.artifacts["case_labels_report_path"] = str(out_path)

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.log is None:
            raise RuntimeError("ctx.log is None. Run normalization + cleaning first.")

        df = ctx.log.df
        required = {"case_id", "activity", "timestamp"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Canonical log missing required columns: {missing}")

        # Ensure timestamp dtype
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        ts_na = int(ts.isna().sum())

        # start/end times
        agg = (
            df.assign(_ts=ts)
            .groupby("case_id", sort=False)["_ts"]
            .agg(["min", "max"])
            .rename(columns={"min": "case_start_time", "max": "case_end_time"})
        )

        # Use case attribute endDate if available (true completion time)
        case_attr_end_col = None
        for potential_col in ["case:endDate", "endDate", "case:end_time", "end_time"]:
            if potential_col in df.columns:
                case_attr_end_col = potential_col
                break

        if case_attr_end_col:
            case_attr_end = pd.to_datetime(
                df.groupby("case_id")[case_attr_end_col].first(), utc=True, errors="coerce"
            )
            valid_mask = case_attr_end.notna() & (case_attr_end >= agg["case_end_time"])
            agg.loc[valid_mask, "case_end_time"] = case_attr_end[valid_mask]
            print(f"  Using case attribute '{case_attr_end_col}' for end times ({valid_mask.sum()} cases updated)")

        # --- Outcome labeling from last event ---
        outcome_col = (self.config.outcome_col or "").strip()
        label_col = self.config.label_col

        # Map raw XES column names to normalized names (NormalizeSchemaStep renames them)
        col_aliases = {
            "concept:name": "activity",
            "case:concept:name": "case_id",
            "time:timestamp": "timestamp",
        }
        if outcome_col and outcome_col not in df.columns and outcome_col in col_aliases:
            resolved = col_aliases[outcome_col]
            if resolved in df.columns:
                print(f"  [CaseLabels] Mapped '{outcome_col}' -> '{resolved}' (post-normalization)")
                outcome_col = resolved

        if outcome_col and outcome_col in df.columns:
            # Sort by timestamp within each case, take last event
            df_sorted = df.assign(_ts=ts).sort_values(["case_id", "_ts"])
            last_events = df_sorted.groupby("case_id", sort=False).last()

            # Extract outcome from the specified column
            raw_outcome = last_events[outcome_col].astype(str).fillna("unknown")

            # Filter to valid outcome values if specified
            outcome_values_str = (self.config.outcome_values or "").strip()
            if outcome_values_str:
                valid_values = set(v.strip() for v in outcome_values_str.split(",") if v.strip())
                raw_outcome = raw_outcome.apply(lambda x: x if x in valid_values else "Other")

            outcome = raw_outcome.rename(label_col)
            used_strategy = "last_event_attribute"
            print(f"  [CaseLabels] Outcome from last event column '{outcome_col}': {outcome.nunique()} unique values")
        else:
            # No outcome column specified — set to "unknown"
            outcome = pd.Series("unknown", index=agg.index, name=label_col)
            used_strategy = "none"
            if outcome_col:
                print(f"  [CaseLabels] Warning: outcome column '{outcome_col}' not found in log. Setting to 'unknown'.")
            else:
                print(f"  [CaseLabels] No outcome column specified. Setting label_outcome to 'unknown'.")

        case_table = agg.join(outcome, how="left").reset_index()

        # Value distribution
        vc = case_table[label_col].value_counts()
        qc: Dict[str, Any] = {
            "step": self.name,
            "num_cases": int(case_table["case_id"].nunique()),
            "timestamp_parse_na": ts_na,
            "outcome_strategy": used_strategy,
            "outcome_col": outcome_col if outcome_col else None,
            "num_outcome_classes": int(vc.shape[0]),
            "outcome_distribution": {str(k): int(v) for k, v in vc.head(20).items()},
            "case_table_columns": list(case_table.columns),
            "sample_rows": case_table.head(3).to_dict(orient="records"),
        }

        ctx.artifacts["case_table"] = case_table
        ctx.artifacts["case_labels_qc"] = qc

        self._write_report(ctx, qc)

        return ctx
