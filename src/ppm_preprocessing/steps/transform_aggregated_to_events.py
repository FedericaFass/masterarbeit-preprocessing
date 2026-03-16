"""
Transform aggregated workflow data (case-level) into event-level format.

This step converts CSV data where:
  - Each row = one case with workflow state durations
  - Columns like wf_open, wf_in_progress, etc. represent time spent in states

Into event-level format where:
  - Each row = one event
  - Columns: case_id, activity, timestamp
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from .base import Step
from ppm_preprocessing.domain.context import PipelineContext


@dataclass
class TransformAggregatedConfig:
    """Configuration for transforming aggregated workflow data."""

    # Column mapping
    case_id_col: str = "id"
    start_time_col: str = "started"
    end_time_col: str = "ended"

    # Workflow state columns (prefix for duration columns)
    workflow_duration_prefix: str = "wf_"
    workflow_event_count_prefix: str = "wfe_"

    # Reporting
    report_filename: str = "01b_transform_aggregated_qc.json"
    sample_rows: int = 10


class TransformAggregatedToEventsStep(Step):
    """
    Transform aggregated workflow data into event-level format.

    Input: ctx.raw_df with aggregated case-level data
    Output: ctx.raw_df transformed to event-level format
    """
    name = "transform_aggregated_to_events"

    def __init__(self, config: TransformAggregatedConfig | None = None):
        self.config = config or TransformAggregatedConfig()

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.raw_df is None:
            raise RuntimeError("ctx.raw_df is None. Run load step first.")

        df = ctx.raw_df.copy()

        # Get workflow state columns
        wf_cols = [col for col in df.columns
                   if col.startswith(self.config.workflow_duration_prefix)]
        wfe_cols = [col for col in df.columns
                    if col.startswith(self.config.workflow_event_count_prefix)]

        if not wf_cols:
            raise ValueError(
                f"No workflow duration columns found with prefix '{self.config.workflow_duration_prefix}'. "
                f"Available columns: {list(df.columns)}"
            )

        # Extract state names (remove prefix)
        state_names = [col.replace(self.config.workflow_duration_prefix, "")
                      for col in wf_cols]

        print(f"Found {len(wf_cols)} workflow states: {state_names[:10]}...")

        # Identify case-level metadata columns to preserve
        exclude_cols = {self.config.case_id_col, self.config.start_time_col, self.config.end_time_col}
        exclude_cols.update(wf_cols)
        exclude_cols.update(wfe_cols)

        # Case metadata = all other columns
        case_metadata_cols = [col for col in df.columns if col not in exclude_cols]
        print(f"Preserving {len(case_metadata_cols)} case metadata columns: {case_metadata_cols[:10]}...")

        # Convert to event-level format
        events = []

        for idx, row in df.iterrows():
            case_id = row[self.config.case_id_col]
            case_start = pd.to_datetime(row[self.config.start_time_col], utc=True)

            if pd.isna(case_start):
                continue

            # Extract case-level metadata for this case
            case_metadata = {}
            for col in case_metadata_cols:
                # Prefix with "case:" so NormalizeSchemaStep treats them as case attributes
                case_metadata[f"case:{col}"] = row[col]

            # Track current time for this case
            current_time = case_start

            # Process each workflow state
            for wf_col, state_name in zip(wf_cols, state_names):
                duration = row[wf_col]

                # Skip if no time spent in this state
                if pd.isna(duration) or duration == 0:
                    continue

                # Get event count for this state (if available)
                wfe_col = f"{self.config.workflow_event_count_prefix}{state_name}"
                event_count = row[wfe_col] if wfe_col in df.columns else 1

                if pd.isna(event_count) or event_count == 0:
                    event_count = 1

                # Create event(s) for this state
                # If multiple events in same state, distribute time evenly
                event_count = int(event_count)
                time_per_event = duration / event_count

                for i in range(event_count):
                    event_time = current_time + pd.Timedelta(seconds=time_per_event * i)

                    # Create event with case metadata included
                    event = {
                        'case_id': case_id,
                        'activity': state_name,
                        'timestamp': event_time,
                    }
                    event.update(case_metadata)  # Add all case metadata
                    events.append(event)

                # Move time forward by total duration
                current_time += pd.Timedelta(seconds=duration)

        # Create event-level DataFrame
        event_df = pd.DataFrame(events)

        if len(event_df) == 0:
            raise RuntimeError("No events generated from aggregated data. Check your data format.")

        # Sort by case and time
        event_df = event_df.sort_values(['case_id', 'timestamp']).reset_index(drop=True)

        # Replace raw_df with event-level data
        ctx.raw_df = event_df
        ctx.artifacts["loaded_columns"] = list(event_df.columns)

        # QC Report
        qc: Dict[str, Any] = {
            "step": self.name,
            "original_format": "aggregated (case-level)",
            "transformed_format": "event-level",
            "num_original_cases": int(len(df)),
            "num_events_generated": int(len(event_df)),
            "num_unique_cases": int(event_df['case_id'].nunique()),
            "num_workflow_states": len(wf_cols),
            "workflow_states": state_names,
            "num_case_metadata_cols_preserved": len(case_metadata_cols),
            "case_metadata_cols_preserved": case_metadata_cols,
            "columns": list(event_df.columns),
            "events_per_case_avg": float(len(event_df) / event_df['case_id'].nunique()),
        }

        # Sample events
        try:
            qc["sample_events"] = event_df.head(self.config.sample_rows).to_dict(orient="records")
        except Exception:
            qc["sample_events"] = []

        ctx.artifacts["transform_aggregated_qc"] = qc

        # Save report
        output_dir = Path(getattr(ctx, "output_dir", "outputs"))
        reports_dir = output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        report_path = reports_dir / self.config.report_filename
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(self._json_safe(qc), f, indent=2)

        ctx.artifacts["transform_aggregated_report_path"] = str(report_path)

        print(f"\nTransformed {len(df)} cases into {len(event_df)} events")
        print(f"   Average events per case: {qc['events_per_case_avg']:.1f}")

        return ctx

    @staticmethod
    def _json_safe(obj: Any) -> Any:
        """Convert to JSON-serializable types."""
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, (list, tuple)):
            return [TransformAggregatedToEventsStep._json_safe(x) for x in obj]
        if isinstance(obj, dict):
            return {str(k): TransformAggregatedToEventsStep._json_safe(v)
                   for k, v in obj.items()}
        if isinstance(obj, (pd.Timestamp,)):
            return obj.isoformat()
        return str(obj)
