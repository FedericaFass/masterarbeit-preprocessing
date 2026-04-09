from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import json
import pandas as pd

from ppm_preprocessing.steps.base import Step
from ppm_preprocessing.domain.context import PipelineContext
from ppm_preprocessing.tasks.specs import TaskSpec


@dataclass
class SingleTaskReportExamplesConfig:
    task: TaskSpec
    out_key: str = "single_task_report"
    in_key: str = "single_task_automl"

    # --- file output ---
    save_json: bool = True
    save_examples_csv: bool = True

    report_basename: Optional[str] = None
    reports_subdir: str = "reports"


class SingleTaskReportExamplesStep(Step):
    """
    Thin reporting + export step.

    Reads the already-computed AutoML output (including test + examples)
    from SingleTaskAutoMLTrainStep and stores a compact report under out_key.

    Additionally writes:
      - JSON: single_task_report__{task_name}.json
      - CSV: single_task_examples__{task_name}.csv
    into outputs/reports/ (or {ctx.output_dir}/reports if present).
    """
    name = "single_task_report_examples"

    def __init__(self, config: SingleTaskReportExamplesConfig):
        self.config = config

    def _get_reports_dir(self, ctx: PipelineContext) -> Path:
        out_dir = ctx.artifacts.get("out_dir") or getattr(ctx, "output_dir", None)
        if out_dir:
            return Path(out_dir) / self.config.reports_subdir
        return Path("outputs") / self.config.reports_subdir

    def _basename(self) -> str:
        if self.config.report_basename:
            return str(self.config.report_basename)
        return f"single_task_report__{self.config.task.name}"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if ctx is None:
            raise RuntimeError(
                f"{self.name} received ctx=None. A previous pipeline step returned None; "
                f"each step must return ctx."
            )

        if not hasattr(ctx, "artifacts") or ctx.artifacts is None:
            raise RuntimeError(
                f"{self.name} received an invalid ctx (missing artifacts). "
                f"ctx type={type(ctx)}"
            )

        # IMPORTANT: do not use `if not st` (empty dict would be treated as missing)
        st = ctx.artifacts.get(self.config.in_key, None)
        if st is None:
            available = sorted(list(ctx.artifacts.keys()))
            raise RuntimeError(
                f"'{self.config.in_key}' missing. Run SingleTaskAutoMLTrainStep first. "
                f"Available artifact keys: {available}"
            )

        if not isinstance(st, dict):
            raise TypeError(
                f"Expected ctx.artifacts['{self.config.in_key}'] to be a dict, got {type(st)}"
            )

        task = self.config.task

        report: Dict[str, Any] = {
            "task": task.name,
            "strategy": st.get("strategy", {}) or {},
            "label_qc": st.get("label_qc", {}) or {},
            "mode_results": st.get("mode_results", {}) or {},
            "examples_policy": st.get("examples_policy"),
            "examples": st.get("examples", []) or [],
        }

        # store compact report in-memory
        ctx.artifacts[self.config.out_key] = report

        # persist to files
        reports_dir = self._get_reports_dir(ctx)
        reports_dir.mkdir(parents=True, exist_ok=True)

        base = self._basename()

        if self.config.save_json:
            json_path = reports_dir / f"{base}.json"
            json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
            ctx.artifacts[f"{self.config.out_key}_json_path"] = str(json_path)

        if self.config.save_examples_csv:
            examples = report.get("examples", []) or []
            csv_path = reports_dir / f"single_task_examples__{task.name}.csv"

            if examples:
                pd.DataFrame(examples).to_csv(csv_path, index=False, encoding="utf-8")
            else:
                # still create an empty file with headers
                pd.DataFrame(
                    columns=[
                        "split",
                        "bucket_id",
                        "case_id",
                        "prefix_len",
                        "prefix_activities",
                        "y_true",
                        "y_pred",
                        "abs_error",
                    ]
                ).to_csv(csv_path, index=False, encoding="utf-8")

            ctx.artifacts[f"{self.config.out_key}_examples_csv_path"] = str(csv_path)

        return ctx
