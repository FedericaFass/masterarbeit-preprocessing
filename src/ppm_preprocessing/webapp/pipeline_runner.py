"""
Reusable pipeline runner for the web app.

Wraps the same logic from cli/run_task.py into a callable function
that returns results instead of printing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import json
import traceback

from ppm_preprocessing.domain.context import PipelineContext

from ppm_preprocessing.steps.load_xes import LoadXesStep
from ppm_preprocessing.steps.load_csv import LoadCsvStep
from ppm_preprocessing.steps.transform_aggregated_to_events import (
    TransformAggregatedToEventsStep,
    TransformAggregatedConfig,
)
from ppm_preprocessing.steps.normalize_schema import NormalizeSchemaStep
from ppm_preprocessing.steps.clean_sort import CleanAndSortStep
from ppm_preprocessing.steps.qc_report import QcReportStep
from ppm_preprocessing.io.format_detection import detect_format
from ppm_preprocessing.steps.case_labels import CaseLabelsStep, CaseLabelsConfig
from ppm_preprocessing.steps.prefix_extraction import PrefixExtractionStep, PrefixExtractionConfig
from ppm_preprocessing.steps.split_cases import CaseSplitStep, CaseSplitConfig
from ppm_preprocessing.steps.outlier_detection import OutlierDetectionStep, OutlierDetectionConfig
from ppm_preprocessing.steps.filter_rare_classes import FilterRareClassesStep, FilterRareClassesConfig
from ppm_preprocessing.steps.deduplicate_events import DeduplicateEventsStep, DeduplicateEventsConfig
from ppm_preprocessing.steps.stable_sort import StableSortStep, StableSortConfig
from ppm_preprocessing.steps.drop_columns import DropColumnsStep, DropColumnsConfig
from ppm_preprocessing.steps.filter_rare_variants import FilterRareVariantsStep, FilterRareVariantsConfig
from ppm_preprocessing.steps.concept_drift_window import ConceptDriftWindowStep, ConceptDriftWindowConfig
from ppm_preprocessing.steps.filter_short_cases import FilterShortCasesStep, FilterShortCasesConfig
from ppm_preprocessing.steps.normalize_activities import NormalizeActivitiesStep, NormalizeActivitiesConfig
from ppm_preprocessing.steps.filter_infrequent_activities import FilterInfrequentActivitiesStep, FilterInfrequentActivitiesConfig
from ppm_preprocessing.steps.filter_zero_duration_cases import FilterZeroDurationCasesStep, FilterZeroDurationCasesConfig
from ppm_preprocessing.steps.filter_case_length import FilterCaseLengthStep, FilterCaseLengthConfig
from ppm_preprocessing.steps.filter_consecutive_duplicates import FilterConsecutiveDuplicatesStep, FilterConsecutiveDuplicatesConfig
from ppm_preprocessing.steps.impute_missing_attributes import ImputeMissingAttributesStep, ImputeMissingAttributesConfig
from ppm_preprocessing.steps.repair_timestamps import RepairTimestampsStep, RepairTimestampsConfig

from ppm_preprocessing.bucketing.no_bucket import NoBucketer
from ppm_preprocessing.bucketing.prefix_length_bins import PrefixLenBinsBucketer
from ppm_preprocessing.bucketing.prefix_length_adaptive import PrefixLenAdaptiveBucketer
from ppm_preprocessing.bucketing.cluster import ClusterBucketer, ClusterBucketConfig
from ppm_preprocessing.bucketing.last_activity import LastActivityBucketer
from ppm_preprocessing.tasks.specs import default_task_specs, make_next_attr_task_spec

from ppm_preprocessing.steps.single_task_strategy_search import (
    SingleTaskStrategySearchStep,
    SingleTaskStrategySearchConfig,
)
from ppm_preprocessing.steps.single_task_automl_train import (
    SingleTaskAutoMLTrainStep,
    SingleTaskAutoMLTrainConfig,
)
from ppm_preprocessing.steps.single_task_report_examples import (
    SingleTaskReportExamplesStep,
    SingleTaskReportExamplesConfig,
)
from ppm_preprocessing.steps.single_task_persist_model import SingleTaskPersistModelStep
from ppm_preprocessing.steps.single_task_test_evaluation import (
    SingleTaskTestEvaluationStep,
    SingleTaskTestEvaluationConfig,
)
from ppm_preprocessing.steps.single_task_feature_importance import (
    SingleTaskFeatureImportanceStep,
)
from ppm_preprocessing.steps.visualize_results import (
    VisualizeResultsStep,
    VisualizeResultsConfig,
)


def _safe_int(val, fallback=None):
    """Convert val to int, returning fallback if val is None, NaN, or Inf."""
    import math
    if val is None:
        return fallback
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return fallback
        return int(f)
    except (TypeError, ValueError):
        return fallback


def _sanitize_for_json(obj):
    """Recursively replace NaN/Inf floats with None so json.dumps produces valid JSON."""
    import math
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    try:
        import numpy as _np
        if isinstance(obj, _np.floating) and not math.isfinite(float(obj)):
            return None
    except ImportError:
        pass
    return obj


def run_pipeline(
    log_path: Path,
    out_dir: Path,
    task_name: str = "remaining_time",
    time_budget_s: int = 300,
    outlier_enabled: bool = True,
    min_class_samples: int = 0,
    outcome_col: str = "",
    outcome_values: str = "",
    columns_to_drop: list | None = None,
    next_event_attr_col: str = "",
    lifecycle_drop_values: list | None = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    max_prefix_len: int = 30,
    min_bucket_samples: int = 100,
    bin_size: int = 5,
    temporal_split: bool = False,
    rare_variant_filter: bool = False,
    min_variant_count: int = 5,
    concept_drift_window: bool = False,
    since_date: str = "",
    filter_long_cases: bool = True,
    filter_consecutive_duplicates: bool = False,
    impute_missing: bool = False,
    col_mapping: dict | None = None,
    on_progress: Any = None,
) -> Dict[str, Any]:

    def _progress(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    def _step_data(data: dict) -> None:
        import json
        _progress(f"__STEP__:{json.dumps(_sanitize_for_json(data), default=str)}")

    def _run_step(step, ctx, label: str) -> PipelineContext:
        """Run a single pipeline step with progress reporting and before/after stats."""
        import json
        before_rows = int(len(ctx.log.df)) if ctx.log is not None else None
        before_cases = int(ctx.log.df["case_id"].nunique()) if ctx.log is not None and "case_id" in ctx.log.df.columns else None
        _progress(label)
        new_ctx = step.run(ctx)
        after_rows = int(len(new_ctx.log.df)) if new_ctx.log is not None else None
        after_cases = int(new_ctx.log.df["case_id"].nunique()) if new_ctx.log is not None and "case_id" in new_ctx.log.df.columns else None

        # collect any extra QC details the step may have written to artifacts
        qc_key = getattr(step, "name", "") + "_qc"
        qc = new_ctx.artifacts.get(qc_key, {}) or {}

        _step_data({
            "step": getattr(step, "name", type(step).__name__),
            "label": label,
            "before_rows": before_rows,
            "after_rows": after_rows,
            "before_cases": before_cases,
            "after_cases": after_cases,
            "qc": {k: v for k, v in qc.items() if isinstance(v, (int, float, str, bool, list, dict)) and k not in ("step",)},
        })
        return new_ctx

    try:
        out_dir.mkdir(parents=True, exist_ok=True)

        # Resolve task spec — support custom next-event attribute prediction
        tasks = default_task_specs()
        if task_name == "next_activity" and next_event_attr_col and next_event_attr_col != "activity":
            task = make_next_attr_task_spec(next_event_attr_col)
            tasks = {**tasks, task.name: task}  # register dynamic task so AutoML step can find it
        elif task_name not in tasks:
            return {"status": "error", "error": f"Unknown task: {task_name}"}
        else:
            task = tasks[task_name]

        if not log_path.exists():
            return {"status": "error", "error": f"File not found: {log_path}"}

        _progress("Detecting file format...")
        file_format = detect_format(str(log_path))

        if file_format == "csv":
            load_step = LoadCsvStep()
        else:
            load_step = LoadXesStep()

        ctx = PipelineContext(input_path=str(log_path), task=task.name)
        ctx.artifacts["out_dir"] = str(out_dir)

        # Validate: outcome task requires an outcome column
        if task_name == "outcome" and not outcome_col.strip():
            return {"status": "error", "error": "Outcome task requires an outcome column. Please select a column in the Outcome configuration."}

        is_classification = "classification" in task.task_type
        needs_case_labels = True  # always needed — PrefixExtractionStep requires case_table

        # Determine extra columns to extract as "next event" labels
        extra_next_cols = [next_event_attr_col] if (
            next_event_attr_col and next_event_attr_col != "activity" and task_name == "next_activity"
        ) else []

        bucketers = {
            "no_bucket": NoBucketer(),
            "last_activity": LastActivityBucketer(),
            "prefix_len_bins": PrefixLenBinsBucketer(bin_size=bin_size, max_len=max_prefix_len),
            "prefix_len_adaptive": PrefixLenAdaptiveBucketer(max_len=max_prefix_len),
            "cluster": ClusterBucketer(ClusterBucketConfig(n_clusters=3)),
        }
        encodings = ["last_state", "aggregation", "index_latest_payload", "embedding"]

        strategy_cfg = SingleTaskStrategySearchConfig(
            task=task,
            bucketers=bucketers,
            encodings=encodings,
            min_bucket_samples=min_bucket_samples,
            skip_single_class=True,
            use_probe_model=True,
        )

        # ---------------------------------------------------------------
        # Configuration summary
        # ---------------------------------------------------------------
        test_ratio_pct = int(round((1.0 - train_ratio - val_ratio) * 100))
        split_method = "temporal ordering" if temporal_split else "random"
        active_opts = []
        if outlier_enabled:
            active_opts.append("outlier detection")
        if temporal_split:
            active_opts.append("temporal ordering")
        if concept_drift_window and since_date:
            active_opts.append(f"time window from {since_date}")
        if rare_variant_filter:
            active_opts.append(f"rare variant filter (min={min_variant_count})")
        if filter_consecutive_duplicates:
            active_opts.append("filter consecutive duplicates")
        if impute_missing:
            active_opts.append("impute missing attributes")
        if lifecycle_drop_values:
            active_opts.append(f"lifecycle filter ({lifecycle_drop_values})")
        if columns_to_drop:
            active_opts.append(f"drop {len(columns_to_drop)} column(s)")
        if is_classification and min_class_samples > 0:
            active_opts.append(f"min class samples={min_class_samples}")

        _progress(
            f"Configuration: task={task_name}"
            + (f" | outcome={outcome_col}" if task_name == "outcome" and outcome_col else "")
            + (f" | attr={next_event_attr_col}" if next_event_attr_col and next_event_attr_col != "activity" else "")
            + f" | split={int(train_ratio*100)}/{int(val_ratio*100)}/{test_ratio_pct}% ({split_method})"
            + f" | max_prefix={max_prefix_len}"
            + f" | budget={time_budget_s}s"
        )
        if active_opts:
            _progress("Active options: " + " · ".join(active_opts))
        else:
            _progress("Active options: none (default settings)")

        # ---------------------------------------------------------------
        # Run steps individually with progress reporting
        # ---------------------------------------------------------------

        # 1. Load
        ctx = _run_step(load_step, ctx, f"Loading {file_format.upper()} file...")

        # 1b. CSV transform if needed (aggregated format: one row per case with wf_* columns)
        _used_transform = False
        if file_format == "csv":
            raw_cols = list(ctx.raw_df.columns) if ctx.raw_df is not None else []
            wf_cols = [c for c in raw_cols if c.startswith("wf_")]
            if wf_cols:
                _used_transform = True
                _case_id_col = (col_mapping.get("case_col") or "id") if col_mapping else "id"
                _start_col = (col_mapping.get("ts_col") or "started") if col_mapping else "started"
                # detect end time column automatically
                _end_candidates = ["ended", "end_time", "endtime", "completed", "end", "finish"]
                _end_col = next((c for c in _end_candidates if c in raw_cols), None) or "ended"
                ctx = _run_step(
                    TransformAggregatedToEventsStep(
                        TransformAggregatedConfig(
                            case_id_col=_case_id_col,
                            start_time_col=_start_col,
                            end_time_col=_end_col,
                            workflow_duration_prefix="wf_",
                            workflow_event_count_prefix="wfe_",
                        )
                    ),
                    ctx,
                    "Transforming aggregated CSV to events...",
                )

        # 2. Normalize + clean
        # After TransformAggregatedToEventsStep output is already canonical → use default detection.
        # For regular event-log CSVs (no wf_ cols), use col_mapping to guide column detection.
        if col_mapping and not _used_transform:
            from ppm_preprocessing.steps.normalize_schema import NormalizeSchemaConfig
            _cm = col_mapping
            _ns_cfg = NormalizeSchemaConfig(
                case_candidates=[_cm["case_col"]] if _cm.get("case_col") else None,
                act_candidates=[_cm["act_col"]] if _cm.get("act_col") else None,
                ts_candidates=[_cm["ts_col"]] if _cm.get("ts_col") else None,
            )
            ctx = _run_step(NormalizeSchemaStep(_ns_cfg), ctx, "Normalizing schema (custom mapping)...")
        else:
            ctx = _run_step(NormalizeSchemaStep(), ctx, "Normalizing schema...")

        # 2a. Filter lifecycle:transition events
        if lifecycle_drop_values and ctx.log is not None:
            lc_col = "lifecycle:transition"
            if lc_col in ctx.log.df.columns:
                before = len(ctx.log.df)
                ctx.log.df = ctx.log.df[
                    ~ctx.log.df[lc_col].astype(str).isin(lifecycle_drop_values)
                ].reset_index(drop=True)
                removed = before - len(ctx.log.df)
                _progress(
                    f"Lifecycle filter: removed {removed} events "
                    f"(dropping {lifecycle_drop_values}), {len(ctx.log.df)} remaining"
                )

        # 2b. Drop user-selected empty columns
        if columns_to_drop:
            ctx = _run_step(
                DropColumnsStep(DropColumnsConfig(columns=list(columns_to_drop))),
                ctx,
                f"Dropping {len(columns_to_drop)} empty column(s)...",
            )

        n_events = len(ctx.log.df) if ctx.log else 0
        n_cases = ctx.log.df["case_id"].nunique() if ctx.log else 0
        _progress(f"Loaded {n_cases} cases, {n_events} events")

        ctx = _run_step(
            DeduplicateEventsStep(DeduplicateEventsConfig()), ctx,
            "Deduplicating events...",
        )
        ctx = _run_step(CleanAndSortStep(), ctx, "Cleaning and sorting...")
        ctx = _run_step(
            StableSortStep(StableSortConfig(tie_breakers=("_event_index",))),
            ctx, "Stable-sorting events...",
        )
        ctx = _run_step(QcReportStep(), ctx, "Running QC report...")

        # Repair backwards timestamps (auto)
        ctx = _run_step(RepairTimestampsStep(), ctx, "Repairing backwards timestamps...")
        rt_qc = ctx.artifacts.get("repair_timestamps_qc", {})
        if rt_qc.get("backwards_timestamps_repaired", 0):
            _progress(f"Timestamp repair: fixed {rt_qc['backwards_timestamps_repaired']} backwards events")

        # --- Auto preprocessing steps (literature-based) ---
        # 2a-i: Filter short cases (< 2 events)
        ctx = _run_step(FilterShortCasesStep(), ctx, "Filtering short cases (< 2 events)...")
        short_qc = ctx.artifacts.get("filter_short_cases_qc", {})
        if short_qc.get("short_cases_removed", 0):
            _progress(f"Short cases removed: {short_qc['short_cases_removed']} ({short_qc.get('short_cases_pct', 0):.1f}%)")

        # 2a-ii: Normalize activity labels (strip whitespace, fill nulls)
        ctx = _run_step(NormalizeActivitiesStep(), ctx, "Normalizing activity labels...")

        # 2a-iii: Filter infrequent activities (< 0.5% of traces)
        ctx = _run_step(FilterInfrequentActivitiesStep(), ctx, "Filtering infrequent activities...")
        infreq_qc = ctx.artifacts.get("filter_infrequent_activities_qc", {})
        if infreq_qc.get("rare_activities_removed", 0):
            _progress(f"Infrequent activities removed: {infreq_qc['rare_activities_removed']} types, {infreq_qc.get('rows_dropped', 0)} events")

        # 2a-iv: Filter zero-duration cases
        ctx = _run_step(FilterZeroDurationCasesStep(), ctx, "Filtering zero-duration cases...")
        zdur_qc = ctx.artifacts.get("filter_zero_duration_cases_qc", {})
        if zdur_qc.get("zero_duration_cases_removed", 0):
            _progress(f"Zero-duration cases removed: {zdur_qc['zero_duration_cases_removed']} ({zdur_qc.get('zero_duration_pct', 0):.1f}%)")

        # 2a-v: Filter max case length (p99) — optional
        if filter_long_cases:
            ctx = _run_step(FilterCaseLengthStep(), ctx, "Filtering extreme-length cases (p99)...")
            clen_qc = ctx.artifacts.get("filter_case_length_qc", {})
            if clen_qc.get("long_cases_removed", 0):
                _progress(f"Long cases removed: {clen_qc['long_cases_removed']} ({clen_qc.get('long_cases_pct', 0):.1f}%), threshold={clen_qc.get('computed_threshold')}")

        # 2b-opt: Optional — filter consecutive duplicate events
        if filter_consecutive_duplicates:
            ctx = _run_step(FilterConsecutiveDuplicatesStep(), ctx, "Filtering consecutive duplicate events...")
            cdup_qc = ctx.artifacts.get("filter_consecutive_duplicates_qc", {})
            if cdup_qc.get("consecutive_duplicate_events_removed", 0):
                _progress(f"Consecutive duplicates removed: {cdup_qc['consecutive_duplicate_events_removed']} events")

        # 2b-opt: Optional — impute missing attribute values
        if impute_missing:
            ctx = _run_step(ImputeMissingAttributesStep(), ctx, "Imputing missing attribute values...")
            imp_qc = ctx.artifacts.get("impute_missing_attributes_qc", {})
            if imp_qc.get("columns_imputed", 0):
                _progress(f"Imputed {imp_qc['columns_imputed']} columns with missing values")

        # 2c. Time Window Filter — discard cases before a cutoff date
        ctx = _run_step(
            ConceptDriftWindowStep(ConceptDriftWindowConfig(
                enabled=concept_drift_window,
                since_date=since_date or None,
            )),
            ctx,
            f"Time window filter: keeping cases from {since_date}..." if concept_drift_window else "Time window filter: disabled",
        )
        cdw_qc = ctx.artifacts.get("concept_drift_window_qc", {})
        if cdw_qc.get("enabled") and cdw_qc.get("cases_removed", 0):
            _progress(
                f"Time window filter: removed {cdw_qc['cases_removed']} old cases "
                f"({cdw_qc['cases_removed_pct']:.1f}%), {cdw_qc['cases_after']} remaining"
            )

        # 2d. Rare variant filter — remove cases with very rare activity sequences
        ctx = _run_step(
            FilterRareVariantsStep(FilterRareVariantsConfig(
                enabled=rare_variant_filter,
                min_variant_count=min_variant_count,
            )),
            ctx,
            f"Rare variant filter (min={min_variant_count})..." if rare_variant_filter else "Rare variant filter: disabled",
        )
        rvf_qc = ctx.artifacts.get("filter_rare_variants_qc", {})
        if rvf_qc.get("enabled") and rvf_qc.get("cases_removed", 0):
            _progress(
                f"Rare variant filter: removed {rvf_qc['cases_removed']} cases "
                f"({rvf_qc['cases_removed_pct']:.1f}%), "
                f"{rvf_qc['variants_rare']} of {rvf_qc['variants_total']} variants were rare"
            )

        # 3. Case labels (if needed)
        if needs_case_labels:
            ctx = _run_step(
                CaseLabelsStep(
                    CaseLabelsConfig(
                        outcome_col=outcome_col,
                        outcome_values=outcome_values,
                        label_col="label_outcome",
                    )
                ),
                ctx,
                "Computing case labels...",
            )

        # 4. Prefix extraction
        ctx = _run_step(
            PrefixExtractionStep(PrefixExtractionConfig(
                max_prefix_len=max_prefix_len,
                min_prefix_len=1,
                next_event_attr_cols=extra_next_cols or None,
            )),
            ctx,
            f"Extracting prefixes (max_len={max_prefix_len})...",
        )
        ps = ctx.artifacts.get("prefix_samples")
        if ps is not None:
            _progress(f"Prefix samples: {len(ps)} rows")

        # Emit feature info step card
        _step_data({
            "step": "prefix_features",
            "label": "Prefix & Feature Extraction",
            "before_rows": None,
            "after_rows": int(len(ps)) if ps is not None else None,
            "before_cases": None,
            "after_cases": None,
            "qc": {
                "label_column": task.label_col,
                "time_features_added": [
                    "feat_elapsed_time_sec",
                    "feat_time_since_last_event_sec",
                    "feat_elapsed_time_log1p",
                    "feat_time_since_last_log1p",
                    "feat_prefix_end_hour",
                    "feat_prefix_end_weekday",
                    "feat_prefix_end_is_weekend",
                    "feat_prefix_end_month",
                ],
                "max_prefix_len": max_prefix_len,
                "total_prefix_rows": int(len(ps)) if ps is not None else None,
            },
        })

        # 5. Case split
        test_ratio = round(1.0 - train_ratio - val_ratio, 4)
        split_method = "temporal" if temporal_split else "random"
        _progress(f"Splitting cases ({int(train_ratio*100)}/{int(val_ratio*100)}/{int(test_ratio*100)}, {split_method})...")
        ctx = CaseSplitStep(CaseSplitConfig(
            train_ratio=train_ratio, val_ratio=val_ratio, random_state=42,
            temporal_split=temporal_split,
        )).run(ctx)
        splits = ctx.artifacts.get("case_splits", {})
        n_train = len(splits.get("train", []))
        n_val   = len(splits.get("val", []))
        n_test  = len(splits.get("test", []))
        _progress(f"Split: {n_train} train, {n_val} val, {n_test} test cases")
        _step_data({
            "step": "case_split",
            "label": f"Case Split ({split_method})",
            "before_rows": None,
            "after_rows": None,
            "before_cases": None,
            "after_cases": None,
            "qc": {
                "split_method": split_method,
                "train_cases": n_train,
                "val_cases": n_val,
                "test_cases": n_test,
                "train_pct": int(train_ratio * 100),
                "val_pct": int(val_ratio * 100),
                "test_pct": int(round((1 - train_ratio - val_ratio) * 100)),
            },
        })

        # 5b. Outlier detection (IQR on case duration / remaining time)
        _progress("Detecting case duration outliers (IQR)...")
        _ps_before = ctx.artifacts.get("prefix_samples")
        _ps_before_rows = int(len(_ps_before)) if _ps_before is not None else None
        ctx = OutlierDetectionStep(OutlierDetectionConfig(enabled=outlier_enabled)).run(ctx)
        outlier_qc = ctx.artifacts.get("outlier_detection_qc", {})
        _ps_after_rows = outlier_qc.get("total_rows_after", _ps_before_rows)
        _step_data({
            "step": "outlier_detection",
            "label": "Detecting case duration outliers (IQR)...",
            "before_rows": _ps_before_rows,
            "after_rows": _safe_int(_ps_after_rows, _ps_before_rows),
            "before_cases": None,
            "after_cases": None,
            "qc": {k: v for k, v in outlier_qc.items() if isinstance(v, (int, float, str, bool, list, dict)) and k != "step"},
        })

        # 5c. Filter rare classes — only for classification
        _progress("Filtering rare classes...")
        _ps_before2 = ctx.artifacts.get("prefix_samples")
        _ps_before_rows2 = int(len(_ps_before2)) if _ps_before2 is not None else None
        ctx = FilterRareClassesStep(FilterRareClassesConfig(
            enabled=is_classification and min_class_samples > 0,
            min_class_samples=min_class_samples,
            label_col=task.label_col,
        )).run(ctx)
        filter_qc = ctx.artifacts.get("filter_rare_classes_qc", {})
        _ps_after_rows2 = filter_qc.get("rows_after", _ps_before_rows2)
        _step_data({
            "step": "filter_rare_classes",
            "label": "Filtering rare classes...",
            "before_rows": _ps_before_rows2,
            "after_rows": _safe_int(_ps_after_rows2, _ps_before_rows2),
            "before_cases": None,
            "after_cases": None,
            "qc": {k: v for k, v in filter_qc.items() if isinstance(v, (int, float, str, bool, list, dict)) and k != "step"},
        })

        # 6. Strategy search
        _progress("Strategy search: testing 5 bucketers × 4 encodings = 20 strategies...")
        ctx = SingleTaskStrategySearchStep(strategy_cfg).run(ctx)

        best = ctx.artifacts.get("best_strategy", {})
        if best:
            score = best.get("primary_score")
            score_str = f"{score:.2f}" if score is not None else "N/A"
            metric_label = "F1" if is_classification else "MAE"
            _progress(
                f"Best strategy: {best.get('bucketing')} + {best.get('encoding')} "
                f"({best.get('mode')}) | {metric_label} = {score_str}"
            )

        # 7. AutoML training
        _progress(f"Training AutoML model (budget={time_budget_s}s)...")
        ctx = SingleTaskAutoMLTrainStep(
            config=SingleTaskAutoMLTrainConfig(
                task_name=task.name,
                bucketers=bucketers,
                encodings=encodings,
                min_bucket_samples=min_bucket_samples,
                skip_single_class=True,
                time_budget_s=time_budget_s,
                seed=42,
                n_jobs=1,  # sequential trials — parallel causes thread overhead that breaks the time budget
                estimator_list=None,
                n_example_rows=3,
                examples_policy="all",
                target_log1p=not is_classification,
            ),
            tasks=tasks,
        ).run(ctx)

        # 8. Report examples
        ctx = _run_step(
            SingleTaskReportExamplesStep(SingleTaskReportExamplesConfig(
                task=task,
                out_key="single_task_report",
                in_key="single_task_automl",
                save_json=True,
                save_examples_csv=True,
            )),
            ctx,
            "Building report...",
        )

        # 9. Persist model bundle
        ctx = _run_step(SingleTaskPersistModelStep(), ctx, "Saving model bundle...")

        # 10. Test evaluation
        ctx = _run_step(
            SingleTaskTestEvaluationStep(
                config=SingleTaskTestEvaluationConfig(),
                task=task,
            ),
            ctx,
            "Evaluating on held-out test set...",
        )

        # 11. Feature importance
        ctx = _run_step(SingleTaskFeatureImportanceStep(), ctx, "Computing feature importance...")

        # 12. Generate visualizations
        # Reports (JSON/CSV) are saved to out_dir/reports/ (session-specific)
        viz_reports_dir = out_dir / "reports"
        viz_reports_dir.mkdir(parents=True, exist_ok=True)
        ctx = _run_step(
            VisualizeResultsStep(VisualizeResultsConfig(
                task_name=task.name,
                output_dir=str(viz_reports_dir),
            )),
            ctx,
            "Generating charts...",
        )

        # Save summary JSONs (same as run_task.py)
        _progress("Saving results...")
        _json_dump = lambda obj: json.dumps(obj, indent=2, default=str)
        (out_dir / "comparison.json").write_text(
            _json_dump(ctx.artifacts.get("single_task_comparison", [])), encoding="utf-8"
        )
        (out_dir / "best_strategy.json").write_text(
            _json_dump(ctx.artifacts.get("best_strategy", {})), encoding="utf-8"
        )
        (out_dir / "report.json").write_text(
            _json_dump(ctx.artifacts.get("single_task_report", {})), encoding="utf-8"
        )
        (out_dir / "test_evaluation.json").write_text(
            _json_dump(ctx.artifacts.get("single_task_test_evaluation", {})), encoding="utf-8"
        )

        # Extract results
        test_eval = ctx.artifacts.get("single_task_test_evaluation", {})
        best_strategy = ctx.artifacts.get("best_strategy", {})
        model_bundle_path = out_dir / "model_bundle.joblib"

        # Compute avg case duration in days from the processed log
        avg_case_duration_days = None
        if ctx.log is not None:
            try:
                import pandas as _pd
                ts = _pd.to_datetime(ctx.log.df["timestamp"], utc=True, errors="coerce")
                df_ts = ctx.log.df[["case_id"]].copy()
                df_ts["ts"] = ts
                grp = df_ts.groupby("case_id")["ts"]
                durations = (grp.max() - grp.min()).dt.total_seconds() / 86400
                avg_case_duration_days = round(float(durations.mean()), 2)
            except Exception:
                pass
        (out_dir / "dataset_stats.json").write_text(
            _json_dump({"avg_case_duration_days": avg_case_duration_days}), encoding="utf-8"
        )

        return {
            "status": "success",
            "metrics": test_eval.get("test_metrics", {}),
            "per_bucket_metrics": test_eval.get("per_bucket_metrics", {}),
            "best_strategy": {
                "bucketing": best_strategy.get("bucketing"),
                "encoding": best_strategy.get("encoding"),
                "mode": best_strategy.get("mode"),
            },
            "test_samples": test_eval.get("test_samples", 0),
            "avg_case_duration_days": avg_case_duration_days,
            "model_bundle_path": str(model_bundle_path),
            "model_bundle_exists": model_bundle_path.exists(),
            "log_qc": ctx.artifacts.get("qc_report", {}),
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def run_strategy_search_only(
    log_path: Path,
    task_name: str = "remaining_time",
    outlier_enabled: bool = True,
    min_class_samples: int = 0,
    outcome_col: str = "",
    outcome_values: str = "",
    columns_to_drop: list | None = None,
    next_event_attr_col: str = "",
    lifecycle_drop_values: list | None = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    max_prefix_len: int = 30,
    min_bucket_samples: int = 100,
    bin_size: int = 5,
    temporal_split: bool = False,
    rare_variant_filter: bool = False,
    min_variant_count: int = 5,
    concept_drift_window: bool = False,
    since_date: str = "",
    filter_long_cases: bool = True,
    filter_consecutive_duplicates: bool = False,
    impute_missing: bool = False,
    col_mapping: Dict[str, str] | None = None,
    on_progress: Any = None,
) -> Dict[str, Any]:
    """
    Run the preprocessing pipeline up to and including the strategy search step only.
    Skips AutoML training — uses fast linear models to score bucketer/encoder combos.
    Returns the best strategy with its validation score for quick configuration comparison.
    """

    def _progress(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    def _step_data(data: dict) -> None:
        import json
        _progress(f"__STEP__:{json.dumps(_sanitize_for_json(data), default=str)}")

    def _run_step(step, ctx, label: str) -> PipelineContext:
        import json
        before_rows = int(len(ctx.log.df)) if ctx.log is not None else None
        before_cases = int(ctx.log.df["case_id"].nunique()) if ctx.log is not None and "case_id" in ctx.log.df.columns else None
        _progress(label)
        new_ctx = step.run(ctx)
        after_rows = int(len(new_ctx.log.df)) if new_ctx.log is not None else None
        after_cases = int(new_ctx.log.df["case_id"].nunique()) if new_ctx.log is not None and "case_id" in new_ctx.log.df.columns else None
        qc_key = getattr(step, "name", "") + "_qc"
        qc = new_ctx.artifacts.get(qc_key, {}) or {}
        _step_data({
            "step": getattr(step, "name", type(step).__name__),
            "label": label,
            "before_rows": before_rows,
            "after_rows": after_rows,
            "before_cases": before_cases,
            "after_cases": after_cases,
            "qc": {k: v for k, v in qc.items() if isinstance(v, (int, float, str, bool, list, dict)) and k not in ("step",)},
        })
        return new_ctx

    try:
        # Resolve task spec — support custom next-event attribute prediction
        tasks = default_task_specs()
        if task_name == "next_activity" and next_event_attr_col and next_event_attr_col != "activity":
            task = make_next_attr_task_spec(next_event_attr_col)
        elif task_name not in tasks:
            return {"status": "error", "error": f"Unknown task: {task_name}"}
        else:
            task = tasks[task_name]

        if not log_path.exists():
            return {"status": "error", "error": f"File not found: {log_path}"}

        if task_name == "outcome" and not outcome_col.strip():
            return {"status": "error", "error": "Outcome task requires an outcome column."}

        _progress("Detecting file format...")
        file_format = detect_format(str(log_path))

        if file_format == "csv":
            load_step = LoadCsvStep()
        else:
            load_step = LoadXesStep()

        ctx = PipelineContext(input_path=str(log_path), task=task.name)

        is_classification = "classification" in task.task_type
        needs_case_labels = True  # always needed — PrefixExtractionStep requires case_table

        extra_next_cols = [next_event_attr_col] if (
            next_event_attr_col and next_event_attr_col != "activity" and task_name == "next_activity"
        ) else []

        bucketers = {
            "no_bucket": NoBucketer(),
            "last_activity": LastActivityBucketer(),
            "prefix_len_bins": PrefixLenBinsBucketer(bin_size=bin_size, max_len=max_prefix_len),
            "prefix_len_adaptive": PrefixLenAdaptiveBucketer(max_len=max_prefix_len),
            "cluster": ClusterBucketer(ClusterBucketConfig(n_clusters=3)),
        }
        encodings = ["last_state", "aggregation", "index_latest_payload", "embedding"]

        strategy_cfg = SingleTaskStrategySearchConfig(
            task=task,
            bucketers=bucketers,
            encodings=encodings,
            min_bucket_samples=min_bucket_samples,
            skip_single_class=True,
            use_probe_model=False,
        )

        # 1. Load
        ctx = _run_step(load_step, ctx, f"Loading {file_format.upper()} file...")

        # 1b. CSV transform if needed (aggregated format: one row per case with wf_* columns)
        _used_transform = False
        if file_format == "csv":
            raw_cols = list(ctx.raw_df.columns) if ctx.raw_df is not None else []
            wf_cols = [c for c in raw_cols if c.startswith("wf_")]
            if wf_cols:
                _used_transform = True
                _case_id_col = (col_mapping.get("case_col") or "id") if col_mapping else "id"
                _start_col = (col_mapping.get("ts_col") or "started") if col_mapping else "started"
                _end_candidates = ["ended", "end_time", "endtime", "completed", "end", "finish"]
                _end_col = next((c for c in _end_candidates if c in raw_cols), None) or "ended"
                ctx = _run_step(
                    TransformAggregatedToEventsStep(
                        TransformAggregatedConfig(
                            case_id_col=_case_id_col,
                            start_time_col=_start_col,
                            end_time_col=_end_col,
                            workflow_duration_prefix="wf_",
                            workflow_event_count_prefix="wfe_",
                        )
                    ),
                    ctx,
                    "Transforming aggregated CSV to events...",
                )

        # 2. Normalize
        if col_mapping and not _used_transform:
            from ppm_preprocessing.steps.normalize_schema import NormalizeSchemaConfig
            _cm = col_mapping
            _ns_cfg = NormalizeSchemaConfig(
                case_candidates=[_cm["case_col"]] if _cm.get("case_col") else None,
                act_candidates=[_cm["act_col"]] if _cm.get("act_col") else None,
                ts_candidates=[_cm["ts_col"]] if _cm.get("ts_col") else None,
            )
            ctx = _run_step(NormalizeSchemaStep(_ns_cfg), ctx, "Normalizing schema (custom mapping)...")
        else:
            ctx = _run_step(NormalizeSchemaStep(), ctx, "Normalizing schema...")

        # 2a. Filter lifecycle:transition events
        if lifecycle_drop_values and ctx.log is not None:
            lc_col = "lifecycle:transition"
            if lc_col in ctx.log.df.columns:
                before = len(ctx.log.df)
                ctx.log.df = ctx.log.df[
                    ~ctx.log.df[lc_col].astype(str).isin(lifecycle_drop_values)
                ].reset_index(drop=True)
                removed = before - len(ctx.log.df)
                _progress(
                    f"Lifecycle filter: removed {removed} events "
                    f"(dropping {lifecycle_drop_values}), {len(ctx.log.df)} remaining"
                )

        # 2b. Drop user-selected columns
        if columns_to_drop:
            ctx = _run_step(
                DropColumnsStep(DropColumnsConfig(columns=list(columns_to_drop))),
                ctx,
                f"Dropping {len(columns_to_drop)} column(s)...",
            )

        # 3. Clean + sort
        ctx = _run_step(DeduplicateEventsStep(DeduplicateEventsConfig()), ctx, "Deduplicating events...")
        ctx = _run_step(CleanAndSortStep(), ctx, "Cleaning and sorting...")
        ctx = _run_step(
            StableSortStep(StableSortConfig(tie_breakers=("_event_index",))),
            ctx, "Stable-sorting events...",
        )
        ctx = _run_step(QcReportStep(), ctx, "Running QC report...")

        # Repair backwards timestamps (auto)
        ctx = _run_step(RepairTimestampsStep(), ctx, "Repairing backwards timestamps...")

        # --- Auto preprocessing steps (literature-based) ---
        ctx = _run_step(FilterShortCasesStep(), ctx, "Filtering short cases (< 2 events)...")
        ctx = _run_step(NormalizeActivitiesStep(), ctx, "Normalizing activity labels...")
        ctx = _run_step(FilterInfrequentActivitiesStep(), ctx, "Filtering infrequent activities...")
        ctx = _run_step(FilterZeroDurationCasesStep(), ctx, "Filtering zero-duration cases...")
        if filter_long_cases:
            ctx = _run_step(FilterCaseLengthStep(), ctx, "Filtering extreme-length cases (p99)...")

        if filter_consecutive_duplicates:
            ctx = _run_step(FilterConsecutiveDuplicatesStep(), ctx, "Filtering consecutive duplicate events...")
        if impute_missing:
            ctx = _run_step(ImputeMissingAttributesStep(), ctx, "Imputing missing attribute values...")

        # 3c. Time window filter
        ctx = _run_step(
            ConceptDriftWindowStep(ConceptDriftWindowConfig(
                enabled=concept_drift_window,
                since_date=since_date or None,
            )),
            ctx,
            f"Time window filter: keeping cases from {since_date}..." if concept_drift_window else "Time window filter: disabled",
        )
        cdw_qc = ctx.artifacts.get("concept_drift_window_qc", {})
        if cdw_qc.get("enabled") and cdw_qc.get("cases_removed", 0):
            _progress(
                f"Time window filter: removed {cdw_qc['cases_removed']} old cases "
                f"({cdw_qc['cases_removed_pct']:.1f}%), {cdw_qc['cases_after']} remaining"
            )

        # 3d. Rare variant filter
        ctx = _run_step(
            FilterRareVariantsStep(FilterRareVariantsConfig(
                enabled=rare_variant_filter,
                min_variant_count=min_variant_count,
            )),
            ctx,
            f"Rare variant filter (min={min_variant_count})..." if rare_variant_filter else "Rare variant filter: disabled",
        )
        rvf_qc = ctx.artifacts.get("filter_rare_variants_qc", {})
        if rvf_qc.get("enabled") and rvf_qc.get("cases_removed", 0):
            _progress(
                f"Rare variant filter: removed {rvf_qc['cases_removed']} cases "
                f"({rvf_qc['cases_removed_pct']:.1f}%)"
            )

        # 4. Case labels
        if needs_case_labels:
            ctx = _run_step(
                CaseLabelsStep(
                    CaseLabelsConfig(
                        outcome_col=outcome_col,
                        outcome_values=outcome_values,
                        label_col="label_outcome",
                    )
                ),
                ctx,
                "Computing case labels...",
            )

        # 5. Prefix extraction
        ctx = _run_step(
            PrefixExtractionStep(PrefixExtractionConfig(
                max_prefix_len=max_prefix_len,
                min_prefix_len=1,
                next_event_attr_cols=extra_next_cols or None,
            )),
            ctx,
            "Extracting prefixes...",
        )
        # Emit prefix_features info card (same as full pipeline)
        _ps = ctx.artifacts.get("prefix_samples")
        _step_data({
            "step": "prefix_features",
            "label": "Prefix & Feature Extraction",
            "before_rows": None,
            "after_rows": int(len(_ps)) if _ps is not None else None,
            "before_cases": None,
            "after_cases": None,
            "qc": {
                "label_column": task.label_col,
                "time_features_added": [
                    "feat_elapsed_time_sec",
                    "feat_time_since_last_event_sec",
                    "feat_elapsed_time_log1p",
                    "feat_time_since_last_log1p",
                    "feat_prefix_end_hour",
                    "feat_prefix_end_weekday",
                    "feat_prefix_end_is_weekend",
                    "feat_prefix_end_month",
                ],
                "max_prefix_len": max_prefix_len,
                "total_prefix_rows": int(len(_ps)) if _ps is not None else None,
            },
        })

        # 6. Case split
        split_method2 = "temporal" if temporal_split else "random"
        _test_ratio2 = round(1.0 - train_ratio - val_ratio, 4)
        _progress(f"Splitting cases ({int(train_ratio*100)}/{int(val_ratio*100)}/{int(round(_test_ratio2*100))}, {split_method2})...")
        ctx = CaseSplitStep(CaseSplitConfig(
            train_ratio=train_ratio, val_ratio=val_ratio, random_state=42,
            temporal_split=temporal_split,
        )).run(ctx)
        _splits2 = ctx.artifacts.get("case_splits", {})
        _n_train2 = len(_splits2.get("train", []))
        _n_val2   = len(_splits2.get("val", []))
        _n_test2  = len(_splits2.get("test", []))
        _step_data({
            "step": "case_split",
            "label": f"Case Split ({split_method2})",
            "before_rows": None,
            "after_rows": None,
            "before_cases": None,
            "after_cases": None,
            "qc": {
                "split_method": split_method2,
                "train_cases": _n_train2,
                "val_cases": _n_val2,
                "test_cases": _n_test2,
                "train_pct": int(train_ratio * 100),
                "val_pct": int(val_ratio * 100),
                "test_pct": int(round((1 - train_ratio - val_ratio) * 100)),
            },
        })

        # 7. Outlier detection (skip for classification)
        _progress("Detecting label outliers...")
        _ps_b = ctx.artifacts.get("prefix_samples")
        _ps_b_rows = int(len(_ps_b)) if _ps_b is not None else None
        ctx = OutlierDetectionStep(OutlierDetectionConfig(enabled=outlier_enabled)).run(ctx)
        _o_qc = ctx.artifacts.get("outlier_detection_qc", {})
        _ps_a_rows = _o_qc.get("total_rows_after", _ps_b_rows)
        _step_data({
            "step": "outlier_detection",
            "label": "Detecting label outliers...",
            "before_rows": _ps_b_rows,
            "after_rows": _safe_int(_ps_a_rows, _ps_b_rows),
            "before_cases": None,
            "after_cases": None,
            "qc": {k: v for k, v in _o_qc.items() if isinstance(v, (int, float, str, bool, list, dict)) and k != "step"},
        })

        # 8. Filter rare classes
        _progress("Filtering rare classes...")
        _ps_b2 = ctx.artifacts.get("prefix_samples")
        _ps_b_rows2 = int(len(_ps_b2)) if _ps_b2 is not None else None
        ctx = FilterRareClassesStep(FilterRareClassesConfig(
            enabled=is_classification and min_class_samples > 0,
            min_class_samples=min_class_samples,
            label_col=task.label_col,
        )).run(ctx)
        _f_qc = ctx.artifacts.get("filter_rare_classes_qc", {})
        _ps_a_rows2 = _f_qc.get("rows_after", _ps_b_rows2)
        _step_data({
            "step": "filter_rare_classes",
            "label": "Filtering rare classes...",
            "before_rows": _ps_b_rows2,
            "after_rows": _safe_int(_ps_a_rows2, _ps_b_rows2),
            "before_cases": None,
            "after_cases": None,
            "qc": {k: v for k, v in _f_qc.items() if isinstance(v, (int, float, str, bool, list, dict)) and k != "step"},
        })

        # 9. Strategy search
        _progress("Strategy search: testing 5 bucketers × 4 encodings = 20 strategies...")
        ctx = SingleTaskStrategySearchStep(strategy_cfg).run(ctx)

        best = ctx.artifacts.get("best_strategy") or {}
        comparison = ctx.artifacts.get("single_task_comparison", [])

        score = best.get("primary_score")
        is_clf = is_classification
        metric_label = "F1" if is_clf else "MAE"
        score_str = f"{score:.4f}" if score is not None else "N/A"
        _progress(
            f"Best: {best.get('bucketing')} + {best.get('encoding')} "
            f"| {metric_label} = {score_str}"
        )

        if score is None:
            skipped = [r for r in comparison if r.get("primary_score") is None]
            return {
                "status": "no_strategy",
                "error": (
                    f"No valid strategy found — all {len(skipped)} strategies were skipped "
                    f"(too few samples per bucket or training failed). "
                    f"Try lowering min_bucket_samples or uploading a larger log."
                ),
                "best_strategy": best,
                "comparison": comparison,
                "primary_score": None,
                "is_classification": is_clf,
            }

        return {
            "status": "success",
            "best_strategy": best,
            "comparison": comparison,
            "primary_score": score,
            "is_classification": is_clf,
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
