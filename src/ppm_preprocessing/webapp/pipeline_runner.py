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
    recent_pct: float = 80.0,
    on_progress: Any = None,
) -> Dict[str, Any]:

    def _progress(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    def _run_step(step, ctx, label: str) -> PipelineContext:
        """Run a single pipeline step with progress reporting."""
        _progress(label)
        return step.run(ctx)

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
        # Run steps individually with progress reporting
        # ---------------------------------------------------------------

        # 1. Load
        ctx = _run_step(load_step, ctx, f"Loading {file_format.upper()} file...")

        # 1b. CSV transform if needed
        if file_format == "csv":
            ctx = _run_step(
                TransformAggregatedToEventsStep(
                    TransformAggregatedConfig(
                        case_id_col="id",
                        start_time_col="started",
                        end_time_col="ended",
                        workflow_duration_prefix="wf_",
                        workflow_event_count_prefix="wfe_",
                    )
                ),
                ctx,
                "Transforming aggregated CSV to events...",
            )

        # 2. Normalize + clean
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

        # 2c. Concept Drift Window — keep only the most recent X% of cases
        ctx = _run_step(
            ConceptDriftWindowStep(ConceptDriftWindowConfig(
                enabled=concept_drift_window,
                recent_pct=recent_pct,
            )),
            ctx,
            f"Concept drift window: keeping most recent {recent_pct:.0f}% of cases..." if concept_drift_window else "Concept drift window: disabled",
        )
        cdw_qc = ctx.artifacts.get("concept_drift_window_qc", {})
        if cdw_qc.get("enabled") and cdw_qc.get("cases_removed", 0):
            _progress(
                f"Concept drift window: removed {cdw_qc['cases_removed']} old cases "
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

        # 5. Case split
        test_ratio = round(1.0 - train_ratio - val_ratio, 4)
        split_method = "temporal" if temporal_split else "random"
        ctx = _run_step(
            CaseSplitStep(CaseSplitConfig(
                train_ratio=train_ratio, val_ratio=val_ratio, random_state=42,
                temporal_split=temporal_split,
            )),
            ctx,
            f"Splitting cases ({int(train_ratio*100)}/{int(val_ratio*100)}/{int(test_ratio*100)}, {split_method})...",
        )
        splits = ctx.artifacts.get("case_splits", {})
        _progress(
            f"Split: {len(splits.get('train', []))} train, "
            f"{len(splits.get('val', []))} val, "
            f"{len(splits.get('test', []))} test cases"
        )

        # 5b. Outlier detection (IQR on train labels) — skip for classification
        ctx = _run_step(
            OutlierDetectionStep(OutlierDetectionConfig(enabled=outlier_enabled and not is_classification)),
            ctx,
            "Detecting label outliers (IQR)...",
        )
        outlier_qc = ctx.artifacts.get("outlier_detection_qc", {})
        n_removed = outlier_qc.get("outlier_rows_removed", 0)
        if n_removed:
            _progress(
                f"Outlier detection: removed {n_removed} train rows "
                f"({outlier_qc.get('outlier_pct', 0):.1f}%)"
            )

        # 5c. Filter rare classes — only for classification
        ctx = _run_step(
            FilterRareClassesStep(FilterRareClassesConfig(
                enabled=is_classification and min_class_samples > 0,
                min_class_samples=min_class_samples,
                label_col=task.label_col,
            )),
            ctx,
            "Filtering rare classes...",
        )
        filter_qc = ctx.artifacts.get("filter_rare_classes_qc", {})
        if filter_qc.get("rows_removed", 0):
            _progress(
                f"Rare class filter: removed {filter_qc['rare_classes_removed']} classes "
                f"({filter_qc['rows_removed']} rows, {filter_qc.get('rows_removed_pct', 0):.1f}%)"
            )

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
    recent_pct: float = 80.0,
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

    def _run_step(step, ctx, label: str) -> PipelineContext:
        _progress(label)
        return step.run(ctx)

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
            use_probe_model=True,
        )

        # 1. Load
        ctx = _run_step(load_step, ctx, f"Loading {file_format.upper()} file...")

        # 1b. CSV transform if needed
        if file_format == "csv":
            ctx = _run_step(
                TransformAggregatedToEventsStep(
                    TransformAggregatedConfig(
                        case_id_col="id",
                        start_time_col="started",
                        end_time_col="ended",
                        workflow_duration_prefix="wf_",
                        workflow_event_count_prefix="wfe_",
                    )
                ),
                ctx,
                "Transforming aggregated CSV to events...",
            )

        # 2. Normalize
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

        # 3c. Concept drift window
        ctx = _run_step(
            ConceptDriftWindowStep(ConceptDriftWindowConfig(
                enabled=concept_drift_window,
                recent_pct=recent_pct,
            )),
            ctx,
            f"Concept drift window: keeping most recent {recent_pct:.0f}%..." if concept_drift_window else "Concept drift window: disabled",
        )
        cdw_qc = ctx.artifacts.get("concept_drift_window_qc", {})
        if cdw_qc.get("enabled") and cdw_qc.get("cases_removed", 0):
            _progress(
                f"Concept drift window: removed {cdw_qc['cases_removed']} old cases "
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

        # 6. Case split
        split_method2 = "temporal" if temporal_split else "random"
        ctx = _run_step(
            CaseSplitStep(CaseSplitConfig(
                train_ratio=train_ratio, val_ratio=val_ratio, random_state=42,
                temporal_split=temporal_split,
            )),
            ctx,
            f"Splitting cases ({int(train_ratio*100)}/{int(val_ratio*100)}/{int(round(1-train_ratio-val_ratio,4)*100)}, {split_method2})...",
        )

        # 7. Outlier detection (skip for classification)
        ctx = _run_step(
            OutlierDetectionStep(OutlierDetectionConfig(enabled=outlier_enabled and not is_classification)),
            ctx,
            "Detecting label outliers...",
        )

        # 8. Filter rare classes
        ctx = _run_step(
            FilterRareClassesStep(FilterRareClassesConfig(
                enabled=is_classification and min_class_samples > 0,
                min_class_samples=min_class_samples,
                label_col=task.label_col,
            )),
            ctx,
            "Filtering rare classes...",
        )

        # 9. Strategy search
        _progress("Strategy search: testing 5 bucketers × 4 encodings = 20 strategies...")
        ctx = SingleTaskStrategySearchStep(strategy_cfg).run(ctx)

        best = ctx.artifacts.get("best_strategy", {})
        comparison = ctx.artifacts.get("single_task_comparison", [])

        score = best.get("primary_score")
        is_clf = is_classification
        metric_label = "F1" if is_clf else "MAE"
        score_str = f"{score:.4f}" if score is not None else "N/A"
        _progress(
            f"Best: {best.get('bucketing')} + {best.get('encoding')} "
            f"| {metric_label} = {score_str}"
        )

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
