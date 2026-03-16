from __future__ import annotations

from pathlib import Path
import json
import sys

from ppm_preprocessing.domain.context import PipelineContext
from ppm_preprocessing.pipeline.preprocessing_pipeline import PreprocessingPipeline

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

# NEW ETL-hardening steps
from ppm_preprocessing.steps.deduplicate_events import DeduplicateEventsStep, DeduplicateEventsConfig
from ppm_preprocessing.steps.stable_sort import StableSortStep, StableSortConfig

from ppm_preprocessing.bucketing.no_bucket import NoBucketer
from ppm_preprocessing.bucketing.prefix_length_bins import PrefixLenBinsBucketer
from ppm_preprocessing.bucketing.prefix_length_adaptive import PrefixLenAdaptiveBucketer
from ppm_preprocessing.bucketing.cluster import ClusterBucketer, ClusterBucketConfig
from ppm_preprocessing.tasks.specs import default_task_specs

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

ROOT = Path(__file__).resolve().parents[3]
LOG_PATH = ROOT / "data" / "raw" / "BPIC15_1.xes"
#LOG_PATH = ROOT / "data" / "raw" / "BPI_Challenge_2013_closed_problems.xes"
#LOG_PATH = ROOT / "data" / "raw" / "issues.csv"

OUT_DIR = ROOT / "outputs" / "single_task"


def _json_dump(obj) -> str:
    return json.dumps(obj, indent=2, default=str)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = default_task_specs()
    task_name = sys.argv[2] if len(sys.argv) > 2 else "remaining_time"
    task = tasks[task_name]

    # Simple argument parsing (backward compatible)
    # Usage: run_task.py [log_path] [task_name] [outcome_col] [outcome_values]
    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])
    else:
        log_path = LOG_PATH  # default hardcoded path

    outcome_col = sys.argv[3] if len(sys.argv) > 3 else ""
    outcome_values = sys.argv[4] if len(sys.argv) > 4 else ""

    if not log_path.exists():
        raise FileNotFoundError(f"Log not found: {log_path}")

    # Detect format and choose appropriate loader
    file_format = detect_format(str(log_path))

    if file_format == 'csv':
        load_step = LoadCsvStep()
    else:  # xes
        load_step = LoadXesStep()

    ctx = PipelineContext(input_path=str(log_path), task=task.name)
    ctx.artifacts["out_dir"] = str(OUT_DIR)

    is_classification = "classification" in task.task_type
    needs_case_labels = task.name in {"remaining_time", "outcome", "next_activity"}

    # ✅ CONSISTENT keys (no parameter suffix in the key)
    bucketers = {
        "no_bucket": NoBucketer(),
        "prefix_len_bins": PrefixLenBinsBucketer(bin_size=5, max_len=30),
        "prefix_len_adaptive": PrefixLenAdaptiveBucketer(max_len=30),
        "cluster": ClusterBucketer(ClusterBucketConfig(n_clusters=3)),
    }

    # ✅ only encodings that your EncodingStep actually produces
    encodings = ["last_state", "aggregation", "index_latest_payload", "embedding"]

    strategy_cfg = SingleTaskStrategySearchConfig(
        task=task,
        bucketers=bucketers,
        encodings=encodings,
        min_bucket_samples=100,
        skip_single_class=True,
    )

    steps = [
        load_step,  # Dynamic loader based on file format
    ]

    # If CSV with aggregated format, add transformation step
    if file_format == 'csv':
        steps.append(
            TransformAggregatedToEventsStep(
                TransformAggregatedConfig(
                    case_id_col="id",
                    start_time_col="started",
                    end_time_col="ended",
                    workflow_duration_prefix="wf_",
                    workflow_event_count_prefix="wfe_",
                )
            )
        )

    steps.extend([
        NormalizeSchemaStep(),

        # ETL hardening
        DeduplicateEventsStep(DeduplicateEventsConfig()),
        CleanAndSortStep(),
        StableSortStep(StableSortConfig(tie_breakers=("_event_index",))),
        QcReportStep(),
    ])

    if needs_case_labels:
        steps.append(
            CaseLabelsStep(
                CaseLabelsConfig(
                    outcome_col=outcome_col,
                    outcome_values=outcome_values,
                    label_col="label_outcome",
                )
            )
        )

    steps.extend([
        PrefixExtractionStep(PrefixExtractionConfig(max_prefix_len=50, min_prefix_len=1)),

        CaseSplitStep(CaseSplitConfig(train_ratio=0.7, val_ratio=0.15, random_state=42)),

        # 0) outlier detection on train labels (IQR) — skip for classification
        OutlierDetectionStep(OutlierDetectionConfig(enabled=not is_classification)),

        # 0b) filter rare classes — only for classification
        FilterRareClassesStep(FilterRareClassesConfig(
            enabled=is_classification,
            min_class_samples=10,
            label_col=task.label_col,
        )),

        # 1) find best strategy
        SingleTaskStrategySearchStep(strategy_cfg),

        # 2) train AutoML on that best strategy
        SingleTaskAutoMLTrainStep(
            config=SingleTaskAutoMLTrainConfig(
                task_name=task.name,
                bucketers=bucketers,
                encodings=encodings,
                min_bucket_samples=100,
                skip_single_class=True,
                time_budget_s=300,
                seed=42,
                estimator_list=None,
                n_example_rows=3,
                examples_policy="all",  
                target_log1p=not is_classification,
            ),
            tasks=tasks,
        ),

        # 3) build + save report artifacts
        SingleTaskReportExamplesStep(SingleTaskReportExamplesConfig(
            task=task,
            out_key="single_task_report",
            in_key="single_task_automl",
            save_json=True,
            save_examples_csv=True,
        )),

        SingleTaskPersistModelStep(),

        # 4) Final test evaluation on held-out test set
        SingleTaskTestEvaluationStep(
            config=SingleTaskTestEvaluationConfig(),
            task=task,
        ),

        # 5) Feature importance report
        SingleTaskFeatureImportanceStep(),
    ])

    pipeline = PreprocessingPipeline(steps=steps)
    ctx = pipeline.run(ctx)

    print("\n=== QC ===")
    print(_json_dump(ctx.artifacts.get("qc_report", {})))

    print("\n=== ETL QC (lifecycle/dedupe/sort) ===")
    print(_json_dump({
        "deduplicate_qc": ctx.artifacts.get("deduplicate_qc"),
        "stable_sort_qc": ctx.artifacts.get("stable_sort_qc"),
    }))

    print("\n=== BEST STRATEGY ===")
    print(_json_dump(ctx.artifacts.get("best_strategy", {})))

    print("\n=== COMPARISON ROWS ===")
    print(len(ctx.artifacts.get("single_task_comparison", [])))

    print("\n=== SINGLE TASK AUTOML ===")
    print(_json_dump(ctx.artifacts.get("single_task_automl", {})))

    print("\n=== SINGLE TASK REPORT ===")
    print(_json_dump(ctx.artifacts.get("single_task_report", {})))

    print("\n=== TEST EVALUATION ===")
    print(_json_dump(ctx.artifacts.get("single_task_test_evaluation", {})))

    # persist summary jsons
    (OUT_DIR / "comparison.json").write_text(_json_dump(ctx.artifacts.get("single_task_comparison", [])), encoding="utf-8")
    (OUT_DIR / "best_strategy.json").write_text(_json_dump(ctx.artifacts.get("best_strategy", {})), encoding="utf-8")
    (OUT_DIR / "report.json").write_text(_json_dump(ctx.artifacts.get("single_task_report", {})), encoding="utf-8")
    (OUT_DIR / "test_evaluation.json").write_text(_json_dump(ctx.artifacts.get("single_task_test_evaluation", {})), encoding="utf-8")


if __name__ == "__main__":
    main()
