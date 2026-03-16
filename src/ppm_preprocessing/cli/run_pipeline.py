from pathlib import Path
import json
import sys

from ppm_preprocessing.domain.context import PipelineContext
from ppm_preprocessing.pipeline.preprocessing_pipeline import PreprocessingPipeline

from ppm_preprocessing.steps.load_xes import LoadXesStep
from ppm_preprocessing.steps.load_csv import LoadCsvStep
from ppm_preprocessing.steps.normalize_schema import NormalizeSchemaStep
from ppm_preprocessing.steps.clean_sort import CleanAndSortStep
from ppm_preprocessing.steps.qc_report import QcReportStep
from ppm_preprocessing.io.format_detection import detect_format

# NEW: case-level labels (outcome + end time)
from ppm_preprocessing.steps.case_labels import CaseLabelsStep, CaseLabelsConfig

from ppm_preprocessing.steps.prefix_extraction import PrefixExtractionStep, PrefixExtractionConfig
from ppm_preprocessing.steps.split_cases import CaseSplitStep, CaseSplitConfig

# Bucketers
from ppm_preprocessing.bucketing.no_bucket import NoBucketer
from ppm_preprocessing.bucketing.prefix_length_bins import PrefixLenBinsBucketer

# Tasks + Multi-task compare
from ppm_preprocessing.tasks.specs import default_task_specs
from ppm_preprocessing.steps.compare_tasks_bucketing_encoding import (
    CompareTasksBucketingEncodingStep,
    CompareTasksConfig,
)
from ppm_preprocessing.steps.automl_train_best import AutoMLTrainBestStep, AutoMLTrainBestConfig

ROOT = Path(__file__).resolve().parents[3]
LOG_PATH = ROOT / "data" / "raw" / "BPI Challenge 2017.xes.gz"
OUT_DIR = ROOT / "outputs" / "reports"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Simple argument parsing (backward compatible)
    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])
    else:
        log_path = LOG_PATH  # default hardcoded path

    if not log_path.exists():
        raise FileNotFoundError(f"Log not found: {log_path}")

    # Detect format and choose appropriate loader
    file_format = detect_format(str(log_path))

    if file_format == 'csv':
        load_step = LoadCsvStep()
    else:  # xes
        load_step = LoadXesStep()

    ctx = PipelineContext(
        input_path=str(log_path),
        task="multi_task",
    )

    pipeline = PreprocessingPipeline(steps=[
        # canonical log
        load_step,  # Dynamic loader based on file format
        NormalizeSchemaStep(),
        CleanAndSortStep(),
        QcReportStep(),

        # NEW: case-level labels (provides ctx.artifacts["case_table"])
        CaseLabelsStep(
            CaseLabelsConfig(
                positive_activity="A_Accepted",   # baseline for BPI 2017
                outcome_col="label_outcome",
            )
        ),

        # prefixes (now produces label_next_activity + label_outcome + label_remaining_time_sec + prefix_row_id)
        PrefixExtractionStep(
            PrefixExtractionConfig(
                max_prefix_len=30,
                min_prefix_len=1,
                # sample_frac=0.1,  # optional for dev speed
            )
        ),

        # fixed case split (fair comparison)
        CaseSplitStep(
            CaseSplitConfig(
                train_ratio=0.7,
                val_ratio=0.15,
                random_state=42,
            )
        ),

        # compare tasks × bucketing × encoding with fixed task-specific models
        CompareTasksBucketingEncodingStep(
            CompareTasksConfig(
                tasks=default_task_specs(),
                bucketers={
                    "no_bucket": NoBucketer(),
                    "prefix_len_bins": PrefixLenBinsBucketer(bin_size=5, max_len=30),
                },
                encodings=["last_n_5", "last_n_10", "aggregation"],
                min_bucket_samples=1000,
            )
        ),

         AutoMLTrainBestStep(AutoMLTrainBestConfig(time_budget_s=300)),
    ])

    ctx = pipeline.run(ctx)

    # ---- Print everything "in between" ----
    print("\n================ QC REPORT ================")
    print(json.dumps(ctx.artifacts.get("qc_report", {}), indent=2))

    print("\n================ CASE LABELS QC ================")
    print(json.dumps(ctx.artifacts.get("case_labels_qc", {}), indent=2))

    print("\n================ PREFIX QC ================")
    print(json.dumps(ctx.artifacts.get("prefix_qc", {}), indent=2))

    print("\n================ PREFIX SAMPLES HEAD ================")
    ps = ctx.artifacts.get("prefix_samples")
    if ps is not None:
        print(ps.head(5))

    print("\n================ CASE SPLITS ================")
    splits = ctx.artifacts.get("case_splits", {})
    print({
        "num_train_cases": len(splits.get("train", [])),
        "num_val_cases": len(splits.get("val", [])),
        "num_test_cases": len(splits.get("test", [])),
    })

 # ---- Multi-task comparison prints ----
    print("\n================ COMPARISON RESULTS (TASKS x BUCKETING x ENCODING) ================\n")
    results = ctx.artifacts.get("comparison_results", [])

    if not results:
        print("No comparison results found.")
    else:
        print(f"{'Task':<14} | {'Bucketing':<16} | {'Encoding':<14} | {'Mode':<18} | {'Primary':<10} | Details")
        print("-" * 120)

        for r in results:
            score = r.get("primary_score")
            score_str = f"{score:.4f}" if isinstance(score, (int, float)) and score is not None else "N/A"
            details = r.get("details", {})

            print(
                f"{r['task']:<14} | "
                f"{r['bucketing']:<16} | "
                f"{r['encoding']:<14} | "
                f"{r['mode']:<18} | "
                f"{score_str:<10} | "
                f"{details}"
            )

    # ---- Pick winners (best per task) ----
    def pick_best_per_task(rows):
        best = {}
        for r in rows:
            task = r.get("task")
            score = r.get("primary_score")
            if task is None or score is None:
                continue
            lower_better = (task == "remaining_time")
            if task not in best:
                best[task] = r
                continue
            prev = best[task].get("primary_score")
            if prev is None:
                best[task] = r
                continue
            if lower_better:
                if score < prev:
                    best[task] = r
            else:
                if score > prev:
                    best[task] = r
        return best

    best = pick_best_per_task(results)

    print("\n================ WINNERS (BEST STRATEGY PER TASK) ================\n")
    if not best:
        print("No winners could be selected (all tasks may have been skipped).")
    else:
        for task, r in best.items():
            print(
                f"- {task}: "
                f"{r['bucketing']}/{r['encoding']}/{r.get('mode','?')} "
                f"(primary_score={r.get('primary_score')})"
            )

    # ---- AutoML report (print ONCE) ----

    report = ctx.artifacts.get("automl_report")
    print("\n================ AUTOML SUMMARY ================\n")
    if not report:
        print("No automl_report found. (Did AutoMLTrainBestStep run?)")
    else:
        runs = report.get("runs", {})
        for task, run in runs.items():
            if run.get("skipped"):
                print(f"- {task}: SKIPPED ({run.get('reason')}) | strategy={run.get('strategy')}")
                continue

            ai = run.get("automl", {})
            data = run.get("data", {})
            strat = run.get("strategy", {})

            print(
                f"- {task}: OK | "
                f"strategy={strat.get('bucketing')}/{strat.get('encoding')}/{strat.get('mode')} | "
                f"best_estimator={ai.get('best_estimator')} | "
                f"best_loss={ai.get('best_loss')} | "
                f"metric={ai.get('metric')} | "
                f"time_to_find_best_s={ai.get('training_log', {}).get('time_to_find_best_s')} | "
                f"X_train={data.get('X_train')} X_val={data.get('X_val')}"
            )

    # save
    out_file = OUT_DIR / "task_bucketing_encoding_comparison.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved comparison to: {out_file}")



if __name__ == "__main__":
    main()
