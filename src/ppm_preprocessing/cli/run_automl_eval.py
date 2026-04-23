"""
AutoML Evaluation.

For each log × task:
  1. Apply non-optional preprocessing (same as strategy eval)
  2. Strategy search with LightGBM probe (all 20 bucketer × encoder combos)
  3. Take the best combo and run FLAML AutoML on it
  4. Report: baseline LightGBM | best LightGBM | AutoML (FLAML)

Usage:
    python -m ppm_preprocessing.cli.run_automl_eval \\
        --output results/automl_eval

    # Single log + task:
    python -m ppm_preprocessing.cli.run_automl_eval \\
        --task next_activity --output results/automl_eval
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Log registry (same as strategy eval)
# ---------------------------------------------------------------------------

@dataclass
class LogConfig:
    name: str
    path: str
    outcome_col: str = "activity"
    case_col: str = ""
    activity_col: str = ""
    timestamp_col: str = ""


DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"

LOG_CONFIGS: List[LogConfig] = [
    LogConfig(name="DomesticDeclarations",
              path=str(DATA_DIR / "DomesticDeclarations.xes")),
    LogConfig(name="PermitLog",
              path=str(DATA_DIR / "PermitLog.xes")),
    LogConfig(name="BPIC15_1",
              path=str(DATA_DIR / "BPIC15_1.xes")),
    LogConfig(name="BPI_Challenge_2013_closed_problems",
              path=str(DATA_DIR / "BPI_Challenge_2013_closed_problems.xes")),
    LogConfig(name="issues",
              path=str(DATA_DIR / "issues.csv"),
              case_col="id",
              timestamp_col="started"),
]

TASKS = ["next_activity", "remaining_time", "outcome"]
BASELINE_BUCKETER = "no_bucket"
BASELINE_ENCODER  = "last_state"
AUTOML_TIME_BUDGET = 300   # seconds per run


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _run_automl_eval(
    log_cfg: LogConfig,
    task_name: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    max_prefix_len: int = 30,
    min_bucket_samples: int = 100,
    time_budget_s: int = AUTOML_TIME_BUDGET,
) -> Dict[str, Any]:
    try:
        from ppm_preprocessing.domain.context import PipelineContext
        from ppm_preprocessing.io.format_detection import detect_format
        from ppm_preprocessing.steps.load_xes import LoadXesStep
        from ppm_preprocessing.steps.load_csv import LoadCsvStep
        from ppm_preprocessing.steps.normalize_schema import (
            NormalizeSchemaStep, NormalizeSchemaConfig)
        from ppm_preprocessing.steps.transform_aggregated_to_events import (
            TransformAggregatedToEventsStep, TransformAggregatedConfig)
        from ppm_preprocessing.steps.deduplicate_events import (
            DeduplicateEventsStep, DeduplicateEventsConfig)
        from ppm_preprocessing.steps.clean_sort import CleanAndSortStep
        from ppm_preprocessing.steps.stable_sort import StableSortStep, StableSortConfig
        from ppm_preprocessing.steps.repair_timestamps import RepairTimestampsStep
        from ppm_preprocessing.steps.filter_short_cases import FilterShortCasesStep
        from ppm_preprocessing.steps.normalize_activities import NormalizeActivitiesStep
        from ppm_preprocessing.steps.filter_infrequent_activities import FilterInfrequentActivitiesStep
        from ppm_preprocessing.steps.filter_zero_duration_cases import FilterZeroDurationCasesStep
        from ppm_preprocessing.steps.case_labels import CaseLabelsStep, CaseLabelsConfig
        from ppm_preprocessing.steps.prefix_extraction import (
            PrefixExtractionStep, PrefixExtractionConfig)
        from ppm_preprocessing.steps.split_cases import CaseSplitStep, CaseSplitConfig
        from ppm_preprocessing.steps.outlier_detection import (
            OutlierDetectionStep, OutlierDetectionConfig)
        from ppm_preprocessing.steps.filter_rare_classes import (
            FilterRareClassesStep, FilterRareClassesConfig)
        from ppm_preprocessing.bucketing.no_bucket import NoBucketer
        from ppm_preprocessing.bucketing.last_activity import LastActivityBucketer
        from ppm_preprocessing.bucketing.prefix_length_bins import PrefixLenBinsBucketer
        from ppm_preprocessing.bucketing.prefix_length_adaptive import PrefixLenAdaptiveBucketer
        from ppm_preprocessing.bucketing.cluster import ClusterBucketer, ClusterBucketConfig
        from ppm_preprocessing.steps.single_task_strategy_search import (
            SingleTaskStrategySearchStep, SingleTaskStrategySearchConfig)
        from ppm_preprocessing.steps.single_task_automl_train import (
            SingleTaskAutoMLTrainStep, SingleTaskAutoMLTrainConfig)
        from ppm_preprocessing.tasks.specs import default_task_specs

        tasks = default_task_specs()
        if task_name not in tasks:
            return {"status": "error", "error": f"Unknown task: {task_name}"}
        task = tasks[task_name]
        is_clf = "classification" in task.task_type

        log_path = Path(log_cfg.path)
        file_format = detect_format(str(log_path))
        load_step = LoadCsvStep() if file_format == "csv" else LoadXesStep()

        ctx = PipelineContext(input_path=str(log_path), task=task.name)
        ctx = load_step.run(ctx)

        _used_transform = False
        if file_format == "csv":
            raw_cols = list(ctx.raw_df.columns) if ctx.raw_df is not None else []
            wf_cols = [c for c in raw_cols if c.startswith("wf_")]
            if wf_cols:
                _used_transform = True
                ctx = TransformAggregatedToEventsStep(TransformAggregatedConfig(
                    case_id_col=log_cfg.case_col or "id",
                    start_time_col=log_cfg.timestamp_col or "started",
                    end_time_col="ended",
                    workflow_duration_prefix="wf_",
                    workflow_event_count_prefix="wfe_",
                )).run(ctx)

        norm_cfg = NormalizeSchemaConfig(
            case_candidates=[log_cfg.case_col] if (log_cfg.case_col and not _used_transform) else None,
            act_candidates=[log_cfg.activity_col] if (log_cfg.activity_col and not _used_transform) else None,
            ts_candidates=[log_cfg.timestamp_col] if (log_cfg.timestamp_col and not _used_transform) else None,
        )
        ctx = NormalizeSchemaStep(norm_cfg).run(ctx)

        # Non-optional preprocessing (Stages 2-10)
        ctx = DeduplicateEventsStep(DeduplicateEventsConfig()).run(ctx)
        ctx = CleanAndSortStep().run(ctx)
        ctx = StableSortStep(StableSortConfig(tie_breakers=("_event_index",))).run(ctx)
        ctx = RepairTimestampsStep().run(ctx)
        ctx = FilterShortCasesStep().run(ctx)
        ctx = NormalizeActivitiesStep().run(ctx)
        ctx = FilterInfrequentActivitiesStep().run(ctx)
        ctx = FilterZeroDurationCasesStep().run(ctx)

        n_cases = ctx.log.df["case_id"].nunique() if ctx.log else 0

        ctx = CaseLabelsStep(CaseLabelsConfig(outcome_col=log_cfg.outcome_col)).run(ctx)
        ctx = PrefixExtractionStep(PrefixExtractionConfig(
            max_prefix_len=max_prefix_len,
            min_prefix_len=1,
            add_time_features=True,
        )).run(ctx)

        ctx = CaseSplitStep(CaseSplitConfig(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            random_state=42,
        )).run(ctx)

        ctx = OutlierDetectionStep(OutlierDetectionConfig(
            enabled=True, label_col=task.label_col)).run(ctx)
        ctx = FilterRareClassesStep(FilterRareClassesConfig(
            enabled=is_clf, min_class_samples=3, label_col=task.label_col)).run(ctx)

        # Strategy search (LightGBM probe) — same as strategy eval
        bucketers = {
            "no_bucket":           NoBucketer(),
            "last_activity":       LastActivityBucketer(),
            "prefix_len_bins":     PrefixLenBinsBucketer(bin_size=5, max_len=max_prefix_len),
            "prefix_len_adaptive": PrefixLenAdaptiveBucketer(max_len=max_prefix_len),
            "cluster":             ClusterBucketer(ClusterBucketConfig(n_clusters=3)),
        }
        encodings = ["last_state", "aggregation", "index_latest_payload", "embedding"]

        strategy_cfg = SingleTaskStrategySearchConfig(
            task=task,
            bucketers=bucketers,
            encodings=encodings,
            min_bucket_samples=min_bucket_samples,
            skip_single_class=True,
            use_probe_model=True,
            save_report=False,
            save_csv=False,
        )
        ctx = SingleTaskStrategySearchStep(strategy_cfg).run(ctx)

        comparison = ctx.artifacts.get("single_task_comparison", [])
        best = ctx.artifacts.get("best_strategy") or {}

        baseline_row = next(
            (r for r in comparison
             if r.get("bucketing") == BASELINE_BUCKETER
             and r.get("encoding") == BASELINE_ENCODER),
            {},
        )
        lgbm_baseline = baseline_row.get("primary_score")
        lgbm_best     = best.get("primary_score")
        best_bucketer = best.get("bucketing")
        best_encoder  = best.get("encoding")

        if best_bucketer is None:
            return {
                "status": "skipped",
                "reason": "no_valid_lgbm_strategy",
                "is_classification": is_clf,
                "n_cases": n_cases,
                "lgbm_baseline": None,
                "lgbm_best": None,
            }

        # AutoML on the best combo
        print(f"  Running FLAML on best: {best_bucketer}+{best_encoder} "
              f"(budget={time_budget_s}s)...", flush=True)

        ctx = SingleTaskAutoMLTrainStep(
            config=SingleTaskAutoMLTrainConfig(
                task_name=task.name,
                bucketers=bucketers,
                encodings=encodings,
                min_bucket_samples=min_bucket_samples,
                skip_single_class=True,
                time_budget_s=time_budget_s,
                seed=42,
                n_jobs=1,
                estimator_list=None,
                target_log1p=False,   # keep original scale for comparable MAE
            ),
            tasks=tasks,
        ).run(ctx)

        automl_out = ctx.artifacts.get("single_task_automl", {})

        # Extract val score + model details from FLAML output
        # FLAML minimises loss:
        #   classification → loss = 1 - f1_macro  → score = 1 - loss
        #   regression     → loss = MAE (seconds)  → score = loss
        mode_results = automl_out.get("mode_results", {})
        global_res   = mode_results.get("global_model", {})
        flaml_info   = global_res.get("flaml", {})
        best_loss: Optional[float] = flaml_info.get("best_loss")

        automl_model_info: Dict[str, Any] = {}

        if best_loss is not None:
            # Global model path
            automl_score = (1 - best_loss) if is_clf else best_loss
            automl_model_info["mode"] = "global_model"
            automl_model_info["best_estimator"] = flaml_info.get("best_estimator")
            automl_model_info["best_config"]    = flaml_info.get("best_config")
            automl_model_info["best_loss"]      = best_loss
            automl_model_info["time_to_best_s"] = flaml_info.get("time_to_find_best_s")
        else:
            # Per-bucket path
            pb = mode_results.get("per_bucket_models", {})
            per_bucket = pb.get("per_bucket", {})
            bucket_details = {}
            losses = []
            for bid, bv in per_bucket.items():
                fi = bv.get("flaml", {})
                bl = fi.get("best_loss")
                if bl is not None:
                    losses.append(bl)
                bucket_details[bid] = {
                    "best_estimator": fi.get("best_estimator"),
                    "best_config":    fi.get("best_config"),
                    "best_loss":      bl,
                    "time_to_best_s": fi.get("time_to_find_best_s"),
                }
            if losses:
                avg_loss = sum(losses) / len(losses)
                automl_score = (1 - avg_loss) if is_clf else avg_loss
            else:
                automl_score = None
            automl_model_info["mode"]           = "per_bucket"
            automl_model_info["best_estimator"] = "per_bucket"
            automl_model_info["buckets"]        = bucket_details
            automl_model_info["n_buckets"]      = len(bucket_details)

        return {
            "status": "success",
            "is_classification": is_clf,
            "n_cases": n_cases,
            "lgbm_baseline": lgbm_baseline,
            "lgbm_best": lgbm_best,
            "best_bucketer": best_bucketer,
            "best_encoder": best_encoder,
            "automl_score": automl_score,
            "automl_best_estimator": automl_model_info.get("best_estimator"),
            "automl_model_info": automl_model_info,
        }

    except Exception as e:
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(score, is_clf: bool) -> str:
    if score is None:
        return "     N/A"
    if is_clf:
        return f"{score * 100:6.2f}%"
    return f"{score / 86400:8.3f} d"


def _delta(a, b, higher_is_better: bool) -> str:
    """Relative gain of a over b (%)."""
    if a is None or b is None or b == 0:
        return "    N/A"
    pct = (a - b) / abs(b) * 100 if higher_is_better else (b - a) / abs(b) * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def _print_row(result: Dict, log_name: str, task_name: str) -> None:
    is_clf = result.get("is_classification", True)
    h = is_clf
    bl = result.get("lgbm_baseline")
    lb = result.get("lgbm_best")
    am = result.get("automl_score")
    bb = result.get("best_bucketer", "N/A")
    be = result.get("best_encoder", "N/A")
    est = result.get("automl_best_estimator", "?")

    d_lgbm  = _delta(lb, bl, h)   # LightGBM best vs baseline
    d_automl = _delta(am, lb, h)  # AutoML vs LightGBM best

    print(f"  {log_name:<34} {task_name:<16} "
          f"{_fmt(bl, is_clf):>9}  {_fmt(lb, is_clf):>9} ({d_lgbm:>8})  "
          f"{_fmt(am, is_clf):>9} ({d_automl:>8})  "
          f"{bb}+{be}  [{est}]")


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def _save(all_results: Dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / "automl_eval_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    W = 130
    lines = [
        "=" * W,
        f"  {'Log':<34} {'Task':<16} {'LGBMbase':>9}  {'LGBMbest':>9} {'(Dbase)':>10}  "
        f"{'AutoML':>9} {'(DLGBM)':>9}  Best combo  [estimator]",
        "-" * W,
    ]
    for key, r in all_results.items():
        log_name, task_name = key.split("|")
        is_clf = r.get("is_classification", True)
        h = is_clf
        bl = r.get("lgbm_baseline")
        lb = r.get("lgbm_best")
        am = r.get("automl_score")
        bb = r.get("best_bucketer", "N/A")
        be = r.get("best_encoder", "N/A")
        est = r.get("automl_best_estimator", "?")
        status = r.get("status", "?")
        if status != "success":
            lines.append(f"  {log_name:<34} {task_name:<16}  [{status}: {r.get('reason', r.get('error','')[:60])}]")
            continue
        d_lgbm   = _delta(lb, bl, h)
        d_automl = _delta(am, lb, h)
        lines.append(
            f"  {log_name:<34} {task_name:<16} "
            f"{_fmt(bl, is_clf):>9}  {_fmt(lb, is_clf):>9} ({d_lgbm:>8})  "
            f"{_fmt(am, is_clf):>9} ({d_automl:>8})  "
            f"{bb}+{be}  [{est}]"
        )
    lines.append("=" * W)

    out_txt = out_dir / "automl_eval_summary.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  -> {out_json}")
    print(f"  -> {out_txt}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AutoML evaluation: strategy search (LightGBM) then FLAML on best combo.")
    parser.add_argument("--task", default="",
                        choices=["", "next_activity", "remaining_time", "outcome"])
    parser.add_argument("--output", default="results/automl_eval")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--max-prefix-len", type=int, default=30)
    parser.add_argument("--min-bucket-samples", type=int, default=100)
    parser.add_argument("--time-budget", type=int, default=AUTOML_TIME_BUDGET,
                        help="FLAML time budget per run in seconds (default: 180)")
    args = parser.parse_args()

    tasks_to_run = [args.task] if args.task else TASKS
    out_dir = Path(args.output)

    # Resume support
    out_json_path = out_dir / "automl_eval_results.json"
    all_results: Dict[str, Dict] = {}
    if out_json_path.exists():
        with open(out_json_path) as f:
            all_results = json.load(f)
        print(f"Resuming: {len(all_results)} run(s) already done.")

    logs = [lc for lc in LOG_CONFIGS if Path(lc.path).exists()]
    total = len(logs) * len(tasks_to_run)
    done = 0

    W = 130
    print()
    print("=" * W)
    print(f"  {'Log':<34} {'Task':<16} {'LGBMbase':>9}  {'LGBMbest':>9} {'(Dbase)':>10}  "
          f"{'AutoML':>9} {'(DLGBM)':>9}  Best combo  [estimator]")
    print("-" * W)

    for log_cfg in logs:
        for task_name in tasks_to_run:
            done += 1
            key = f"{log_cfg.name}|{task_name}"

            if key in all_results:
                r = all_results[key]
                print(f"\n[{done}/{total}] {log_cfg.name} / {task_name}  [already done]")
                _print_row(r, log_cfg.name, task_name)
                continue

            print(f"\n[{done}/{total}] {log_cfg.name} / {task_name}", flush=True)
            t0 = time.time()

            result = _run_automl_eval(
                log_cfg=log_cfg,
                task_name=task_name,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                max_prefix_len=args.max_prefix_len,
                min_bucket_samples=args.min_bucket_samples,
                time_budget_s=args.time_budget,
            )
            elapsed = time.time() - t0
            result["elapsed_s"] = round(elapsed, 1)
            all_results[key] = result

            if result["status"] == "error":
                print(f"  ERROR: {result.get('error', '')}")
                tb = result.get("traceback", "")
                if tb:
                    print(tb[-800:])
            else:
                _print_row(result, log_cfg.name, task_name)
                print(f"  [{elapsed:.0f}s]")

            _save(all_results, out_dir)

    print("=" * W)
    print(f"\nDone. {done} combination(s) processed.")


if __name__ == "__main__":
    main()
