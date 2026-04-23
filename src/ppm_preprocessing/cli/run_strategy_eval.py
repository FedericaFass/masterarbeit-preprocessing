"""
Strategy Search Evaluation.

Applies the full non-optional preprocessing pipeline (Stages 2-10)
and then runs a strategy search over all bucketer × encoder combinations.
Compares each combination against the baseline (no_bucket + last_state).

Uses the same probe model and split as the ablation study
(LightGBM for classification, Ridge for regression; 70/15/15, seed=42).

Usage:
    python -m ppm_preprocessing.cli.run_strategy_eval \\
        --output results/strategy_eval

    # Single log + task:
    python -m ppm_preprocessing.cli.run_strategy_eval \\
        --log data/DomesticDeclarations.xes \\
        --task next_activity \\
        --output results/strategy_eval
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Log registry
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
    LogConfig(
        name="DomesticDeclarations",
        path=str(DATA_DIR / "DomesticDeclarations.xes"),
        outcome_col="activity",
    ),
    LogConfig(
        name="PermitLog",
        path=str(DATA_DIR / "PermitLog.xes"),
        outcome_col="activity",
    ),
    LogConfig(
        name="BPIC15_1",
        path=str(DATA_DIR / "BPIC15_1.xes"),
        outcome_col="activity",
    ),
    LogConfig(
        name="BPI_Challenge_2013_closed_problems",
        path=str(DATA_DIR / "BPI_Challenge_2013_closed_problems.xes"),
        outcome_col="activity",
    ),
    LogConfig(
        name="issues",
        path=str(DATA_DIR / "issues.csv"),
        outcome_col="activity",
        case_col="id",
        timestamp_col="started",
    ),
]

TASKS = ["next_activity", "remaining_time", "outcome"]

BASELINE_BUCKETER = "no_bucket"
BASELINE_ENCODER = "last_state"


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def _run_strategy_search(
    log_cfg: LogConfig,
    task_name: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    max_prefix_len: int = 30,
    min_bucket_samples: int = 100,
) -> Dict[str, Any]:
    """
    Apply non-optional preprocessing (Stages 2-10) then run strategy search
    over all bucketer × encoder combinations.

    Returns a dict with:
        status, comparison (list of dicts per strategy), baseline_score,
        best_bucketer, best_encoder, best_score, is_classification,
        primary_metric, n_cases
    """
    try:
        from ppm_preprocessing.domain.context import PipelineContext
        from ppm_preprocessing.io.format_detection import detect_format
        from ppm_preprocessing.steps.load_xes import LoadXesStep
        from ppm_preprocessing.steps.load_csv import LoadCsvStep
        from ppm_preprocessing.steps.normalize_schema import (
            NormalizeSchemaStep, NormalizeSchemaConfig,
        )
        from ppm_preprocessing.steps.transform_aggregated_to_events import (
            TransformAggregatedToEventsStep, TransformAggregatedConfig,
        )
        from ppm_preprocessing.steps.deduplicate_events import (
            DeduplicateEventsStep, DeduplicateEventsConfig,
        )
        from ppm_preprocessing.steps.clean_sort import CleanAndSortStep
        from ppm_preprocessing.steps.stable_sort import StableSortStep, StableSortConfig
        from ppm_preprocessing.steps.repair_timestamps import RepairTimestampsStep
        from ppm_preprocessing.steps.filter_short_cases import FilterShortCasesStep
        from ppm_preprocessing.steps.normalize_activities import NormalizeActivitiesStep
        from ppm_preprocessing.steps.filter_infrequent_activities import (
            FilterInfrequentActivitiesStep,
        )
        from ppm_preprocessing.steps.filter_zero_duration_cases import (
            FilterZeroDurationCasesStep,
        )
        from ppm_preprocessing.steps.case_labels import CaseLabelsStep, CaseLabelsConfig
        from ppm_preprocessing.steps.prefix_extraction import (
            PrefixExtractionStep, PrefixExtractionConfig,
        )
        from ppm_preprocessing.steps.split_cases import CaseSplitStep, CaseSplitConfig
        from ppm_preprocessing.steps.outlier_detection import (
            OutlierDetectionStep, OutlierDetectionConfig,
        )
        from ppm_preprocessing.steps.filter_rare_classes import (
            FilterRareClassesStep, FilterRareClassesConfig,
        )
        from ppm_preprocessing.bucketing.no_bucket import NoBucketer
        from ppm_preprocessing.bucketing.last_activity import LastActivityBucketer
        from ppm_preprocessing.bucketing.prefix_length_bins import PrefixLenBinsBucketer
        from ppm_preprocessing.bucketing.prefix_length_adaptive import (
            PrefixLenAdaptiveBucketer,
        )
        from ppm_preprocessing.bucketing.cluster import ClusterBucketer, ClusterBucketConfig
        from ppm_preprocessing.steps.single_task_strategy_search import (
            SingleTaskStrategySearchStep, SingleTaskStrategySearchConfig,
        )
        from ppm_preprocessing.tasks.specs import default_task_specs

        tasks = default_task_specs()
        if task_name not in tasks:
            return {"status": "error", "error": f"Unknown task: {task_name}"}
        task = tasks[task_name]

        log_path = Path(log_cfg.path)
        file_format = detect_format(str(log_path))
        load_step = LoadCsvStep() if file_format == "csv" else LoadXesStep()

        ctx = PipelineContext(input_path=str(log_path), task=task.name)
        ctx = load_step.run(ctx)

        # Aggregated CSV transform (issues.csv has wf_* columns)
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

        # ── Non-optional preprocessing (Stages 2–10) ──────────────────────────
        ctx = DeduplicateEventsStep(DeduplicateEventsConfig()).run(ctx)
        ctx = CleanAndSortStep().run(ctx)
        ctx = StableSortStep(StableSortConfig(tie_breakers=("_event_index",))).run(ctx)
        ctx = RepairTimestampsStep().run(ctx)
        ctx = FilterShortCasesStep().run(ctx)
        ctx = NormalizeActivitiesStep().run(ctx)
        ctx = FilterInfrequentActivitiesStep().run(ctx)
        ctx = FilterZeroDurationCasesStep().run(ctx)

        n_cases = ctx.log.df["case_id"].nunique() if ctx.log else 0

        # ── Labels + prefix extraction (with time features) ───────────────────
        ctx = CaseLabelsStep(CaseLabelsConfig(
            outcome_col=log_cfg.outcome_col,
        )).run(ctx)
        ctx = PrefixExtractionStep(PrefixExtractionConfig(
            max_prefix_len=max_prefix_len,
            min_prefix_len=1,
            add_time_features=True,
        )).run(ctx)

        # ── Split ─────────────────────────────────────────────────────────────
        ctx = CaseSplitStep(CaseSplitConfig(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            random_state=42,
        )).run(ctx)

        is_clf = "classification" in task.task_type

        ctx = OutlierDetectionStep(OutlierDetectionConfig(
            enabled=True,
            label_col=task.label_col,
        )).run(ctx)
        ctx = FilterRareClassesStep(FilterRareClassesConfig(
            enabled=is_clf,
            min_class_samples=3,
            label_col=task.label_col,
        )).run(ctx)

        # ── Strategy search: all bucketer × encoder combinations ──────────────
        bucketers = {
            "no_bucket":              NoBucketer(),
            "last_activity":          LastActivityBucketer(),
            "prefix_len_bins":        PrefixLenBinsBucketer(bin_size=5, max_len=max_prefix_len),
            "prefix_len_adaptive":    PrefixLenAdaptiveBucketer(max_len=max_prefix_len),
            "cluster":                ClusterBucketer(ClusterBucketConfig(n_clusters=3)),
        }
        encodings = ["last_state", "aggregation", "index_latest_payload", "embedding"]

        strategy_cfg = SingleTaskStrategySearchConfig(
            task=task,
            bucketers=bucketers,
            encodings=encodings,
            min_bucket_samples=min_bucket_samples,
            skip_single_class=True,
            use_probe_model=True,   # always LightGBM (probe model), same as pipeline_runner
            save_report=False,
            save_csv=False,
        )
        ctx = SingleTaskStrategySearchStep(strategy_cfg).run(ctx)

        best = ctx.artifacts.get("best_strategy") or {}
        comparison = ctx.artifacts.get("single_task_comparison", [])

        # Extract baseline score
        baseline_row = next(
            (r for r in comparison
             if r.get("bucketing") == BASELINE_BUCKETER
             and r.get("encoding") == BASELINE_ENCODER),
            {},
        )
        baseline_score = baseline_row.get("primary_score")

        return {
            "status": "success",
            "is_classification": is_clf,
            "primary_metric": best.get("primary_metric", ""),
            "n_cases": n_cases,
            "comparison": comparison,
            "baseline_bucketer": BASELINE_BUCKETER,
            "baseline_encoder": BASELINE_ENCODER,
            "baseline_score": baseline_score,
            "best_bucketer": best.get("bucketing"),
            "best_encoder": best.get("encoding"),
            "best_score": best.get("primary_score"),
        }

    except Exception as e:
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_score(score, is_clf: bool) -> str:
    if score is None:
        return "     N/A"
    if is_clf:
        return f"{score * 100:6.2f}%"
    return f"{score / 86400:8.3f} d"


def _delta_str(score, ref, higher_is_better: bool) -> str:
    if ref is None or ref == 0 or score is None:
        return "       "
    pct = (score - ref) / abs(ref) * 100 if higher_is_better else (ref - score) / abs(ref) * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def _print_comparison(
    result: Dict,
    log_name: str,
    task_name: str,
) -> None:
    comparison = result.get("comparison", [])
    is_clf = result.get("is_classification", True)
    higher = is_clf
    baseline_score = result.get("baseline_score")
    best_score = result.get("best_score")
    best_b = result.get("best_bucketer")
    best_e = result.get("best_encoder")
    n_cases = result.get("n_cases", 0)

    metric_label = "F1_macro (val)" if is_clf else "MAE days (val)"
    W = 90
    print()
    print("=" * W)
    print(f"  STRATEGY SEARCH  |  Log: {log_name}  |  Task: {task_name}")
    print(f"  Cases after preprocessing: {n_cases:,}  |  Metric: {metric_label}")
    print("=" * W)
    print(f"  {'Bucketer':<22} {'Encoder':<26} {'Score':>9}  {'vs baseline':>11}  {'vs best':>8}")
    print("-" * W)

    # Sort: best first
    valid = [r for r in comparison if r.get("primary_score") is not None]
    invalid = [r for r in comparison if r.get("primary_score") is None]
    valid_sorted = sorted(valid, key=lambda r: r["primary_score"], reverse=higher)

    for row in valid_sorted + invalid:
        b = row.get("bucketing", "?")
        e = row.get("encoding", "?")
        score = row.get("primary_score")
        vs_base = _delta_str(score, baseline_score, higher)
        vs_best = _delta_str(score, best_score, higher)
        marker = " << best" if (b == best_b and e == best_e) else (
                 " << baseline" if (b == BASELINE_BUCKETER and e == BASELINE_ENCODER) else "")
        score_s = _fmt_score(score, is_clf)
        status = row.get("status", "ok")
        if score is None:
            print(f"  {b:<22} {e:<26} {'SKIPPED':>9}  {'':>11}  {'':>8}  ({status})")
        else:
            print(f"  {b:<22} {e:<26} {score_s:>9}  {vs_base:>11}  {vs_best:>8}{marker}")

    print("-" * W)
    if best_score is not None and baseline_score is not None:
        gain = _delta_str(best_score, baseline_score, higher)
        print(f"  Baseline ({BASELINE_BUCKETER} + {BASELINE_ENCODER}): {_fmt_score(baseline_score, is_clf)}")
        print(f"  Best     ({best_b} + {best_e}):  {_fmt_score(best_score, is_clf)}   [{gain} vs baseline]")
    print("=" * W)


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def _save_results(
    all_results: Dict[str, Dict],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Full JSON
    out_json = out_dir / "strategy_eval_results.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_json}")

    # Detailed summary text
    out_txt = out_dir / "strategy_eval_summary.txt"
    W = 95
    sections: List[str] = []

    for key, res in all_results.items():
        log_name, task_name = key.split("|")
        is_clf = res.get("is_classification", True)
        higher = is_clf
        best_b = res.get("best_bucketer", "N/A")
        best_e = res.get("best_encoder", "N/A")
        best_s = res.get("best_score")
        base_s = res.get("baseline_score")
        n_cases = res.get("n_cases", 0)
        metric_label = "F1_macro (val)" if is_clf else "MAE days (val)"

        header = [
            "=" * W,
            f"  Log: {log_name}  |  Task: {task_name}",
            f"  Cases after preprocessing: {n_cases:,}  |  Metric: {metric_label}",
            "=" * W,
            f"  {'Bucketer':<22} {'Encoder':<28} {'Score':>9}  {'vs baseline':>11}  {'vs best':>8}",
            "-" * W,
        ]

        comparison = res.get("comparison", [])
        if not comparison:
            header.append("  No results.")
            sections.append("\n".join(header))
            continue

        valid = [r for r in comparison if r.get("primary_score") is not None]
        invalid = [r for r in comparison if r.get("primary_score") is None]
        valid_sorted = sorted(valid, key=lambda r: r["primary_score"], reverse=higher)

        rows: List[str] = []
        for row in valid_sorted + invalid:
            b = row.get("bucketing", "?")
            e = row.get("encoding", "?")
            score = row.get("primary_score")
            vs_base = _delta_str(score, base_s, higher)
            vs_best = _delta_str(score, best_s, higher)
            marker = " << best" if (b == best_b and e == best_e) else (
                     " << baseline" if (b == BASELINE_BUCKETER and e == BASELINE_ENCODER) else "")
            score_s = _fmt_score(score, is_clf)
            if score is None:
                rows.append(f"  {b:<22} {e:<28} {'SKIPPED':>9}  {'':>11}  {'':>8}  ({row.get('status','')})")
            else:
                rows.append(f"  {b:<22} {e:<28} {score_s:>9}  {vs_base:>11}  {vs_best:>8}{marker}")

        gain_str = _delta_str(best_s, base_s, higher) if (best_s and base_s) else "N/A"
        best_b_s = (best_b or "N/A")
        best_e_s = (best_e or "N/A")
        footer = [
            "-" * W,
            f"  Baseline  ({BASELINE_BUCKETER:<22} {BASELINE_ENCODER:<16}): {_fmt_score(base_s, is_clf):>9}",
            f"  Best      ({best_b_s:<22} {best_e_s:<16}): {_fmt_score(best_s, is_clf):>9}   [{gain_str} vs baseline]",
            "=" * W,
        ]

        sections.append("\n".join(header + rows + footer))

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n\n".join(sections) + "\n")
    print(f"Summary saved to {out_txt}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strategy search evaluation across all logs and tasks."
    )
    parser.add_argument(
        "--log", default="",
        help="Path to a single event log (default: run all 5 logs)",
    )
    parser.add_argument(
        "--task", default="",
        choices=["", "next_activity", "remaining_time", "outcome"],
        help="Single task to run (default: all three tasks)",
    )
    parser.add_argument(
        "--output", default="results/strategy_eval",
        help="Output directory (default: results/strategy_eval)",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--max-prefix-len", type=int, default=30)
    parser.add_argument("--min-bucket-samples", type=int, default=100)
    args = parser.parse_args()

    # Determine which logs and tasks to run
    if args.log:
        log_path = Path(args.log)
        logs = [LogConfig(name=log_path.stem, path=str(log_path))]
    else:
        logs = LOG_CONFIGS

    tasks_to_run = [args.task] if args.task else TASKS
    out_dir = Path(args.output)

    # Load existing results so we can resume interrupted runs
    out_json = out_dir / "strategy_eval_results.json"
    all_results: Dict[str, Dict] = {}
    if out_json.exists():
        with open(out_json) as f:
            all_results = json.load(f)
        print(f"Resuming: {len(all_results)} run(s) already completed.")

    total = len(logs) * len(tasks_to_run)
    done = 0

    for log_cfg in logs:
        if not Path(log_cfg.path).exists():
            print(f"WARNING: log not found, skipping: {log_cfg.path}", file=sys.stderr)
            continue

        for task_name in tasks_to_run:
            done += 1
            key = f"{log_cfg.name}|{task_name}"
            if key in all_results:
                print(f"\n[{done}/{total}] {log_cfg.name} / {task_name}  [skipped, already done]")
                continue
            print(f"\n[{done}/{total}] {log_cfg.name} / {task_name}", flush=True)
            t0 = time.time()

            result = _run_strategy_search(
                log_cfg=log_cfg,
                task_name=task_name,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                max_prefix_len=args.max_prefix_len,
                min_bucket_samples=args.min_bucket_samples,
            )
            elapsed = time.time() - t0

            result["elapsed_s"] = round(elapsed, 1)
            all_results[key] = result

            if result["status"] == "error":
                print(f"  ERROR: {result.get('error', 'unknown')}")
                if result.get("traceback"):
                    print(result["traceback"][-800:])
            else:
                _print_comparison(result, log_cfg.name, task_name)
                print(f"  [{elapsed:.0f}s]")

            # Save incrementally so partial results are not lost
            _save_results(all_results, out_dir)

    print(f"\nDone. {done} run(s) completed.")


if __name__ == "__main__":
    main()
