"""
Ablation study: cumulative preprocessing step evaluation.

Starts from a minimal baseline and adds one preprocessing step at a time,
measuring the effect on model performance using a fixed strategy
(no_bucket + last_state encoding) with a fast probe model.

Usage:
    python -m ppm_preprocessing.cli.run_ablation \\
        --log path/to/log.xes \\
        --task remaining_time \\
        --output results/ablation

Optional flags to include optional steps in the sequence:
    --consecutive-duplicates
    --impute-missing
    --drift-window <YYYY-MM-DD>
    --rare-variants
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Stage definition
# ---------------------------------------------------------------------------

@dataclass
class AblationStage:
    name: str
    # ── always-on steps (toggled off only in early baseline stages) ──────────
    deduplicate_events: bool = False
    clean_and_sort: bool = False
    stable_sort: bool = False
    add_time_features: bool = False     # time features in PrefixExtraction
    # ── automatic quality steps ──────────────────────────────────────────────
    repair_timestamps: bool = False
    filter_short_cases: bool = False
    normalize_activities: bool = False
    filter_infrequent_activities: bool = False
    filter_zero_duration_cases: bool = False
    filter_case_length: bool = False
    # ── optional user steps ───────────────────────────────────────────────────
    filter_consecutive_duplicates: bool = False
    impute_missing: bool = False
    concept_drift_window: bool = False
    since_date: str = ""
    rare_variant_filter: bool = False
    min_variant_count: int = 2
    # ── post-split ────────────────────────────────────────────────────────────
    outlier_detection: bool = False
    filter_rare_classes: bool = False


def build_stages(
    include_consecutive_duplicates: bool,
    include_impute: bool,
    include_drift: bool,
    since_date: str,
    include_rare_variants: bool,
) -> List[AblationStage]:
    """Return the cumulative sequence of stages."""
    stages: List[AblationStage] = []

    def _next(name: str, **overrides) -> AblationStage:
        """Copy all flags from previous stage and apply overrides."""
        prev = stages[-1] if stages else AblationStage(name="__init__")
        fields = {k: getattr(prev, k) for k in prev.__dataclass_fields__ if k != "name"}
        fields.update(overrides)
        return AblationStage(name=name, **fields)

    # ── True minimal baseline: just load + NormalizeSchema ───────────────────
    stages.append(AblationStage(name=" 1  Baseline (load + NormalizeSchema only)"))

    # ── Always-on infrastructure steps ───────────────────────────────────────
    stages.append(_next(" 2  + Deduplicate Events",       deduplicate_events=True))
    stages.append(_next(" 3  + Clean & Sort (types, nulls, timestamp order)", clean_and_sort=True))
    stages.append(_next(" 4  + Stable Sort (tie-breaking by record index)",   stable_sort=True))
    stages.append(_next(" 5  + Time Features in Prefix Extraction",           add_time_features=True))

    # ── Automatic quality steps ───────────────────────────────────────────────
    stages.append(_next(" 6  + Repair Timestamps",                            repair_timestamps=True))
    stages.append(_next(" 7  + Filter Short Cases (< 2 events)",              filter_short_cases=True))
    stages.append(_next(" 8  + Normalize Activities",                         normalize_activities=True))
    stages.append(_next(" 9  + Filter Infrequent Activities (< 0.5%)",        filter_infrequent_activities=True))
    stages.append(_next("10  + Filter Zero-Duration Cases",                   filter_zero_duration_cases=True))
    stages.append(_next("11  + Filter Long Cases (p99 threshold)",            filter_case_length=True))

    # ── Optional user steps ───────────────────────────────────────────────────
    n = len(stages)
    if include_consecutive_duplicates:
        stages.append(_next(f"{n+1:2d}  + Filter Consecutive Duplicate Events", filter_consecutive_duplicates=True))
        n += 1
    if include_impute:
        stages.append(_next(f"{n+1:2d}  + Impute Missing Attributes",           impute_missing=True))
        n += 1
    if include_drift:
        stages.append(_next(f"{n+1:2d}  + Concept Drift Window (since {since_date})",
                            concept_drift_window=True, since_date=since_date))
        n += 1
    if include_rare_variants:
        stages.append(_next(f"{n+1:2d}  + Filter Rare Variants",                rare_variant_filter=True))
        n += 1

    # ── Post-split steps ──────────────────────────────────────────────────────
    stages.append(_next(f"{n+1:2d}  + Outlier Removal (IQR on case duration)", outlier_detection=True))
    n += 1
    stages.append(_next(f"{n+1:2d}  + Filter Rare Classes",                    filter_rare_classes=True))

    return stages


# ---------------------------------------------------------------------------
# Single-stage pipeline runner
# ---------------------------------------------------------------------------

def _run_stage(
    log_path: Path,
    task_name: str,
    stage: AblationStage,
    bucketer_name: str,
    encoding_name: str,
    train_ratio: float,
    val_ratio: float,
    max_prefix_len: int,
    min_bucket_samples: int,
    case_col: str = "",
    activity_col: str = "",
    timestamp_col: str = "",
    outcome_col: str = "",
) -> Dict[str, Any]:
    """Run the full pipeline for one stage and return score + row counts."""
    import traceback as _tb

    from ppm_preprocessing.domain.context import PipelineContext
    from ppm_preprocessing.io.format_detection import detect_format
    from ppm_preprocessing.steps.load_xes import LoadXesStep
    from ppm_preprocessing.steps.load_csv import LoadCsvStep
    from ppm_preprocessing.steps.normalize_schema import NormalizeSchemaStep, NormalizeSchemaConfig
    from ppm_preprocessing.steps.transform_aggregated_to_events import (
        TransformAggregatedToEventsStep, TransformAggregatedConfig,
    )
    from ppm_preprocessing.steps.clean_sort import CleanAndSortStep
    from ppm_preprocessing.steps.stable_sort import StableSortStep, StableSortConfig
    from ppm_preprocessing.steps.deduplicate_events import DeduplicateEventsStep, DeduplicateEventsConfig
    from ppm_preprocessing.steps.repair_timestamps import RepairTimestampsStep
    from ppm_preprocessing.steps.filter_short_cases import FilterShortCasesStep
    from ppm_preprocessing.steps.normalize_activities import NormalizeActivitiesStep
    from ppm_preprocessing.steps.filter_infrequent_activities import FilterInfrequentActivitiesStep
    from ppm_preprocessing.steps.filter_zero_duration_cases import FilterZeroDurationCasesStep
    from ppm_preprocessing.steps.filter_case_length import FilterCaseLengthStep
    from ppm_preprocessing.steps.filter_consecutive_duplicates import FilterConsecutiveDuplicatesStep
    from ppm_preprocessing.steps.impute_missing_attributes import ImputeMissingAttributesStep
    from ppm_preprocessing.steps.concept_drift_window import ConceptDriftWindowStep, ConceptDriftWindowConfig
    from ppm_preprocessing.steps.filter_rare_variants import FilterRareVariantsStep, FilterRareVariantsConfig
    from ppm_preprocessing.steps.case_labels import CaseLabelsStep, CaseLabelsConfig
    from ppm_preprocessing.steps.prefix_extraction import PrefixExtractionStep, PrefixExtractionConfig
    from ppm_preprocessing.steps.split_cases import CaseSplitStep, CaseSplitConfig
    from ppm_preprocessing.steps.outlier_detection import OutlierDetectionStep, OutlierDetectionConfig
    from ppm_preprocessing.steps.filter_rare_classes import FilterRareClassesStep, FilterRareClassesConfig
    from ppm_preprocessing.bucketing.no_bucket import NoBucketer
    from ppm_preprocessing.bucketing.last_activity import LastActivityBucketer
    from ppm_preprocessing.steps.single_task_strategy_search import (
        SingleTaskStrategySearchStep, SingleTaskStrategySearchConfig,
    )
    from ppm_preprocessing.tasks.specs import default_task_specs

    try:
        tasks = default_task_specs()
        if task_name not in tasks:
            return {"status": "error", "error": f"Unknown task: {task_name}"}
        task = tasks[task_name]

        file_format = detect_format(str(log_path))
        load_step = LoadCsvStep() if file_format == "csv" else LoadXesStep()

        ctx = PipelineContext(input_path=str(log_path), task=task.name)
        ctx = load_step.run(ctx)

        # ── Aggregated CSV transform (e.g. issues.csv with wf_* columns) ────────
        _used_transform = False
        if file_format == "csv":
            raw_cols = list(ctx.raw_df.columns) if ctx.raw_df is not None else []
            wf_cols = [c for c in raw_cols if c.startswith("wf_")]
            if wf_cols:
                _used_transform = True
                _case_id = case_col or "id"
                _start   = timestamp_col or "started"
                _end_candidates = ["ended", "end_time", "endtime", "completed", "end", "finish"]
                _end = next((c for c in _end_candidates if c in raw_cols), None) or "ended"
                ctx = TransformAggregatedToEventsStep(TransformAggregatedConfig(
                    case_id_col=_case_id,
                    start_time_col=_start,
                    end_time_col=_end,
                    workflow_duration_prefix="wf_",
                    workflow_event_count_prefix="wfe_",
                )).run(ctx)

        # ── Always-on infrastructure (each toggled off only in early baselines) ─
        norm_cfg = NormalizeSchemaConfig(
            case_candidates=[case_col] if (case_col and not _used_transform) else None,
            act_candidates=[activity_col] if (activity_col and not _used_transform) else None,
            ts_candidates=[timestamp_col] if (timestamp_col and not _used_transform) else None,
        )
        ctx = NormalizeSchemaStep(norm_cfg).run(ctx)
        if stage.deduplicate_events:
            ctx = DeduplicateEventsStep(DeduplicateEventsConfig()).run(ctx)
        if stage.clean_and_sort:
            ctx = CleanAndSortStep().run(ctx)
        if stage.stable_sort:
            ctx = StableSortStep(StableSortConfig(tie_breakers=("_event_index",))).run(ctx)

        n_events_after_load = len(ctx.log.df) if ctx.log else 0
        n_cases_after_load = ctx.log.df["case_id"].nunique() if ctx.log else 0

        # ── Phase 2: Automatic quality steps (configurable per stage) ────────
        if stage.repair_timestamps:
            ctx = RepairTimestampsStep().run(ctx)

        if stage.filter_short_cases:
            ctx = FilterShortCasesStep().run(ctx)

        if stage.normalize_activities:
            ctx = NormalizeActivitiesStep().run(ctx)

        if stage.filter_infrequent_activities:
            ctx = FilterInfrequentActivitiesStep().run(ctx)

        if stage.filter_zero_duration_cases:
            ctx = FilterZeroDurationCasesStep().run(ctx)

        if stage.filter_case_length:
            ctx = FilterCaseLengthStep().run(ctx)

        # ── Phase 3: Optional user steps ─────────────────────────────────────
        if stage.filter_consecutive_duplicates:
            ctx = FilterConsecutiveDuplicatesStep().run(ctx)

        if stage.impute_missing:
            ctx = ImputeMissingAttributesStep().run(ctx)

        if stage.concept_drift_window and stage.since_date:
            ctx = ConceptDriftWindowStep(ConceptDriftWindowConfig(
                enabled=True, since_date=stage.since_date,
            )).run(ctx)

        if stage.rare_variant_filter:
            ctx = FilterRareVariantsStep(FilterRareVariantsConfig(
                enabled=True, min_variant_count=stage.min_variant_count,
            )).run(ctx)

        n_events_after_filter = len(ctx.log.df) if ctx.log else 0
        n_cases_after_filter = ctx.log.df["case_id"].nunique() if ctx.log else 0

        # ── Phase 4: Case labels + prefix extraction ─────────────────────────
        ctx = CaseLabelsStep(CaseLabelsConfig(outcome_col=outcome_col)).run(ctx)
        ctx = PrefixExtractionStep(PrefixExtractionConfig(
            max_prefix_len=max_prefix_len,
            min_prefix_len=1,
            add_time_features=stage.add_time_features,
        )).run(ctx)

        ps = ctx.artifacts.get("prefix_samples")
        n_prefix_rows = len(ps) if ps is not None else 0

        # ── Phase 5: Split ────────────────────────────────────────────────────
        test_ratio = round(1.0 - train_ratio - val_ratio, 4)
        ctx = CaseSplitStep(CaseSplitConfig(
            train_ratio=train_ratio, val_ratio=val_ratio, random_state=42,
        )).run(ctx)

        # Post-split: outlier + rare classes
        is_clf = "classification" in task.task_type
        ctx = OutlierDetectionStep(OutlierDetectionConfig(
            enabled=stage.outlier_detection,
            label_col=task.label_col,
        )).run(ctx)

        ctx = FilterRareClassesStep(FilterRareClassesConfig(
            enabled=is_clf and stage.filter_rare_classes,
            min_class_samples=3,
            label_col=task.label_col,
        )).run(ctx)

        # ── Phase 6: Strategy evaluation (fixed bucketer + encoder) ──────────
        bucketer_map = {
            "no_bucket": NoBucketer(),
            "last_activity": LastActivityBucketer(),
        }
        bucketer = bucketer_map.get(bucketer_name, NoBucketer())

        is_classification = "classification" in task.task_type
        strategy_cfg = SingleTaskStrategySearchConfig(
            task=task,
            bucketers={bucketer_name: bucketer},
            encodings=[encoding_name],
            min_bucket_samples=min_bucket_samples,
            skip_single_class=True,
            use_probe_model=is_classification,  # LightGBM for clf tasks, Ridge for regression
            save_report=False,
            save_csv=False,
        )
        ctx = SingleTaskStrategySearchStep(strategy_cfg).run(ctx)

        best = ctx.artifacts.get("best_strategy") or {}
        score = best.get("primary_score")
        comparison = ctx.artifacts.get("single_task_comparison", [])
        # Pull extra metrics from the matched comparison row
        row = next((r for r in comparison
                    if r.get("bucketing") == bucketer_name and r.get("encoding") == encoding_name), {})
        extra_metrics = {k: v for k, v in row.items()
                         if k not in ("bucketing", "encoding", "mode", "primary_score", "primary_metric", "status")}

        return {
            "status": "success",
            "primary_score": score,
            "primary_metric": best.get("primary_metric", ""),
            "is_classification": is_clf,
            "n_events_load": n_events_after_load,
            "n_cases_load": n_cases_after_load,
            "n_events_filtered": n_events_after_filter,
            "n_cases_filtered": n_cases_after_filter,
            "n_prefix_rows": n_prefix_rows,
            "extra_metrics": extra_metrics,
        }

    except Exception as e:
        return {"status": "error", "error": str(e), "traceback": _tb.format_exc()}


# ---------------------------------------------------------------------------
# Pretty print helpers
# ---------------------------------------------------------------------------

def _bar(value: float, best: float, width: int = 30, higher_is_better: bool = True) -> str:
    if best == 0:
        filled = 0
    elif higher_is_better:
        filled = int(round(value / best * width))
    else:
        # lower is better — invert: baseline is the worst (full bar)
        filled = int(round(best / value * width)) if value > 0 else width
    filled = max(0, min(width, filled))
    return "#" * filled + "." * (width - filled)


def _fmt_score(score, metric: str, is_clf: bool) -> str:
    if score is None:
        return "  N/A   "
    if is_clf:
        return f"{score * 100:6.2f}%"
    return f"{score / 86400:>8.3f} d  ({score:,.0f} s)"


def _delta_str(score, ref, higher_is_better: bool, suffix: str = "") -> str:
    if ref is None or ref == 0 or score is None:
        return ""
    if higher_is_better:
        pct = (score - ref) / abs(ref) * 100
    else:
        pct = (ref - score) / abs(ref) * 100   # positive = improvement
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%{suffix}"


def _print_results(results: List[Dict], stages: List[AblationStage], metric: str, is_clf: bool, log_name: str = "") -> None:
    scores = [r["primary_score"] for r in results if r.get("primary_score") is not None]
    if not scores:
        print("No valid scores to display.")
        return

    higher_is_better = is_clf
    best = max(scores) if higher_is_better else min(scores)
    baseline = scores[0] if scores else None

    metric_header = "F1 (val)" if is_clf else "MAE (val, days)"
    log_tag = f"  Log: {log_name}" if log_name else ""
    W = 118
    print()
    print("=" * W)
    print(f"  ABLATION STUDY - {metric_header}{log_tag}")
    print("=" * W)
    print(f"  {'Stage':<44}  {'Score':>14}  {'D vs baseline':>13}  {'D vs prev':>9}  {'Cases':>7}  Progress")
    print("-" * W)

    prev_score = None
    for i, (stage, r) in enumerate(zip(stages, results)):
        score = r.get("primary_score")
        status = r.get("status", "error")
        label = stage.name[:44]
        n_cases = r.get("n_cases_filtered")

        if status != "success" or score is None:
            print(f"  {label:<44}  {'ERROR':>14}  {'':>13}  {'':>9}  {'':>7}")
            prev_score = None
            continue

        vs_base = _delta_str(score, baseline if i > 0 else None, higher_is_better)
        vs_prev = _delta_str(score, prev_score, higher_is_better)

        bar = _bar(score, best, width=18, higher_is_better=higher_is_better)
        score_fmt = _fmt_score(score, metric, is_clf)
        best_marker = " *" if score == best else "  "

        cases_str = ""
        if n_cases is not None:
            cases_str = f"(!){n_cases}" if n_cases < 100 else str(n_cases)
        print(f"  {label:<44}  {score_fmt}  {vs_base:>13}  {vs_prev:>9}  {cases_str:>7}  {bar}{best_marker}")

        prev_score = score

    print("=" * W)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation study: measure contribution of each preprocessing step."
    )
    parser.add_argument("--log", required=True, help="Path to XES or CSV event log")
    parser.add_argument("--task", default="remaining_time",
                        choices=["remaining_time", "next_activity", "outcome"],
                        help="Prediction task (default: remaining_time)")
    parser.add_argument("--bucketer", default="no_bucket",
                        choices=["no_bucket", "last_activity"],
                        help="Fixed bucketing strategy (default: no_bucket)")
    parser.add_argument("--encoding", default="last_state",
                        choices=["last_state", "aggregation", "index_latest_payload"],
                        help="Fixed encoding strategy (default: last_state)")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--max-prefix-len", type=int, default=30)
    parser.add_argument("--min-bucket-samples", type=int, default=30)
    parser.add_argument("--output", default="outputs/ablation",
                        help="Output directory for results JSON and chart")
    # Optional steps
    parser.add_argument("--consecutive-duplicates", action="store_true",
                        help="Include 'filter consecutive duplicates' stage")
    parser.add_argument("--impute-missing", action="store_true",
                        help="Include 'impute missing attributes' stage")
    parser.add_argument("--drift-window", default="",
                        help="Include concept drift window stage with this date (YYYY-MM-DD)")
    parser.add_argument("--rare-variants", action="store_true",
                        help="Include 'filter rare variants' stage")
    # Column mapping for non-standard CSV logs
    parser.add_argument("--case-col", default="", help="Column name for case ID (CSV logs with non-standard names)")
    parser.add_argument("--activity-col", default="", help="Column name for activity (CSV logs with non-standard names)")
    parser.add_argument("--timestamp-col", default="", help="Column name for timestamp (CSV logs with non-standard names)")
    # Outcome column (required when --task outcome)
    parser.add_argument("--outcome-col", default="", help="Column name for outcome label (required for task=outcome)")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"ERROR: file not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    # Sub-folder per log + task so results never overwrite each other
    log_stem = log_path.stem  # e.g. "DomesticDeclarations"
    out_dir = Path(args.output) / log_stem / args.task
    out_dir.mkdir(parents=True, exist_ok=True)

    stages = build_stages(
        include_consecutive_duplicates=args.consecutive_duplicates,
        include_impute=args.impute_missing,
        include_drift=bool(args.drift_window),
        since_date=args.drift_window,
        include_rare_variants=args.rare_variants,
    )

    print(f"\nAblation study: {len(stages)} stages")
    print(f"  Log:      {log_path}  [{log_stem}]")
    print(f"  Task:     {args.task}")
    print(f"  Strategy: {args.bucketer} + {args.encoding}")
    print(f"  Output:   {out_dir}\n")

    results: List[Dict] = []
    metric_name = ""
    is_clf = False

    for i, stage in enumerate(stages):
        t0 = time.time()
        print(f"[{i+1}/{len(stages)}] {stage.name} ...", end="", flush=True)
        r = _run_stage(
            log_path=log_path,
            task_name=args.task,
            stage=stage,
            bucketer_name=args.bucketer,
            encoding_name=args.encoding,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            max_prefix_len=args.max_prefix_len,
            min_bucket_samples=args.min_bucket_samples,
            case_col=args.case_col,
            activity_col=args.activity_col,
            timestamp_col=args.timestamp_col,
            outcome_col=args.outcome_col,
        )
        elapsed = time.time() - t0

        if r.get("status") == "success":
            score = r.get("primary_score")
            metric_name = r.get("primary_metric", "")
            is_clf = r.get("is_classification", False)
            score_str = _fmt_score(score, metric_name, is_clf)
            print(f" {score_str}  [{elapsed:.1f}s]")
        else:
            print(f" ERROR: {r.get('error', '?')}  [{elapsed:.1f}s]")
            if r.get("traceback"):
                print(r["traceback"][:500])

        results.append({**r, "stage_name": stage.name})

    # ── Print summary table ──────────────────────────────────────────────────
    _print_results(results, stages, metric_name, is_clf, log_name=log_stem)

    # ── Save JSON ────────────────────────────────────────────────────────────
    out_json = out_dir / "ablation_results.json"
    payload = {
        "log": log_stem,
        "task": args.task,
        "bucketer": args.bucketer,
        "encoding": args.encoding,
        "stages": [
            {
                "stage": s.name,
                "status": r.get("status"),
                "primary_score": r.get("primary_score"),
                "primary_metric": r.get("primary_metric"),
                "n_events_load": r.get("n_events_load"),
                "n_cases_load": r.get("n_cases_load"),
                "n_events_filtered": r.get("n_events_filtered"),
                "n_cases_filtered": r.get("n_cases_filtered"),
                "n_prefix_rows": r.get("n_prefix_rows"),
                "extra_metrics": r.get("extra_metrics", {}),
                "error": r.get("error"),
            }
            for s, r in zip(stages, results)
        ],
    }
    out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Results saved to {out_json}")

    # ── ASCII chart to file ───────────────────────────────────────────────────
    out_txt = out_dir / "ablation_summary.txt"
    import io
    buf = io.StringIO()
    _orig_stdout = sys.stdout
    sys.stdout = buf
    _print_results(results, stages, metric_name, is_clf, log_name=log_stem)
    sys.stdout = _orig_stdout
    out_txt.write_text(buf.getvalue(), encoding="utf-8")
    print(f"Summary saved to {out_txt}")


if __name__ == "__main__":
    main()
