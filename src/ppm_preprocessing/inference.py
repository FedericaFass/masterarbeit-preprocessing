"""
Inference module for running case prediction.

Usage:
    import joblib
    from ppm_preprocessing.inference import load_bundle, predict_running_case

    bundle = load_bundle("outputs/single_task/model_bundle.joblib")

    # A running case: list of events (dicts with at least 'activity' and 'timestamp')
    running_case = [
        {"activity": "A_Create Application", "timestamp": "2017-01-01 10:00:00", ...},
        {"activity": "A_Submitted",          "timestamp": "2017-01-01 10:05:00", ...},
        {"activity": "W_Handle leads",       "timestamp": "2017-01-02 14:30:00", ...},
    ]

    result = predict_running_case(bundle, running_case, case_id="C42")
    print(result)
    # {
    #     "case_id": "C42",
    #     "predicted_remaining_time_sec": 345600.0,
    #     "predicted_remaining_time_days": 4.0,
    #     "bucket_id": 1,
    #     "prefix_len": 3,
    # }
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import warnings

import joblib
import numpy as np
import pandas as pd


def load_bundle(path: str) -> Dict[str, Any]:
    """Load a persisted model bundle from disk."""
    bundle = joblib.load(path)

    required = ["models", "encoder", "bucketer", "strategy"]
    for key in required:
        if bundle.get(key) is None:
            raise ValueError(
                f"Bundle is missing '{key}'. "
                f"Re-run the training pipeline to generate a complete bundle."
            )

    return bundle


def _build_prefix_row(
    events: List[Dict[str, Any]],
    case_id: str = "running",
    case_start_time: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Convert a list of events into a single prefix row (DataFrame),
    matching the format produced by PrefixExtractionStep.

    Each event dict must have at least:
      - "activity": str
      - "timestamp": str or datetime
    Optional event-level attributes are carried through as event_last__ columns.
    """
    if not events:
        raise ValueError("Running case must have at least one event.")

    # Parse timestamps
    for ev in events:
        if "timestamp" not in ev or "activity" not in ev:
            raise ValueError("Each event must have 'activity' and 'timestamp' keys.")

    activities = [str(ev["activity"]) for ev in events]
    timestamps = [pd.Timestamp(ev["timestamp"], tz="UTC") for ev in events]

    prefix_len = len(events)
    prefix_end_time = timestamps[-1]

    if case_start_time is None:
        case_start_time = timestamps[0]

    elapsed_sec = max(0.0, (prefix_end_time - case_start_time).total_seconds())
    time_since_last_sec = 0.0
    if prefix_len > 1:
        time_since_last_sec = max(0.0, (timestamps[-1] - timestamps[-2]).total_seconds())

    row: Dict[str, Any] = {
        "case_id": case_id,
        "prefix_row_id": 0,
        "prefix_len": prefix_len,
        "prefix_activities": activities,
        "prefix_end_time": prefix_end_time,
        # Time features
        "feat_elapsed_time_sec": elapsed_sec,
        "feat_time_since_last_event_sec": time_since_last_sec,
        "feat_elapsed_time_log1p": float(np.log1p(elapsed_sec)),
        "feat_time_since_last_log1p": float(np.log1p(time_since_last_sec)),
        # Calendar features
        "feat_prefix_end_hour": int(prefix_end_time.hour),
        "feat_prefix_end_weekday": int(prefix_end_time.weekday()),
        "feat_prefix_end_is_weekend": int(prefix_end_time.weekday() >= 5),
        "feat_prefix_end_month": int(prefix_end_time.month),
        # Placeholder labels (not used for inference, but encoder expects them)
        "label_remaining_time_sec": 0.0,
        "label_remaining_time_log1p": 0.0,
        "label_next_activity": "",
        "label_outcome": "unknown",
    }

    # Carry through case-level attributes (case:* keys)
    last_event = events[-1]
    for key, val in last_event.items():
        if key in ("activity", "timestamp"):
            continue
        if str(key).startswith("case:") or str(key).startswith("case__"):
            safe_name = str(key).replace("case:", "").replace("case__", "")
            row[f"case__{safe_name}"] = val
        else:
            row[f"event_last__{key}"] = val

    # Also carry case-level attributes from the first event (if different keys)
    first_event = events[0]
    for key, val in first_event.items():
        if key in ("activity", "timestamp"):
            continue
        if str(key).startswith("case:") or str(key).startswith("case__"):
            safe_name = str(key).replace("case:", "").replace("case__", "")
            col = f"case__{safe_name}"
            if col not in row:
                row[col] = val

    return pd.DataFrame([row])


def predict_running_case(
    bundle: Dict[str, Any],
    events: List[Dict[str, Any]],
    case_id: str = "running",
    case_start_time: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Predict for a running case. Supports both regression (remaining time)
    and classification (next activity) tasks.

    Args:
        bundle: Loaded model bundle (from load_bundle())
        events: List of event dicts, each with at least 'activity' and 'timestamp'.
                Additional keys are treated as case/event attributes.
        case_id: Identifier for this case
        case_start_time: Optional override for case start time (ISO format string).
                         Defaults to the first event's timestamp.

    Returns:
        Dict with prediction results. For regression:
        - predicted_remaining_time_sec, predicted_remaining_time_days
        For classification:
        - predicted_next_activity
    """
    encoder = bundle["encoder"]
    bucketer = bundle["bucketer"]
    models = bundle["models"]
    strategy = bundle["strategy"]
    task_type = bundle.get("task_type", "")
    is_clf = "classification" in str(task_type)
    target_log1p = bool(bundle.get("target_log1p", False))
    clamp_nonnegative = bool(bundle.get("clamp_nonnegative", True))
    mode = strategy.get("mode", "global_model")

    # 1. Build prefix row from running case events
    start_ts = pd.Timestamp(case_start_time, tz="UTC") if case_start_time else None
    prefix_df = _build_prefix_row(events, case_id=case_id, case_start_time=start_ts)

    # 2. Apply bucketer (no refitting — uses learned boundaries)
    bucket_ids = bucketer.assign(prefix_df)
    bucket_id = int(bucket_ids.iloc[0])
    prefix_df["bucket_id"] = bucket_id

    # 3. Apply encoder (no refitting — uses learned vocab/target encodings)
    encoded = encoder.transform(prefix_df)
    X = encoded.X

    # 4. Select model and predict
    if mode == "per_bucket_models":
        model = models.get(str(bucket_id))
        if model is None:
            # Fallback: find the closest bucket that has a model
            available = sorted(models.keys(), key=lambda k: abs(int(k) - bucket_id))
            if available:
                model = models[available[0]]
                bucket_id = int(available[0])
            else:
                raise RuntimeError("No trained models found in bundle.")
    else:
        model = models.get("global")
        if model is None:
            raise RuntimeError("No global model found in bundle.")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        raw_pred = model.predict(X)

    base_result = {
        "case_id": case_id,
        "bucket_id": bucket_id,
        "prefix_len": len(events),
        "task_type": task_type,
    }

    if is_clf:
        # Classification: return predicted class label as string
        task_name = bundle.get("task_name", "")
        pred_label = str(raw_pred[0])
        if task_name == "outcome":
            base_result["predicted_outcome"] = pred_label
        else:
            base_result["predicted_next_activity"] = pred_label
    else:
        # Regression: post-process (inverse log1p + clamp)
        y_pred = float(raw_pred[0])
        if target_log1p:
            y_pred = float(np.expm1(y_pred))
        if clamp_nonnegative:
            y_pred = max(0.0, y_pred)
        base_result["predicted_remaining_time_sec"] = y_pred
        base_result["predicted_remaining_time_days"] = y_pred / 86400.0

    return base_result


def predict_batch(
    bundle: Dict[str, Any],
    cases: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """
    Predict remaining time for multiple running cases.

    Args:
        bundle: Loaded model bundle
        cases: Dict mapping case_id -> list of event dicts

    Returns:
        List of prediction result dicts
    """
    results = []
    for case_id, events in cases.items():
        result = predict_running_case(bundle, events, case_id=case_id)
        results.append(result)
    return results
