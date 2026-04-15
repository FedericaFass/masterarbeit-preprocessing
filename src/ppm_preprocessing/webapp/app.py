"""
Flask web app for PPM remaining time prediction.

Usage:
    python -m ppm_preprocessing.webapp.app
    -> opens http://localhost:5000

Session isolation: each browser session gets a unique ID stored in a signed
cookie. All state (training progress, results, uploaded files, run history,
model bundles) is scoped to that session ID. Users never see each other's data.
"""
from __future__ import annotations

from pathlib import Path
import os
import secrets
import tempfile
import threading
import json
import time

from flask import Flask, render_template, request, jsonify, send_file, session

from ppm_preprocessing.inference import load_bundle, predict_running_case
from ppm_preprocessing.webapp.pipeline_runner import run_pipeline, run_strategy_search_only

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]  # project root

app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB upload limit
# Use a stable secret key from env so sessions survive restarts.
# Falls back to a random key (sessions lost on restart — fine for dev).
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32))

# ---------------------------------------------------------------------------
# Per-session state
# ---------------------------------------------------------------------------
_sessions: dict[str, dict] = {}
_sessions_lock = threading.Lock()


def _sid() -> str:
    """Return this request's session ID, creating one if needed."""
    if "sid" not in session:
        session["sid"] = secrets.token_urlsafe(16)
    return session["sid"]


def _ss(sid: str) -> dict:
    """Return (or lazily create) the in-memory state dict for a session."""
    with _sessions_lock:
        if sid not in _sessions:
            _sessions[sid] = {
                # Training
                "train_status": "idle",
                "train_progress": "",
                "train_steps": [],
                "train_result": None,
                "train_lock": threading.Lock(),
                "scanned_file": None,
                "bundle_path": None,
                "bundle_cache": {"path": None, "bundle": None},
                # Quick compare
                "compare_status": "idle",
                "compare_progress": "",
                "compare_results": None,
                "compare_steps": [],
                "compare_lock": threading.Lock(),
                # Bookkeeping
                "created_at": time.time(),
                "last_active": time.time(),
            }
        _sessions[sid]["last_active"] = time.time()
        return _sessions[sid]


def _out_dir(sid: str) -> Path:
    """Per-session output directory."""
    return ROOT / "outputs" / "sessions" / sid


def _bundle_path(sid: str) -> Path:
    return _out_dir(sid) / "model_bundle.joblib"


def _run_history_path(sid: str) -> Path:
    return _out_dir(sid) / "run_history.json"


# ---------------------------------------------------------------------------
# Session cleanup — remove sessions idle for > 24 h
# ---------------------------------------------------------------------------
def _cleanup_sessions():
    """Background thread: purge old session dirs and in-memory state."""
    while True:
        time.sleep(3600)  # check every hour
        cutoff = time.time() - 86400  # 24 h
        with _sessions_lock:
            stale = [sid for sid, s in _sessions.items() if s.get("last_active", 0) < cutoff]
        for sid in stale:
            try:
                import shutil
                shutil.rmtree(_out_dir(sid), ignore_errors=True)
            except Exception:
                pass
            with _sessions_lock:
                _sessions.pop(sid, None)


threading.Thread(target=_cleanup_sessions, daemon=True).start()


# ---------------------------------------------------------------------------
# Helper: run history
# ---------------------------------------------------------------------------
def _load_run_history(sid: str):
    p = _run_history_path(sid)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _save_run_entry(sid: str, config: dict, result: dict):
    from datetime import datetime
    history = _load_run_history(sid)
    metrics = result.get("metrics", {})
    best = result.get("best_strategy", {})
    entry = {
        "run_id": len(history) + 1,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "file": config.get("file", ""),
        "task": config.get("task_name", ""),
        "outlier_detection": config.get("outlier_enabled", True),
        "columns_dropped": config.get("columns_to_drop", []),
        "min_class_samples": config.get("min_class_samples", 0),
        "bucketing": best.get("bucketing", ""),
        "encoding": best.get("encoding", ""),
        "mae_days": metrics.get("mae_days"),
        "median_ae_days": metrics.get("median_ae_days"),
        "f1_macro": metrics.get("f1_macro"),
        "f1_micro": metrics.get("f1_micro"),
        "accuracy": metrics.get("accuracy"),
        "test_samples": result.get("test_samples", 0),
    }
    history.append(entry)
    _out_dir(sid).mkdir(parents=True, exist_ok=True)
    _run_history_path(sid).write_text(json.dumps(history, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# Helper: model bundle cache
# ---------------------------------------------------------------------------
def _get_bundle(ss: dict, bundle_path_str: str | None = None):
    """Load or return cached model bundle for this session."""
    path = bundle_path_str or ss.get("bundle_path") or ""
    cache = ss["bundle_cache"]
    if cache["path"] == path and cache["bundle"] is not None:
        return cache["bundle"]
    bundle = load_bundle(path)
    cache["path"] = path
    cache["bundle"] = bundle
    return bundle


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    sid = _sid()
    has_model = _bundle_path(sid).exists()
    return render_template("index.html", has_model=has_model)


def _auto_detect_cols(df, fmt):
    """Return (case_col, ts_col, act_col) best guesses. Any may be None."""
    import pandas as pd
    cols_lower = {c.lower(): c for c in df.columns}

    if fmt == "csv":
        _case_cands = ["case_id", "caseid", "case:concept:name", "case id", "case",
                       "trace_id", "traceid", "instance_id", "process_instance_id",
                       "issue_num", "id"]
        case_col = next((cols_lower[k] for k in _case_cands if k in cols_lower), None)
        if case_col is None:
            case_col = next((c for c in df.columns
                             if any(k in c.lower() for k in ("case", "trace", "instance", "issue"))), None)

        _ts_cands = ["timestamp", "time:timestamp", "time", "datetime",
                     "start_time", "starttime", "started", "start", "begin_time",
                     "event_time", "eventtime", "created", "creation_time", "date", "event_date"]
        ts_col = next((cols_lower[k] for k in _ts_cands if k in cols_lower), None)
        if ts_col is None:
            ts_col = next((c for c in df.columns
                           if any(k in c.lower() for k in ("time", "date", "start", "created"))), None)
    else:
        case_col = next((c for c in ["case:concept:name", "case_id"] if c in df.columns), None)
        ts_col = next((c for c in ["time:timestamp", "timestamp"] if c in df.columns), None)

    _act_exact = {"concept:name", "activity", "Activity", "ACTIVITY", "act", "event",
                  "task", "action", "event_type", "eventtype", "ActivityName", "event_name"}
    act_col = next((c for c in df.columns if c in _act_exact), None)
    if act_col is None:
        act_col = next((c for c in df.columns if c.lower() in {a.lower() for a in _act_exact}), None)
    if act_col is None:
        act_col = next((c for c in df.columns
                        if any(k in c.lower() for k in ("activity", "event_type", "task", "action"))), None)

    return case_col, ts_col, act_col


def _run_scan(df, case_col, ts_col, act_col_raw, ss_state, file_path):
    """Run the full column analysis and return a JSON-ready dict. Saves file_path to session."""
    import pandas as pd
    import numpy as np

    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df_sorted = df.sort_values([case_col, ts_col])
    last_events = df_sorted.groupby(case_col, sort=False).last()

    skip_cols = {case_col, ts_col, "time:timestamp", "case:concept:name", "@@index",
                 "concept:name", "activity", "case:id", "id"}
    if act_col_raw:
        skip_cols.add(act_col_raw)
    n_total = len(df)

    columns_info = []
    empty_columns_info = []
    for col in df.columns:
        if col in skip_cols or col.startswith("@@"):
            continue
        null_count = int(df[col].isna().sum())
        empty_str_count = int((df[col].astype(str).str.strip() == "").sum()) if df[col].dtype == object else 0
        total_empty = null_count + empty_str_count
        empty_pct = round(100.0 * total_empty / n_total, 1) if n_total > 0 else 0.0
        empty_columns_info.append({"column": col, "null_count": total_empty, "null_pct": empty_pct})
        vals = last_events[col].dropna().astype(str)
        unique_vals = vals.unique().tolist()
        n_unique = len(unique_vals)
        if n_unique == 0 or n_unique > 500:
            continue
        columns_info.append({"column": col, "n_unique": n_unique,
                              "values": sorted(unique_vals)[:30], "sample": sorted(unique_vals)[:5]})

    columns_info.sort(key=lambda x: x["n_unique"])
    empty_columns_info.sort(key=lambda x: x["null_pct"], reverse=True)

    outcome_skip = {case_col, ts_col, "time:timestamp", "timestamp", "@@index"}
    outcome_columns = []
    for col in df.columns:
        if col in outcome_skip or col.startswith("@@"):
            continue
        vals_oc = last_events[col].dropna().astype(str)
        n_uniq_oc = int(vals_oc.nunique())
        if n_uniq_oc < 2 or n_uniq_oc > 100:
            continue
        outcome_columns.append({"column": col, "n_unique": n_uniq_oc})
    outcome_columns.sort(key=lambda x: x["n_unique"])

    activity_counts = []
    if act_col_raw:
        vc = df[act_col_raw].dropna().astype(str).value_counts()
        total_events = int(vc.sum())
        for name, cnt in vc.head(30).items():
            activity_counts.append({"name": str(name), "count": int(cnt),
                                    "pct": round(100.0 * cnt / total_events, 1) if total_events else 0})

    case_duration_stats = None
    case_duration_hist = []
    try:
        ts_series = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        df_tmp = df[[case_col]].copy()
        df_tmp["ts"] = ts_series
        grp = df_tmp.groupby(case_col)["ts"]
        durations_days = (grp.max() - grp.min()).dt.total_seconds() / 86400
        durations_days = durations_days.dropna()
        if len(durations_days) > 0:
            case_duration_stats = {
                "min": round(float(durations_days.min()), 2),
                "max": round(float(durations_days.max()), 2),
                "mean": round(float(durations_days.mean()), 2),
                "median": round(float(durations_days.median()), 2),
                "p25": round(float(durations_days.quantile(0.25)), 2),
                "p75": round(float(durations_days.quantile(0.75)), 2),
                "p90": round(float(durations_days.quantile(0.90)), 2),
            }
            p95 = float(durations_days.quantile(0.95))
            clipped = durations_days[durations_days <= p95]
            if len(clipped) > 0 and p95 > 0:
                counts, edges = np.histogram(clipped, bins=8)
                for i in range(len(counts)):
                    lo, hi = edges[i], edges[i + 1]
                    if hi < 1:
                        label = f"{lo*24:.0f}h\u2013{hi*24:.0f}h"
                    elif hi < 30:
                        label = f"{lo:.1f}\u2013{hi:.1f}d"
                    else:
                        label = f"{lo/30:.1f}\u2013{hi/30:.1f}mo"
                    case_duration_hist.append({"label": label, "count": int(counts[i])})
    except Exception:
        pass

    event_attr_columns = []
    if act_col_raw:
        n_uniq_act = int(df[act_col_raw].dropna().astype(str).nunique())
        sample_acts = sorted(df[act_col_raw].dropna().astype(str).unique().tolist()[:5])
        event_attr_columns.append({"column": "activity", "n_unique": n_uniq_act, "sample": sample_acts})
    for col in df.columns:
        if col in skip_cols or col.startswith("@@") or col.startswith("case:") or col == act_col_raw:
            continue
        vals_all = df[col].dropna().astype(str)
        n_uniq = int(vals_all.nunique())
        if n_uniq == 0 or n_uniq > 50:
            continue
        event_attr_columns.append({"column": col, "n_unique": n_uniq,
                                   "sample": sorted(vals_all.unique().tolist()[:5])})

    lc_col = next((c for c in df.columns if c in ("lifecycle:transition", "lifecycle")), None)
    lifecycle_info = None
    if lc_col:
        lifecycle_info = {"column": lc_col,
                          "values": sorted(df[lc_col].dropna().astype(str).unique().tolist())}

    auto_max_prefix_len = 30
    case_length_stats = None
    case_length_hist = []
    try:
        case_lengths = df.groupby(case_col).size()
        p95_len = int(case_lengths.quantile(0.95))
        auto_max_prefix_len = max(5, min(100, p95_len))
        case_length_stats = {
            "min": int(case_lengths.min()),
            "max": int(case_lengths.max()),
            "mean": round(float(case_lengths.mean()), 1),
            "median": round(float(case_lengths.median()), 1),
        }
        clipped_len = case_lengths[case_lengths <= p95_len]
        if len(clipped_len) > 0:
            counts_l, edges_l = np.histogram(clipped_len, bins=min(8, p95_len))
            for i in range(len(counts_l)):
                case_length_hist.append({
                    "label": f"{int(edges_l[i])}\u2013{int(edges_l[i+1])}",
                    "count": int(counts_l[i])
                })
    except Exception:
        pass

    # Unique trace variants + top variants
    unique_variants = 0
    top_variants = []
    try:
        if act_col_raw:
            variant_series = df.sort_values([case_col, ts_col]).groupby(case_col)[act_col_raw].apply(
                lambda x: " → ".join(x.dropna().astype(str))
            )
            vc_var = variant_series.value_counts()
            unique_variants = int(len(vc_var))
            n_cases_var = int(len(variant_series))
            for var_name, var_cnt in vc_var.head(10).items():
                full = str(var_name)
                acts = full.split(" → ")
                if len(acts) <= 4:
                    short = full
                else:
                    short = " → ".join(acts[:2]) + f" → … ({len(acts)} steps) → " + acts[-1]
                top_variants.append({
                    "name": short,
                    "full": full,
                    "count": int(var_cnt),
                    "pct": round(100.0 * var_cnt / n_cases_var, 1) if n_cases_var else 0,
                })
    except Exception:
        pass

    # Overall missing value rate
    missing_rate = 0.0
    try:
        feature_cols = [c for c in df.columns if c not in {case_col, ts_col} and not c.startswith("@@")]
        if feature_cols:
            total_cells = len(df) * len(feature_cols)
            missing_cells = int(df[feature_cols].isna().sum().sum())
            missing_rate = round(100.0 * missing_cells / total_cells, 1) if total_cells else 0.0
    except Exception:
        pass

    # Events over time (monthly buckets)
    events_over_time = []
    try:
        ts_series2 = pd.to_datetime(df[ts_col], utc=True, errors="coerce").dropna()
        if len(ts_series2) > 0:
            ts_month = ts_series2.dt.to_period("M").astype(str)
            monthly = ts_month.value_counts().sort_index()
            for period, cnt in monthly.items():
                events_over_time.append({"label": str(period), "count": int(cnt)})
    except Exception:
        pass

    ss_state["scanned_file"] = file_path
    ss_state["col_mapping"] = {"case_col": case_col, "ts_col": ts_col, "act_col": act_col_raw}

    return {
        "columns": columns_info,
        "empty_columns": empty_columns_info,
        "outcome_columns": outcome_columns,
        "event_attr_columns": event_attr_columns,
        "lifecycle_info": lifecycle_info,
        "num_cases": int(last_events.shape[0]),
        "num_events": n_total,
        "file_path": file_path,
        "activity_counts": activity_counts,
        "case_duration_stats": case_duration_stats,
        "case_duration_hist": case_duration_hist,
        "case_length_stats": case_length_stats,
        "case_length_hist": case_length_hist,
        "unique_variants": unique_variants,
        "top_variants": top_variants,
        "missing_rate": missing_rate,
        "events_over_time": events_over_time,
        "auto_max_prefix_len": auto_max_prefix_len,
        "detected_case_col": case_col,
        "detected_ts_col": ts_col,
        "detected_act_col": act_col_raw,
    }


@app.route("/api/scan-columns", methods=["POST"])
def api_scan_columns():
    sid = _sid()
    ss_state = _ss(sid)

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    upload_dir = ROOT / "data" / "raw"
    upload_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename).suffix or ".xes"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=str(upload_dir))
    file.save(tmp.name)
    tmp.close()

    # Save immediately so mapping endpoint can load it without re-upload
    ss_state["pending_file"] = tmp.name

    try:
        import pandas as pd
        from ppm_preprocessing.io.format_detection import detect_format

        fmt = detect_format(tmp.name)

        if fmt == "csv":
            df = pd.read_csv(tmp.name, low_memory=False)
        else:
            from pm4py.objects.log.importer.xes import importer as xes_importer
            from pm4py.objects.conversion.log import converter as log_converter
            log = xes_importer.apply(tmp.name)
            df = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)

        case_col, ts_col, act_col = _auto_detect_cols(df, fmt)

        # For CSV: always show mapping UI (non-standard column names need user confirmation)
        # For XES: only show if required columns are missing
        if fmt == "csv" or not case_col or not ts_col:
            return jsonify({
                "mapping_needed": True,
                "all_columns": list(df.columns),
                "guessed_case_col": case_col,
                "guessed_ts_col": ts_col,
                "guessed_act_col": act_col,
            })

        return jsonify(_run_scan(df, case_col, ts_col, act_col, ss_state, tmp.name))

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/apply-column-mapping", methods=["POST"])
def api_apply_column_mapping():
    """Re-run scan with user-selected column mapping (no file re-upload needed)."""
    sid = _sid()
    ss_state = _ss(sid)

    file_path = ss_state.get("pending_file")
    if not file_path or not Path(file_path).exists():
        return jsonify({"error": "No pending file found. Please upload the file again."}), 400

    case_col = request.form.get("case_col", "").strip()
    ts_col = request.form.get("ts_col", "").strip()
    act_col = request.form.get("act_col", "").strip() or None

    if not case_col or not ts_col:
        return jsonify({"error": "case_col and ts_col are required."}), 400

    try:
        import pandas as pd
        from ppm_preprocessing.io.format_detection import detect_format

        fmt = detect_format(file_path)
        if fmt == "csv":
            df = pd.read_csv(file_path, low_memory=False)
        else:
            from pm4py.objects.log.importer.xes import importer as xes_importer
            from pm4py.objects.conversion.log import converter as log_converter
            log = xes_importer.apply(file_path)
            df = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)

        if case_col not in df.columns:
            return jsonify({"error": f"Column '{case_col}' not found in file."}), 400
        if ts_col not in df.columns:
            return jsonify({"error": f"Column '{ts_col}' not found in file."}), 400
        if act_col and act_col not in df.columns:
            act_col = None

        return jsonify(_run_scan(df, case_col, ts_col, act_col, ss_state, file_path))

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/train", methods=["POST"])
def api_train():
    sid = _sid()
    ss_state = _ss(sid)

    with ss_state["train_lock"]:
        if ss_state["train_status"] == "running":
            return jsonify({"error": "Training already in progress"}), 409

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    upload_dir = ROOT / "data" / "raw"
    upload_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename).suffix or ".xes"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=str(upload_dir))
    file.save(tmp.name)
    tmp.close()

    log_path = Path(tmp.name)
    out_dir = _out_dir(sid)

    with ss_state["train_lock"]:
        ss_state["train_status"] = "running"
        ss_state["train_progress"] = "Starting pipeline..."
        ss_state["train_result"] = None
        ss_state["bundle_path"] = None
        ss_state["train_steps"] = []

    def _on_progress(msg: str):
        import json as _json
        import logging as _logging
        with ss_state["train_lock"]:
            if msg.startswith("__STEP__:"):
                try:
                    ss_state["train_steps"].append(_json.loads(msg[9:]))
                except Exception as _e:
                    _logging.warning(f"[train_steps] JSON parse failed: {_e!r} | msg[:200]={msg[:200]!r}")
            else:
                ss_state["train_progress"] = msg

    task_name = request.form.get("task_name", "remaining_time")
    outlier_enabled = request.form.get("outlier_enabled", "true").lower() == "true"
    min_class_samples = int(request.form.get("min_class_samples", "0"))
    outcome_col = request.form.get("outcome_col", "")
    outcome_values = request.form.get("outcome_values", "")
    next_event_attr_col = request.form.get("next_event_attr_col", "")
    import json as _json
    columns_to_drop_raw = request.form.get("columns_to_drop", "[]")
    try:
        columns_to_drop = _json.loads(columns_to_drop_raw)
    except Exception:
        columns_to_drop = []
    lifecycle_drop_raw = request.form.get("lifecycle_drop_values", "[]")
    try:
        lifecycle_drop_values = _json.loads(lifecycle_drop_raw)
    except Exception:
        lifecycle_drop_values = []
    try:
        train_ratio = max(0.1, min(0.9, float(request.form.get("train_ratio", "0.7"))))
        val_ratio = max(0.05, min(0.5, float(request.form.get("val_ratio", "0.15"))))
        if train_ratio + val_ratio >= 1.0:
            val_ratio = round(1.0 - train_ratio - 0.05, 2)
    except Exception:
        train_ratio, val_ratio = 0.7, 0.15

    max_prefix_len = max(5, min(200, int(request.form.get("max_prefix_len", "30"))))
    time_budget_s = max(30, min(3600, int(request.form.get("time_budget_s", "300"))))
    min_bucket_samples = max(10, min(1000, int(request.form.get("min_bucket_samples", "100"))))
    bin_size = max(1, min(50, int(request.form.get("bin_size", "5"))))
    temporal_split = request.form.get("temporal_split", "false").lower() == "true"
    rare_variant_filter = request.form.get("rare_variant_filter", "false").lower() == "true"
    min_variant_count = max(2, min(100, int(request.form.get("min_variant_count", "5"))))
    concept_drift_window = request.form.get("concept_drift_window", "false").lower() == "true"
    since_date = request.form.get("since_date", "").strip()
    filter_consecutive_duplicates = request.form.get("filter_consecutive_duplicates", "false").lower() == "true"
    impute_missing = request.form.get("impute_missing", "false").lower() == "true"
    col_mapping = ss_state.get("col_mapping")  # set by scan or apply-column-mapping

    run_config = {
        "file": file.filename,
        "task_name": task_name,
        "outcome_col": outcome_col,
        "next_event_attr_col": next_event_attr_col,
        "time_budget_s": time_budget_s,
        "max_prefix_len": max_prefix_len,
        "train_val_test": f"{int(train_ratio*100)}/{int(val_ratio*100)}/{int(round(1-train_ratio-val_ratio,2)*100)}",
        "temporal_ordering": temporal_split,
        "outlier_detection": outlier_enabled,
        "min_class_samples": min_class_samples,
        "columns_to_drop": columns_to_drop,
        "lifecycle_drop_values": lifecycle_drop_values,
        "time_window_filter": since_date if concept_drift_window else None,
        "rare_variant_filter": rare_variant_filter,
        "min_variant_count": min_variant_count if rare_variant_filter else None,
        "filter_consecutive_duplicates": filter_consecutive_duplicates,
        "impute_missing": impute_missing,
        "min_bucket_samples": min_bucket_samples,
        "bin_size": bin_size,
    }

    def _run():
        try:
            result = run_pipeline(
                log_path=log_path,
                out_dir=out_dir,
                task_name=task_name,
                time_budget_s=time_budget_s,
                outlier_enabled=outlier_enabled,
                min_class_samples=min_class_samples,
                outcome_col=outcome_col,
                outcome_values=outcome_values,
                next_event_attr_col=next_event_attr_col,
                columns_to_drop=columns_to_drop,
                lifecycle_drop_values=lifecycle_drop_values,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                max_prefix_len=max_prefix_len,
                min_bucket_samples=min_bucket_samples,
                bin_size=bin_size,
                temporal_split=temporal_split,
                rare_variant_filter=rare_variant_filter,
                min_variant_count=min_variant_count,
                concept_drift_window=concept_drift_window,
                since_date=since_date,
                filter_consecutive_duplicates=filter_consecutive_duplicates,
                impute_missing=impute_missing,
                col_mapping=col_mapping,
                on_progress=_on_progress,
            )
            with ss_state["train_lock"]:
                ss_state["train_result"] = result
                if result.get("status") == "success":
                    ss_state["train_status"] = "done"
                    ss_state["train_progress"] = "Training complete!"
                    ss_state["bundle_path"] = result.get("model_bundle_path")
                    ss_state["bundle_cache"] = {"path": None, "bundle": None}
                    try:
                        _save_run_entry(sid, run_config, result)
                    except Exception:
                        pass
                else:
                    ss_state["train_status"] = "error"
                    tb = result.get("traceback", "")
                    err = result.get("error", "Unknown error")
                    ss_state["train_progress"] = f"{err}\n{tb}" if tb else err
        except Exception as e:
            import traceback as _tb
            with ss_state["train_lock"]:
                ss_state["train_status"] = "error"
                ss_state["train_progress"] = _tb.format_exc()
                ss_state["train_result"] = {"status": "error", "error": str(e), "traceback": _tb.format_exc()}

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"message": "Training started", "filename": file.filename})


@app.route("/api/train/status")
def api_train_status():
    import logging as _logging
    sid = _sid()
    ss_state = _ss(sid)
    with ss_state["train_lock"]:
        steps = list(ss_state.get("train_steps", []))
        _logging.info(f"[api_train_status] sid={sid[:8]} status={ss_state['train_status']} steps={len(steps)}")
        resp = {
            "status": ss_state["train_status"],
            "progress": ss_state["train_progress"],
            "steps": steps,
        }
        if ss_state["train_status"] in ("done", "error"):
            resp["result"] = ss_state["train_result"]
        return jsonify(resp)



@app.route("/api/predict", methods=["POST"])
def api_predict():
    sid = _sid()
    ss_state = _ss(sid)
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    events = data.get("events", [])
    case_id = data.get("case_id", "web_case")

    if not events:
        return jsonify({"error": "No events provided"}), 400

    for i, ev in enumerate(events):
        if "activity" not in ev:
            return jsonify({"error": f"Event {i+1} missing 'activity' field"}), 400
        if "timestamp" not in ev:
            return jsonify({"error": f"Event {i+1} missing 'timestamp' field"}), 400

    bundle_p = ss_state.get("bundle_path") or str(_bundle_path(sid))
    if not Path(bundle_p).exists():
        return jsonify({"error": "No model bundle found. Train a model first."}), 404

    try:
        bundle = _get_bundle(ss_state, bundle_p)
        result = predict_running_case(bundle, events, case_id=case_id)

        resp = {
            "case_id": result["case_id"],
            "bucket_id": result["bucket_id"],
            "prefix_len": result["prefix_len"],
            "task_type": result.get("task_type", ""),
        }
        if "predicted_outcome" in result:
            resp["predicted_outcome"] = result["predicted_outcome"]
        elif "predicted_next_activity" in result:
            resp["predicted_next_activity"] = result["predicted_next_activity"]
        else:
            sec = result["predicted_remaining_time_sec"]
            resp["predicted_remaining_time_sec"] = round(sec, 1)
            resp["predicted_remaining_time_hours"] = round(sec / 3600, 2)
            resp["predicted_remaining_time_days"] = round(sec / 86400, 2)

        return jsonify(resp)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict/batch", methods=["POST"])
def api_predict_batch():
    sid = _sid()
    ss_state = _ss(sid)

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    bundle_p = ss_state.get("bundle_path") or str(_bundle_path(sid))
    if not Path(bundle_p).exists():
        return jsonify({"error": "No model bundle found. Train a model first."}), 404

    suffix = Path(file.filename).suffix or ".xes"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    file.save(tmp.name)
    tmp.close()

    try:
        import pandas as pd
        from ppm_preprocessing.io.format_detection import detect_format

        fmt = detect_format(tmp.name)
        if fmt == "csv":
            df = pd.read_csv(tmp.name, low_memory=False)
        else:
            from pm4py.objects.log.importer.xes import importer as xes_importer
            from pm4py.objects.conversion.log import converter as log_converter
            log = xes_importer.apply(tmp.name)
            df = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)

        # Use stored col_mapping from training scan if available
        cm = ss_state.get("col_mapping") or {}
        _case_cands = [cm.get("case_col"), "case:concept:name", "case_id", "caseid", "id"]
        _act_cands  = [cm.get("act_col"),  "concept:name", "activity", "Activity"]
        _ts_cands   = [cm.get("ts_col"),   "time:timestamp", "timestamp", "started", "time"]
        case_col = next((c for c in _case_cands if c and c in df.columns), None)
        act_col  = next((c for c in _act_cands  if c and c in df.columns), None)
        ts_col   = next((c for c in _ts_cands   if c and c in df.columns), None)

        if not all([case_col, act_col, ts_col]):
            return jsonify({"error": f"Cannot detect required columns. Found: {list(df.columns)}"}), 400

        df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
        df = df.sort_values([case_col, ts_col])

        bundle = _get_bundle(ss_state, bundle_p)
        results = []
        errors = []

        for cid, group in df.groupby(case_col, sort=False):
            events = []
            for _, row in group.iterrows():
                ev = {"activity": str(row[act_col]), "timestamp": str(row[ts_col])}
                for col in df.columns:
                    if col in (case_col, act_col, ts_col):
                        continue
                    val = row[col]
                    if pd.notna(val):
                        ev[col] = val
                events.append(ev)

            try:
                pred = predict_running_case(bundle, events, case_id=str(cid))
                row_out = {
                    "case_id": str(cid),
                    "num_events": len(events),
                    "last_activity": events[-1]["activity"],
                    "bucket_id": pred["bucket_id"],
                }
                if "predicted_outcome" in pred:
                    row_out["predicted_outcome"] = pred["predicted_outcome"]
                elif "predicted_next_activity" in pred:
                    row_out["predicted_next_activity"] = pred["predicted_next_activity"]
                else:
                    sec = pred["predicted_remaining_time_sec"]
                    row_out["predicted_remaining_time_sec"] = round(sec, 1)
                    row_out["predicted_remaining_time_hours"] = round(sec / 3600, 2)
                    row_out["predicted_remaining_time_days"] = round(sec / 86400, 2)
                results.append(row_out)
            except Exception as e:
                errors.append({"case_id": str(cid), "error": str(e)})

        return jsonify({
            "predictions": results,
            "errors": errors,
            "total_cases": len(results) + len(errors),
            "successful": len(results),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        Path(tmp.name).unlink(missing_ok=True)


@app.route("/api/model/info")
def api_model_info():
    sid = _sid()
    out = _out_dir(sid)

    meta_path = out / "model_bundle_meta.json"
    eval_path = out / "test_evaluation.json"
    best_path = out / "best_strategy.json"
    stats_path = out / "dataset_stats.json"

    info = {"has_model": _bundle_path(sid).exists()}

    if meta_path.exists():
        info["meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
    if eval_path.exists():
        ev = json.loads(eval_path.read_text(encoding="utf-8"))
        if stats_path.exists():
            try:
                ev.update(json.loads(stats_path.read_text(encoding="utf-8")))
            except Exception:
                pass
        info["evaluation"] = ev
    if best_path.exists():
        info["best_strategy"] = json.loads(best_path.read_text(encoding="utf-8"))

    reports_dir = out / "reports"
    charts = []
    if reports_dir.exists():
        for p in sorted(reports_dir.glob("*.png")):
            charts.append(p.stem)
    info["charts"] = charts

    return jsonify(info)


@app.route("/api/quick-compare", methods=["POST"])
def api_quick_compare():
    sid = _sid()
    ss_state = _ss(sid)

    with ss_state["compare_lock"]:
        if ss_state["compare_status"] == "running":
            return jsonify({"error": "Quick compare already running"}), 409

    log_path_str = ss_state.get("scanned_file")
    if not log_path_str or not Path(log_path_str).exists():
        return jsonify({"error": "No file uploaded. Please upload and scan a file first."}), 400

    task_name = request.form.get("task_name", "remaining_time")
    outlier_enabled = request.form.get("outlier_enabled", "true").lower() == "true"
    rare_class_enabled = request.form.get("rare_class_enabled", "false").lower() == "true"
    min_class_samples_user = int(request.form.get("min_class_samples", "0"))
    outcome_col = request.form.get("outcome_col", "")
    outcome_values = request.form.get("outcome_values", "")
    next_event_attr_col = request.form.get("next_event_attr_col", "")
    import json as _json
    columns_to_drop_raw = request.form.get("columns_to_drop", "[]")
    try:
        columns_to_drop = _json.loads(columns_to_drop_raw)
    except Exception:
        columns_to_drop = []
    lifecycle_drop_raw = request.form.get("lifecycle_drop_values", "[]")
    try:
        lifecycle_drop_values = _json.loads(lifecycle_drop_raw)
    except Exception:
        lifecycle_drop_values = []
    try:
        train_ratio = max(0.1, min(0.9, float(request.form.get("train_ratio", "0.7"))))
        val_ratio = max(0.05, min(0.5, float(request.form.get("val_ratio", "0.15"))))
        if train_ratio + val_ratio >= 1.0:
            val_ratio = round(1.0 - train_ratio - 0.05, 2)
    except Exception:
        train_ratio, val_ratio = 0.7, 0.15

    max_prefix_len = max(5, min(200, int(request.form.get("max_prefix_len", "30"))))
    min_bucket_samples = max(10, min(1000, int(request.form.get("min_bucket_samples", "100"))))
    bin_size = max(1, min(50, int(request.form.get("bin_size", "5"))))
    effective_min_class = min_class_samples_user if rare_class_enabled else 0
    cmp_temporal_split = request.form.get("temporal_split", "false").lower() == "true"
    cmp_rare_variant = request.form.get("rare_variant_filter", "false").lower() == "true"
    cmp_min_variant = max(2, min(100, int(request.form.get("min_variant_count", "5"))))
    cmp_drift_window = request.form.get("concept_drift_window", "false").lower() == "true"
    cmp_since_date = request.form.get("since_date", "").strip()
    cmp_consec_dup = request.form.get("filter_consecutive_duplicates", "false").lower() == "true"
    cmp_impute = request.form.get("impute_missing", "false").lower() == "true"
    col_mapping = ss_state.get("col_mapping")
    log_path = Path(log_path_str)

    with ss_state["compare_lock"]:
        ss_state["compare_status"] = "running"
        ss_state["compare_progress"] = "Starting Baseline vs Your Config comparison..."
        ss_state["compare_results"] = None
        ss_state["compare_steps"] = []

    def _on_progress(msg: str):
        with ss_state["compare_lock"]:
            ss_state["compare_progress"] = msg

    def _run():
        is_classification = task_name in ("next_activity", "outcome")
        # Baseline: no optional steps, no time filter
        variants = [
            ("Baseline",    False,           [],             0,                  False, False, 5,               False, "",              False,        False),
            ("Your Config", outlier_enabled, columns_to_drop, effective_min_class, cmp_temporal_split, cmp_rare_variant, cmp_min_variant, cmp_drift_window, cmp_since_date, cmp_consec_dup, cmp_impute),
        ]

        results = []
        for i, (label, oe, cols, mcs, ts, rvf, mvc, cdw, sd, fcd, imp) in enumerate(variants):
            _on_progress(f"[{i + 1}/2] {label} — loading & preprocessing...")

            def _variant_progress(msg, _label=label, _i=i):
                import json as _json
                import logging as _logging
                if msg.startswith("__STEP__:"):
                    try:
                        step_data = _json.loads(msg[9:])
                        step_data["variant"] = _label
                        step_data["variant_index"] = _i
                        with ss_state["compare_lock"]:
                            ss_state["compare_steps"].append(step_data)
                    except Exception as _e:
                        _logging.warning(f"[compare_steps] JSON parse failed: {_e!r}")
                else:
                    _on_progress(f"[{_i + 1}/2] {_label}: {msg}")

            r = run_strategy_search_only(
                log_path=log_path,
                task_name=task_name,
                outlier_enabled=oe,
                min_class_samples=mcs,
                outcome_col=outcome_col,
                outcome_values=outcome_values,
                next_event_attr_col=next_event_attr_col,
                columns_to_drop=cols,
                lifecycle_drop_values=lifecycle_drop_values,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                max_prefix_len=max_prefix_len,
                min_bucket_samples=min_bucket_samples,
                bin_size=bin_size,
                temporal_split=ts,
                rare_variant_filter=rvf,
                min_variant_count=mvc,
                concept_drift_window=cdw,
                since_date=sd,
                filter_consecutive_duplicates=fcd,
                impute_missing=imp,
                col_mapping=col_mapping,
                on_progress=_variant_progress,
            )
            results.append({
                "label": label,
                "outlier_enabled": oe,
                "columns_dropped": len(cols),
                "min_class_samples": mcs,
                "status": r.get("status"),
                "best_strategy": r.get("best_strategy", {}),
                "primary_score": r.get("primary_score"),
                "is_classification": r.get("is_classification", is_classification),
                "error": r.get("error"),
            })

        with ss_state["compare_lock"]:
            ss_state["compare_status"] = "done"
            ss_state["compare_results"] = results
            ss_state["compare_progress"] = "Done — Baseline vs Your Config compared."

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"message": "Quick compare started"})


@app.route("/api/quick-compare/status")
def api_quick_compare_status():
    sid = _sid()
    ss_state = _ss(sid)
    with ss_state["compare_lock"]:
        resp = {
            "status": ss_state["compare_status"],
            "progress": ss_state["compare_progress"],
            "steps": list(ss_state["compare_steps"]),
        }
        if ss_state["compare_status"] in ("done", "error"):
            resp["results"] = ss_state["compare_results"]
        return jsonify(resp)


@app.route("/api/run-history")
def api_run_history():
    sid = _sid()
    return jsonify(_load_run_history(sid))


@app.route("/api/run-history/clear", methods=["POST"])
def api_run_history_clear():
    sid = _sid()
    p = _run_history_path(sid)
    if p.exists():
        p.unlink()
    return jsonify({"message": "History cleared"})


@app.route("/api/reports/<name>")
def api_report_image(name: str):
    sid = _sid()
    safe = "".join(c for c in name if c.isalnum() or c in ("_", "-"))
    path = _out_dir(sid) / "reports" / f"{safe}.png"
    if not path.exists():
        return jsonify({"error": "Chart not found"}), 404
    return send_file(str(path), mimetype="image/png")


# ---------------------------------------------------------------------------
# LLM Chat assistant
# ---------------------------------------------------------------------------
_CHAT_SYSTEM = """You are a concise assistant embedded in a Predictive Process Monitoring (PPM) preprocessing tool. Help users understand settings, interpret results, and navigate the workflow.

TOOL WORKFLOW:
1. Upload event log (XES or CSV)
2. Select prediction task
3. Configure preprocessing & compare against baseline
4. Train model (AutoML)
5. View results & charts
6. Predict on running (incomplete) cases

PREDICTION TASKS:
- Remaining Time — regression, predicts how long until a case completes. Metric: MAE (days, lower is better).
- Outcome — multiclass classification, predicts the final outcome of a case using the last activity or a chosen case attribute as label. Metric: F1-macro (higher is better).
- Next Event Attribute — predicts the next value of any event-level column (e.g. activity, resource). Metric: F1-macro.

MANDATORY PIPELINE STEPS (always run, cannot be disabled):
1. Load XES / Load CSV — parses the uploaded file.
2. Normalize Schema — maps XES standard columns (case:concept:name, concept:name, time:timestamp) to internal names (case_id, activity, timestamp).
3. Deduplicate Events — removes exact duplicate event rows.
4. Clean & Sort — validates and parses timestamps, drops rows with unparseable timestamps, sorts events by case and time.
5. Stable Sort — re-sorts using a stable mergesort with an internal tie-breaker index so events with identical timestamps always appear in a deterministic order.
6. Repair Timestamps — detects and corrects backwards timestamps within a case.
7. Filter Short Cases — removes cases with fewer than 2 events (single-event cases carry no sequential information).
8. Normalize Activities — strips whitespace and standardises activity label formatting.
9. Filter Infrequent Activities — removes events whose activity type appears in fewer than 10 cases.
10. Filter Zero-Duration Cases — removes cases where start and end timestamps are identical.
11. Filter Case Length Outliers — removes cases longer than the 99th percentile of case length (computed on training split only).
12. Case Labels — assigns the prediction target label to each case (e.g. remaining time in seconds, outcome class).
13. Prefix Extraction — generates one training instance per (case, prefix length) pair up to Max Prefix Length.
14. Case Split — partitions cases into train / validation / test sets (default 70/15/15%).

OPTIONAL PREPROCESSING STEPS (user-configurable):
- Outlier Detection (IQR): after prefix extraction and case split, removes training prefix rows whose case duration falls outside IQR bounds (computed on train only). Works for all task types. Validation and test sets are never modified.
- Filter Rare Classes: removes training prefix rows whose outcome class has fewer than N training samples. Prevents the model from learning from near-zero evidence. Classification tasks only.
- Time Window Filter: discards cases that started before a chosen date. Useful when older data reflects a superseded process version.
- Rare Variant Filter: removes cases whose exact activity sequence (trace variant) appears fewer than N times. Eliminates one-off traces that cannot be generalised.
- Filter Consecutive Duplicates: collapses repeated consecutive identical activities within a case (A→A→B becomes A→B). Removes logging artefacts.
- Impute Missing Attributes: fills missing case attribute values with column median (numeric) or mode (categorical). Ensures complete feature vectors for all encoders.
- Drop Columns: exclude specific case attributes before encoding. Columns with ≥50% missing values are pre-checked. Activity, timestamp, case ID, and the target column are always protected.
- Temporal Ordering: sorts cases chronologically before splitting so oldest cases go to train and newest to test. Prevents future-data leakage and simulates real deployment.

ADVANCED SETTINGS:
- Max Prefix Length: longest prefix used for training. Auto = 95th percentile of case lengths.
- AutoML Time Budget: seconds allowed for AutoML model search. Auto = 300 s.
- Min Bucket Samples: buckets with fewer training samples than this threshold are skipped. Auto = 100.
- Fixed-Width Bin Size: events per bucket for the Prefix Length Bins bucketing strategy. Auto = 5.

STRATEGY SEARCH (5 bucketers × 4 encoders = 20 combinations, scored by linear probe on validation set):
Bucketers: No Bucket (global model) | Last Activity | Prefix Length Bins (fixed-width) | Prefix Length Adaptive | Cluster
Encoders:  Last State | Aggregation | Index Latest Payload | Embedding (sentence-transformer)
→ Best combination is selected for full AutoML training, evaluated on the held-out test set.
→ Per-bucket mode: each bucket trains its own model. Improves accuracy when different process stages behave differently.

BASELINE COMPARISON:
Runs the pipeline twice simultaneously — once with all optional steps disabled (Baseline) and once with the user's configuration. Both use the same file, split ratios, and random seed. Scores are computed with fast linear probe models, not full AutoML. Use the comparison to decide whether the configured preprocessing actually improves over the uncleaned baseline before committing to full training.

METRICS:
- MAE in days (lower is better) — remaining time regression
- F1-macro (higher is better) — outcome and next-event classification
- F1-micro also shown for classification — reflects accuracy on the dominant class; macro treats all classes equally regardless of frequency.

KEY CONCEPTS:
- Prefix: the first k events of a case. The model sees only the partial trace observed so far and predicts the future.
- Bucketing: groups prefixes by a criterion (e.g. last activity, prefix length) so each group can receive a dedicated model or encoding strategy.
- Encoding: converts a variable-length prefix into a fixed-size numeric feature vector suitable for standard ML models.
- Concept drift: the process behaviour changes over time. Older cases may follow different rules; the Time Window Filter and Temporal Ordering address this.
- IQR (Interquartile Range): a robust measure of spread. Outliers are defined as values below Q1 − 1.5×IQR or above Q3 + 1.5×IQR.

Keep answers short and practical. When the user asks about their specific results, refer to the CURRENT TRAINING RESULT and BASELINE COMPARISON RESULTS blocks provided below."""


@app.route("/api/chat-ping")
def api_chat_ping():
    key_set = bool(os.environ.get("ANTHROPIC_API_KEY"))
    try:
        import anthropic as _a
        pkg_ok = True
        pkg_ver = getattr(_a, "__version__", "?")
    except ImportError:
        pkg_ok = False
        pkg_ver = "not installed"
    return jsonify({"ok": True, "anthropic_installed": pkg_ok, "version": pkg_ver, "key_set": key_set})


def _build_context_block(ss_state: dict, sid: str) -> str:
    """Return a compact JSON summary of this session's results for the Claude system prompt."""
    parts = []

    with ss_state["train_lock"]:
        tr = ss_state.get("train_result")
        ts = ss_state.get("train_status", "idle")

    if tr and ts == "done" and tr.get("status") == "success":
        summary = {
            "training_status": "done",
            "task": tr.get("task_name") or tr.get("task"),
            "is_classification": tr.get("is_classification"),
            "best_strategy": tr.get("best_strategy"),
            "metrics": tr.get("metrics"),
            "test_samples": tr.get("test_samples"),
            "n_cases": tr.get("n_cases"),
            "n_events": tr.get("n_events"),
        }
        parts.append("CURRENT TRAINING RESULT:\n" + json.dumps(summary, indent=2, default=str))
    elif ts == "running":
        parts.append("CURRENT TRAINING STATUS: running (not finished yet)")
    else:
        parts.append("CURRENT TRAINING STATUS: no training done yet in this session")

    with ss_state["compare_lock"]:
        cr = ss_state.get("compare_results")
        cs = ss_state.get("compare_status", "idle")

    if cr and cs == "done":
        parts.append("BASELINE COMPARISON RESULTS:\n" + json.dumps(cr, indent=2, default=str))

    # Preprocessing steps from training run
    train_steps = ss_state.get("train_steps", [])
    if train_steps:
        step_summary = []
        for s in train_steps:
            if s.get("step") == "qc_report":
                continue
            before = s.get("before_rows")
            after = s.get("after_rows")
            removed = (before - after) if (before is not None and after is not None) else None
            step_summary.append({
                "step": s.get("step"),
                "before_rows": before,
                "after_rows": after,
                "rows_removed": removed,
                "cases_removed": (s.get("before_cases") - s.get("after_cases"))
                    if (s.get("before_cases") is not None and s.get("after_cases") is not None) else None,
            })
        parts.append("TRAINING PIPELINE STEPS (row counts):\n" + json.dumps(step_summary, indent=2, default=str))

    # Preprocessing steps from comparison run
    compare_steps = ss_state.get("compare_steps", [])
    if compare_steps:
        step_summary = []
        for s in compare_steps:
            if s.get("step") == "qc_report":
                continue
            before = s.get("before_rows")
            after = s.get("after_rows")
            removed = (before - after) if (before is not None and after is not None) else None
            step_summary.append({
                "step": s.get("step"),
                "variant": s.get("variant", ""),
                "before_rows": before,
                "after_rows": after,
                "rows_removed": removed,
            })
        parts.append("COMPARISON PIPELINE STEPS (row counts):\n" + json.dumps(step_summary, indent=2, default=str))

    history = _load_run_history(sid)
    if history:
        parts.append("RUN HISTORY (last 5):\n" + json.dumps(history[-5:], indent=2, default=str))

    return "\n\n".join(parts)


@app.route("/api/chat", methods=["POST"])
def api_chat():
    import traceback as _tb

    sid = _sid()
    ss_state = _ss(sid)

    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    messages = data.get("messages", [])
    if not messages:
        return jsonify({"error": "No messages"}), 400

    for msg in messages:
        if msg.get("role") not in ("user", "assistant") or not isinstance(msg.get("content"), str):
            return jsonify({"error": "Invalid message format"}), 400

    try:
        import anthropic as _anthropic
    except ImportError:
        return jsonify({"error": "anthropic not installed — run: pip install anthropic"}), 500

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return jsonify({"error": "ANTHROPIC_API_KEY not set"}), 500

    system = _CHAT_SYSTEM + "\n\n" + _build_context_block(ss_state, sid)

    try:
        client = _anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1024,
            system=system,
            messages=messages,
        )
        return jsonify({"text": response.content[0].text})
    except Exception as e:
        _tb.print_exc()
        msg = str(e)
        if "credit balance is too low" in msg:
            return jsonify({"error": "No Anthropic credits — add credits at console.anthropic.com/settings/billing"}), 402
        if "401" in msg or "invalid x-api-key" in msg.lower() or "authentication" in msg.lower():
            return jsonify({"error": "Invalid API key — check console.anthropic.com"}), 401
        return jsonify({"error": msg}), 500


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import os
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    print("\n  PPM Predictive Process Monitoring")
    print(f"  http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    main()
