from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np
from flaml import AutoML

from .base import AutoMLAdapter, AutoMLConfig, TaskKind


def _map_task(task: Any) -> str:
    """
    Map internal task kind to FLAML task strings.

    Accepts:
      - str: "binary" | "multiclass" | "regression"
      - Enum-like objects exposing `.value` or `.name`
    """
    if task is None:
        raise ValueError("task is None")

    if hasattr(task, "value"):
        task = getattr(task, "value")
    elif hasattr(task, "name"):
        task = getattr(task, "name")

    t = str(task).lower().strip()
    mapping = {
        "binary": "binary",
        "bin": "binary",
        "multiclass": "multiclass",
        "multi": "multiclass",
        "regression": "regression",
        "reg": "regression",
    }
    if t not in mapping:
        raise ValueError(f"Unknown task kind: {task!r}")
    return mapping[t]


def _map_metric(metric: str) -> str:
    m = (metric or "").lower().strip()
    mapping = {
        # classification
        "f1_macro": "macro_f1",
        "macro_f1": "macro_f1",
        "f1_micro": "micro_f1",
        "micro_f1": "micro_f1",
        "f1": "f1",
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
        "auc": "roc_auc",
        # regression
        "mae": "mae",
        "mae_sec": "mae",
        "mae_days": "mae",
        "rmse": "rmse",
        "mse": "mse",
        "r2": "r2",
    }
    return mapping.get(m, m)


class FLAMLAutoMLAdapter(AutoMLAdapter):
    name = "flaml"

    def fit_predict(
        self,
        task: str | TaskKind,
        X_train, y_train,
        X_val, y_val,
        config: AutoMLConfig,
    ) -> Dict[str, Any]:
        automl = AutoML()

        flaml_task = _map_task(task)
        flaml_metric = _map_metric(config.metric)

        automl_settings: Dict[str, Any] = {
            "time_budget": config.time_budget_s,
            "task": flaml_task,
            "metric": flaml_metric,
            "seed": config.seed,
            "n_jobs": config.n_jobs,
            "log_file_name": "",  # empty string disables log file
            "verbose": 0,        # suppress console output
        }
        if config.estimator_list:
            automl_settings["estimator_list"] = config.estimator_list

        automl.fit(X_train=X_train, y_train=y_train, **automl_settings)

        y_val_pred = np.asarray(automl.predict(X_val))

        y_val_proba: Optional[np.ndarray] = None
        try:
            y_val_proba = np.asarray(automl.predict_proba(X_val))
        except Exception:
            y_val_proba = None

        # IMPORTANT: return automl object so later steps can do test scoring + examples
        run_info: Dict[str, Any] = {
            "task": flaml_task,
            "metric": flaml_metric,
            "time_budget_s": config.time_budget_s,
            "best_estimator": getattr(automl, "best_estimator", None),
            "best_loss": getattr(automl, "best_loss", None),
            "best_config": getattr(automl, "best_config", None),
            "best_iteration": getattr(automl, "best_iteration", None),
            "training_log": {
                "time_to_find_best_s": getattr(automl, "time_to_find_best_model", None),
            },
            "best_model_repr": repr(getattr(automl, "model", None)),
            "val": {
                "y_pred": y_val_pred.tolist(),
                "y_proba": None if y_val_proba is None else y_val_proba.tolist(),
            },
            "_automl_object": automl,                 # <-- key fix
            "_best_model_object": getattr(automl, "model", None),
        }
        return run_info
