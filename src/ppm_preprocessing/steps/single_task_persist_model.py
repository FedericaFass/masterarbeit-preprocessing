from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import json
import joblib

from ppm_preprocessing.steps.base import Step
from ppm_preprocessing.domain.context import PipelineContext


@dataclass
class SingleTaskPersistModelConfig:
    # where to write files
    out_dir_key: str = "out_dir"
    filename_bundle: str = "model_bundle.joblib"
    filename_meta: str = "model_bundle_meta.json"

    # artifact keys
    automl_key: str = "single_task_automl"     # JSON summary (strategy, etc.)
    models_key: str = "single_task_models"     # actual automl objects per bucket/global

    # optional: persist these too if you already store them
    encoder_key: str = "fitted_encoder"        # recommended
    bucketer_key: str = "fitted_bucketer"      # recommended


class SingleTaskPersistModelStep(Step):
    """
    Persists the trained single-task models to disk so a user can download and reuse them.

    Requires:
      ctx.artifacts["single_task_automl"]   (JSON-friendly summary)
      ctx.artifacts["single_task_models"]  (contains actual AutoML objects)

    Optionally stores:
      ctx.artifacts["fitted_encoder"]
      ctx.artifacts["fitted_bucketer"]
    """
    name = "single_task_persist_model"

    def __init__(self, config: SingleTaskPersistModelConfig | None = None):
        self.config = config or SingleTaskPersistModelConfig()

    @staticmethod
    def _extract_estimator(automl_obj: Any, keep_full: bool = False) -> Any:
        """
        Extract the fitted estimator from a FLAML AutoML object.

        For classification, we keep the full AutoML object so that
        predict() applies the inverse label_transformer (int -> string).
        For regression, we extract just .model to save space.
        """
        if automl_obj is None:
            return None
        if keep_full:
            return automl_obj
        return getattr(automl_obj, "model", automl_obj)

    def run(self, ctx: PipelineContext) -> PipelineContext:
        c = self.config

        out_dir = Path(ctx.artifacts.get(c.out_dir_key, "."))
        out_dir.mkdir(parents=True, exist_ok=True)

        st = ctx.artifacts.get(c.automl_key)
        if not st:
            raise RuntimeError(f"{c.automl_key} missing. Run SingleTaskAutoMLTrainStep first.")

        models_art = ctx.artifacts.get(c.models_key)
        if not models_art:
            raise RuntimeError(f"{c.models_key} missing. Run SingleTaskAutoMLTrainStep first.")

        # Optional: encoder/bucketer if you already store them
        encoder = ctx.artifacts.get(c.encoder_key, None)
        bucketer = ctx.artifacts.get(c.bucketer_key, None)

        # Convert your stored structure -> plain {key: fitted_estimator}
        # You stored: models[str(bucket)] = {"automl": automl_obj, "info": ...}
        # For classification, keep the full AutoML object so predict()
        # returns original string labels (via label_transformer).
        is_clf = "classification" in str(st.get("task_type", ""))
        fitted_models: Dict[str, Any] = {}
        for k, v in models_art.items():
            automl_obj = v.get("automl") if isinstance(v, dict) else None
            est = self._extract_estimator(automl_obj, keep_full=is_clf)
            if est is not None:
                fitted_models[str(k)] = est

        if not fitted_models:
            raise RuntimeError("No fitted models found in single_task_models.")

        bundle = {
            "task_name": st.get("task_name"),
            "task_type": st.get("task_type"),
            "strategy": st.get("strategy"),
            "target_log1p": bool(st.get("target_log1p", False)),
            "clamp_nonnegative": bool(st.get("clamp_nonnegative", True)),
            "models": fitted_models,  # global or per-bucket estimators
            "encoder": encoder,       # fitted on train only
            "bucketer": bucketer,     # fitted on train only
            "meta": {
                "note": "Load with joblib.load(...). Use ppm_preprocessing.inference.predict_running_case() for inference.",
            },
        }

        bundle_path = out_dir / c.filename_bundle
        joblib.dump(bundle, bundle_path)

        meta = {
            "task_name": st.get("task_name"),
            "strategy": st.get("strategy"),
            "target_log1p": bool(st.get("target_log1p", False)),
            "clamp_nonnegative": bool(st.get("clamp_nonnegative", True)),
            "model_keys": sorted(list(fitted_models.keys())),
            "has_encoder": encoder is not None,
            "has_bucketer": bucketer is not None,
            "bundle_file": str(bundle_path),
        }
        meta_path = out_dir / c.filename_meta
        meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

        ctx.artifacts["persisted_model"] = {
            "bundle_path": str(bundle_path),
            "meta_path": str(meta_path),
        }
        return ctx
