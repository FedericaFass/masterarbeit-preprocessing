"""
Feature importance extraction step.

Extracts feature importances from trained FLAML models and saves
a ranked report (JSON + optional CSV). Works with tree-based models
(LightGBM, XGBoost, RandomForest, ExtraTrees, CatBoost) that expose
`feature_importances_`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

import numpy as np

from .base import Step
from ppm_preprocessing.domain.context import PipelineContext


@dataclass
class FeatureImportanceConfig:
    models_key: str = "single_task_models"
    automl_key: str = "single_task_automl"
    encoder_key: str = "fitted_encoder"
    output_key: str = "feature_importance"
    save_json: bool = True
    save_csv: bool = True
    top_n_print: int = 15


class SingleTaskFeatureImportanceStep(Step):
    """
    Extract and report feature importances from trained model(s).

    Supports both global_model and per_bucket_models modes.
    Feature names come from the fitted encoder's `feature_names_` attribute.
    """

    name = "feature_importance"

    def __init__(self, config: FeatureImportanceConfig | None = None):
        self.config = config or FeatureImportanceConfig()

    @staticmethod
    def _extract_importances(automl_obj: Any) -> Optional[np.ndarray]:
        """Try to extract feature importances from a FLAML AutoML object."""
        model = getattr(automl_obj, "model", None)
        if model is None:
            return None

        # FLAML wraps the estimator; the actual model may be inside a pipeline
        estimator = model
        if hasattr(estimator, "steps"):
            # sklearn Pipeline — get the last step
            estimator = estimator.steps[-1][1]

        if hasattr(estimator, "feature_importances_"):
            return np.asarray(estimator.feature_importances_, dtype=np.float64)

        # Some models (e.g., linear) expose coef_ instead
        if hasattr(estimator, "coef_"):
            coef = np.asarray(estimator.coef_, dtype=np.float64)
            if coef.ndim > 1:
                coef = np.mean(np.abs(coef), axis=0)
            return np.abs(coef)

        return None

    def _build_ranked_list(
        self,
        importances: np.ndarray,
        feature_names: List[str],
    ) -> List[Dict[str, Any]]:
        """Build a sorted list of {feature, importance, rank}."""
        total = float(importances.sum()) if importances.sum() > 0 else 1.0
        order = np.argsort(importances)[::-1]

        ranked = []
        for rank, idx in enumerate(order, start=1):
            ranked.append({
                "rank": rank,
                "feature": feature_names[idx] if idx < len(feature_names) else f"feature_{idx}",
                "importance": float(importances[idx]),
                "importance_pct": float(importances[idx] / total * 100),
            })
        return ranked

    def run(self, ctx: PipelineContext) -> PipelineContext:
        c = self.config

        models = ctx.artifacts.get(c.models_key)
        if not models:
            print("Warning: No trained models found. Skipping feature importance.")
            ctx.artifacts[c.output_key] = {"error": "no_models"}
            return ctx

        # Get feature names from fitted encoder
        encoder = ctx.artifacts.get(c.encoder_key)
        feature_names: List[str] = []
        if encoder is not None and hasattr(encoder, "feature_names_"):
            feature_names = list(encoder.feature_names_)

        automl_result = ctx.artifacts.get(c.automl_key, {})
        strategy = automl_result.get("strategy", {})
        mode = strategy.get("mode", "global_model")

        result: Dict[str, Any] = {
            "mode": mode,
            "encoding": strategy.get("encoding"),
            "bucketing": strategy.get("bucketing"),
            "models": {},
        }

        print(f"\n{'='*60}")
        print("FEATURE IMPORTANCE REPORT")
        print(f"{'='*60}")

        if mode == "global_model":
            model_data = models.get("global")
            if model_data:
                imp = self._extract_importances(model_data.get("automl"))
                if imp is not None:
                    if not feature_names:
                        feature_names = [f"feature_{i}" for i in range(len(imp))]
                    ranked = self._build_ranked_list(imp, feature_names)
                    result["models"]["global"] = {
                        "n_features": len(ranked),
                        "features": ranked,
                    }
                    self._print_top(ranked, "Global model", c.top_n_print)
                else:
                    print("  Could not extract feature importances (model type may not support it).")
                    result["models"]["global"] = {"error": "not_supported"}
        else:
            for model_key, model_data in models.items():
                imp = self._extract_importances(model_data.get("automl"))
                if imp is not None:
                    if not feature_names:
                        feature_names = [f"feature_{i}" for i in range(len(imp))]
                    ranked = self._build_ranked_list(imp, feature_names)
                    result["models"][str(model_key)] = {
                        "n_features": len(ranked),
                        "features": ranked,
                    }
                    self._print_top(ranked, f"Bucket {model_key}", c.top_n_print)

        ctx.artifacts[c.output_key] = result

        # Save outputs
        out_dir = Path(ctx.artifacts.get("out_dir", "outputs/single_task"))
        out_dir.mkdir(parents=True, exist_ok=True)

        if c.save_json:
            path = out_dir / "feature_importance.json"
            path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
            print(f"\nSaved to {path}")

        if c.save_csv and result.get("models"):
            import csv
            # Save a flat CSV with all models' importances
            csv_path = out_dir / "feature_importance.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["model", "rank", "feature", "importance", "importance_pct"])
                for model_key, model_info in result["models"].items():
                    if "features" not in model_info:
                        continue
                    for feat in model_info["features"]:
                        writer.writerow([
                            model_key,
                            feat["rank"],
                            feat["feature"],
                            f"{feat['importance']:.6f}",
                            f"{feat['importance_pct']:.2f}",
                        ])
            print(f"Saved to {csv_path}")

        print(f"{'='*60}\n")
        return ctx

    @staticmethod
    def _print_top(ranked: List[Dict[str, Any]], label: str, top_n: int) -> None:
        print(f"\n  {label} - Top {min(top_n, len(ranked))} features:")
        print(f"  {'Rank':>4}  {'Feature':<40} {'Importance':>10} {'%':>7}")
        print(f"  {'-'*65}")
        for feat in ranked[:top_n]:
            print(f"  {feat['rank']:>4}  {feat['feature']:<40} {feat['importance']:>10.4f} {feat['importance_pct']:>6.1f}%")
