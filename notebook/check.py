from pathlib import Path
import joblib

MODEL_PATH = Path("outputs/single_task/model_bundle.joblib")  # anpassen falls anders

bundle = joblib.load(MODEL_PATH)

print("Keys:", bundle.keys())
print("Task:", bundle.get("task_name"))
print("Strategy:", bundle.get("strategy"))
print("Models:", bundle.get("models", {}).keys())

# Beispiel: 1 Modell anzeigen
models = bundle.get("models", {})
if models:
    k = sorted(models.keys())[0]
    m = models[k]
    print(f"Example bucket: {k} -> {type(m).__name__}")
    if hasattr(m, "n_features_in_"):
        print("n_features_in_:", m.n_features_in_)
