# PPM Preprocessing — Predictive Process Monitoring Pipeline

A browser-based tool for preprocessing event logs, training prediction models, and evaluating them for Predictive Process Monitoring (PPM). Built as part of a master's thesis.

---

## Table of Contents

1. [Requirements](#1-requirements)
2. [Installation](#2-installation)
3. [Running Locally](#3-running-locally)
4. [Running with Docker](#4-running-with-docker)
5. [Using the Web UI](#5-using-the-web-ui)
6. [CLI Tools (Research Scripts)](#6-cli-tools-research-scripts)
7. [Folder Structure](#7-folder-structure)
8. [Key Files Explained](#8-key-files-explained)

---

## 1. Requirements

| Requirement | Version |
|-------------|---------|
| Python | 3.10 or higher |
| pip | any recent version |
| Git | for cloning the repo |
| (optional) Docker | for containerized deployment |

> **Note:** The embedding encoder downloads a ~90 MB model (`all-MiniLM-L6-v2`) from HuggingFace on first use. Internet access is required the first time you run it.

---

## 2. Installation

### Step 1 — Clone the repository

```bash
git clone <your-repo-url>
cd masterarbeit-preprocessing
```

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it:

- **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
- **Windows (CMD):**
  ```cmd
  .venv\Scripts\activate.bat
  ```
- **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Install the package itself

```bash
pip install -e .
```

This installs the `ppm_preprocessing` package in editable mode so imports work from anywhere in the project.

---

## 3. Running Locally

```bash
python -m ppm_preprocessing.webapp
```

Then open your browser at:

```
http://localhost:5000
```

The app will keep running until you press `Ctrl+C` in the terminal.

> **Alternative — run with Gunicorn (production-like):**
> ```bash
> bash start.sh
> ```
> This starts the app on port 8080 with 4 threads.

---

## 4. Running with Docker

### Build the image

```bash
docker build -t ppm-preprocessing .
```

### Run the container

```bash
docker run -p 8080:8080 ppm-preprocessing
```

Then open `http://localhost:8080`.

> The Docker image pre-downloads the sentence-transformer model at build time, so no internet access is needed at runtime.

---

## 5. Using the Web UI

### Step-by-step workflow

1. **Upload an event log** — supported formats: `.xes`, `.csv`
2. **Choose a prediction task** — Next Attribute, Remaining Time, or Outcome
3. **Configure the pipeline** — select which preprocessing steps to apply (deduplication, filtering, imputation, etc.)
4. **Run Comparison** - see if your configuration is better than the baseline and see which steps do what to your eventlog
5. **Train with AutoML** — FLAML trains a model on the best strategy (300 s budget)
6. **See results** —  evaluation reports, and figures
7. **See your uncomplited cases** - you can test uncompleted cases and get a prediction

### Session isolation

Each browser session is independent. Your uploads, results, and models are never visible to other users. Sessions are stored in `outputs/sessions/`.

---

## 6. CLI Tools (Research Scripts)

These scripts were used to produce the thesis results. Run them from the project root with the virtual environment activated.

### Run the ablation study

Evaluates each preprocessing stage cumulatively for a given log and task:

```bash
python -m ppm_preprocessing.cli.run_ablation \
    --log data/raw/DomesticDeclarations.xes \
    --task next_activity \
    --out results/ablation/DomesticDeclarations
```

### Run the strategy search evaluation (all 5 logs × 3 tasks)

```bash
python -m ppm_preprocessing.cli.run_strategy_eval
```

Output is written to `results/strategy_eval/strategy_eval_results.json`.

### Run the AutoML evaluation

```bash
python -m ppm_preprocessing.cli.run_automl_eval
```

Output is written to `results/automl_eval/automl_eval_results.json`.

### Regenerate thesis figures

```bash
python results/figures/generate_figures.py
```

Produces PDF and PNG figures in `results/figures/`.

---

## 7. Folder Structure

```
masterarbeit-preprocessing/
│
├── src/
│   └── ppm_preprocessing/          # Main Python package
│       ├── automl/                 # FLAML adapter
│       ├── bucketing/              # Bucketing strategies (5 variants)
│       ├── cli/                    # Command-line research scripts
│       ├── domain/                 # Core data types (CanonicalLog, PipelineContext)
│       ├── encoders/               # Encoding strategies (4 variants)
│       ├── io/                     # File loaders (XES, CSV) and writers
│       ├── pipeline/               # Pipeline orchestration
│       ├── steps/                  # Individual pipeline steps (one file per step)
│       ├── tasks/                  # Task definitions (next activity, remaining time, outcome)
│       ├── webapp/                 # Flask web application
│       │   ├── app.py              # Flask routes and session handling
│       │   ├── pipeline_runner.py  # Background pipeline execution
│       │   └── templates/          # HTML frontend (index.html)
│       └── inference.py            # Load model bundle and run predictions
│
├── data/
│   ├── raw/                        # Input event logs (.xes, .csv)
│   ├── interim/                    # Intermediate processed files
│   └── processed/                  # Final processed datasets
│
├── results/
│   ├── ablation/                   # Ablation study results per log per task
│   │   └── <LogName>/<task>/ablation_results.json
│   ├── strategy_eval/              # Strategy search evaluation results
│   │   ├── strategy_eval_results.json
│   │   └── strategy_eval_summary.txt
│   ├── automl_eval/                # AutoML evaluation results
│   │   ├── automl_eval_results.json
│   │   └── automl_eval_summary.txt
│   └── figures/                    # Generated thesis figures (.pdf, .png)
│       └── generate_figures.py     # Script to regenerate all figures
│
├── outputs/
│   ├── reports/                    # QC reports from pipeline steps (.json, .csv, .png)
│   ├── sessions/                   # Per-session state for the web app
│   └── single_task/                # Model bundle and reports from the last web UI run
│       ├── model_bundle.joblib     # Trained model (load with joblib.load)
│       ├── model_bundle_meta.json  # Metadata about the bundle
│       ├── best_strategy.json      # Best bucketing/encoding combination found
│       ├── comparison.json         # Full strategy comparison table
│       ├── test_evaluation.json    # Evaluation metrics on the test set
│       ├── feature_importance.csv  # Feature importance scores
│       └── report.json             # Full run report
│
├── tests/                          # Smoke tests
│   └── test_smoke.py
│
├── configs/
│   └── pipeline.yaml               # Pipeline configuration (currently minimal)
│
├── notebook/
│   └── check.py                    # Ad-hoc inspection scripts
│
├── pyproject.toml                  # Package metadata and build config
├── requirements.txt                # All Python dependencies (pinned versions)
├── Dockerfile                      # Docker image for deployment
├── start.sh                        # Gunicorn startup script (used by Docker)
├── railway.toml                    # Railway.app deployment config
├── wsgi.py                         # WSGI entry point
└── README.md                       # This file
```

---

## 8. Key Files Explained

### `src/ppm_preprocessing/webapp/app.py`
Flask application. Handles file uploads, session management, routing, and serving results. Each browser session gets its own isolated working directory under `outputs/sessions/`.

### `src/ppm_preprocessing/webapp/pipeline_runner.py`
Runs the pipeline in a background thread so the UI stays responsive. Sends progress updates to the frontend via polling.

### `src/ppm_preprocessing/webapp/templates/index.html`
Single-page frontend. Handles file upload, step configuration, progress display, and result download — all in one HTML file.

### `src/ppm_preprocessing/pipeline/preprocessing_pipeline.py`
Orchestrates the sequence of pipeline steps. Each step receives a `PipelineContext` and returns an updated one.

### `src/ppm_preprocessing/domain/context.py`
`PipelineContext` is the shared state object passed between steps. It holds the event log, all intermediate artifacts, and configuration.

### `src/ppm_preprocessing/domain/canonical_log.py`
`CanonicalLog` is the internal representation of an event log (a DataFrame with a fixed schema: `case_id`, `activity`, `timestamp`, and optional attributes).

### `src/ppm_preprocessing/steps/`
Each file implements one pipeline step. Examples:
- `deduplicate_events.py` — removes duplicate events within a case
- `filter_short_cases.py` — removes cases below a minimum length
- `impute_missing_attributes.py` — fills missing attribute values
- `single_task_strategy_search.py` — tests all 20 bucketing/encoding combos with LightGBM
- `single_task_automl_train.py` — trains with FLAML on the best strategy
- `single_task_persist_model.py` — saves the trained model bundle to disk

### `src/ppm_preprocessing/bucketing/`
Five bucketing strategies: `no_bucket`, `last_activity`, `prefix_len_bins`, `prefix_len_adaptive`, `cluster`.

### `src/ppm_preprocessing/encoders/`
Four encoding strategies: `last_state`, `aggregation`, `index_latest_payload`, `embedding` (uses sentence-transformers).

### `src/ppm_preprocessing/inference.py`
Provides `load_bundle()` and `predict_running_case()` — use these to load a saved model bundle and make predictions on new cases programmatically.

### `outputs/single_task/model_bundle.joblib`
The trained model from the last web UI run. Load it like this:

```python
import joblib
from ppm_preprocessing.inference import load_bundle, predict_running_case

bundle = load_bundle("outputs/single_task/model_bundle.joblib")
# bundle contains: task_type, strategy, models, encoder, bucketer
```

### `results/figures/generate_figures.py`
Reads the three result JSON files and produces all thesis figures. Run after any result update to regenerate the PDFs.

### `requirements.txt`
All dependencies with pinned versions. Key libraries:
- `flaml` — AutoML framework
- `pm4py` — XES event log parsing
- `scikit-learn` — ML utilities
- `sentence-transformers` + `torch` — embedding encoder
- `flask` + `gunicorn` — web server
- `matplotlib` + `seaborn` — figure generation

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'ppm_preprocessing'`**
→ Run `pip install -e .` from the project root.

**Upload fails for large `.xes` files**
→ The upload limit is 2 GB. If you hit it, check that you are using the local version (not a restrictive proxy).

**Embedding encoder is slow on first run**
→ It downloads ~90 MB from HuggingFace. Subsequent runs use the cached model.

**Port already in use**
→ Change the port: `python -m ppm_preprocessing.webapp --port 5001` or set `PORT=5001 bash start.sh`.
