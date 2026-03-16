"""
Visualization step for generating comprehensive plots of prediction results.
Supports both regression and classification tasks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

import numpy as np
import pandas as pd

from .base import Step
from ppm_preprocessing.domain.context import PipelineContext

_PRETTY = {
    # Bucketing
    "prefix_len_fixed":    "Fixed-Width",
    "prefix_len_adaptive": "Adaptive",
    "state_based":         "State-Based",
    "zero":                "All Prefixes",
    # Encoding
    "last_state":          "Last State",
    "agg_features":        "Aggregated",
    "boolean":             "Boolean Bag",
    "freq_encoded":        "Frequency",
    "embedding":           "Embedding",
    "index_latest_payload":"Latest Payload",
}

def _p(name: str) -> str:
    """Return a human-readable label for an internal strategy name."""
    return _PRETTY.get(name, name.replace("_", " ").title())


@dataclass
class VisualizeResultsConfig:
    task_name: str = "remaining_time"

    # Report paths are derived from task_name dynamically
    strategy_search_report: str = ""
    final_report: str = ""
    examples_csv: str = ""

    # Output settings
    output_dir: str = "outputs/reports"
    dpi: int = 150
    figsize_wide: tuple = (12, 6)
    figsize_square: tuple = (8, 8)

    # Plot settings
    style: str = "seaborn-v0_8-darkgrid"
    palette: str = "husl"

    # Number of examples to show
    n_worst_examples: int = 10

    def __post_init__(self):
        tn = self.task_name
        if not self.strategy_search_report:
            self.strategy_search_report = f"single_task_strategy_search__{tn}.json"
        if not self.final_report:
            self.final_report = f"single_task_report__{tn}.json"
        if not self.examples_csv:
            self.examples_csv = f"single_task_examples__{tn}.csv"


class VisualizeResultsStep(Step):
    """
    Generate comprehensive visualizations for prediction results.
    Automatically detects regression vs classification from the task name
    and generates the appropriate charts.
    """
    name = "visualize_results"

    def __init__(self, config: VisualizeResultsConfig | None = None):
        self.config = config or VisualizeResultsConfig()

    def _is_classification(self) -> bool:
        tn = self.config.task_name
        return tn == "outcome" or tn.startswith("next_")

    def _get_reports_dir(self, ctx: PipelineContext) -> Path:
        return Path(self.config.output_dir)

    def _load_json_report(self, reports_dir: Path, filename: str) -> Optional[Dict[str, Any]]:
        path = reports_dir / filename
        if not path.exists():
            print(f"Warning: {filename} not found")
            return None
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_csv_report(self, reports_dir: Path, filename: str) -> Optional[pd.DataFrame]:
        path = reports_dir / filename
        if not path.exists():
            print(f"Warning: {filename} not found")
            return None
        return pd.read_csv(path)

    def _setup_plot_style(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        self._plt = plt
        self._sns = sns
        try:
            plt.style.use(self.config.style)
        except Exception:
            plt.style.use('default')
        sns.set_palette(self.config.palette)

    def _save_figure(self, fig, reports_dir: Path, filename: str):
        path = reports_dir / f"{self.config.task_name}__{filename}"
        fig.savefig(path, dpi=self.config.dpi, bbox_inches='tight')
        self._plt.close(fig)
        print(f"  [OK] Saved: {filename}")

    # =========================================================
    # REGRESSION PLOTS
    # =========================================================
    def _plot_strategy_comparison_regression(self, reports_dir: Path, strategy_report: Dict[str, Any]):
        """Plot comparison of all strategies (regression — MAE)."""
        strategies = strategy_report.get('all_strategies', [])
        if not strategies:
            return

        data = []
        for s in strategies:
            score = s.get('primary_score')
            if score is None:
                continue
            data.append({
                'strategy': f"{_p(s['bucketing'])} · {_p(s['encoding'])}",
                'mae_sec': score,
                'mae_days': score / 86400,
            })

        if not data:
            return

        df = pd.DataFrame(data).sort_values('mae_sec')
        n = len(df)
        colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(n)]

        fig, (ax1, ax2) = self._plt.subplots(1, 2, figsize=(14, max(4, n * 0.42)))

        ax1.barh(df['strategy'], df['mae_sec'], color=colors, height=0.6)
        ax1.set_xlabel('MAE (seconds)', fontsize=11)
        ax1.set_title('Strategy Comparison — MAE (seconds)', fontsize=13, fontweight='bold', pad=10)
        ax1.axvline(df['mae_sec'].min(), color='#2ecc71', linestyle='--', linewidth=1, alpha=0.7)
        ax1.tick_params(axis='y', labelsize=10)
        ax1.grid(axis='x', alpha=0.3)

        ax2.barh(df['strategy'], df['mae_days'], color=colors, height=0.6)
        ax2.set_xlabel('MAE (days)', fontsize=11)
        ax2.set_title('Strategy Comparison — MAE (days)', fontsize=13, fontweight='bold', pad=10)
        ax2.axvline(df['mae_days'].min(), color='#2ecc71', linestyle='--', linewidth=1, alpha=0.7)
        ax2.tick_params(axis='y', labelsize=10)
        ax2.grid(axis='x', alpha=0.3)

        self._plt.tight_layout()
        self._save_figure(fig, reports_dir, 'strategy_comparison.png')

    def _plot_predicted_vs_actual(self, reports_dir: Path, examples_df: pd.DataFrame):
        if 'y_true' not in examples_df.columns or 'y_pred' not in examples_df.columns:
            return

        y_true_days = examples_df['y_true'] / 86400
        y_pred_days = examples_df['y_pred'] / 86400

        fig, (ax1, ax2) = self._plt.subplots(1, 2, figsize=(14, 6))

        ax1.scatter(y_true_days, y_pred_days, alpha=0.5, s=30, color='#3498db')
        max_val = max(y_true_days.max(), y_pred_days.max())
        ax1.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect prediction')
        ax1.set_xlabel('Actual Remaining Time (days)', fontsize=12)
        ax1.set_ylabel('Predicted Remaining Time (days)', fontsize=12)
        ax1.set_title('Predicted vs Actual', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2.scatter(y_true_days, y_pred_days, alpha=0.5, s=30, color='#9b59b6')
        ax2.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect prediction')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel('Actual (days) - log scale', fontsize=12)
        ax2.set_ylabel('Predicted (days) - log scale', fontsize=12)
        ax2.set_title('Predicted vs Actual (Log Scale)', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3, which='both')

        self._plt.tight_layout()
        self._save_figure(fig, reports_dir, 'predicted_vs_actual.png')

    def _plot_error_distribution(self, reports_dir: Path, examples_df: pd.DataFrame):
        if 'abs_error' not in examples_df.columns:
            return

        errors_days = examples_df['abs_error'] / 86400

        fig, (ax1, ax2) = self._plt.subplots(1, 2, figsize=(14, 6))

        ax1.hist(errors_days, bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
        ax1.axvline(errors_days.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {errors_days.mean():.1f}')
        ax1.axvline(errors_days.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {errors_days.median():.1f}')
        ax1.set_xlabel('Absolute Error (days)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Error Distribution', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2.boxplot(errors_days, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='#3498db', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
        ax2.set_ylabel('Absolute Error (days)', fontsize=12)
        ax2.set_title('Error Distribution (Box Plot)', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        self._plt.tight_layout()
        self._save_figure(fig, reports_dir, 'error_distribution.png')

    def _plot_error_by_prefix_len(self, reports_dir: Path, examples_df: pd.DataFrame):
        if 'prefix_len' not in examples_df.columns or 'abs_error' not in examples_df.columns:
            return

        grouped = examples_df.groupby('prefix_len').agg({
            'abs_error': ['mean', 'std', 'count']
        }).reset_index()
        grouped.columns = ['prefix_len', 'mean_error', 'std_error', 'count']
        grouped['mean_error_days'] = grouped['mean_error'] / 86400
        grouped['std_error_days'] = grouped['std_error'] / 86400
        grouped = grouped[grouped['count'] >= 5]

        if len(grouped) == 0:
            return

        fig, (ax1, ax2) = self._plt.subplots(1, 2, figsize=(14, 6))

        ax1.plot(grouped['prefix_len'], grouped['mean_error_days'], marker='o',
                linewidth=2, markersize=8, color='#3498db')
        ax1.fill_between(grouped['prefix_len'],
                         grouped['mean_error_days'] - grouped['std_error_days'],
                         grouped['mean_error_days'] + grouped['std_error_days'],
                         alpha=0.3, color='#3498db')
        ax1.set_xlabel('Prefix Length', fontsize=12)
        ax1.set_ylabel('Mean Absolute Error (days)', fontsize=12)
        ax1.set_title('Error by Prefix Length', fontsize=14, fontweight='bold')
        ax1.grid(alpha=0.3)

        ax2.bar(grouped['prefix_len'], grouped['count'], color='#2ecc71', alpha=0.7)
        ax2.set_xlabel('Prefix Length', fontsize=12)
        ax2.set_ylabel('Number of Test Samples', fontsize=12)
        ax2.set_title('Sample Distribution by Prefix Length', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        self._plt.tight_layout()
        self._save_figure(fig, reports_dir, 'error_by_prefix_length.png')

    def _plot_worst_predictions(self, reports_dir: Path, examples_df: pd.DataFrame):
        if 'abs_error' not in examples_df.columns:
            return

        worst = examples_df.nlargest(self.config.n_worst_examples, 'abs_error').copy()
        worst['error_days'] = worst['abs_error'] / 86400
        worst['y_true_days'] = worst['y_true'] / 86400
        worst['y_pred_days'] = worst['y_pred'] / 86400

        fig, ax = self._plt.subplots(figsize=(12, 8))

        x = np.arange(len(worst))
        width = 0.35

        ax.bar(x - width/2, worst['y_true_days'], width, label='Actual', color='#2ecc71', alpha=0.7)
        ax.bar(x + width/2, worst['y_pred_days'], width, label='Predicted', color='#e74c3c', alpha=0.7)

        ax.set_ylabel('Remaining Time (days)', fontsize=12)
        ax.set_title(f'Top {self.config.n_worst_examples} Worst Predictions', fontsize=14, fontweight='bold')
        ax.set_xticks(x)

        labels = []
        for idx, row in worst.iterrows():
            case_id = str(row.get('case_id', ''))[:10]
            prefix_len = int(row.get('prefix_len', 0))
            labels.append(f"{case_id}\n(len={prefix_len})")

        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        self._plt.tight_layout()
        self._save_figure(fig, reports_dir, 'worst_predictions.png')

    def _plot_residuals(self, reports_dir: Path, examples_df: pd.DataFrame):
        if 'y_true' not in examples_df.columns or 'y_pred' not in examples_df.columns:
            return

        y_true_days = examples_df['y_true'] / 86400
        y_pred_days = examples_df['y_pred'] / 86400
        residuals = y_true_days - y_pred_days

        fig, (ax1, ax2) = self._plt.subplots(1, 2, figsize=(14, 6))

        ax1.scatter(y_pred_days, residuals, alpha=0.5, s=30, color='#3498db')
        ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax1.set_xlabel('Predicted Remaining Time (days)', fontsize=12)
        ax1.set_ylabel('Residuals (days)', fontsize=12)
        ax1.set_title('Residual Plot', fontsize=14, fontweight='bold')
        ax1.grid(alpha=0.3)

        ax2.hist(residuals, bins=50, color='#9b59b6', alpha=0.7, edgecolor='black')
        ax2.axvline(residuals.mean(), color='blue', linestyle='--', linewidth=2,
                   label=f'Mean: {residuals.mean():.1f}')
        ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
        ax2.set_xlabel('Residuals (days)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Residual Distribution', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)

        self._plt.tight_layout()
        self._save_figure(fig, reports_dir, 'residuals.png')

    # =========================================================
    # CLASSIFICATION PLOTS
    # =========================================================
    def _plot_strategy_comparison_classification(self, reports_dir: Path, strategy_report: Dict[str, Any]):
        """Plot comparison of all strategies (classification — F1 macro)."""
        strategies = strategy_report.get('all_strategies', [])
        if not strategies:
            return

        data = []
        for s in strategies:
            score = s.get('primary_score')
            if score is None:
                continue
            data.append({
                'strategy': f"{_p(s['bucketing'])} · {_p(s['encoding'])}",
                'f1_macro': score,
            })

        if not data:
            return

        df = pd.DataFrame(data).sort_values('f1_macro', ascending=True)

        n = len(df)
        fig, ax = self._plt.subplots(figsize=(10, max(4, n * 0.42)))

        colors = ['#2ecc71' if v == df['f1_macro'].max() else '#3498db' for v in df['f1_macro']]
        bars = ax.barh(df['strategy'], df['f1_macro'], color=colors, height=0.6)
        ax.set_xlabel('F1 Score (Macro)', fontsize=11)
        ax.set_title('Strategy Comparison — F1 Macro', fontsize=13, fontweight='bold', pad=10)
        ax.axvline(df['f1_macro'].max(), color='#2ecc71', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_xlim(0, 1.08)
        ax.tick_params(axis='y', labelsize=10)
        ax.grid(axis='x', alpha=0.3)

        for i, (idx, row) in enumerate(df.iterrows()):
            ax.text(row['f1_macro'] + 0.01, i, f"{row['f1_macro']:.3f}", va='center', fontsize=9)

        self._plt.tight_layout()
        self._save_figure(fig, reports_dir, 'strategy_comparison.png')

    def _plot_confusion_matrix(self, reports_dir: Path, examples_df: pd.DataFrame):
        """Plot confusion matrix for classification."""
        if 'y_true' not in examples_df.columns or 'y_pred' not in examples_df.columns:
            return

        from sklearn.metrics import confusion_matrix

        y_true = examples_df['y_true'].astype(str)
        y_pred = examples_df['y_pred'].astype(str)
        labels = sorted(set(y_true) | set(y_pred))

        cm = confusion_matrix(y_true, y_pred, labels=labels)

        fig, ax = self._plt.subplots(figsize=(max(8, len(labels) * 0.6), max(6, len(labels) * 0.5)))

        self._sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        self._plt.xticks(rotation=45, ha='right', fontsize=8)
        self._plt.yticks(rotation=0, fontsize=8)

        self._plt.tight_layout()
        self._save_figure(fig, reports_dir, 'confusion_matrix.png')

    def _plot_per_class_f1(self, reports_dir: Path, examples_df: pd.DataFrame):
        """Plot per-class F1 scores."""
        if 'y_true' not in examples_df.columns or 'y_pred' not in examples_df.columns:
            return

        from sklearn.metrics import classification_report

        y_true = examples_df['y_true'].astype(str)
        y_pred = examples_df['y_pred'].astype(str)

        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        classes = []
        f1_scores = []
        supports = []
        for cls, metrics in report.items():
            if cls in ('accuracy', 'macro avg', 'weighted avg'):
                continue
            classes.append(cls)
            f1_scores.append(metrics['f1-score'])
            supports.append(metrics['support'])

        if not classes:
            return

        df = pd.DataFrame({'class': classes, 'f1': f1_scores, 'support': supports})
        df = df.sort_values('f1', ascending=True)

        fig, (ax1, ax2) = self._plt.subplots(1, 2, figsize=(16, max(6, len(df) * 0.4)))

        ax1.barh(df['class'], df['f1'], color='#3498db', alpha=0.8)
        ax1.set_xlabel('F1 Score', fontsize=12)
        ax1.set_title('Per-Class F1 Scores', fontsize=14, fontweight='bold')
        ax1.set_xlim(0, 1)
        ax1.grid(axis='x', alpha=0.3)
        for i, (idx, row) in enumerate(df.iterrows()):
            ax1.text(row['f1'] + 0.01, i, f"{row['f1']:.3f}", va='center', fontsize=9)

        ax2.barh(df['class'], df['support'], color='#2ecc71', alpha=0.8)
        ax2.set_xlabel('Number of Samples', fontsize=12)
        ax2.set_title('Class Distribution (Test Set)', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)

        self._plt.tight_layout()
        self._save_figure(fig, reports_dir, 'per_class_f1.png')

    def _plot_accuracy_by_prefix_len(self, reports_dir: Path, examples_df: pd.DataFrame):
        """Plot accuracy by prefix length for classification."""
        if 'prefix_len' not in examples_df.columns:
            return
        if 'y_true' not in examples_df.columns or 'y_pred' not in examples_df.columns:
            return

        examples_df = examples_df.copy()
        examples_df['correct'] = (examples_df['y_true'].astype(str) == examples_df['y_pred'].astype(str)).astype(int)

        grouped = examples_df.groupby('prefix_len').agg(
            accuracy=('correct', 'mean'),
            count=('correct', 'count'),
        ).reset_index()
        grouped = grouped[grouped['count'] >= 5]

        if len(grouped) == 0:
            return

        fig, (ax1, ax2) = self._plt.subplots(1, 2, figsize=(14, 6))

        ax1.plot(grouped['prefix_len'], grouped['accuracy'], marker='o',
                linewidth=2, markersize=8, color='#3498db')
        ax1.set_xlabel('Prefix Length', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Accuracy by Prefix Length', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 1)
        ax1.grid(alpha=0.3)

        ax2.bar(grouped['prefix_len'], grouped['count'], color='#2ecc71', alpha=0.7)
        ax2.set_xlabel('Prefix Length', fontsize=12)
        ax2.set_ylabel('Number of Test Samples', fontsize=12)
        ax2.set_title('Sample Distribution by Prefix Length', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        self._plt.tight_layout()
        self._save_figure(fig, reports_dir, 'accuracy_by_prefix_length.png')

    # =========================================================
    # RUN
    # =========================================================
    def run(self, ctx: PipelineContext) -> PipelineContext:
        """Generate all visualizations."""
        print(f"\n{'='*60}")
        print("GENERATING VISUALIZATIONS")
        print(f"{'='*60}")

        reports_dir = self._get_reports_dir(ctx)
        reports_dir.mkdir(parents=True, exist_ok=True)

        self._setup_plot_style()

        print("\nLoading reports...")
        strategy_report = self._load_json_report(reports_dir, self.config.strategy_search_report)
        final_report = self._load_json_report(reports_dir, self.config.final_report)
        examples_df = self._load_csv_report(reports_dir, self.config.examples_csv)

        print("\nGenerating plots...")

        is_clf = self._is_classification()

        if strategy_report:
            if is_clf:
                self._plot_strategy_comparison_classification(reports_dir, strategy_report)
            else:
                self._plot_strategy_comparison_regression(reports_dir, strategy_report)

        if examples_df is not None:
            if is_clf:
                self._plot_confusion_matrix(reports_dir, examples_df)
                self._plot_per_class_f1(reports_dir, examples_df)
                self._plot_accuracy_by_prefix_len(reports_dir, examples_df)
            else:
                self._plot_predicted_vs_actual(reports_dir, examples_df)
                self._plot_error_distribution(reports_dir, examples_df)
                self._plot_error_by_prefix_len(reports_dir, examples_df)
                self._plot_worst_predictions(reports_dir, examples_df)
                self._plot_residuals(reports_dir, examples_df)

        print(f"\n{'='*60}")
        print("[SUCCESS] VISUALIZATION COMPLETE")
        print(f"{'='*60}")
        print(f"All plots saved to: {reports_dir}/")

        return ctx
