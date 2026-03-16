#!/usr/bin/env python3
"""
CLI to generate visualizations for prediction results.

Usage:
    python src/ppm_preprocessing/cli/run_visualize.py                    # defaults to remaining_time
    python src/ppm_preprocessing/cli/run_visualize.py next_activity
"""
import sys
from ppm_preprocessing.steps.visualize_results import VisualizeResultsStep, VisualizeResultsConfig
from ppm_preprocessing.domain.context import PipelineContext


def main():
    task_name = sys.argv[1] if len(sys.argv) > 1 else "remaining_time"

    print("=" * 60)
    print(f"GENERATING VISUALIZATIONS: {task_name}")
    print("=" * 60)

    config = VisualizeResultsConfig(
        task_name=task_name,
        output_dir="outputs/reports",
        dpi=150,
        figsize_wide=(12, 6),
        figsize_square=(8, 8),
        style="seaborn-v0_8-darkgrid",
        palette="husl",
        n_worst_examples=10,
    )

    step = VisualizeResultsStep(config)

    ctx = PipelineContext(
        input_path="outputs/reports",
        task=task_name,
    )

    ctx = step.run(ctx)

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"\nPlots saved to: {config.output_dir}/")


if __name__ == "__main__":
    main()
