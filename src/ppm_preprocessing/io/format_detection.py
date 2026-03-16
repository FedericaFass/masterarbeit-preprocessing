"""File format detection utilities for event log files."""

from __future__ import annotations
from pathlib import Path


def detect_format(file_path: str) -> str:
    """
    Detect file format from file extension.

    Args:
        file_path: Path to the file

    Returns:
        Format string: 'csv', 'xes', or 'parquet'

    Raises:
        ValueError: If file format is not supported
    """
    path = Path(file_path).resolve()

    # Handle .xes.gz specifically
    if path.name.lower().endswith('.xes.gz'):
        return 'xes'

    suffix = path.suffix.lower()

    if suffix == '.csv':
        return 'csv'
    elif suffix in {'.xes', '.gz'}:
        return 'xes'
    elif suffix == '.parquet':
        return 'parquet'
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            f"Supported formats: .csv, .xes, .xes.gz, .parquet"
        )
