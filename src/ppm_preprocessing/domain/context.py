from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import pandas as pd

from .canonical_log import CanonicalLog

@dataclass
class PipelineContext:
    input_path: str
    task: str  # z.B. "next_activity", "remaining_time", "outcome"
    artifacts: Dict[str, Any] = field(default_factory=dict)

    raw_df: Optional[pd.DataFrame] = None
    log: Optional[CanonicalLog] = None
