from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict
import pandas as pd

@dataclass
class CanonicalLog:
    df: pd.DataFrame
    meta: Dict[str, Any] = field(default_factory=dict)
