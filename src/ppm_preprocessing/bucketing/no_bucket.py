from __future__ import annotations

import pandas as pd

from ppm_preprocessing.bucketing.base import Bucketer


class NoBucketer(Bucketer):
    name = "no_bucket"

    def assign(self, prefixes: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=prefixes.index, dtype="int32")
