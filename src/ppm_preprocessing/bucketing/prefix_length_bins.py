from __future__ import annotations

import pandas as pd

from ppm_preprocessing.bucketing.base import Bucketer


class PrefixLenBinsBucketer(Bucketer):
    """
    Groups prefix lengths into bins:
      bin_size=5 => 1-5 -> bucket 1, 6-10 -> bucket 2, ...
    bucket_id is 1-based.
    """
    name = "prefix_len_bins"

    def __init__(
        self,
        bin_size: int = 5,
        prefix_len_col: str = "prefix_len",
        max_len: int | None = None,
        clamp_lower: int = 1,
    ):
        if bin_size < 1:
            raise ValueError("bin_size must be >= 1")
        self.bin_size = int(bin_size)
        self.prefix_len_col = prefix_len_col
        self.max_len = int(max_len) if max_len is not None else None
        self.clamp_lower = int(clamp_lower)

    def assign(self, prefixes: pd.DataFrame) -> pd.Series:
        if self.prefix_len_col not in prefixes.columns:
            raise ValueError(f"PrefixLenBinsBucketer requires column '{self.prefix_len_col}'.")

        pl = prefixes[self.prefix_len_col].astype(int)

        max_len = self.max_len
        if max_len is None:
            max_len = int(pl.max()) if len(pl) else self.clamp_lower

        pl = pl.clip(lower=self.clamp_lower, upper=max_len)

        bucket_id = ((pl - self.clamp_lower) // self.bin_size + 1).astype("int32")
        return pd.Series(bucket_id.values, index=prefixes.index, dtype="int32")
