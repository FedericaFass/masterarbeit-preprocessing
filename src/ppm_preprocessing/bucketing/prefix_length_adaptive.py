from __future__ import annotations

import math
import pandas as pd

from ppm_preprocessing.bucketing.base import Bucketer


class PrefixLenAdaptiveBucketer(Bucketer):
    """
    Adaptive prefix-length buckets learned from training data.

    Default factors correspond to:
      edges = [ceil(0.2*median), ceil(0.6*median), ceil(1.0*median),
              ceil(1.5*median), ceil(2.0*median), max]
    Edges are inclusive upper bounds.
    Buckets are 1-based.
    """
    name = "prefix_len_adaptive"

    def __init__(
        self,
        prefix_len_col: str = "prefix_len",
        factors=(0.2, 0.6, 1.0, 1.5, 2.0),
        clamp_lower: int = 1,
        min_bucket_width: int = 1,
        max_len: int | None = None,
    ):
        self.prefix_len_col = prefix_len_col
        self.factors = tuple(float(x) for x in factors)
        self.clamp_lower = int(clamp_lower)
        self.min_bucket_width = int(min_bucket_width)
        self.max_len = int(max_len) if max_len is not None else None

        # learned state
        self.edges_: list[int] | None = None
        self.median_: int | None = None
        self.max_: int | None = None

    def fit(self, prefixes_train: pd.DataFrame) -> "PrefixLenAdaptiveBucketer":
        if self.prefix_len_col not in prefixes_train.columns:
            raise ValueError(f"PrefixLenAdaptiveBucketer requires '{self.prefix_len_col}' in train data.")

        pl = prefixes_train[self.prefix_len_col].astype(int)

        median_pl = max(self.clamp_lower, int(pl.median()))
        max_pl = max(median_pl, int(pl.max()))

        if self.max_len is not None:
            max_pl = min(max_pl, self.max_len)

        edges = [int(math.ceil(f * median_pl)) for f in self.factors]
        edges.append(int(max_pl))

        # sanitize edges
        edges = [max(self.clamp_lower, e) for e in edges]
        edges = sorted(set(edges))
        if edges[-1] != max_pl:
            edges.append(max_pl)

        # enforce minimum width
        out = []
        prev = self.clamp_lower - 1
        for e in edges:
            e = max(e, prev + self.min_bucket_width)
            out.append(e)
            prev = e

        self.edges_ = out
        self.median_ = median_pl
        self.max_ = max_pl
        return self

    def assign(self, prefixes: pd.DataFrame) -> pd.Series:
        if self.edges_ is None:
            raise RuntimeError("PrefixLenAdaptiveBucketer is not fitted. Call fit(train_prefixes) first.")

        if self.prefix_len_col not in prefixes.columns:
            raise ValueError(f"PrefixLenAdaptiveBucketer requires '{self.prefix_len_col}'.")

        pl = prefixes[self.prefix_len_col].astype(int)

        upper = self.edges_[-1]
        pl = pl.clip(lower=self.clamp_lower, upper=upper)

        # searchsorted: returns index of first edge >= pl
        idx = pd.Index(self.edges_).searchsorted(pl, side="left")
        bucket_id = (idx + 1).astype("int32")
        return pd.Series(bucket_id, index=prefixes.index, dtype="int32")
