from __future__ import annotations

import pandas as pd

from ppm_preprocessing.bucketing.base import Bucketer


class LastActivityBucketer(Bucketer):
    """
    Bucket by the last observed activity in the prefix (cf. Teinemaa et al. 2019).

    Cases whose prefix ends with the same activity are grouped together.
    This is effective when process behaviour is strongly activity-dependent
    (e.g. different approval stages lead to very different remaining times).

    A single 'unknown' bucket catches activities not seen during training.
    """
    name = "last_activity"

    def __init__(self, activity_col: str = "prefix_activities"):
        self.activity_col = activity_col
        self.activity_to_bucket_: dict[str, int] | None = None
        self.unknown_bucket_: int = 1

    @staticmethod
    def _last_activity(prefix_activities) -> str:
        if hasattr(prefix_activities, "__len__") and len(prefix_activities) > 0:
            return str(prefix_activities[-1])
        return "__unknown__"

    def fit(self, prefixes_train: pd.DataFrame) -> "LastActivityBucketer":
        if self.activity_col not in prefixes_train.columns:
            raise ValueError(
                f"LastActivityBucketer requires column '{self.activity_col}' in prefix data."
            )
        activities = prefixes_train[self.activity_col].apply(self._last_activity)
        unique = sorted(activities.unique())
        # 1-based bucket IDs; bucket 0 reserved as unknown
        self.activity_to_bucket_ = {a: i + 1 for i, a in enumerate(unique)}
        self.unknown_bucket_ = len(unique) + 1
        return self

    def assign(self, prefixes: pd.DataFrame) -> pd.Series:
        if self.activity_to_bucket_ is None:
            raise RuntimeError(
                "LastActivityBucketer is not fitted. Call fit(train_prefixes) first."
            )
        if self.activity_col not in prefixes.columns:
            raise ValueError(
                f"LastActivityBucketer requires column '{self.activity_col}'."
            )
        activities = prefixes[self.activity_col].apply(self._last_activity)
        bucket_ids = (
            activities
            .map(self.activity_to_bucket_)
            .fillna(self.unknown_bucket_)
            .astype("int32")
        )
        return pd.Series(bucket_ids, index=prefixes.index, dtype="int32")
