# ppm_preprocessing/encoders/last_state.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import Encoder, EncodedDataset

UNK_TOKEN = "<UNK>"


@dataclass
class LastStateConfig:
    """
    Last State encoding (paper 4.3.2):
      Uses only the last snapshot of data attributes for a prefix.
      (No sequence/history features like prefix_activities.)

    Numeric -> as is
    Categorical -> one-hot (top-K + min_freq, rest -> UNK)

    DEFAULTS adjusted for remaining_time:
      - label_remaining_time_sec
      - numeric y
    """
    label_col: str = "label_remaining_time_sec"
    label_is_numeric: bool = True

    include_prefix_len: bool = True
    prefix_len_col: str = "prefix_len"

    snapshot_prefixes: Tuple[str, ...] = ("case__", "event_last__", "feat_")
    include_extra_features: bool = True

    ignore_cols: Optional[List[str]] = None
    extra_numeric_cols: Optional[List[str]] = None
    extra_categorical_cols: Optional[List[str]] = None

    max_categories_per_col: int = 20
    min_freq_per_category: int = 50
    max_categorical_cols: int = 30

    feature_prefix: str = "feat__"


class LastStateEncoder(Encoder):
    name = "last_state"

    def __init__(self, config: LastStateConfig | None = None):
        self.config = config or LastStateConfig()

        self.cat2id: Dict[str, Dict[str, int]] = {}
        self.id2cat: Dict[str, Dict[int, str]] = {}

        self.label2id: Dict[str, int] = {}
        self.id2label: Dict[int, str] = {}

        self.numeric_cols_: List[str] = []
        self.categorical_cols_: List[str] = []
        self.feature_names_: List[str] = []

    def _default_ignore(self) -> List[str]:
        c = self.config
        base = [
            "case_id",
            "prefix_row_id",
            "prefix_end_time",
            "prefix_activities",
            "bucket_id",
        ]
        base.append(c.label_col)

        # other common labels to ignore
        base += ["label_outcome", "label_next_activity", "label_remaining_time_sec", "label_remaining_time_log1p"]

        # prefix_len handled separately
        base.append(c.prefix_len_col)

        if c.ignore_cols:
            base += list(c.ignore_cols)

        return list(dict.fromkeys(base))

    def _is_snapshot_col(self, col: str) -> bool:
        sp = self.config.snapshot_prefixes
        return any(str(col).startswith(p) for p in sp)

    def _infer_extra_cols(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        c = self.config
        ignore = set(self._default_ignore())

        snapshot_candidates = [
            col for col in df.columns
            if col not in ignore and self._is_snapshot_col(str(col))
        ]

        if c.extra_numeric_cols is not None or c.extra_categorical_cols is not None:
            num = [x for x in (c.extra_numeric_cols or []) if x in df.columns and x not in ignore]
            cat = [x for x in (c.extra_categorical_cols or []) if x in df.columns and x not in ignore]
            return num, cat

        num_cols: List[str] = []
        cat_cols: List[str] = []
        for col in snapshot_candidates:
            s = df[col]
            if pd.api.types.is_bool_dtype(s):
                cat_cols.append(col)
            elif pd.api.types.is_numeric_dtype(s):
                num_cols.append(col)
            else:
                cat_cols.append(col)

        return num_cols, cat_cols

    def fit(self, df: pd.DataFrame) -> "LastStateEncoder":
        c = self.config
        if c.label_col not in df.columns:
            raise ValueError(f"Missing column '{c.label_col}'")

        # label vocab only for classification
        if not c.label_is_numeric:
            labels = df[c.label_col].astype(str).tolist()
            ordered_labels = [UNK_TOKEN] + sorted(set(labels))
            self.label2id = {lab: i for i, lab in enumerate(ordered_labels)}
            self.id2label = {i: lab for lab, i in self.label2id.items()}

        self.numeric_cols_, self.categorical_cols_ = [], []
        self.cat2id, self.id2cat = {}, {}

        if c.include_extra_features:
            num_cols, cat_cols = self._infer_extra_cols(df)
            self.numeric_cols_ = num_cols
            self.categorical_cols_ = cat_cols[: int(c.max_categorical_cols)]

            for col in self.categorical_cols_:
                s = df[col].astype(str).fillna(UNK_TOKEN)
                vc = s.value_counts(dropna=False)

                keep = vc[vc >= int(c.min_freq_per_category)].index.tolist()
                keep = keep[: int(c.max_categories_per_col)]

                ordered = [UNK_TOKEN] + [k for k in keep if k != UNK_TOKEN]
                m = {cat: i for i, cat in enumerate(ordered)}
                self.cat2id[col] = m
                self.id2cat[col] = {i: cat for cat, i in m.items()}

        feat_names: List[str] = []

        if c.include_prefix_len and c.prefix_len_col in df.columns:
            feat_names.append(f"{c.feature_prefix}{c.prefix_len_col}")

        for col in self.numeric_cols_:
            feat_names.append(f"{c.feature_prefix}num__{col}")

        for col in self.categorical_cols_:
            cats = self.id2cat[col]
            for idx in range(len(cats)):
                feat_names.append(f"{c.feature_prefix}cat__{col}__{cats[idx]}")

        self.feature_names_ = feat_names
        print(f"    [LastState.fit] numeric={len(self.numeric_cols_)}, "
              f"categorical={len(self.categorical_cols_)}, "
              f"total_features={len(feat_names)}", flush=True)
        if self.categorical_cols_:
            print(f"    [LastState.fit] cat cols: {self.categorical_cols_[:10]}{'...' if len(self.categorical_cols_) > 10 else ''}", flush=True)
        return self

    def transform(self, df: pd.DataFrame) -> EncodedDataset:
        c = self.config
        n = len(df)
        total_onehot_cols = sum(len(self.cat2id.get(col, {})) for col in self.categorical_cols_)
        print(f"    [LastState.transform] n={n}, num_cols={len(self.numeric_cols_)}, "
              f"cat_cols={len(self.categorical_cols_)}, total_onehot={total_onehot_cols}, "
              f"matrix_size=({n} x {1 + len(self.numeric_cols_) + total_onehot_cols})",
              flush=True)
        parts: List[np.ndarray] = []

        if c.include_prefix_len and c.prefix_len_col in df.columns:
            v = (
                pd.to_numeric(df[c.prefix_len_col], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=np.float32)
                .reshape(-1, 1)
            )
            parts.append(v)

        for col in self.numeric_cols_:
            if col not in df.columns:
                parts.append(np.zeros((n, 1), dtype=np.float32))
                continue
            v = (
                pd.to_numeric(df[col], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=np.float32)
                .reshape(-1, 1)
            )
            parts.append(v)

        for col in self.categorical_cols_:
            vocab = self.cat2id.get(col, {UNK_TOKEN: 0})
            unk = vocab.get(UNK_TOKEN, 0)
            k = len(vocab)

            X_cat = np.zeros((n, k), dtype=np.float32)
            if col in df.columns:
                s = df[col].astype(str).fillna(UNK_TOKEN).to_numpy()
                for i in range(n):
                    cid = vocab.get(s[i], unk)
                    X_cat[i, cid] = 1.0
            else:
                X_cat[:, unk] = 1.0

            parts.append(X_cat)

        X = np.hstack(parts).astype(np.float32) if parts else np.zeros((n, 0), dtype=np.float32)

        if c.label_is_numeric:
            y = pd.to_numeric(df[c.label_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        else:
            if not self.label2id:
                raise RuntimeError("Label vocab not fitted, but label_is_numeric=False.")
            unk_lab = self.label2id.get(UNK_TOKEN, 0)
            labs = df[c.label_col].astype(str).to_numpy()
            y = np.array([self.label2id.get(lab, unk_lab) for lab in labs], dtype=np.int32)

        meta: Dict[str, Any] = {
            "encoder": self.name,
            "feature_dim": int(X.shape[1]),
            "label_col": c.label_col,
            "label_is_numeric": bool(c.label_is_numeric),
            "numeric_cols": list(self.numeric_cols_),
            "categorical_cols": list(self.categorical_cols_),
            "feature_names": list(self.feature_names_),
            "max_categories_per_col": int(c.max_categories_per_col),
            "min_freq_per_category": int(c.min_freq_per_category),
            "snapshot_prefixes": list(c.snapshot_prefixes),
            "n_numeric": int(len(self.numeric_cols_)),
            "n_categorical": int(len(self.categorical_cols_)),
        }
        return EncodedDataset(X=X, y=y, meta=meta)
