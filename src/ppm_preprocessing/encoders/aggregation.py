# ppm_preprocessing/encoders/aggregation.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import Encoder, EncodedDataset

UNK_TOKEN = "<UNK>"


@dataclass
class AggregationConfig:
    # control-flow sequence input
    prefix_col: str = "prefix_activities"

    # optional extra: prefix length
    include_prefix_len: bool = True
    prefix_len_col: str = "prefix_len"

    # label (DEFAULTS adjusted for remaining_time)
    # IMPORTANT: for remaining_time you want numeric y
    label_col: str = "label_remaining_time_sec"
    label_is_numeric: bool = True

    # include all other available features (case__/event_last__/feat_* ...)
    include_extra_features: bool = True

    # ignore cols beyond prefix/label (can be used to drop high-cardinality IDs etc.)
    ignore_cols: Optional[List[str]] = None

    # explicit override (optional)
    extra_numeric_cols: Optional[List[str]] = None
    extra_categorical_cols: Optional[List[str]] = None

    # One-hot control
    max_categories_per_col: int = 20
    min_freq_per_category: int = 50
    # hard cap on total categorical columns kept (prevents OOM on wide datasets)
    max_categorical_cols: int = 30

    feature_prefix: str = "feat__"


class AggregationEncoder(Encoder):
    """
    Aggregation encoding (cf. Paper 4.3.3):
      - activity frequency counts across the prefix (order ignored)
      - numeric attributes: "as is"
      - categorical attributes: one-hot encoding (top-K + min freq, else UNK)
      - optional prefix_len

    Fixes:
      - prefix_len excluded from extra_numeric_cols to avoid duplicates
      - ignore high-cardinality IDs via ignore_cols
      - remaining_time defaults: numeric label
    """
    name = "aggregation"

    def __init__(self, config: AggregationConfig | None = None):
        self.config = config or AggregationConfig()

        # activity vocab
        self.act2id: Dict[str, int] = {}
        self.id2act: Dict[int, str] = {}

        # categorical vocabs per column
        self.cat2id: Dict[str, Dict[str, int]] = {}
        self.id2cat: Dict[str, Dict[int, str]] = {}

        # label vocab (only used if label_is_numeric=False)
        self.label2id: Dict[str, int] = {}
        self.id2label: Dict[int, str] = {}

        # selected extra columns
        self.numeric_cols_: List[str] = []
        self.categorical_cols_: List[str] = []

        # final feature names
        self.feature_names_: List[str] = []

    def _default_ignore(self) -> List[str]:
        c = self.config
        base = [
            c.prefix_col,
            "prefix_end_time",
            "case_id",
            "prefix_row_id",
            "bucket_id",
            # avoid duplicate prefix_len (explicit + auto numeric)
            c.prefix_len_col,
        ]

        # label col excluded from features
        base.append(c.label_col)

        # other common labels to ignore (safe even if not present)
        base += ["label_outcome", "label_next_activity", "label_remaining_time_sec", "label_remaining_time_log1p"]

        # user-defined ignores
        if c.ignore_cols:
            base += list(c.ignore_cols)

        return list(dict.fromkeys(base))

    def _infer_extra_cols(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        c = self.config
        ignore = set(self._default_ignore())
        candidates = [col for col in df.columns if col not in ignore]

        # explicit overrides
        if c.extra_numeric_cols is not None or c.extra_categorical_cols is not None:
            num = [x for x in (c.extra_numeric_cols or []) if x in df.columns and x not in ignore]
            cat = [x for x in (c.extra_categorical_cols or []) if x in df.columns and x not in ignore]
            return num, cat

        num_cols: List[str] = []
        cat_cols: List[str] = []
        for col in candidates:
            s = df[col]
            # treat bool as categorical
            if pd.api.types.is_bool_dtype(s):
                cat_cols.append(col)
            elif pd.api.types.is_numeric_dtype(s):
                num_cols.append(col)
            else:
                cat_cols.append(col)

        return num_cols, cat_cols

    def fit(self, df: pd.DataFrame) -> "AggregationEncoder":
        c = self.config

        if c.prefix_col not in df.columns:
            raise ValueError(f"Missing column '{c.prefix_col}'")
        if c.label_col not in df.columns:
            raise ValueError(f"Missing column '{c.label_col}'")

        # --- activity vocab (control-flow) ---
        vocab = set()
        for seq in df[c.prefix_col]:
            if isinstance(seq, list):
                vocab.update([str(a) for a in seq])
            else:
                raise TypeError(f"Expected list in {c.prefix_col}, got {type(seq)}")

        ordered_acts = [UNK_TOKEN] + sorted(vocab)
        self.act2id = {a: i for i, a in enumerate(ordered_acts)}
        self.id2act = {i: a for a, i in self.act2id.items()}

        # --- label vocab (only if classification) ---
        if not c.label_is_numeric:
            labels = df[c.label_col].astype(str).tolist()
            ordered_labels = [UNK_TOKEN] + sorted(set(labels))
            self.label2id = {lab: i for i, lab in enumerate(ordered_labels)}
            self.id2label = {i: lab for lab, i in self.label2id.items()}

        # --- extra features ---
        self.numeric_cols_, self.categorical_cols_ = [], []
        self.cat2id, self.id2cat = {}, {}

        if c.include_extra_features:
            num_cols, cat_cols = self._infer_extra_cols(df)
            self.numeric_cols_ = num_cols
            # cap total categorical columns to prevent OOM on wide datasets
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

        # --- feature names ---
        feat_names: List[str] = []

        # activity counts
        feat_names += [f"{c.feature_prefix}act_count__{a}" for a in ordered_acts]

        # optional prefix_len
        if c.include_prefix_len and c.prefix_len_col in df.columns:
            feat_names.append(f"{c.feature_prefix}{c.prefix_len_col}")

        # numeric extras
        for col in self.numeric_cols_:
            feat_names.append(f"{c.feature_prefix}num__{col}")

        # categorical extras (one-hot)
        for col in self.categorical_cols_:
            cats = self.id2cat[col]
            for idx in range(len(cats)):
                feat_names.append(f"{c.feature_prefix}cat__{col}__{cats[idx]}")

        self.feature_names_ = feat_names
        return self

    def transform(self, df: pd.DataFrame) -> EncodedDataset:
        c = self.config
        if not self.act2id:
            raise RuntimeError("Encoder not fitted. Call fit() first.")

        n = len(df)
        unk_act = self.act2id[UNK_TOKEN]
        num_acts = len(self.act2id)

        X_parts: List[np.ndarray] = []

        # --- control-flow: frequency counts ---
        if c.prefix_col not in df.columns:
            raise ValueError(f"Missing required column '{c.prefix_col}' in transform DataFrame")

        X_act = np.zeros((n, num_acts), dtype=np.float32)
        for i, seq in enumerate(df[c.prefix_col]):
            if not isinstance(seq, list):
                raise TypeError(
                    f"Expected list in {c.prefix_col} at row {i}, got {type(seq)}. "
                    f"This is consistent with fit() behavior."
                )
            for a in seq:
                j = self.act2id.get(str(a), unk_act)
                X_act[i, j] += 1.0
        X_parts.append(X_act)

        # --- prefix_len ---
        if c.include_prefix_len and c.prefix_len_col in df.columns:
            v = (
                pd.to_numeric(df[c.prefix_len_col], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=np.float32)
                .reshape(-1, 1)
            )
            X_parts.append(v)

        # --- numeric extras ---
        for col in self.numeric_cols_:
            if col not in df.columns:
                X_parts.append(np.zeros((n, 1), dtype=np.float32))
                continue
            v = (
                pd.to_numeric(df[col], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=np.float32)
                .reshape(-1, 1)
            )
            X_parts.append(v)

        # --- categorical extras ---
        for col in self.categorical_cols_:
            vocab = self.cat2id.get(col, {UNK_TOKEN: 0})
            unk = vocab.get(UNK_TOKEN, 0)
            k = len(vocab)
            X_cat = np.zeros((n, k), dtype=np.float32)

            if col in df.columns:
                s = df[col].astype(str).fillna(UNK_TOKEN).to_numpy()
                for i in range(n):
                    idx = vocab.get(s[i], unk)
                    X_cat[i, idx] = 1.0
            else:
                X_cat[:, unk] = 1.0

            X_parts.append(X_cat)

        X = np.hstack(X_parts).astype(np.float32) if X_parts else np.zeros((n, 0), dtype=np.float32)

        # --- y ---
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
            "include_extra_features": bool(c.include_extra_features),
        }
        return EncodedDataset(X=X, y=y, meta=meta)
