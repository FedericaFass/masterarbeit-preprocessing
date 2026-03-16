# ppm_preprocessing/encoders/index_latest_payload.py
"""
Index Latest-Payload encoding:
  Combines index-based encoding for activities with latest-payload for dynamic features.

  Feature vector structure:
  g_i = (s1_i, ..., su_i, a_i1, a_i2, ..., a_im, d1_im, ..., dr_im, label_i)

  Where:
  - s_i: static features (case attributes)
  - a_ij: activity at position j (for all j=1 to m)
  - d_im: dynamic features from the last event only (position m)
  - label_i: target label
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import Encoder, EncodedDataset

UNK_TOKEN = "<UNK>"


class SimpleTargetEncoder:
   
    def __init__(self, smoothing: float = 1.0):
        self.smoothing = smoothing
        self.global_mean_ = 0.0
        self.encoding_map_ = {}

    def fit(self, series: pd.Series, y: pd.Series) -> "SimpleTargetEncoder":
        """Fit the encoder on a categorical series and target."""
        # Global mean (fallback for unseen categories)
        self.global_mean_ = float(y.mean())

        # Use groupby for efficient category aggregation (much faster than loop)
        df = pd.DataFrame({'cat': series.astype(str), 'y': y})
        grouped = df.groupby('cat', observed=True)['y']

        cat_means = grouped.mean()
        cat_counts = grouped.count()

        # Compute smoothed encoding (m-estimation) for all categories at once
        smoothed = (cat_counts * cat_means + self.smoothing * self.global_mean_) / (cat_counts + self.smoothing)
        self.encoding_map_ = smoothed.to_dict()

        return self

    def transform(self, series: pd.Series) -> np.ndarray:
        """Transform a categorical series to target-encoded values."""
        return np.array([
            self.encoding_map_.get(str(val), self.global_mean_)
            for val in series
        ], dtype=np.float32)

    def transform_value(self, val) -> float:
        """Transform a single value (optimized for performance)."""
        return self.encoding_map_.get(str(val), self.global_mean_)

    def fit_transform(self, series: pd.Series, y: pd.Series) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(series, y)
        return self.transform(series)


@dataclass
class IndexLatestPayloadConfig:
    """
    Index Latest-Payload encoding configuration.

    Combines:
    - Static features (case attributes)
    - Activity sequence at all positions (control flow)
    - Dynamic features from last event only (latest state)
    """
    label_col: str = "label_remaining_time_sec"
    label_is_numeric: bool = True

    # Static features (case attributes)
    include_case_attributes: bool = True
    case_prefix: str = "case__"

    # Activity sequence (control flow)
    include_activity_sequence: bool = True

    # Latest payload (dynamic features from last event)
    include_latest_payload: bool = True
    event_last_prefix: str = "event_last__"

    # Temporal and engineered features (CRITICAL for remaining time!)
    include_temporal_features: bool = True
    temporal_prefix: str = "feat_"  # Single underscore to match PrefixExtractionStep output

    # Additional features
    include_prefix_len: bool = True
    prefix_len_col: str = "prefix_len"

    ignore_cols: Optional[List[str]] = None
    extra_numeric_cols: Optional[List[str]] = None
    extra_categorical_cols: Optional[List[str]] = None

    max_categories_per_col: int = 50
    min_freq_per_category: int = 50

    # Categorical encoding strategy
    categorical_encoding: str = "target"  # Options: "onehot", "target", "label"
    target_encoding_smoothing: float = 1.0  # Smoothing for target encoding (higher = more regularization)

    # Activity sequence encoding strategy
    activity_encoding: str = "target"  # Options: "onehot_per_position", "target", "counts"

    feature_prefix: str = "feat__"


class IndexLatestPayloadEncoder(Encoder):
    """
    Index Latest-Payload encoder.

    Creates features combining:
    1. Static case attributes
    2. Activity at each position (preserves sequence)
    3. Dynamic attributes from last event only
    """
    name = "index_latest_payload"

    def __init__(self, config: IndexLatestPayloadConfig | None = None):
        self.config = config or IndexLatestPayloadConfig()

        # Vocabularies for categorical features (one-hot encoding)
        self.cat2id: Dict[str, Dict[str, int]] = {}  # {col_name: {category: id}}
        self.id2cat: Dict[str, Dict[int, str]] = {}  # {col_name: {id: category}}

        # Target encoders for categorical features (target encoding)
        self.target_encoders_: Dict[str, SimpleTargetEncoder] = {}  # {col_name: encoder}

        # Activity target encoder (for activity sequences)
        self.activity_target_encoder_: Optional[SimpleTargetEncoder] = None

        self.label2id: Dict[str, int] = {}
        self.id2label: Dict[int, str] = {}

        # Discovered columns
        self.case_numeric_cols_: List[str] = []
        self.case_categorical_cols_: List[str] = []
        self.last_event_numeric_cols_: List[str] = []
        self.last_event_categorical_cols_: List[str] = []
        self.temporal_numeric_cols_: List[str] = []  # Temporal features (feat__)
        self.temporal_categorical_cols_: List[str] = []

        self.feature_names_: List[str] = []
        self.max_prefix_len_: int = 0

    def _default_ignore(self) -> List[str]:
        c = self.config
        base = [
            "case_id",
            "prefix_row_id",
            "prefix_end_time",
            "prefix_activities",
            "bucket_id",
            "event_id",
            "timestamp",
            "lifecycle",
        ]
        base.append(c.label_col)
        base.append(c.prefix_len_col)

        # other common labels
        base += ["label_outcome", "label_next_activity", "label_remaining_time_sec", "label_remaining_time_log1p"]

        if c.ignore_cols:
            base += list(c.ignore_cols)

        return list(dict.fromkeys(base))

    def _is_case_col(self, col: str) -> bool:
        return str(col).startswith(self.config.case_prefix)

    def _is_last_event_col(self, col: str) -> bool:
        return str(col).startswith(self.config.event_last_prefix)

    def _is_temporal_col(self, col: str) -> bool:
        return str(col).startswith(self.config.temporal_prefix)

    def _infer_columns(self, df: pd.DataFrame) -> None:
        """
        Discover case and last-event attribute columns.
        """
        c = self.config
        ignore = set(self._default_ignore())

        # Case attributes (static)
        case_candidates = [col for col in df.columns
                          if col not in ignore and self._is_case_col(str(col))]

        # Last event attributes (dynamic)
        last_event_candidates = [col for col in df.columns
                                if col not in ignore and self._is_last_event_col(str(col))]

        # Temporal/engineered features (feat__)
        temporal_candidates = [col for col in df.columns
                              if col not in ignore and self._is_temporal_col(str(col))]

        # Classify case attributes
        for col in case_candidates:
            if pd.api.types.is_numeric_dtype(df[col]):
                self.case_numeric_cols_.append(col)
            else:
                self.case_categorical_cols_.append(col)

        # Classify last event attributes
        for col in last_event_candidates:
            if pd.api.types.is_numeric_dtype(df[col]):
                self.last_event_numeric_cols_.append(col)
            else:
                self.last_event_categorical_cols_.append(col)

        # Classify temporal features
        for col in temporal_candidates:
            if pd.api.types.is_numeric_dtype(df[col]):
                self.temporal_numeric_cols_.append(col)
            else:
                self.temporal_categorical_cols_.append(col)

    @staticmethod
    def _parse_activities(activities_val) -> List[str]:
        """
        Parse activities from various formats (list, JSON string, CSV string).

        Returns a list of activity strings.
        """
        import json

        if isinstance(activities_val, list):
            # Activities stored as list (normal case during pipeline)
            return [str(a) for a in activities_val]
        elif isinstance(activities_val, str) and activities_val:
            # Activities stored as string (e.g., from CSV or JSON)
            try:
                parsed = json.loads(activities_val)
                if isinstance(parsed, list):
                    return [str(a) for a in parsed]
                else:
                    return [str(activities_val)]
            except (json.JSONDecodeError, ValueError):
                # Not JSON, try splitting by common separators
                if "," in activities_val:
                    return [a.strip() for a in activities_val.split(",")]
                elif ";" in activities_val:
                    return [a.strip() for a in activities_val.split(";")]
                else:
                    return [activities_val.strip()]
        return []

    def _build_vocab(self, series: pd.Series, col_name: str) -> None:
        """Build vocabulary for a categorical column."""
        c = self.config

        # Handle empty series
        if len(series) == 0 or series.isna().all():
            # Create vocab with only UNK_TOKEN
            vocab = {UNK_TOKEN: 0}
            self.cat2id[col_name] = vocab
            self.id2cat[col_name] = {0: UNK_TOKEN}
            return

        # Count frequencies
        value_counts = series.value_counts()

        # Keep top K values with min frequency
        top_values = value_counts[
            value_counts >= c.min_freq_per_category
        ].head(c.max_categories_per_col)

        # Build vocabulary - always include UNK_TOKEN
        vocab = {UNK_TOKEN: 0}
        for idx, val in enumerate(top_values.index, start=1):
            vocab[str(val)] = idx

        self.cat2id[col_name] = vocab
        self.id2cat[col_name] = {v: k for k, v in vocab.items()}

    def _build_target_encoder(self, series: pd.Series, y: pd.Series, col_name: str) -> None:
        """Build target encoder for a categorical column."""
        c = self.config

        # Handle empty series
        if len(series) == 0 or series.isna().all():
            # Create a dummy encoder that returns global mean
            encoder = SimpleTargetEncoder(smoothing=c.target_encoding_smoothing)
            encoder.global_mean_ = float(y.mean()) if len(y) > 0 else 0.0
            encoder.encoding_map_ = {}
            self.target_encoders_[col_name] = encoder
            return

        # Build target encoder
        encoder = SimpleTargetEncoder(smoothing=c.target_encoding_smoothing)
        encoder.fit(series, y)
        self.target_encoders_[col_name] = encoder

    def fit(self, df: pd.DataFrame) -> IndexLatestPayloadEncoder:
        """
        Fit the encoder on training data.

        Args:
            df: DataFrame with prefix samples (one row per prefix)
        """
        c = self.config

        # Extract y values for target encoding
        if c.label_col not in df.columns:
            raise ValueError(f"Missing column '{c.label_col}'")
        y = df[c.label_col]

        # Target encoding requires numeric y — fall back to onehot for classification
        if not c.label_is_numeric:
            c.categorical_encoding = "onehot"
            c.activity_encoding = "onehot_per_position"

        # Discover columns
        self._infer_columns(df)

        # Determine max prefix length
        if c.prefix_len_col in df.columns:
            self.max_prefix_len_ = int(df[c.prefix_len_col].max())
        else:
            self.max_prefix_len_ = 10  # default fallback

        # Build encoders for case categorical attributes
        for col in self.case_categorical_cols_:
            if col in df.columns:
                if c.categorical_encoding == "target":
                    self._build_target_encoder(df[col], y, col)
                else:  # "onehot" or "label"
                    self._build_vocab(df[col], col)

        # Build encoders for last event categorical attributes
        for col in self.last_event_categorical_cols_:
            if col in df.columns:
                if c.categorical_encoding == "target":
                    self._build_target_encoder(df[col], y, col)
                else:  # "onehot" or "label"
                    self._build_vocab(df[col], col)

        # Build encoders for temporal categorical features (e.g., hour, weekday)
        for col in self.temporal_categorical_cols_:
            if col in df.columns:
                if c.categorical_encoding == "target":
                    self._build_target_encoder(df[col], y, col)
                else:  # "onehot" or "label"
                    self._build_vocab(df[col], col)

        # Build encoder for activities (from prefix_activities column)
        if "prefix_activities" in df.columns:
            # Extract all activities from the activity sequences
            all_activities = []
            all_activities_y = []

            # Convert y to numpy array for fast indexing
            y_values = y.values
            for idx, activities_val in enumerate(df["prefix_activities"].dropna().values):
                activities = self._parse_activities(activities_val)
                y_val = y_values[idx]

                for act in activities:
                    all_activities.append(act)
                    all_activities_y.append(y_val)

            if all_activities:
                if c.activity_encoding == "target":
                    # Build target encoder for activities
                    self.activity_target_encoder_ = SimpleTargetEncoder(smoothing=c.target_encoding_smoothing)
                    self.activity_target_encoder_.fit(pd.Series(all_activities), pd.Series(all_activities_y))
                else:  # "onehot_per_position" or "counts"
                    self._build_vocab(pd.Series(all_activities), "activity")

        # Build label vocabulary if classification
        if not c.label_is_numeric and c.label_col in df.columns:
            self._build_vocab(df[c.label_col], c.label_col)
            self.label2id = self.cat2id[c.label_col]
            self.id2label = self.id2cat[c.label_col]

        # Build feature names
        self._build_feature_names()

        return self

    def _build_feature_names(self) -> None:
        """Build feature names for the encoded dataset."""
        c = self.config
        names = []

        # Prefix length
        if c.include_prefix_len:
            names.append(f"{c.feature_prefix}prefix_len")

        # Case attributes (static)
        for col in self.case_numeric_cols_:
            names.append(f"{c.feature_prefix}{col}")

        for col in self.case_categorical_cols_:
            if c.categorical_encoding == "target":
                # Target encoding: 1 feature per column
                names.append(f"{c.feature_prefix}target__{col}")
            elif c.categorical_encoding == "label":
                # Label encoding: 1 feature per column
                names.append(f"{c.feature_prefix}label__{col}")
            else:  # "onehot"
                vocab = self.cat2id.get(col, {})
                for cat_name in sorted(vocab.keys()):
                    names.append(f"{c.feature_prefix}cat__{col}__{cat_name}")

        # Activity sequence
        if c.include_activity_sequence:
            if c.activity_encoding == "target":
                # Target encoding: encode each position's activity with target encoding
                for i in range(1, self.max_prefix_len_ + 1):
                    names.append(f"{c.feature_prefix}activity_{i}__target")
            elif c.activity_encoding == "counts":
                # Count encoding: one feature per activity type (position-independent)
                activity_vocab = self.cat2id.get("activity", {})
                for act_name in sorted(activity_vocab.keys()):
                    names.append(f"{c.feature_prefix}activity_count__{act_name}")
            else:  # "onehot_per_position"
                activity_vocab = self.cat2id.get("activity", {})
                # Only add activity features if vocabulary exists
                if len(activity_vocab) > 0:
                    for i in range(1, self.max_prefix_len_ + 1):
                        for act_name in sorted(activity_vocab.keys()):
                            names.append(f"{c.feature_prefix}activity_{i}__{act_name}")

        # Temporal features (CRITICAL for remaining time prediction!)
        if c.include_temporal_features:
            # Numeric temporal features
            for col in self.temporal_numeric_cols_:
                names.append(f"{c.feature_prefix}{col}")

            # Categorical temporal features (e.g., hour, weekday)
            for col in self.temporal_categorical_cols_:
                if c.categorical_encoding == "target":
                    names.append(f"{c.feature_prefix}target__{col}")
                elif c.categorical_encoding == "label":
                    names.append(f"{c.feature_prefix}label__{col}")
                else:  # "onehot"
                    vocab = self.cat2id.get(col, {})
                    for cat_name in sorted(vocab.keys()):
                        names.append(f"{c.feature_prefix}cat__{col}__{cat_name}")

        # Latest payload (dynamic features from last event)
        if c.include_latest_payload:
            # Numeric attributes from last event
            for col in self.last_event_numeric_cols_:
                names.append(f"{c.feature_prefix}{col}")

            # Categorical attributes from last event
            for col in self.last_event_categorical_cols_:
                if c.categorical_encoding == "target":
                    names.append(f"{c.feature_prefix}target__{col}")
                elif c.categorical_encoding == "label":
                    names.append(f"{c.feature_prefix}label__{col}")
                else:  # "onehot"
                    vocab = self.cat2id.get(col, {})
                    for cat_name in sorted(vocab.keys()):
                        names.append(f"{c.feature_prefix}cat__{col}__{cat_name}")

        self.feature_names_ = names

    def _encode_row(self, row: pd.Series) -> np.ndarray:
        """Encode a single row to feature vector."""
        c = self.config
        features = []

        # Prefix length
        if c.include_prefix_len:
            prefix_len = float(row.get(c.prefix_len_col, 0))
            features.append(prefix_len)
        else:
            prefix_len = int(row.get(c.prefix_len_col, self.max_prefix_len_))

        # Case attributes (static)
        for col in self.case_numeric_cols_:
            val = row.get(col, 0.0)
            features.append(float(val) if pd.notna(val) else 0.0)

        for col in self.case_categorical_cols_:
            if c.categorical_encoding == "target":
                # Target encoding: single encoded value
                val = row.get(col)
                encoder = self.target_encoders_.get(col)
                if encoder:
                    encoded_val = encoder.transform_value(val)
                else:
                    encoded_val = 0.0
                features.append(float(encoded_val))
            elif c.categorical_encoding == "label":
                # Label encoding: single integer
                val = str(row.get(col, UNK_TOKEN))
                vocab = self.cat2id.get(col, {UNK_TOKEN: 0})
                cat_id = vocab.get(val, vocab.get(UNK_TOKEN, 0))
                features.append(float(cat_id))
            else:  # "onehot"
                val = str(row.get(col, UNK_TOKEN))
                vocab = self.cat2id.get(col, {UNK_TOKEN: 0})
                # Ensure UNK_TOKEN exists in vocab
                if UNK_TOKEN not in vocab:
                    vocab = {UNK_TOKEN: 0}
                cat_id = vocab.get(val, vocab[UNK_TOKEN])
                # One-hot encoding
                one_hot = [0.0] * len(vocab)
                if cat_id < len(one_hot):  # Safety check
                    one_hot[cat_id] = 1.0
                features.extend(one_hot)

        # Activity sequence
        if c.include_activity_sequence:
            # Parse prefix_activities using the same logic as fit()
            activities_val = row.get("prefix_activities", [])
            activities = self._parse_activities(activities_val)

            if c.activity_encoding == "target":
                # Target encoding: encode each position with target encoder
                if self.activity_target_encoder_:
                    for i in range(1, self.max_prefix_len_ + 1):
                        if i <= len(activities):
                            act = activities[i - 1]
                            encoded_val = self.activity_target_encoder_.transform_value(act)
                        else:
                            # Padding - use global mean
                            encoded_val = self.activity_target_encoder_.global_mean_
                        features.append(float(encoded_val))

            elif c.activity_encoding == "counts":
                # Count encoding: count each activity type (position-independent)
                activity_vocab = self.cat2id.get("activity", {})
                if len(activity_vocab) > 0:
                    activity_counts = {act: 0 for act in sorted(activity_vocab.keys())}
                    for act in activities:
                        act_str = str(act)
                        if act_str in activity_counts:
                            activity_counts[act_str] += 1
                    features.extend([float(activity_counts[act]) for act in sorted(activity_vocab.keys())])

            else:  # "onehot_per_position"
                activity_vocab = self.cat2id.get("activity", {})

                # Skip activity encoding if vocabulary is empty
                if len(activity_vocab) > 0:
                    # Ensure UNK_TOKEN exists in vocab
                    if UNK_TOKEN not in activity_vocab:
                        activity_vocab = {UNK_TOKEN: 0}

                    for i in range(1, self.max_prefix_len_ + 1):
                        if i <= len(activities):
                            # This position has an activity
                            act = activities[i - 1]
                            cat_id = activity_vocab.get(str(act), activity_vocab[UNK_TOKEN])
                        else:
                            # Padding - use UNK
                            cat_id = activity_vocab[UNK_TOKEN]

                        # One-hot encoding
                        one_hot = [0.0] * len(activity_vocab)
                        if cat_id < len(one_hot):  # Safety check
                            one_hot[cat_id] = 1.0
                        features.extend(one_hot)

        # Temporal features (CRITICAL for remaining time!)
        if c.include_temporal_features:
            # Numeric temporal features (elapsed time, time since last, etc.)
            for col in self.temporal_numeric_cols_:
                val = row.get(col, 0.0)
                features.append(float(val) if pd.notna(val) else 0.0)

            # Categorical temporal features (hour, weekday, etc.)
            for col in self.temporal_categorical_cols_:
                if c.categorical_encoding == "target":
                    val = row.get(col)
                    encoder = self.target_encoders_.get(col)
                    if encoder:
                        encoded_val = encoder.transform_value(val)
                    else:
                        encoded_val = 0.0
                    features.append(float(encoded_val))
                elif c.categorical_encoding == "label":
                    val = str(row.get(col, UNK_TOKEN))
                    vocab = self.cat2id.get(col, {UNK_TOKEN: 0})
                    cat_id = vocab.get(val, vocab.get(UNK_TOKEN, 0))
                    features.append(float(cat_id))
                else:  # "onehot"
                    val = str(row.get(col, UNK_TOKEN))
                    vocab = self.cat2id.get(col, {UNK_TOKEN: 0})
                    # Ensure UNK_TOKEN exists in vocab
                    if UNK_TOKEN not in vocab:
                        vocab = {UNK_TOKEN: 0}
                    cat_id = vocab.get(val, vocab[UNK_TOKEN])
                    # One-hot encoding
                    one_hot = [0.0] * len(vocab)
                    if cat_id < len(one_hot):  # Safety check
                        one_hot[cat_id] = 1.0
                    features.extend(one_hot)

        # Latest payload (dynamic features from last event)
        if c.include_latest_payload:
            # Numeric attributes
            for col in self.last_event_numeric_cols_:
                val = row.get(col, 0.0)
                features.append(float(val) if pd.notna(val) else 0.0)

            # Categorical attributes
            for col in self.last_event_categorical_cols_:
                if c.categorical_encoding == "target":
                    val = row.get(col)
                    encoder = self.target_encoders_.get(col)
                    if encoder:
                        encoded_val = encoder.transform_value(val)
                    else:
                        encoded_val = 0.0
                    features.append(float(encoded_val))
                elif c.categorical_encoding == "label":
                    val = str(row.get(col, UNK_TOKEN))
                    vocab = self.cat2id.get(col, {UNK_TOKEN: 0})
                    cat_id = vocab.get(val, vocab.get(UNK_TOKEN, 0))
                    features.append(float(cat_id))
                else:  # "onehot"
                    val = str(row.get(col, UNK_TOKEN))
                    vocab = self.cat2id.get(col, {UNK_TOKEN: 0})
                    # Ensure UNK_TOKEN exists in vocab
                    if UNK_TOKEN not in vocab:
                        vocab = {UNK_TOKEN: 0}
                    cat_id = vocab.get(val, vocab[UNK_TOKEN])
                    # One-hot encoding
                    one_hot = [0.0] * len(vocab)
                    if cat_id < len(one_hot):  # Safety check
                        one_hot[cat_id] = 1.0
                    features.extend(one_hot)

        return np.array(features, dtype=np.float32)

    def transform(self, df: pd.DataFrame) -> EncodedDataset:
        """Transform data to encoded feature matrix."""
        c = self.config

        # Encode all rows - use to_dict('records') instead of iterrows() for 10-100x speedup
        X_list = []
        for row_dict in df.to_dict('records'):
            # Convert dict to Series for compatibility with _encode_row
            row = pd.Series(row_dict)
            X_list.append(self._encode_row(row))

        X = np.vstack(X_list) if X_list else np.array([]).reshape(0, len(self.feature_names_))

        # Extract labels
        if c.label_is_numeric:
            y = df[c.label_col].to_numpy(dtype=np.float32)
        else:
            y_raw = df[c.label_col].astype(str)
            y = np.array([self.label2id.get(v, 0) for v in y_raw], dtype=np.int32)

        # Extract metadata
        case_id = df["case_id"].astype(str).to_numpy() if "case_id" in df.columns else np.array([])
        row_idx = df["prefix_row_id"].to_numpy() if "prefix_row_id" in df.columns else np.arange(len(df))

        return EncodedDataset(
            X=X,
            y=y,
            meta={
                "case_id": case_id,
                "row_idx": row_idx,
                "feature_names": self.feature_names_,
                "label_col": self.config.label_col,
                "label_is_numeric": self.config.label_is_numeric,
            }
        )

    def fit_transform(self, df: pd.DataFrame) -> EncodedDataset:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)
