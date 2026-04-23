"""
Embedding encoder using pre-trained sentence-transformers.

Embeds activity names into dense vectors, then aggregates per prefix
using mean pooling + last-event embedding.  Categorical attributes are
also embedded using the same sentence-transformer model.  Numeric features
from the prefix (temporal, case attributes) are concatenated alongside.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import Encoder, EncodedDataset

UNK_TOKEN = "<UNK>"

# Module-level cache so the model is loaded once per process, not once per fit() call.
_ST_MODEL_CACHE: dict = {}
# Cache computed embedding vectors: (model_name, text) -> np.ndarray
_ST_VECTOR_CACHE: dict = {}


def _get_st_model(model_name: str):
    if model_name not in _ST_MODEL_CACHE:
        from sentence_transformers import SentenceTransformer
        _ST_MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _ST_MODEL_CACHE[model_name]


def _encode_texts(model, model_name: str, texts: list) -> "np.ndarray":
    """Encode texts using cached vectors where possible."""
    import numpy as np
    result = [None] * len(texts)
    to_encode_idx = []
    to_encode_txt = []
    for i, t in enumerate(texts):
        key = (model_name, t)
        if key in _ST_VECTOR_CACHE:
            result[i] = _ST_VECTOR_CACHE[key]
        else:
            to_encode_idx.append(i)
            to_encode_txt.append(t)
    if to_encode_txt:
        vecs = model.encode(to_encode_txt, show_progress_bar=False, convert_to_numpy=True)
        for j, (idx, txt) in enumerate(zip(to_encode_idx, to_encode_txt)):
            v = vecs[j].astype(np.float32)
            _ST_VECTOR_CACHE[(model_name, txt)] = v
            result[idx] = v
    return np.array(result, dtype=np.float32)


@dataclass
class EmbeddingConfig:
    model_name: str = "all-MiniLM-L6-v2"

    prefix_col: str = "prefix_activities"
    prefix_len_col: str = "prefix_len"
    include_prefix_len: bool = True

    label_col: str = "label_remaining_time_sec"
    label_is_numeric: bool = True

    include_numeric_features: bool = True
    include_categorical_features: bool = True
    ignore_cols: Optional[List[str]] = None

    feature_prefix: str = "feat__"

    # PCA compression: reduce embedding vectors from 384d to this many dims after fitting.
    # Keeps all columns and all semantic information (~92 % variance retained at 64d).
    # Set to None to disable.  Dramatically reduces the feature-matrix size on large datasets.
    compress_dim: Optional[int] = 64


class EmbeddingEncoder(Encoder):
    """
    Sentence-transformer embedding encoder.

    Feature vector per prefix row:
        [ mean_activity_emb | last_activity_emb | cat_col_1_emb | ... | prefix_len | numeric_features ]

    The sentence-transformer model is loaded once during ``fit()`` and
    used to pre-compute embeddings for every unique activity string and
    every unique categorical value.
    ``transform()`` is then a pure lookup + aggregation step (fast).
    """
    name = "embedding"

    def __init__(self, config: EmbeddingConfig | None = None):
        self.config = config or EmbeddingConfig()

        self.activity_embeddings_: Dict[str, np.ndarray] = {}
        self.unk_embedding_: Optional[np.ndarray] = None
        self.emb_dim_: int = 0
        self.numeric_cols_: List[str] = []
        self.categorical_cols_: List[str] = []
        # {col_name: {category_string: embedding_vector}}
        self.categorical_embeddings_: Dict[str, Dict[str, np.ndarray]] = {}
        self.categorical_unk_embeddings_: Dict[str, np.ndarray] = {}
        self.feature_names_: List[str] = []
        self.pca_: Optional[Any] = None  # fitted PCA when compress_dim is set

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _default_ignore(self) -> List[str]:
        c = self.config
        base = [
            c.prefix_col,
            "prefix_end_time",
            "case_id",
            "prefix_row_id",
            "bucket_id",
            c.prefix_len_col,
            c.label_col,
            "label_outcome",
            "label_next_activity",
            "label_remaining_time_sec",
            "label_remaining_time_log1p",
        ]
        if c.ignore_cols:
            base += list(c.ignore_cols)
        return list(dict.fromkeys(base))

    @staticmethod
    def _looks_like_dates(series: "pd.Series") -> bool:
        """Return True if the column values look like date/timestamp strings."""
        sample = series.dropna().astype(str).head(20)
        if len(sample) == 0:
            return False
        try:
            # format='mixed' (pandas >= 2.0) suppresses the dateutil-fallback UserWarning.
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            return parsed.notna().mean() >= 0.8
        except (TypeError, ValueError):
            try:
                parsed = pd.to_datetime(sample, errors="coerce")
                return parsed.notna().mean() >= 0.8
            except Exception:
                return False

    def _discover_columns(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        ignore = set(self._default_ignore())
        num_cols: List[str] = []
        cat_cols: List[str] = []
        for col in df.columns:
            if col in ignore:
                continue
            if pd.api.types.is_bool_dtype(df[col]):
                cat_cols.append(col)
            elif pd.api.types.is_numeric_dtype(df[col]):
                num_cols.append(col)
            else:
                # Skip columns that would produce giant feature matrices with no
                # semantic embedding signal. Other encoders (LastState, Aggregation,
                # IndexPayload) are NOT affected — they keep every column.
                n_unique = df[col].nunique()
                if self._looks_like_dates(df[col]):
                    # Date strings: "2015-03-15", "T08:00:00+01:00", etc.
                    continue
                if n_unique > 500:
                    # High-cardinality IDs / free text: no embedding signal,
                    # would create a matrix too large for available RAM.
                    continue
                cat_cols.append(col)
        return num_cols, cat_cols

    # ------------------------------------------------------------------
    # fit / transform
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "EmbeddingEncoder":
        c = self.config

        if c.prefix_col not in df.columns:
            raise ValueError(f"Missing column '{c.prefix_col}'")
        if c.label_col not in df.columns:
            raise ValueError(f"Missing column '{c.label_col}'")

        # --- load sentence-transformer model (cached per process) ---
        model = _get_st_model(c.model_name)

        # --- collect unique activities ---
        vocab: set[str] = set()
        for seq in df[c.prefix_col]:
            if isinstance(seq, list):
                vocab.update(str(a) for a in seq)

        activities = sorted(vocab)
        if not activities:
            raise ValueError("No activities found in prefix_activities.")

        # --- encode all activities (vectors cached globally across fit() calls) ---
        vectors = _encode_texts(model, c.model_name, activities)
        self.emb_dim_ = int(vectors.shape[1])

        self.activity_embeddings_ = {
            act: vectors[i].astype(np.float32)
            for i, act in enumerate(activities)
        }
        self.unk_embedding_ = vectors.mean(axis=0).astype(np.float32)

        print(
            f"[EmbeddingEncoder] Encoded {len(activities)} unique activities "
            f"-> {self.emb_dim_}d embeddings ({c.model_name})"
        )

        # --- discover numeric + categorical columns ---
        num_cols, cat_cols = self._discover_columns(df)

        if c.include_numeric_features:
            self.numeric_cols_ = num_cols
        else:
            self.numeric_cols_ = []

        if c.include_categorical_features:
            self.categorical_cols_ = cat_cols
        else:
            self.categorical_cols_ = []

        # --- embed categorical values ---
        if self.categorical_cols_:
            # Collect all unique values across all categorical columns
            all_unique_values: set[str] = set()
            col_values: Dict[str, List[str]] = {}
            for col in self.categorical_cols_:
                unique_vals = sorted(df[col].astype(str).fillna(UNK_TOKEN).unique().tolist())
                col_values[col] = unique_vals
                all_unique_values.update(unique_vals)

            # Encode all unique categorical values (cached globally across fit() calls)
            all_values_list = sorted(all_unique_values)
            if all_values_list:
                cat_vectors = _encode_texts(model, c.model_name, all_values_list)
                value_to_emb = {
                    val: cat_vectors[i].astype(np.float32)
                    for i, val in enumerate(all_values_list)
                }

                # Build per-column lookup tables
                for col in self.categorical_cols_:
                    self.categorical_embeddings_[col] = {
                        val: value_to_emb[val] for val in col_values[col]
                    }
                    # UNK embedding = mean of that column's value embeddings
                    col_vecs = np.array([value_to_emb[val] for val in col_values[col]], dtype=np.float32)
                    self.categorical_unk_embeddings_[col] = col_vecs.mean(axis=0).astype(np.float32)

                print(
                    f"[EmbeddingEncoder] Embedded {len(all_values_list)} unique categorical values "
                    f"across {len(self.categorical_cols_)} columns"
                )

        # --- PCA compression (optional) ---
        # Fit PCA on ALL embedding vectors (activities + categorical values + UNK),
        # then replace stored embeddings with their compressed versions.
        # This shrinks the feature matrix from (n, k*384) to (n, k*compress_dim),
        # e.g. 17 cat cols: 1.45 GB → ~230 MB at 64d, with ~92% variance retained.
        if c.compress_dim is not None and c.compress_dim < self.emb_dim_:
            from sklearn.decomposition import PCA

            # Collect every embedding vector we have
            _all_vecs: List[np.ndarray] = list(self.activity_embeddings_.values())
            _all_vecs.append(self.unk_embedding_)
            for _col_embs in self.categorical_embeddings_.values():
                _all_vecs.extend(_col_embs.values())
            for _unk_vec in self.categorical_unk_embeddings_.values():
                _all_vecs.append(_unk_vec)

            _pca_matrix = np.array(_all_vecs, dtype=np.float32)
            n_components = min(c.compress_dim, _pca_matrix.shape[0] - 1, _pca_matrix.shape[1])

            self.pca_ = PCA(n_components=n_components, random_state=42)
            self.pca_.fit(_pca_matrix)

            def _compress(v: np.ndarray) -> np.ndarray:
                return self.pca_.transform(v.reshape(1, -1))[0].astype(np.float32)

            self.activity_embeddings_ = {a: _compress(v) for a, v in self.activity_embeddings_.items()}
            self.unk_embedding_ = _compress(self.unk_embedding_)
            for col in self.categorical_cols_:
                self.categorical_embeddings_[col] = {
                    val: _compress(v) for val, v in self.categorical_embeddings_[col].items()
                }
                self.categorical_unk_embeddings_[col] = _compress(self.categorical_unk_embeddings_[col])

            self.emb_dim_ = n_components
            print(
                f"[EmbeddingEncoder] PCA compressed: 384d -> {n_components}d "
                f"({self.pca_.explained_variance_ratio_.sum():.1%} variance retained)"
            )

        # --- build feature names ---
        feat: List[str] = []
        feat += [f"{c.feature_prefix}emb_mean_{i}" for i in range(self.emb_dim_)]
        feat += [f"{c.feature_prefix}emb_last_{i}" for i in range(self.emb_dim_)]
        for col in self.categorical_cols_:
            feat += [f"{c.feature_prefix}cat_emb__{col}_{i}" for i in range(self.emb_dim_)]
        if c.include_prefix_len and c.prefix_len_col in df.columns:
            feat.append(f"{c.feature_prefix}{c.prefix_len_col}")
        for col in self.numeric_cols_:
            feat.append(f"{c.feature_prefix}num__{col}")
        self.feature_names_ = feat

        return self

    def transform(self, df: pd.DataFrame) -> EncodedDataset:
        c = self.config
        n = len(df)
        dim = len(self.feature_names_)

        X = np.zeros((n, dim), dtype=np.float32)

        unk = self.unk_embedding_
        emb_lookup = self.activity_embeddings_
        emb_dim = self.emb_dim_

        # Activity embeddings (mean + last)
        seqs = df[c.prefix_col].to_numpy()
        mean_embs = np.empty((n, emb_dim), dtype=np.float32)
        last_embs = np.empty((n, emb_dim), dtype=np.float32)
        for i in range(n):
            seq = seqs[i]
            if not isinstance(seq, list) or len(seq) == 0:
                mean_embs[i] = unk
                last_embs[i] = unk
            else:
                vecs = np.array(
                    [emb_lookup.get(str(a), unk) for a in seq],
                    dtype=np.float32,
                )
                mean_embs[i] = vecs.mean(axis=0)
                last_embs[i] = vecs[-1]
        X[:, :emb_dim] = mean_embs
        X[:, emb_dim: 2 * emb_dim] = last_embs

        # Categorical embeddings — vectorized: one list-comp + single matrix assign per col
        offset = 2 * emb_dim
        for col in self.categorical_cols_:
            col_lookup = self.categorical_embeddings_.get(col, {})
            col_unk = self.categorical_unk_embeddings_.get(col, unk)
            if col in df.columns:
                vals = df[col].astype(str).fillna(UNK_TOKEN).to_numpy()
                col_matrix = np.array(
                    [col_lookup.get(v, col_unk) for v in vals], dtype=np.float32
                )
                X[:, offset: offset + emb_dim] = col_matrix
            else:
                X[:, offset: offset + emb_dim] = col_unk
            offset += emb_dim

        # Prefix length
        if c.include_prefix_len and c.prefix_len_col in df.columns:
            X[:, offset] = pd.to_numeric(df[c.prefix_len_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            offset += 1

        # Numeric features
        for col in self.numeric_cols_:
            if col in df.columns:
                X[:, offset] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            offset += 1

        # Labels
        if c.label_is_numeric:
            y = pd.to_numeric(df[c.label_col], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        else:
            y = df[c.label_col].to_numpy()

        meta: Dict[str, Any] = {
            "encoder": self.name,
            "feature_dim": int(X.shape[1]),
            "feature_names": list(self.feature_names_),
            "label_col": c.label_col,
            "label_is_numeric": c.label_is_numeric,
            "model_name": c.model_name,
            "emb_dim": self.emb_dim_,
            "num_activities": len(self.activity_embeddings_),
            "num_numeric_cols": len(self.numeric_cols_),
            "num_categorical_cols": len(self.categorical_cols_),
        }

        return EncodedDataset(X=X, y=y, meta=meta)
