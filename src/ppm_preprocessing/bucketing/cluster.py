from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans

from ppm_preprocessing.bucketing.base import Bucketer
from ppm_preprocessing.encoders.aggregation import AggregationEncoder, AggregationConfig


@dataclass
class ClusterBucketConfig:
    """Configuration for cluster-based bucketing."""
    n_clusters: int = 3
    algorithm: str = "kmeans"       # "kmeans" or "minibatch_kmeans"
    random_state: int = 42

    # Internal encoder settings (used to produce feature vectors for clustering)
    label_col: str = "label_remaining_time_sec"


class ClusterBucketer(Bucketer):
    """
    Cluster-based bucketer (cf. Teinemaa et al. 2019, Section 4.2.4).

    Encodes prefix traces using AggregationEncoder with only activity frequency
    counts and prefix length (no extra numeric/categorical features) to keep
    clustering focused on control-flow similarity. Applies KMeans clustering.
    One model is trained per cluster at training time.
    At runtime, a new prefix is encoded and assigned to the nearest cluster.

    The internal encoder is fitted on training data only (during fit()),
    avoiding any leakage.
    """
    name = "cluster"

    def __init__(self, config: ClusterBucketConfig | None = None):
        self.config = config or ClusterBucketConfig()

        # Fitted state (populated by fit())
        self.encoder_: Optional[AggregationEncoder] = None
        self.cluster_model_: Optional[KMeans | MiniBatchKMeans] = None
        self.n_clusters_actual_: int = 0

    def fit(self, prefixes_train: pd.DataFrame) -> "ClusterBucketer":
        c = self.config

        # Build internal encoder: activity counts + prefix_len only
        # No extra features — keeps clustering focused on control-flow
        enc = AggregationEncoder(AggregationConfig(
            label_col=c.label_col,
            label_is_numeric=True,
            include_prefix_len=True,
            include_extra_features=False,
        ))
        enc.fit(prefixes_train)
        encoded = enc.transform(prefixes_train)
        X = encoded.X

        # Clamp n_clusters to available samples
        n_clusters = min(c.n_clusters, len(X))
        if n_clusters < 2:
            n_clusters = 1

        # Fit clustering model
        if c.algorithm == "minibatch_kmeans":
            model = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=c.random_state,
                n_init="auto",
            )
        else:
            model = KMeans(
                n_clusters=n_clusters,
                random_state=c.random_state,
                n_init="auto",
            )
        model.fit(X)

        self.encoder_ = enc
        self.cluster_model_ = model
        self.n_clusters_actual_ = n_clusters
        return self

    def assign(self, prefixes: pd.DataFrame) -> pd.Series:
        if self.encoder_ is None or self.cluster_model_ is None:
            raise RuntimeError(
                "ClusterBucketer is not fitted. Call fit(train_prefixes) first."
            )

        encoded = self.encoder_.transform(prefixes)
        X = encoded.X

        # Predict cluster assignment (0-based) → convert to 1-based bucket IDs
        cluster_ids = self.cluster_model_.predict(X)
        bucket_ids = cluster_ids + 1

        return pd.Series(bucket_ids, index=prefixes.index, dtype="int32")
