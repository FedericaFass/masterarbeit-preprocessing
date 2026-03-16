from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .base import Step
from ppm_preprocessing.domain.context import PipelineContext

@dataclass
class CaseSplitConfig:
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    random_state: int = 42
    temporal_split: bool = False  # sort by case start time instead of random shuffle

class CaseSplitStep(Step):
    name = "case_split"

    def __init__(self, config: CaseSplitConfig | None = None):
        self.config = config or CaseSplitConfig()

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.log is None:
            raise RuntimeError("Need canonical log in ctx.log (run load/normalize/clean first).")

        if self.config.temporal_split:
            # Sort cases by their first event timestamp (oldest → newest)
            df = ctx.log.df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            case_start = df.groupby("case_id")["timestamp"].min().sort_values()
            cases = case_start.index.astype(str).tolist()
            method = "temporal"
        else:
            cases = ctx.log.df["case_id"].astype(str).unique()
            rng = np.random.default_rng(self.config.random_state)
            rng.shuffle(cases)
            method = "random"

        n = len(cases)
        n_train = int(n * self.config.train_ratio)
        n_val = int(n * self.config.val_ratio)

        train_cases = set(cases[:n_train])
        val_cases = set(cases[n_train:n_train + n_val])
        test_cases = set(cases[n_train + n_val:])

        ctx.artifacts["case_splits"] = {
            "train": train_cases,
            "val": val_cases,
            "test": test_cases,
        }
        ctx.artifacts["case_splits_method"] = method
        return ctx
