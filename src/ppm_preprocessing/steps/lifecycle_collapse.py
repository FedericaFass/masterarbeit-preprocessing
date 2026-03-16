from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from ppm_preprocessing.domain.context import PipelineContext
from ppm_preprocessing.steps.base import Step


@dataclass
class LifecycleCollapseConfig:
    # Where to read/write:
    # If use_ctx_log=True -> read ctx.log.df and write back to ctx.log.df
    use_ctx_log: bool = True

    # Optional: work on an artifact dataframe instead (if you really want)
    in_key: Optional[str] = None
    out_key: Optional[str] = None  # if None: overwrite in_key

    case_id_col: str = "case_id"
    activity_col: str = "activity"
    timestamp_col: str = "timestamp"
    lifecycle_col: str = "lifecycle:transition"

    # BPIC2017-style event identifier within a case
    event_id_col: str = "EventID"

    # prefer "complete" over "start" if available
    prefer_lifecycle: str = "complete"

    # optional: if you want the collapsed row to have lifecycle forced to prefer_lifecycle
    normalize_lifecycle_to_prefer: bool = False

    # stable tie-breakers if timestamps equal
    tie_breakers: Tuple[str, ...] = ("_event_index",)

    qc_key: str = "lifecycle_collapse_qc"


class LifecycleCollapseStep(Step):
    """
    Collapse lifecycle rows into one logical event per group.

    Strategy:
      - If EventID exists (and is non-null): group by [case_id, EventID]
      - Else (no EventID): create an occurrence id per case+activity to separate repeated activities
            occurrence increments when (activity changes) OR (lifecycle == 'start') OR (no lifecycle column and activity changes)
        Then group by [case_id, activity, occurrence]
      - Within each group: pick a representative row:
            prefer lifecycle == prefer_lifecycle (e.g. "complete"), otherwise take the latest timestamp.
    """

    name = "lifecycle_collapse"

    def __init__(self, config: LifecycleCollapseConfig | None = None):
        self.config = config or LifecycleCollapseConfig()

    # ---------- helpers ----------
    def _get_df(self, ctx: PipelineContext) -> Tuple[pd.DataFrame, str, str]:
        """
        Returns (df, mode, key_info)
        mode: 'ctx.log.df' or 'ctx.artifacts'
        key_info: artifact key if artifacts-mode
        """
        c = self.config

        if c.use_ctx_log:
            if ctx.log is None:
                raise RuntimeError("ctx.log is None. Run Load/Normalize first.")
            if not hasattr(ctx.log, "df") or ctx.log.df is None:
                raise RuntimeError("ctx.log.df is None.")
            return ctx.log.df, "ctx.log.df", ""

        # artifact mode
        if c.in_key is None:
            raise RuntimeError(
                "LifecycleCollapseStep in artifacts mode requires config.in_key. "
                "In your current pipeline you should set use_ctx_log=True."
            )
        df = ctx.artifacts.get(c.in_key)
        if not isinstance(df, pd.DataFrame):
            raise RuntimeError(f"ctx.artifacts['{c.in_key}'] is not a DataFrame (got {type(df)}).")
        return df, "ctx.artifacts", c.in_key

    def _set_df(self, ctx: PipelineContext, df: pd.DataFrame, mode: str, key: str) -> None:
        c = self.config
        if mode == "ctx.log.df":
            ctx.log.df = df
            return
        # artifacts
        out_key = c.out_key or key
        ctx.artifacts[out_key] = df

    def _ensure_cols(self, df: pd.DataFrame, cols: List[str]) -> None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise RuntimeError(f"LifecycleCollapseStep missing required columns: {missing}")

    def _normalize_ts(self, s: pd.Series) -> pd.Series:
        # Always use UTC-aware timestamps internally
        return pd.to_datetime(s, errors="coerce", utc=True)

    def _pick_grouping(self, df: pd.DataFrame) -> Tuple[List[str], str, Optional[str]]:
        c = self.config
        # Prefer EventID grouping if present and non-empty
        if c.event_id_col in df.columns and df[c.event_id_col].notna().any():
            return [c.case_id_col, c.event_id_col], "case+EventID", None
        # Otherwise we will create an occurrence column
        return [c.case_id_col, c.activity_col, "_occurrence"], "case+activity+occurrence", "_occurrence"

    def _make_occurrence(self, df: pd.DataFrame) -> pd.Series:
        """
        Create an occurrence id per case that separates repeated activities.
        Works best when df is already sorted stably by timestamp and tie-breakers.
        """
        c = self.config
        case_col = c.case_id_col
        act_col = c.activity_col

        # activity change marker within each case
        prev_act = df.groupby(case_col, sort=False)[act_col].shift(1)
        act_changed = (df[act_col] != prev_act).fillna(True)

        if c.lifecycle_col in df.columns:
            lc = df[c.lifecycle_col].astype(str).str.lower().str.strip()
            is_start = lc.eq("start")
            # Start often indicates a new logical event
            new_event = act_changed | is_start
        else:
            new_event = act_changed

        # cumulative sum within case forms occurrence id (1..)
        occ = new_event.groupby(df[case_col], sort=False).cumsum().astype(int)
        return occ

    # ---------- main ----------
    def run(self, ctx: PipelineContext) -> PipelineContext:
        c = self.config
        df, mode, key = self._get_df(ctx)

        if len(df) == 0:
            self._set_df(ctx, df, mode, key)
            ctx.artifacts[c.qc_key] = {
                "enabled": True,
                "mode": mode,
                "in_key": key if mode == "ctx.artifacts" else None,
                "note": "empty input",
                "before_events": 0,
                "after_events": 0,
            }
            return ctx

        self._ensure_cols(df, [c.case_id_col, c.activity_col, c.timestamp_col])

        work = df.copy()

        # normalize timestamps + stable sort
        work["_lc_ts"] = self._normalize_ts(work[c.timestamp_col])
        ts_na = int(work["_lc_ts"].isna().sum())

        sort_cols = [c.case_id_col, "_lc_ts"]
        used_ties = []
        for t in c.tie_breakers:
            if t in work.columns:
                sort_cols.append(t)
                used_ties.append(t)

        work = work.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

        # create occurrence if needed
        group_keys, grouping_method, occ_col = self._pick_grouping(work)
        if occ_col is not None:
            work[occ_col] = self._make_occurrence(work)

        has_lifecycle = c.lifecycle_col in work.columns
        if has_lifecycle:
            lc_norm = work[c.lifecycle_col].astype(str).str.lower().str.strip()
            work["_lc_is_prefer"] = lc_norm.eq(str(c.prefer_lifecycle).lower().strip())
        else:
            work["_lc_is_prefer"] = False

        # To pick representative: sort so that the "best" row is last in group
        # We want prefer_lifecycle=True to win, and latest timestamp to win.
        work = work.sort_values(
            group_keys + ["_lc_is_prefer", "_lc_ts"],
            ascending=[True] * len(group_keys) + [True, True],
            kind="mergesort",
        )

        before_events = int(len(work))
        collapsed = work.groupby(group_keys, as_index=False, sort=False).tail(1)
        after_events = int(len(collapsed))

        # optional normalize lifecycle column on selected rows
        if has_lifecycle and c.normalize_lifecycle_to_prefer:
            collapsed[c.lifecycle_col] = str(c.prefer_lifecycle)

        # cleanup
        collapsed = collapsed.drop(columns=["_lc_ts", "_lc_is_prefer"], errors="ignore")

        lifecycle_top10: Dict[str, int] = {}
        if has_lifecycle:
            vc = collapsed[c.lifecycle_col].astype(str).value_counts().head(10)
            lifecycle_top10 = {str(k): int(v) for k, v in vc.items()}

        self._set_df(ctx, collapsed, mode, key)

        ctx.artifacts[c.qc_key] = {
            "enabled": True,
            "mode": mode,
            "in_key": key if mode == "ctx.artifacts" else None,
            "grouping_method": grouping_method,
            "group_keys": group_keys,
            "created_occurrence": bool(occ_col is not None),
            "prefer_lifecycle": c.prefer_lifecycle,
            "normalize_lifecycle_to_prefer": bool(c.normalize_lifecycle_to_prefer),
            "before_events": before_events,
            "after_events": after_events,
            "collapsed_events": int(before_events - after_events),
            "timestamp_parse_na": ts_na,
            "tie_breakers_used": used_ties,
            "has_lifecycle_col": bool(has_lifecycle),
            "lifecycle_values_top10_after": lifecycle_top10,
        }
        return ctx
