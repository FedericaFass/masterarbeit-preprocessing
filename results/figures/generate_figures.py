#!/usr/bin/env python3
"""Generate thesis figures for ablation, strategy eval, and AutoML eval."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # results/../ = repo root
RESULTS = ROOT / "results"
FIGURES = Path(__file__).parent
FIGURES.mkdir(exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
plt.rcParams.update({
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

LOGS = [
    "DomesticDeclarations",
    "PermitLog",
    "BPIC15_1",
    "BPI_Challenge_2013_closed_problems",
    "issues",
]
LOG_SHORT = {
    "DomesticDeclarations": "DD",
    "PermitLog": "PL",
    "BPIC15_1": "BPIC15",
    "BPI_Challenge_2013_closed_problems": "BPI13",
    "issues": "HD",
}

TASKS = ["next_activity", "remaining_time", "outcome"]
TASK_LABELS = {
    "next_activity": "Next Activity",
    "remaining_time": "Remaining Time",
    "outcome": "Outcome",
}
TASK_YLABELS = {
    "next_activity": "ΔF1-macro",
    "remaining_time": "ΔMAE reduction (days)",
    "outcome": "ΔF1-macro",
}
IS_CLF = {"next_activity": True, "remaining_time": False, "outcome": True}

BUCKETERS = ["no_bucket", "last_activity", "prefix_len_bins", "prefix_len_adaptive", "cluster"]
BUCKETER_LABELS = ["No Bucket", "Last Activity", "Prefix Bins", "Prefix Adaptive", "Cluster"]

ENCODERS = ["last_state", "aggregation", "index_latest_payload", "embedding"]
ENCODER_LABELS = ["Last State", "Aggregation", "Index Latest", "Embedding"]


ABLATION_STAGE_LABELS = [
    "Baseline",
    "+Dedup",
    "+Clean & Sort",
    "+Stable Sort",
    "+Time Features",
    "+Repair TS",
    "+Filter Short",
    "+Norm. Acts",
    "+Filter Infreq.",
    "+Filter Zero-Dur.",
    "+Filter Long",
    "+Consec. Dup.",
    "+Impute",
    "+Rare Variants",
    "+IQR Outlier",
    "+Rare Classes",
]

PALETTE = sns.color_palette("tab10", n_colors=5)
LOG_COLORS = {log: PALETTE[i] for i, log in enumerate(LOGS)}


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Ablation progression
# ─────────────────────────────────────────────────────────────────────────────

def load_ablation_scores(log: str, task: str):
    """Return list of absolute improvements from stage-1 baseline, or None.

    Classification: Δ F1-macro (positive = better)
    Regression:     Δ MAE days reduced (positive = better, i.e. baseline - score)
    """
    path = RESULTS / "ablation" / log / task / "ablation_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)

    stages = data.get("stages", [])
    if not stages:
        return None

    scores = []
    for s in stages:
        if s.get("status") != "success":
            scores.append(None)
        else:
            scores.append(s["primary_score"])

    if scores[0] is None:
        return None

    # Remaining-time scores are stored in seconds — convert to days
    if not IS_CLF[task]:
        scores = [s / 86400.0 if s is not None else None for s in scores]

    baseline = scores[0]
    result = []
    for sc in scores:
        if sc is None:
            result.append(None)
        elif IS_CLF[task]:
            result.append(sc - baseline)        # F1 gain (higher = better)
        else:
            result.append(baseline - sc)        # MAE reduction in days (higher = better)
    return result


def plot_ablation():
    n_stages = len(ABLATION_STAGE_LABELS)
    x = np.arange(n_stages)

    for task in TASKS:
        height = 5.5 if task == "remaining_time" else 4
        fig, ax = plt.subplots(figsize=(11, height))
        any_line = False

        for log in LOGS:
            improvements = load_ablation_scores(log, task)
            if improvements is None or all(v is None for v in improvements):
                continue
            y = [v if v is not None else np.nan for v in improvements]
            ax.plot(x, y, marker="o", markersize=4, linewidth=1.6,
                    color=LOG_COLORS[log], label=LOG_SHORT[log])
            any_line = True

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(ABLATION_STAGE_LABELS, rotation=45, ha="right", fontsize=8.5)
        ax.set_ylabel(TASK_YLABELS[task], fontsize=10)
        ax.grid(axis="y", alpha=0.4)
        ax.grid(axis="x", alpha=0.0)

        # Black x and y axes
        for spine in ("left", "bottom"):
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color("black")
            ax.spines[spine].set_linewidth(1.2)

        if any_line:
            ax.legend(loc="upper left", fontsize=8.5, framealpha=0.7, ncol=3)

        fig.tight_layout()
        stem = f"ablation_{task}"
        fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight", dpi=200)
        fig.savefig(FIGURES / f"{stem}.png", bbox_inches="tight", dpi=200)
        print(f"Saved {FIGURES / stem}.pdf")
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Strategy eval heatmap
# ─────────────────────────────────────────────────────────────────────────────

def load_strategy_scores():
    """Return dict: task -> bucketer -> encoder -> list of improvement_pct per log."""
    path = RESULTS / "strategy_eval" / "strategy_eval_results.json"
    with open(path) as f:
        data = json.load(f)

    # task -> bucketer -> encoder -> [pct, ...]
    from collections import defaultdict
    table = {t: {b: {e: [] for e in ENCODERS} for b in BUCKETERS} for t in TASKS}

    for key, entry in data.items():
        if entry.get("status") != "success":
            continue
        log, task = key.split("|")
        if task not in TASKS:
            continue
        is_clf = entry.get("is_classification", True)

        comparison = entry.get("comparison", [])
        if not comparison:
            continue

        # Find baseline: no_bucket + last_state
        baseline_score = None
        for c in comparison:
            if c["bucketing"] == "no_bucket" and c["encoding"] == "last_state":
                baseline_score = c["primary_score"]
                break

        if baseline_score is None:
            continue

        for c in comparison:
            b = c["bucketing"]
            e = c["encoding"]
            sc = c["primary_score"]
            if b not in BUCKETERS or e not in ENCODERS:
                continue
            if is_clf:
                pct = (sc - baseline_score) / abs(baseline_score) * 100 if baseline_score != 0 else 0.0
            else:
                # Lower is better
                pct = (baseline_score - sc) / abs(baseline_score) * 100 if baseline_score != 0 else 0.0
            table[task][b][e].append(pct)

    return table


def plot_strategy_heatmap():
    table = load_strategy_scores()

    fig, axes = plt.subplots(3, 1, figsize=(8, 10))

    CLIP = 100.0  # cap colorbar at ±100% for readability

    matrices = {}
    matrices_raw = {}
    for task in TASKS:
        mat = np.zeros((len(BUCKETERS), len(ENCODERS)))
        for i, b in enumerate(BUCKETERS):
            for j, e in enumerate(ENCODERS):
                vals = table[task][b][e]
                mat[i, j] = np.median(vals) if vals else np.nan
        matrices_raw[task] = mat.copy()
        matrices[task] = np.clip(mat, -CLIP, CLIP)

    for row, task in enumerate(TASKS):
        ax = axes[row]
        mat_clipped = matrices[task]
        mat_raw = matrices_raw[task]

        # Build annotation strings: add "*" if clipped
        annot = np.full(mat_raw.shape, "", dtype=object)
        for i in range(mat_raw.shape[0]):
            for j in range(mat_raw.shape[1]):
                v = mat_raw[i, j]
                if np.isnan(v):
                    annot[i, j] = "–"
                elif abs(v) > CLIP:
                    annot[i, j] = f"{v:.0f}%*"
                else:
                    annot[i, j] = f"{v:.0f}%"

        sns.heatmap(
            mat_clipped,
            ax=ax,
            annot=annot,
            fmt="",
            cmap="RdYlGn",
            center=0,
            vmin=-CLIP,
            vmax=CLIP,
            linewidths=0.5,
            linecolor="white",
            xticklabels=ENCODER_LABELS if row == len(TASKS) - 1 else False,
            yticklabels=BUCKETER_LABELS,
            cbar=True,
            cbar_kws={"shrink": 0.7},
            annot_kws={"size": 9},
        )
        ax.set_title(TASK_LABELS[task], fontsize=11, pad=6)
        ax.set_xlabel("Encoder" if row == len(TASKS) - 1 else "", fontsize=9)
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=30, labelsize=9)
        ax.tick_params(axis="y", rotation=0, labelsize=9)
        ax.set_ylabel("Bucketer" if row == 1 else "", fontsize=9)

    fig.tight_layout()
    out = FIGURES / "strategy_heatmap.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    print(f"Saved {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: AutoML eval — staged gains
# ─────────────────────────────────────────────────────────────────────────────

def load_automl_gains():
    path = RESULTS / "automl_eval" / "automl_eval_results.json"
    with open(path) as f:
        data = json.load(f)

    rows = []
    for key, entry in data.items():
        if entry.get("status") == "skipped":
            continue
        log, task = key.split("|")
        is_clf = entry.get("is_classification", True)
        base = entry.get("lgbm_baseline")
        best = entry.get("lgbm_best")
        automl = entry.get("automl_score")
        if base is None or best is None or automl is None:
            continue

        if is_clf:
            gain_strategy = (best - base) / abs(base) * 100 if base != 0 else 0.0
            gain_automl = (automl - best) / abs(best) * 100 if best != 0 else 0.0
        else:
            gain_strategy = (base - best) / abs(base) * 100 if base != 0 else 0.0
            gain_automl = (best - automl) / abs(best) * 100 if best != 0 else 0.0

        rows.append({
            "label": f"{LOG_SHORT[log]}\n{TASK_LABELS[task].split(' (')[0]}",
            "gain_strategy": gain_strategy,
            "gain_automl": gain_automl,
            "log": log,
            "task": task,
        })

    return rows


def plot_automl_gains():
    rows = load_automl_gains()
    if not rows:
        print("No AutoML data found")
        return

    # Group rows by task
    task_rows = {t: [r for r in rows if r["task"] == t] for t in TASKS}

    fig, axes = plt.subplots(3, 1, figsize=(8, 9))

    for row, task in enumerate(TASKS):
        ax = axes[row]
        trows = task_rows[task]
        if not trows:
            ax.set_visible(False)
            continue

        n = len(trows)
        x = np.arange(n)
        log_labels = [LOG_SHORT[r["log"]] for r in trows]
        g_auto = [r["gain_automl"] for r in trows]

        colors = ["#4c9be8" if v >= 0 else "#e8604c" for v in g_auto]
        bars = ax.bar(x, g_auto, color=colors, edgecolor="white", linewidth=0.5, width=0.55)

        # Value labels
        max_abs = max(abs(v) for v in g_auto) if g_auto else 1
        offset_pos = max_abs * 0.03
        offset_neg = max_abs * 0.07   # larger so labels clear the zero line

        for bar, val in zip(bars, g_auto):
            if val >= 0:
                ypos = val + offset_pos
                va = "bottom"
            else:
                ypos = val - offset_neg
                va = "top"
            ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f"{val:+.0f}%", ha="center", va=va, fontsize=9, color="#222")

        ax.axhline(0, color="black", linewidth=0.8, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(log_labels, fontsize=10)
        ax.set_title(TASK_LABELS[task], fontsize=11, pad=6)
        ax.set_ylabel("AutoML gain over LGBM best (%)", fontsize=9)
        ax.grid(axis="y", alpha=0.35)
        ax.grid(axis="x", alpha=0.0)
        ax.set_xlim(-0.5, n - 0.5)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color("black")
            ax.spines[spine].set_linewidth(1.2)

    fig.tight_layout()
    out = FIGURES / "automl_gains.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    print(f"Saved {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating ablation figure...")
    plot_ablation()

    print("Generating strategy eval heatmap...")
    plot_strategy_heatmap()

    print("Generating AutoML gains figure...")
    plot_automl_gains()

    print("Done.")
