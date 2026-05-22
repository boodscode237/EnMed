"""Stage 07 — figures.

Regenerates the paper's figures from the aggregated results:
  * per-task means with std            (fig05)
  * global normalised ranking          (fig06)
  * normalised score heatmap           (fig02)
  * item-level paired-test bars + CI   (fig07)
  * significance heatmap               (fig08)
  * significance summary stacked bars  (fig10)
  * best-system-per-cell grid          (fig11)
  * Critical Difference diagrams       (cd_{0,3,5}shot)

All figures are saved as PNG to the configured figures directory.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "savefig.bbox": "tight"})


def _save(fig, out_dir, name):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / name
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def fig_per_task_means(long: pd.DataFrame, reference: str, out_dir: str) -> str:
    tasks = ["mcqa", "extqa", "absqa"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, task in zip(axes, tasks):
        sub = long[long.task == task].groupby("system").score.agg(["mean", "std"])
        sub = sub.sort_values("mean", ascending=False)
        colors = ["#888" if s == reference else "#2c7fb8" for s in sub.index]
        ax.barh(sub.index, sub["mean"], xerr=sub["std"], color=colors)
        ax.axvline(sub.loc[reference, "mean"] if reference in sub.index else 0,
                   color="red", ls="--", lw=1)
        ax.set_title(task.upper())
        ax.invert_yaxis()
    fig.suptitle("Per-task mean score (± std) across shot counts")
    return _save(fig, out_dir, "fig05_per_task_mean_std.png")


def fig_global_ranking(matrix: pd.DataFrame, out_dir: str) -> str:
    means = matrix.mean(axis=1).sort_values(ascending=False)
    stds = matrix.std(axis=1).reindex(means.index)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(means.index, means.values, xerr=stds.values, color="#41b6c4")
    ax.invert_yaxis()
    ax.set_xlabel("Mean normalised score across 9 cells")
    ax.set_title("Global descriptive ranking")
    return _save(fig, out_dir, "fig06_global_mean_std.png")


def fig_normalized_heatmap(norm_matrix: pd.DataFrame, out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(norm_matrix.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(norm_matrix.columns)))
    ax.set_xticklabels(norm_matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(norm_matrix.index)))
    ax.set_yticklabels(norm_matrix.index)
    fig.colorbar(im, ax=ax, label="normalised score")
    ax.set_title("Normalised scores across 9 (task, shot) cells")
    return _save(fig, out_dir, "fig02_normalized_heatmap.png")


def fig_item_ttest(tests: pd.DataFrame, out_dir: str) -> str:
    """Bar chart of per-cell deltas with 95% CI and significance stars."""
    candidates = tests.candidate.unique()
    fig, axes = plt.subplots(1, len(candidates), figsize=(4 * len(candidates), 4),
                             sharey=True)
    if len(candidates) == 1:
        axes = [axes]
    for ax, cand in zip(axes, candidates):
        sub = tests[tests.candidate == cand].copy()
        sub["cell"] = sub.task + "-" + sub.shot.astype(str)
        err = [sub.delta - sub.ci_low, sub.ci_high - sub.delta]
        ax.barh(sub.cell, sub.delta, xerr=err,
                color=["#d7301f" if d < 0 else "#2c7fb8" for d in sub.delta])
        ax.axvline(0, color="k", lw=0.8)
        for y, (_, r) in enumerate(sub.iterrows()):
            if r.p_value < 0.001:
                star = "***"
            elif r.p_value < 0.01:
                star = "**"
            elif r.p_value < 0.05:
                star = "*"
            else:
                star = ""
            if star:
                ax.text(r.delta, y, star, va="center")
        ax.set_title(cand, fontsize=8)
    fig.suptitle("Item-level paired t-tests vs reference (Δ ± 95% CI)")
    return _save(fig, out_dir, "fig07_item_level_ttest.png")


def fig_significance_summary(record: pd.DataFrame, out_dir: str) -> str:
    cats = ["sig_win", "num_win", "num_loss", "sig_loss"]
    colors = ["#238b45", "#a1d99b", "#fcae91", "#cb181d"]
    rec = record.set_index("candidate")[cats]
    fig, ax = plt.subplots(figsize=(8, 4))
    left = np.zeros(len(rec))
    for cat, color in zip(cats, colors):
        ax.barh(rec.index, rec[cat], left=left, color=color, label=cat)
        left += rec[cat].values
    ax.axvline(4.5, ls=":", color="k")
    ax.set_xlabel("cells (out of 9)")
    ax.legend(ncol=4, fontsize=7, loc="lower right")
    ax.set_title("Significance record per system")
    return _save(fig, out_dir, "fig10_sig_summary.png")


def fig_cd(cd_results: dict, out_dir: str) -> list[str]:
    """One Critical Difference diagram per shot count."""
    paths = []
    for shot, res in cd_results.items():
        ranks = res["ranks"]
        cd = res["cd"]
        fig, ax = plt.subplots(figsize=(9, 2.2))
        ax.scatter(ranks.values, [1] * len(ranks), color="#2b3a55", zorder=3)
        for name, r in ranks.items():
            ax.annotate(f"{name} ({r:.2f})", (r, 1), (r, 1.05),
                        fontsize=7, ha="left", rotation=0)
        ax.hlines(0.9, ranks.min(), ranks.min() + cd, color="red", lw=3)
        ax.text(ranks.min() + cd / 2, 0.86, f"CD = {cd:.2f}", color="red",
                ha="center", fontsize=7)
        ax.set_xlabel("avg rank (lower = better)")
        ax.set_yticks([])
        ax.set_title(f"Critical Difference — {shot}-shot")
        paths.append(_save(fig, out_dir, f"cd_{shot}shot.png"))
    return paths
