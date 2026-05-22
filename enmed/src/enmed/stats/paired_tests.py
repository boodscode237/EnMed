"""Stage 06 — item-level paired t-tests.

For each candidate system and each (task, shot) cell, run a paired t-test of the
candidate's per-item scores against the reference (Qwen3-14B-vanilla) on the
*same* items. Reports the paired delta, two-sided 95% CI, p-value and Cohen's d.

This is the inferential core of the paper: nine independent paired tests per
candidate, never a single aggregated test over incomparable metric scales.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from enmed.utils.io import get_logger

log = get_logger(__name__)


def paired_test(cand: np.ndarray, ref: np.ndarray) -> dict:
    """Paired t-test of candidate vs reference on aligned per-item scores."""
    cand, ref = np.asarray(cand, float), np.asarray(ref, float)
    assert cand.shape == ref.shape, "candidate/reference must be item-aligned"
    d = cand - ref
    n = len(d)
    mean_d = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 0 else float("nan")

    if sd == 0:
        t_stat, p = (0.0, 1.0)
    else:
        t_stat, p = stats.ttest_rel(cand, ref)

    tcrit = stats.t.ppf(0.975, df=max(n - 1, 1))
    ci = (mean_d - tcrit * se, mean_d + tcrit * se)
    cohen_d = mean_d / sd if sd > 0 else 0.0

    return {
        "n": n,
        "mean_ref": float(ref.mean()),
        "mean_cand": float(cand.mean()),
        "delta": mean_d,
        "ci_low": ci[0],
        "ci_high": ci[1],
        "p_value": float(p),
        "cohen_d": cohen_d,
    }


def all_paired_tests(long: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Run a paired test for every (candidate, task, shot) cell vs the reference."""
    rows = []
    ref_df = long[long.system == reference]
    candidates = [s for s in long.system.unique() if s != reference]

    for cand in candidates:
        cand_df = long[long.system == cand]
        for (task, shot), g_ref in ref_df.groupby(["task", "shot"]):
            g_cand = cand_df[(cand_df.task == task) & (cand_df.shot == shot)]
            merged = g_ref.merge(g_cand, on="id", suffixes=("_ref", "_cand"))
            if merged.empty:
                continue
            res = paired_test(merged.score_cand.values, merged.score_ref.values)
            rows.append({"candidate": cand, "task": task, "shot": shot, **res})

    out = pd.DataFrame(rows).sort_values(["candidate", "task", "shot"])
    log.info("Ran %d paired tests (%d candidates × cells)",
             len(out), out.candidate.nunique())
    return out.reset_index(drop=True)


def significance_record(tests: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Per-candidate count of significant wins/losses, numeric wins, ties."""
    def classify(r):
        if r.p_value < alpha and r.delta > 0:
            return "sig_win"
        if r.p_value < alpha and r.delta < 0:
            return "sig_loss"
        if r.delta > 0:
            return "num_win"
        if r.delta < 0:
            return "num_loss"
        return "tie"

    t = tests.assign(verdict=tests.apply(classify, axis=1))
    rec = (
        t.groupby(["candidate", "verdict"]).size().unstack(fill_value=0)
        .reindex(columns=["sig_win", "num_win", "num_loss", "sig_loss", "tie"],
                 fill_value=0)
    )
    rec["mean_delta"] = t.groupby("candidate").delta.mean()
    rec["median_p"] = t.groupby("candidate").p_value.median()
    return rec.reset_index()
