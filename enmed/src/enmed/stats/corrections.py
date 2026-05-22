"""Stage 06 — multiplicity correction.

Apply Benjamini–Hochberg (FDR) and Bonferroni (FWER) corrections to the
per-cell p-values, so the reader can see which significant wins/losses survive
correction. Per the paper, the headline system's wins survive Benjamini–Hochberg
at q = 0.05 while the most marginal single-task wins do not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def benjamini_hochberg(p_values, q: float = 0.05):
    """Return a boolean mask of rejections under BH-FDR at level q."""
    p = np.asarray(p_values, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresh = q * (np.arange(1, n + 1) / n)
    passed = ranked <= thresh
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    mask = np.zeros(n, dtype=bool)
    if k > 0:
        mask[order[:k]] = True
    return mask


def bonferroni(p_values, alpha: float = 0.05):
    """Return a boolean mask of rejections under Bonferroni at level alpha."""
    p = np.asarray(p_values, float)
    return p <= alpha / len(p)


def annotate_corrections(tests: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Add `sig_uncorrected`, `sig_bh`, `sig_bonferroni` columns to the tests table."""
    out = tests.copy()
    out["sig_uncorrected"] = out.p_value < alpha
    out["sig_bh"] = benjamini_hochberg(out.p_value.values, q=alpha)
    out["sig_bonferroni"] = bonferroni(out.p_value.values, alpha=alpha)
    return out
