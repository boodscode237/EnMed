"""Stage 06 — rank analysis (Nemenyi Critical Difference).

For each shot count, rank the systems by their per-task scores (lower rank =
better), average ranks across tasks, and compute the Nemenyi critical
difference. The CD diagrams produced from this are descriptive consensus
rankings, not pairwise significance proofs (the observed rank spread is smaller
than CD at this sample size).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_Q_ALPHA_05 = {  # Nemenyi q_alpha for alpha=0.05 (studentized range / sqrt2)
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
    7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
}


def average_ranks(matrix: pd.DataFrame, shot: int, tasks=("mcqa", "extqa", "absqa")) -> pd.Series:
    """Average rank of each system across tasks at a fixed shot count.

    `matrix` is the systems × cell mean matrix (cells named e.g. 'mcqa-0').
    Higher score = better, so ranks are assigned to negated scores.
    """
    cols = [f"{t}-{shot}" for t in tasks if f"{t}-{shot}" in matrix.columns]
    sub = matrix[cols]
    ranks = (-sub).rank(axis=0, method="average")
    return ranks.mean(axis=1).sort_values()


def critical_difference(n_models: int, n_datasets: int, alpha: float = 0.05) -> float:
    """Nemenyi critical difference for `n_models` over `n_datasets` comparisons."""
    q = _Q_ALPHA_05.get(n_models)
    if q is None:
        raise ValueError(f"No tabulated q_alpha for k={n_models}")
    return q * np.sqrt(n_models * (n_models + 1) / (6.0 * n_datasets))


def cd_analysis(matrix: pd.DataFrame, shots=(0, 3, 5)) -> dict[int, dict]:
    """Average ranks + CD per shot count. Returns {shot: {ranks, cd}}."""
    out = {}
    n_models = matrix.shape[0]
    for shot in shots:
        ranks = average_ranks(matrix, shot)
        cd = critical_difference(n_models, n_datasets=3)
        out[shot] = {"ranks": ranks, "cd": cd}
    return out
