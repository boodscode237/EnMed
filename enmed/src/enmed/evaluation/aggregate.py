"""Stage 05 — aggregation.

Collect per-item scores from every system × (task, shot) cell into:
  * a long per-item table  (for the paired statistical tests), and
  * a wide cell-mean matrix (systems × 9 cells, for descriptive ranking).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from enmed.utils.io import get_logger, read_jsonl

log = get_logger(__name__)


def collect_item_scores(predictions_dir: str | Path) -> pd.DataFrame:
    """Read every `*.scored.jsonl` into one long item-level dataframe.

    Expected filename convention: `{system}__{task}__{shot}shot.scored.jsonl`.
    """
    predictions_dir = Path(predictions_dir)
    frames = []
    for f in sorted(predictions_dir.glob("*.scored.jsonl")):
        system, task, shot = _parse_name(f.name)
        rows = read_jsonl(f)
        df = pd.DataFrame(rows)
        df["system"], df["task"], df["shot"] = system, task, int(shot)
        frames.append(df[["system", "task", "shot", "id", "score"]])
    if not frames:
        raise FileNotFoundError(f"No *.scored.jsonl files in {predictions_dir}")
    long = pd.concat(frames, ignore_index=True)
    log.info("Collected %d item-level scores across %d cells",
             len(long), long.groupby(["system", "task", "shot"]).ngroups)
    return long


def cell_matrix(long: pd.DataFrame) -> pd.DataFrame:
    """Systems × (task, shot) matrix of mean scores."""
    pivot = (
        long.groupby(["system", "task", "shot"])["score"]
        .mean()
        .reset_index()
    )
    pivot["cell"] = pivot["task"] + "-" + pivot["shot"].astype(str)
    return pivot.pivot(index="system", columns="cell", values="score")


def normalized_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """Per-cell min-max normalisation (worst→0, best→1) for the global ranking."""
    return (matrix - matrix.min()) / (matrix.max() - matrix.min()).replace(0, 1)


def _parse_name(fname: str) -> tuple[str, str, str]:
    stem = fname.replace(".scored.jsonl", "")
    system, task, shot = stem.split("__")
    return system, task, shot.replace("shot", "")
