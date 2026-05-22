"""Reproducibility, config loading, logging, and IO helpers."""
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Iterable

import yaml


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Seed Python, NumPy and Torch (if available) for reproducible runs."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, recursively merging a `defaults:` parent if present.

    `defaults` may name another file in the same directory, e.g.
    `defaults: base.yaml`.
    """
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    parent_name = cfg.pop("defaults", None)
    if parent_name:
        parent = load_config(path.parent / parent_name)
        cfg = _deep_merge(parent, cfg)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def snapshot_config(cfg: dict, out_dir: str | Path) -> None:
    """Write the resolved config next to a run's outputs for provenance."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.snapshot.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def get_logger(name: str = "enmed") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# --------------------------------------------------------------------------- #
# JSONL IO
# --------------------------------------------------------------------------- #
def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows: Iterable[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
