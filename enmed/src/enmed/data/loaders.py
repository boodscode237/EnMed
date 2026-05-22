"""Dataset loaders that normalise every source to a common schema.

Common schema per task
-----------------------
MCQA : {id, question, options:{A:..,B:..}, answer:"B", specialty}
ExtQA: {id, context, question, answer (gold span), specialty}
AbsQA: {id, question, context?, answer (reference), specialty}

All three QA datasets come from the DrBenchmark ecosystem / associated releases:
FrenchMedMCQA (MCQA), CAS clinical cases (ExtQA), MediQAl (AbsQA). The DAPT
corpus is the French health corpus of Mannion et al. (2026).

Loaders accept either a Hugging Face id (`hf_id`) or a local path
(`local:NAME` → `data/raw/NAME`). Adapt the field-mapping functions to the
exact column names of the release you download.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from datasets import Dataset, load_dataset

from enmed.utils.io import get_logger

log = get_logger(__name__)


def _resolve(hf_id: str, data_root: str, split: str) -> Dataset:
    """Load a HF dataset by id, or a local JSONL/parquet under data/raw."""
    if hf_id.startswith("local:"):
        name = hf_id.split("local:", 1)[1]
        base = Path(data_root) / "raw" / name
        for ext in (".jsonl", ".json", ".parquet", ".csv"):
            candidate = base.with_suffix(ext)
            if candidate.exists():
                fmt = {".jsonl": "json", ".json": "json"}.get(ext, ext.lstrip("."))
                return load_dataset(fmt, data_files=str(candidate), split="train")
        raise FileNotFoundError(f"No local dataset found at {base}.*")
    return load_dataset(hf_id, split=split)


# --------------------------------------------------------------------------- #
# Field mappers — adjust to the actual column names of each release.
# --------------------------------------------------------------------------- #
def _map_mcqa(row: dict, idx: int) -> dict:
    options = row.get("options") or {
        k: row[k] for k in ("A", "B", "C", "D", "E") if k in row
    }
    return {
        "id": row.get("id", f"mcqa-{idx}"),
        "question": row["question"],
        "options": options,
        "answer": row.get("correct_answers") or row.get("answer"),
        "specialty": row.get("specialty", "unknown"),
    }


def _map_extqa(row: dict, idx: int) -> dict:
    return {
        "id": row.get("id", f"extqa-{idx}"),
        "context": row.get("context") or row.get("case"),
        "question": row["question"],
        "answer": row.get("answer") or row.get("span"),
        "specialty": row.get("specialty", "unknown"),
    }


def _map_absqa(row: dict, idx: int) -> dict:
    return {
        "id": row.get("id", f"absqa-{idx}"),
        "question": row["question"],
        "context": row.get("context"),
        "answer": row.get("answer") or row.get("long_answer"),
        "specialty": row.get("specialty", "unknown"),
    }


_MAPPERS: dict[str, Callable[[dict, int], dict]] = {
    "mcqa": _map_mcqa,
    "extqa": _map_extqa,
    "absqa": _map_absqa,
}


def load_task(task: str, cfg: dict, split: str = "test") -> list[dict]:
    """Load and normalise a single task split to the common schema."""
    ds_cfg = cfg["datasets"][task]
    raw = _resolve(ds_cfg["hf_id"], cfg["data_root"], split)
    mapper = _MAPPERS[task]
    rows = [mapper(r, i) for i, r in enumerate(raw)]
    log.info("Loaded %d %s examples (split=%s)", len(rows), task, split)
    return rows


def load_dapt_corpus(cfg: dict, split: str = "train") -> Dataset:
    """Load the Mannion et al. French health corpus for continual pre-training."""
    c = cfg["dapt_corpus"]
    ds = _resolve(c["hf_id"], cfg["data_root"], split)
    log.info("Loaded DAPT corpus with %d documents", len(ds))
    return ds
