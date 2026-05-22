#!/usr/bin/env python
"""Stage 01 — data engineering.

Load each task dataset (MCQA/ExtQA/AbsQA) and the DAPT corpus, normalise to the
common schema, build train/dev/test splits, and write processed JSONL files
that every downstream stage consumes.

Usage:
    python scripts/01_prepare_data.py --config configs/data.yaml
"""
import argparse
from pathlib import Path

from enmed.data.loaders import load_task
from enmed.utils.io import get_logger, load_config, set_seed, write_jsonl

log = get_logger("01_prepare_data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 12181531))
    out = Path(cfg["data_root"]) / "processed"

    for task in cfg["datasets"]:
        for split in ("train", "dev", "test"):
            try:
                rows = load_task(task, cfg, split=split)
            except Exception as e:  # split may not exist for every release
                log.warning("Skipping %s/%s: %s", task, split, e)
                continue
            write_jsonl(rows, out / f"{task}_{split}.jsonl")
            log.info("Wrote %d rows to %s_%s.jsonl", len(rows), task, split)

    log.info("Data engineering complete → %s", out)


if __name__ == "__main__":
    main()
