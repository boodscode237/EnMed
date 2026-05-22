#!/usr/bin/env python
"""Stage 02 — domain-adaptive continual pre-training (DAPT)."""
import argparse

from enmed.training.dapt import run_dapt
from enmed.utils.io import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dapt.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    # data.yaml provides dataset definitions used by the corpus loader
    cfg = {**load_config("configs/data.yaml"), **cfg}
    path = run_dapt(cfg)
    print(f"DAPT checkpoint: {path}")


if __name__ == "__main__":
    main()
