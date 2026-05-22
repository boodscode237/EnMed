#!/usr/bin/env python
"""Stage 03 — LoRA fine-tuning.

Fine-tune one adapter from a config. Run once per system:
    python scripts/03_finetune_adapters.py --config configs/finetune_unified.yaml
    python scripts/03_finetune_adapters.py --config configs/finetune_mcqa.yaml
    ... extqa, absqa
"""
import argparse

from enmed.training.finetune import run_finetune
from enmed.utils.io import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = {**load_config("configs/data.yaml"), **load_config(args.config)}
    path = run_finetune(cfg)
    print(f"Adapter saved: {path}")


if __name__ == "__main__":
    main()
