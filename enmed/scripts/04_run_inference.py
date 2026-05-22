#!/usr/bin/env python
"""Stage 04 — inference across the 9 (task, shot) cells for one system.

Writes one predictions file per cell:
    results/predictions/{system}__{task}__{shot}shot.jsonl
"""
import argparse
from functools import partial
from pathlib import Path

from enmed.data.loaders import load_task
from enmed.data.splits import select_shots
from enmed.inference.generate import predict_cell
from enmed.training.model_utils import load_backbone, setup_chat_template
from enmed.utils.io import get_logger, load_config, set_seed, write_jsonl

log = get_logger("04_run_inference")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--system", required=True, help="system name (label for output files)")
    ap.add_argument("--model-path", required=True, help="HF id or local path to load")
    ap.add_argument("--shot-strategy", default="random",
                    choices=["random", "stratified", "similarity"])
    args = ap.parse_args()

    cfg = {**load_config("configs/data.yaml"), **load_config(args.config)}
    set_seed(cfg.get("seed", 12181531))
    out_dir = Path(cfg["results_root"]) / "predictions"

    cfg_model = {**cfg, "backbone": args.model_path}
    model, tokenizer = load_backbone(cfg_model)
    tokenizer = setup_chat_template(tokenizer)

    selector = partial(select_shots, strategy=args.shot_strategy, seed=cfg["seed"])

    for task in cfg["tasks"]:
        test_rows = load_task(task, cfg, split="test")
        shot_pool = load_task(task, cfg, split="train")
        for shot in cfg["shots"]:
            preds = list(predict_cell(model, tokenizer, task, shot,
                                      test_rows, shot_pool, cfg,
                                      lambda ex, pool, n: selector(ex, pool, n)))
            fname = f"{args.system}__{task}__{shot}shot.jsonl"
            write_jsonl(preds, out_dir / fname)
            log.info("Wrote %d predictions → %s", len(preds), fname)


if __name__ == "__main__":
    main()
