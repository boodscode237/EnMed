#!/usr/bin/env python
"""Stage 05 — score predictions and aggregate into the cell matrix.

Reads results/predictions/*.jsonl, attaches a per-item primary-metric score
(AbsQA uses the LLM-as-judge composite when GEMINI_API_KEY is set), writes
*.scored.jsonl, then saves the aggregated matrices to results/metrics/.
"""
import argparse
from pathlib import Path

from enmed.evaluation.aggregate import cell_matrix, collect_item_scores, normalized_matrix
from enmed.evaluation.metrics import score_items
from enmed.utils.io import get_logger, load_config, read_jsonl, write_jsonl

log = get_logger("05_compute_metrics")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--use-judge", action="store_true",
                    help="score AbsQA with the LLM-as-judge instead of ROUGE-L proxy")
    args = ap.parse_args()

    cfg = load_config(args.config)
    pred_dir = Path(cfg["results_root"]) / "predictions"
    metr_dir = Path(cfg["results_root"]) / "metrics"
    metr_dir.mkdir(parents=True, exist_ok=True)

    for f in sorted(pred_dir.glob("*.jsonl")):
        if f.name.endswith(".scored.jsonl"):
            continue
        task = f.stem.split("__")[1]
        rows = read_jsonl(f)
        if task == "absqa" and args.use_judge:
            from enmed.evaluation.llm_judge import judge_rows
            scored = judge_rows(rows, cfg["judge"])
        else:
            scored = score_items(task, rows)
        write_jsonl(scored, f.with_suffix(".scored.jsonl"))
        log.info("Scored %s (%d items)", f.name, len(scored))

    long = collect_item_scores(pred_dir)
    matrix = cell_matrix(long)
    norm = normalized_matrix(matrix)
    long.to_csv(metr_dir / "item_scores_long.csv", index=False)
    matrix.to_csv(metr_dir / "cell_matrix.csv")
    norm.to_csv(metr_dir / "normalized_matrix.csv")
    log.info("Metrics written → %s", metr_dir)


if __name__ == "__main__":
    main()
