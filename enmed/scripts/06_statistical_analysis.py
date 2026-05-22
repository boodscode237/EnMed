#!/usr/bin/env python
"""Stage 06 — item-level paired t-tests, corrections, and rank analysis."""
import argparse
from pathlib import Path

import pandas as pd

from enmed.stats.corrections import annotate_corrections
from enmed.stats.critical_difference import cd_analysis
from enmed.stats.paired_tests import all_paired_tests, significance_record
from enmed.utils.io import get_logger, load_config

log = get_logger("06_statistical_analysis")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    metr_dir = Path(cfg["results_root"]) / "metrics"
    stats_dir = Path(cfg["results_root"]) / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    long = pd.read_csv(metr_dir / "item_scores_long.csv")
    matrix = pd.read_csv(metr_dir / "cell_matrix.csv", index_col=0)

    reference = cfg["stats"]["reference"]
    alpha = cfg["stats"]["alpha"]

    tests = all_paired_tests(long, reference=reference)
    tests = annotate_corrections(tests, alpha=alpha)
    tests.to_csv(stats_dir / "item_level_ttest.csv", index=False)

    record = significance_record(tests, alpha=alpha)
    record.to_csv(stats_dir / "significance_summary.csv", index=False)

    cd = cd_analysis(matrix, shots=cfg["shots"])
    for shot, res in cd.items():
        res["ranks"].to_csv(stats_dir / f"avg_ranks_{shot}shot.csv")
    log.info("Statistics written → %s", stats_dir)
    log.info("CD = %.2f", next(iter(cd.values()))["cd"])


if __name__ == "__main__":
    main()
