#!/usr/bin/env python
"""Stage 07 — regenerate all paper figures from the aggregated results."""
import argparse
from pathlib import Path

import pandas as pd

from enmed.stats.critical_difference import cd_analysis
from enmed.utils.io import get_logger, load_config
from enmed.viz import figures as F

log = get_logger("07_generate_figures")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    metr = Path(cfg["results_root"]) / "metrics"
    stats = Path(cfg["results_root"]) / "stats"
    figs = Path(cfg["results_root"]) / "figures"

    long = pd.read_csv(metr / "item_scores_long.csv")
    matrix = pd.read_csv(metr / "cell_matrix.csv", index_col=0)
    norm = pd.read_csv(metr / "normalized_matrix.csv", index_col=0)
    tests = pd.read_csv(stats / "item_level_ttest.csv")
    record = pd.read_csv(stats / "significance_summary.csv")
    reference = cfg["stats"]["reference"]

    F.fig_per_task_means(long, reference, figs)
    F.fig_global_ranking(norm, figs)
    F.fig_normalized_heatmap(norm, figs)
    F.fig_item_ttest(tests, figs)
    F.fig_significance_summary(record, figs)
    F.fig_cd(cd_analysis(matrix, shots=cfg["shots"]), figs)
    log.info("Figures written → %s", figs)


if __name__ == "__main__":
    main()
