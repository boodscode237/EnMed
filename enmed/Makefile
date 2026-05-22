# EnMed reproduction pipeline. Run `make all` for the full pipeline,
# or individual targets. GPU stages: dapt, finetune, infer.

PY := python
CFG_DATA := configs/data.yaml
CFG_EVAL := configs/eval.yaml

.PHONY: all data dapt finetune infer metrics stats figures analysis clean help

help:
	@echo "Targets: data dapt finetune infer metrics stats figures | all | analysis | clean"

all: data dapt finetune infer metrics stats figures

## 01 — data engineering
data:
	$(PY) scripts/01_prepare_data.py --config $(CFG_DATA)

## 02 — DAPT pre-training (GPU)
dapt:
	$(PY) scripts/02_run_dapt.py --config configs/dapt.yaml

## 03 — LoRA adapters (GPU): unified + 3 task-specific
finetune:
	$(PY) scripts/03_finetune_adapters.py --config configs/finetune_unified.yaml
	$(PY) scripts/03_finetune_adapters.py --config configs/finetune_mcqa.yaml
	$(PY) scripts/03_finetune_adapters.py --config configs/finetune_extqa.yaml
	$(PY) scripts/03_finetune_adapters.py --config configs/finetune_absqa.yaml

## 04 — inference for every system (GPU). Override MODELS/PATHS as needed.
infer:
	@echo "Run 04_run_inference.py per system, e.g.:"
	@echo "  $(PY) scripts/04_run_inference.py --system EnMed-Unified --model-path models/EnMed-Unified"

## 05-07 — analysis only (CPU; no GPU required)
metrics:
	$(PY) scripts/05_compute_metrics.py --config $(CFG_EVAL)

stats:
	$(PY) scripts/06_statistical_analysis.py --config $(CFG_EVAL)

figures:
	$(PY) scripts/07_generate_figures.py --config $(CFG_EVAL)

analysis: metrics stats figures

clean:
	rm -f results/predictions/*.jsonl results/metrics/*.csv \
	      results/stats/*.csv results/figures/*.png
