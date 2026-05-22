# Step-by-step reproduction

## 0. Environment
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
export HF_TOKEN=...        # model download/push
export GEMINI_API_KEY=...  # only for AbsQA LLM-as-judge
```

## 1. Data engineering
Download datasets into `data/raw/` (see `data/README.md`), then:
```bash
make data
```
Produces `data/processed/{task}_{split}.jsonl`.

## 2. DAPT pre-training (GPU)
```bash
make dapt          # → models/EnMed-DAPT
```

## 3. LoRA fine-tuning (GPU)
```bash
make finetune      # → models/EnMed-{Unified,MCQA,ExtQA,AbsQA}
```

## 4. Inference (GPU) — once per system
The reference and the two controls are evaluated from their public checkpoints;
the EnMed systems from your local adapters. Example:
```bash
python scripts/04_run_inference.py --system Qwen3-14B-vanilla        --model-path unsloth/Qwen3-14B
python scripts/04_run_inference.py --system EnMed-Unified            --model-path models/EnMed-Unified
python scripts/04_run_inference.py --system EnMed-DAPT               --model-path models/EnMed-DAPT
python scripts/04_run_inference.py --system EnMed-MCQA               --model-path models/EnMed-MCQA
python scripts/04_run_inference.py --system EnMed-ExtQA              --model-path models/EnMed-ExtQA
python scripts/04_run_inference.py --system EnMed-AbsQA              --model-path models/EnMed-AbsQA
python scripts/04_run_inference.py --system Qwen3-8B                 --model-path unsloth/Qwen3-8B
python scripts/04_run_inference.py --system Mistral-7B-Instruct-v0.3 --model-path mistralai/Mistral-7B-Instruct-v0.3
```
Each call writes `results/predictions/{system}__{task}__{shot}shot.jsonl`.

## 5–7. Analysis (CPU)
```bash
make metrics    # score predictions → results/metrics/
make stats      # paired t-tests, BH/Bonferroni, CD → results/stats/
make figures    # all figures → results/figures/
```

To score AbsQA with the LLM-as-judge instead of the ROUGE-L proxy:
```bash
python scripts/05_compute_metrics.py --use-judge
```

## Reproduce analysis only
If `results/predictions/*.scored.jsonl` are already present (item-level scores),
the entire statistical analysis and every figure regenerate without a GPU:
```bash
make analysis
```
