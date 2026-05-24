# EnMed: Cross-Lingual Domain Adaptation and Multi-Task Fine-Tuning for French Medical LLMs

Reproducible research code for the **EnMed** family of French medical language
models, built on a Qwen3-14B backbone via domain-adaptive continual pre-training
(DAPT) and LoRA fine-tuning, and evaluated with a rigorous item-level statistical
protocol.

> **Paper:** *Cross-Lingual Domain Adaptation and Multi-Task Fine-Tuning for
> High-Fidelity Medical Language Models* — B. D. Abodo Eloundou & V. Malykh
> (ITMO University / MTS Web Services). Submitted to Neuroinformatics-2026, under review.

This repository reproduces the full Phase 1 pipeline **end to end**, from data
engineering through training, inference, evaluation, and the statistical analysis
that produces every table and figure in the paper.

---

## Pipeline overview

```
 ┌─────────────┐   ┌──────────┐   ┌────────────┐   ┌───────────┐   ┌─────────┐   ┌────────┐
 │ 01 Data     │ → │ 02 DAPT  │ → │ 03 LoRA    │ → │ 04 Infer  │ → │ 05 Eval │ → │ 06     │
 │ engineering │   │ pre-     │   │ fine-tune  │   │ (9 cells) │   │ metrics │   │ stats  │
 │             │   │ training │   │ adapters   │   │           │   │         │   │ + figs │
 └─────────────┘   └──────────┘   └────────────┘   └───────────┘   └─────────┘   └────────┘
   loaders.py        dapt.py        finetune.py      generate.py     metrics.py    paired_tests.py
   formatting.py                                     mcqa/extqa/      llm_judge.py  critical_difference.py
   splits.py                                         absqa.py                       figures.py
```

Each stage is a numbered script in [`scripts/`](scripts/) and a corresponding
module in [`src/enmed/`](src/enmed/). Every stage is **config-driven** (YAML in
[`configs/`](configs/)), seeded for reproducibility, and writes its outputs to
[`results/`](results/) so the next stage can pick them up.

| Stage | Script | Module | Output |
|---|---|---|---|
| 1. Data engineering | `01_prepare_data.py` | `enmed.data` | `data/processed/*.jsonl` |
| 2. DAPT pre-training | `02_run_dapt.py` | `enmed.training.dapt` | `models/enmed-dapt/` |
| 3. LoRA fine-tuning | `03_finetune_adapters.py` | `enmed.training.finetune` | `models/enmed-{unified,mcqa,extqa,absqa}/` |
| 4. Inference | `04_run_inference.py` | `enmed.inference` | `results/predictions/*.jsonl` |
| 5. Metrics | `05_compute_metrics.py` | `enmed.evaluation` | `results/metrics/*.csv` |
| 6. Statistics | `06_statistical_analysis.py` | `enmed.stats` | `results/stats/*.csv` |
| 7. Figures | `07_generate_figures.py` | `enmed.viz` | `results/figures/*.png` |

---

## The eight systems

| System | Description |
|---|---|
| `Qwen3-14B-vanilla` | Un-adapted reference baseline |
| `EnMed-DAPT` | Backbone + domain-adaptive continual pre-training only |
| `EnMed-Unified` ⭐ | DAPT + **multi-task** LoRA (headline system) |
| `EnMed-MCQA` | DAPT + MCQA-only LoRA |
| `EnMed-ExtQA` | DAPT + ExtQA-only LoRA |
| `EnMed-AbsQA` | DAPT + AbsQA-only LoRA |
| `Qwen3-8B` | Lighter same-family control |
| `Mistral-7B-Instruct-v0.3` | Cross-architecture control |

evaluated on three tasks × three shot counts = **9 (task, shot) cells**:
**MCQA** (FrenchMedMCQA, accuracy), **ExtQA** (CAS, token-F1), **AbsQA**
(MediQAl, LLM-as-judge 1–5).

---

## Quick start

### 1. Install

```bash
git clone https://github.com/boodscode237/EnMed.git
cd enmed
python -m venv .venv && source .venv/bin/activate
pip install -e .            # installs the `enmed` package + dependencies
# or: pip install -r requirements.txt
```

Set credentials (only needed for training / gated models / the LLM judge):

```bash
export HF_TOKEN=hf_...           # Hugging Face (model download + push)
export WANDB_API_KEY=...         # optional: training logs
export GEMINI_API_KEY=...        # optional: AbsQA LLM-as-judge (Gemma/Gemini)
```

### 2. Reproduce everything (single command)

```bash
make all            # runs stages 1 → 7 with default configs
```

Or run stages individually:

```bash
make data           # 01 prepare data
make dapt           # 02 DAPT pre-training        (needs GPU)
make finetune       # 03 LoRA adapters            (needs GPU)
make infer          # 04 inference on 9 cells     (needs GPU)
make metrics        # 05 compute metrics
make stats          # 06 statistical analysis
make figures        # 07 figures
```

### 3. Reproduce only the analysis (no GPU needed)

If you already have the raw per-item scores (provided under
`results/predictions/`), you can regenerate every table and figure in the paper
on a laptop:

```bash
make metrics stats figures
```

---

## Hardware & runtime

| Stage | Hardware | Approx. time |
|---|---|---|
| DAPT | 1× A100 80GB | ~hours (corpus-dependent) |
| LoRA fine-tune (per adapter) | 1× A100 80GB | ~1–3 h |
| Inference (per system, 9 cells) | 1× A100 / 1× 24GB GPU (4-bit) | ~30–90 min |
| Metrics + stats + figures | CPU / laptop | < 5 min |

All generative baselines run **inference-only**; 14B-class models use 4-bit
quantization (`bitsandbytes`) via Unsloth.

---

## Reproducibility

- **Global seed** (`12181531` for data, `3407` for training) set in every stage via `enmed.utils.seed.set_seed`.
- **Config snapshots**: each run copies its resolved config into the output dir.
- **Pinned dependencies** in `requirements.txt`.
- **Item-level pairing preserved**: the same held-out test items are scored for every system, enabling the paired statistical tests.

---

## Repository layout

```
enmed/
├── configs/                 # YAML configs for every stage
├── src/enmed/
│   ├── data/                # loaders, formatting, splits, DAPT corpus
│   ├── training/            # DAPT + LoRA fine-tuning (Unsloth + TRL)
│   ├── inference/           # per-task generation (mcqa/extqa/absqa)
│   ├── evaluation/          # metrics + LLM-as-judge + aggregation
│   ├── stats/               # paired tests, BH/Bonferroni, Critical Difference
│   ├── viz/                 # all paper figures
│   └── utils/               # seeding, logging, IO
├── scripts/                 # 01..07 numbered CLI entry points
├── notebooks/               # original Colab notebooks (reference)
├── data/                    # raw + processed datasets (gitignored)
├── results/                 # predictions, metrics, stats, figures
├── tests/                   # unit tests for metrics & stats
├── Makefile                 # one-command reproduction
├── requirements.txt
└── setup.py
```

See [`docs/REPRODUCE.md`](docs/REPRODUCE.md) for a step-by-step walkthrough and
[`data/README.md`](data/README.md) for dataset access and licensing.

---

## Citation

```bibtex
@unpublished{abodoeloundou2025enmed,
  title  = {Cross-Lingual Domain Adaptation and Multi-Task Fine-Tuning
            for High-Fidelity Medical Language Models},
  author = {Abodo Eloundou, Brice Donald and Malykh, Valentin},
  note   = {Submitted to Neuroinformatics-2026.
            Under review. ITMO University / MTS Web Services, Saint Petersburg, Russia},
  year   = {2025}
}
```

## License

Code released under the MIT License (see [`LICENSE`](LICENSE)). Model weights
follow the Qwen3-14B (Apache 2.0) license; datasets follow their respective
licenses (see [`data/README.md`](data/README.md)).
