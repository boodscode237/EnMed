# Datasets

This project uses three French medical QA datasets for evaluation and one
French health corpus for domain-adaptive pre-training. None of the data is
redistributed in this repository; download it from the sources below into
`data/raw/` and run `make data` to produce the processed splits.

## Evaluation datasets

| Task | Dataset | Source | License |
|---|---|---|---|
| MCQA | FrenchMedMCQA | DrBenchmark (Labrak et al., 2022/2024) | see source |
| ExtQA | CAS clinical cases | Grabar et al., 2020 | see source |
| AbsQA | MediQAl | Bazoge, 2025 | see source |

## DAPT pre-training corpus

| Corpus | Source |
|---|---|
| French health corpus | Mannion et al., 2026 (arXiv:2604.06903) |

## Expected layout under `data/raw/`

```
data/raw/
├── CAS.jsonl                    # ExtQA   (fields: context/case, question, answer/span)
├── MediQAl.jsonl                # AbsQA   (fields: question, context?, answer/long_answer)
└── french-health-corpus.jsonl   # DAPT    (field: text)
```

FrenchMedMCQA is loaded directly from the Hugging Face Hub via
`DrBenchmark/FrenchMedMCQA` (configurable in `configs/data.yaml`). Adjust the
field-mapping functions in `src/enmed/data/loaders.py` if the column names of
the release you download differ.

## Splits

`make data` writes `data/processed/{task}_{train,dev,test}.jsonl`. The **same**
held-out test items are scored for every system — this item-level alignment is
what makes the paired statistical tests valid.
