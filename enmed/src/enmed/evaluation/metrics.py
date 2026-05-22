"""Stage 05 — metric computation per task.

MCQA  : accuracy (and macro-F1, Hamming for multi-answer)
ExtQA : token-level F1 and exact match (French-aware normalisation)
AbsQA : ROUGE-L, BLEU-4, BERTScore (CamemBERT backbone); the LLM-as-judge
        composite is computed separately in `llm_judge.py`.

Every function returns *per-item* scores as well as the aggregate, because the
statistical layer operates on item-level vectors.
"""
from __future__ import annotations

import re
import string
import unicodedata

# --------------------------------------------------------------------------- #
# French-aware normalisation
# --------------------------------------------------------------------------- #
_ARTICLES = {"le", "la", "les", "l", "un", "une", "des", "du", "de", "d"}


def normalize_fr(text: str) -> str:
    """Lowercase, strip accents/punctuation, and drop leading French articles."""
    text = text.lower().strip()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )
    text = "".join(ch for ch in text if ch not in string.punctuation)
    tokens = [t for t in text.split() if t not in _ARTICLES]
    return " ".join(tokens)


# --------------------------------------------------------------------------- #
# MCQA
# --------------------------------------------------------------------------- #
def mcqa_item_correct(pred: str, ref) -> float:
    """1.0 if the predicted letter(s) match the gold answer(s), else 0.0."""
    def letters(x):
        return set(re.findall(r"[A-E]", str(x).upper()))

    return float(letters(pred) == letters(ref))


def hamming_accuracy(pred: str, ref) -> float:
    """Partial credit for multi-answer MCQA (Jaccard over option letters)."""
    def letters(x):
        return set(re.findall(r"[A-E]", str(x).upper()))

    p, r = letters(pred), letters(ref)
    if not p and not r:
        return 1.0
    return len(p & r) / max(len(p | r), 1)


# --------------------------------------------------------------------------- #
# ExtQA — token-level F1 / exact match
# --------------------------------------------------------------------------- #
def token_f1(pred: str, ref: str) -> float:
    p_tokens = normalize_fr(pred).split()
    r_tokens = normalize_fr(ref).split()
    if not p_tokens and not r_tokens:
        return 1.0
    if not p_tokens or not r_tokens:
        return 0.0
    common = _multiset_intersection(p_tokens, r_tokens)
    if common == 0:
        return 0.0
    precision = common / len(p_tokens)
    recall = common / len(r_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, ref: str) -> float:
    return float(normalize_fr(pred) == normalize_fr(ref))


def _multiset_intersection(a, b) -> int:
    from collections import Counter

    ca, cb = Counter(a), Counter(b)
    return sum((ca & cb).values())


# --------------------------------------------------------------------------- #
# AbsQA — surface + semantic similarity
# --------------------------------------------------------------------------- #
def rouge_l(pred: str, ref: str) -> float:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    return scorer.score(ref, pred)["rougeL"].fmeasure


def bleu4(preds: list[str], refs: list[str]) -> float:
    import sacrebleu

    return sacrebleu.corpus_bleu(preds, [refs]).score / 100.0


def bertscore_fr(preds: list[str], refs: list[str], model="dangvantuan/sentence-camembert-base"):
    """Per-item BERTScore F1 using a French (CamemBERT) backbone."""
    from bert_score import score

    _, _, f1 = score(preds, refs, lang="fr", model_type=model, verbose=False)
    return f1.tolist()


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
def score_items(task: str, rows: list[dict]) -> list[dict]:
    """Attach a per-item `score` (primary metric) to each prediction row."""
    out = []
    for r in rows:
        if task == "mcqa":
            s = mcqa_item_correct(r["prediction"], r["reference"])
        elif task == "extqa":
            s = token_f1(r["prediction"], r["reference"])
        elif task == "absqa":
            s = rouge_l(r["prediction"], r["reference"])  # surface proxy; judge is separate
        else:
            raise ValueError(task)
        out.append({**r, "score": s})
    return out
