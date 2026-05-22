"""Stage 04 — Inference across the 9 (task, shot) cells.

For each system and each (task, shot) cell, produce one prediction per shared
test item. Predictions are written to JSONL so that downstream metric and
statistical stages operate on a fixed, item-aligned set across all systems
(this item-level alignment is what makes the paired tests valid).

MCQA uses log-probability decoding by default: each option is scored under
teacher forcing and the highest-likelihood option is chosen. This avoids
format-compliance failures in zero/few-shot settings.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from enmed.data.formatting import build_prompt
from enmed.utils.io import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# MCQA — log-probability decoding
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_mcqa_logprob(model, tokenizer, example, shots) -> str:
    """Return the letter of the highest-likelihood option."""
    messages = build_prompt("mcqa", example, shots)
    prefix = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    scores: dict[str, float] = {}
    for letter, text in example["options"].items():
        full = prefix + f" {letter}) {text}"
        enc = tokenizer(full, return_tensors="pt").to(model.device)
        plen = tokenizer(prefix, return_tensors="pt")["input_ids"].shape[1]
        logits = model(**enc).logits[0, plen - 1 : -1]
        ids = enc["input_ids"][0, plen:]
        lp = F.log_softmax(logits, dim=-1)
        scores[letter] = lp[range(len(ids)), ids].sum().item() / max(len(ids), 1)
    return max(scores, key=scores.get)


# --------------------------------------------------------------------------- #
# Free-form generation (ExtQA, AbsQA, and MCQA "generate" mode)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def generate(model, tokenizer, task, example, shots, max_new_tokens, temperature) -> str:
    messages = build_prompt(task, example, shots)
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    out = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-6),
    )
    text = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
    return text.strip()


def predict_cell(model, tokenizer, task, shot, test_rows, shot_pool, eval_cfg, shot_selector):
    """Yield {id, prediction, reference, ...} for one (task, shot) cell."""
    inf = eval_cfg["inference"]
    for ex in test_rows:
        shots = shot_selector(ex, shot_pool, shot) if shot else []
        if task == "mcqa" and inf.get("mcqa_decoding", "logprob") == "logprob":
            pred = predict_mcqa_logprob(model, tokenizer, ex, shots)
        else:
            pred = generate(
                model, tokenizer, task, ex, shots,
                max_new_tokens=inf["max_new_tokens"][task],
                temperature=inf.get("temperature", 0.0),
            )
        yield {
            "id": ex["id"],
            "task": task,
            "shot": shot,
            "prediction": pred,
            "reference": ex["answer"],
            "specialty": ex.get("specialty", "unknown"),
        }
