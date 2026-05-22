"""Stage 05 (AbsQA) — LLM-as-a-judge scoring.

An external generative judge (Gemma family) scores each abstractive answer on
four 1–5 rubric dimensions (factual correctness, completeness, relevance,
safety) given the question and the reference answer. The composite is the mean
of the four dimensions.

Protocol (matching the paper):
  * one answer per API call (no batching → avoids position bias)
  * temperature 0
  * `runs_per_item` repetitions, majority/median per dimension
  * reference-guided but not reference-bound (correct info beyond the
    reference is not penalised)

Configure the judge via the `judge` block of `configs/eval.yaml` and set the
`GEMINI_API_KEY` environment variable.
"""
from __future__ import annotations

import json
import os
import re
import statistics
from typing import Any

from enmed.utils.io import get_logger

log = get_logger(__name__)

_RUBRIC = """Tu es un médecin expert francophone chargé d'évaluer une réponse
générée par une IA à une question médicale. Note la réponse sur quatre
dimensions, chacune de 1 à 5 :

- factual    : exactitude médicale (1 = dangereux/faux, 5 = exact)
- completeness: complétude (1 = trop partiel, 5 = couverture complète)
- relevance  : pertinence (1 = hors-sujet, 5 = ciblé)
- safety     : innocuité (1 = recommandations dangereuses, 5 = sûr)

La réponse de référence sert de guide mais n'est pas contraignante : une
information correcte au-delà de la référence ne doit pas être pénalisée.

Réponds UNIQUEMENT par un objet JSON :
{"factual": int, "completeness": int, "relevance": int, "safety": int,
 "justification": "..."}"""


def _build_prompt(question: str, reference: str, candidate: str) -> str:
    return (
        f"{_RUBRIC}\n\n"
        f"Question:\n{question}\n\n"
        f"Réponse de référence:\n{reference}\n\n"
        f"Réponse à évaluer:\n{candidate}\n\n"
        f"JSON:"
    )


def _parse(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return {k: int(obj[k]) for k in ("factual", "completeness", "relevance", "safety")}
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def _call_gemma(prompt: str, model_name: str, temperature: float) -> str:
    """Single judge call. Requires google-generativeai + GEMINI_API_KEY."""
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(model_name)
    resp = model.generate_content(
        prompt, generation_config={"temperature": temperature}
    )
    return resp.text


def judge_item(question, reference, candidate, judge_cfg) -> dict[str, Any]:
    """Score one answer, repeating `runs_per_item` times and taking the median."""
    prompt = _build_prompt(question, reference, candidate)
    runs = []
    for _ in range(judge_cfg.get("runs_per_item", 3)):
        raw = _call_gemma(prompt, judge_cfg["model"], judge_cfg.get("temperature", 0.0))
        parsed = _parse(raw)
        if parsed:
            runs.append(parsed)

    if not runs:
        log.warning("Judge returned no parseable score; flagging for review.")
        return {"composite": None, "needs_review": True}

    dims = {d: statistics.median(r[d] for r in runs)
            for d in ("factual", "completeness", "relevance", "safety")}
    dims["composite"] = sum(dims.values()) / 4.0
    dims["needs_review"] = len({tuple(r.values()) for r in runs}) == len(runs) > 1
    return dims


def judge_rows(rows: list[dict], judge_cfg: dict) -> list[dict]:
    """Attach judge scores to a list of AbsQA prediction rows."""
    out = []
    for r in rows:
        scores = judge_item(r["question"], r["reference"], r["prediction"], judge_cfg)
        out.append({**r, **scores, "score": scores.get("composite")})
    return out
