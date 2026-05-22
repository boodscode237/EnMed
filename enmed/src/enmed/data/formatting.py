"""Prompt templates for the three French medical QA tasks.

Identical templates are applied across every generative system so that score
differences reflect model capability, not prompt advantage. Few-shot
demonstrations are prepended in the same format. All prompts instruct the model
to answer in French.
"""
from __future__ import annotations

from typing import Sequence

# --------------------------------------------------------------------------- #
# MCQA — multiple choice
# --------------------------------------------------------------------------- #
MCQA_SYSTEM = (
    "Tu es un expert médical francophone. Réponds à la question à choix multiple "
    "en indiquant uniquement la lettre de la meilleure réponse."
)


def format_mcqa(question: str, options: dict[str, str]) -> str:
    """Render an MCQA prompt with a lettered option list."""
    opts = "\n".join(f"{letter}) {text}" for letter, text in options.items())
    return f"Question: {question}\n{opts}\nRéponse:"


# --------------------------------------------------------------------------- #
# ExtQA — extractive span
# --------------------------------------------------------------------------- #
EXTQA_SYSTEM = (
    "Tu es un expert médical francophone. À partir du cas clinique fourni, "
    "extrais et recopie mot pour mot le passage exact qui répond à la question. "
    "Ne reformule pas."
)


def format_extqa(context: str, question: str) -> str:
    return f"Cas clinique:\n{context}\n\nQuestion: {question}\nPassage exact:"


# --------------------------------------------------------------------------- #
# AbsQA — abstractive generation
# --------------------------------------------------------------------------- #
ABSQA_SYSTEM = (
    "Tu es un expert médical francophone. Réponds à la question médicale de "
    "manière claire, complète et sûre. Réponds en français."
)


def format_absqa(question: str, context: str | None = None) -> str:
    if context:
        return f"Contexte:\n{context}\n\nQuestion: {question}\nRéponse:"
    return f"Question: {question}\nRéponse:"


# --------------------------------------------------------------------------- #
# Few-shot assembly
# --------------------------------------------------------------------------- #
SYSTEM_BY_TASK = {"mcqa": MCQA_SYSTEM, "extqa": EXTQA_SYSTEM, "absqa": ABSQA_SYSTEM}
_FORMATTERS = {"mcqa": format_mcqa, "extqa": format_extqa, "absqa": format_absqa}


def build_prompt(
    task: str,
    example: dict,
    shots: Sequence[dict] = (),
) -> list[dict]:
    """Build a chat-format prompt (list of role/content dicts) for any task.

    `example` and each shot are task-specific dicts; see `enmed.data.loaders`
    for the field schema. `shots` are the few-shot demonstrations (already
    selected by the chosen strategy).
    """
    fmt = _FORMATTERS[task]
    messages = [{"role": "system", "content": SYSTEM_BY_TASK[task]}]

    for shot in shots:
        messages.append({"role": "user", "content": _render(task, fmt, shot)})
        messages.append({"role": "assistant", "content": shot["answer"]})

    messages.append({"role": "user", "content": _render(task, fmt, example)})
    return messages


def _render(task: str, fmt, ex: dict) -> str:
    if task == "mcqa":
        return fmt(ex["question"], ex["options"])
    if task == "extqa":
        return fmt(ex["context"], ex["question"])
    return fmt(ex["question"], ex.get("context"))
