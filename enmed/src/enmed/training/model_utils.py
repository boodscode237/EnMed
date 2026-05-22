"""Backbone loading and PEFT/LoRA attachment via Unsloth.

Mirrors the configuration used in the original training notebook (Unsloth
`FastLanguageModel`), with all hyper-parameters surfaced through the YAML config.
"""
from __future__ import annotations

from enmed.utils.io import get_logger

log = get_logger(__name__)


def load_backbone(cfg: dict):
    """Load the 4-bit quantised backbone and tokenizer with Unsloth."""
    from unsloth import FastLanguageModel  # imported lazily (heavy, GPU-only)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["backbone"],
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=cfg.get("load_in_4bit", True),
        load_in_8bit=False,
        full_finetuning=False,
    )
    log.info("Loaded backbone %s", cfg["backbone"])
    return model, tokenizer


def attach_lora(model, cfg: dict):
    """Wrap the model with a LoRA adapter using the config's `lora` block."""
    from unsloth import FastLanguageModel

    lora = cfg["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora["r"],
        target_modules=lora["target_modules"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora.get("dropout", 0.0),
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.get("train_seed", 3407),
        use_rslora=False,
        loftq_config=None,
    )
    log.info("Attached LoRA (r=%d, alpha=%d)", lora["r"], lora["alpha"])
    return model


def setup_chat_template(tokenizer):
    """Apply the Qwen3 chat template used throughout training and inference."""
    from unsloth.chat_templates import get_chat_template

    return get_chat_template(tokenizer, chat_template="qwen3-thinking")
