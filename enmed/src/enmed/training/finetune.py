"""Stage 03 — Supervised LoRA fine-tuning.

Trains one LoRA adapter on top of the EnMed-DAPT backbone. The set of tasks in
the training mixture is controlled by `tasks_in_mix`:

  * EnMed-Unified : ["mcqa", "extqa", "absqa"]  (multi-task)
  * EnMed-MCQA    : ["mcqa"]
  * EnMed-ExtQA   : ["extqa"]
  * EnMed-AbsQA   : ["absqa"]

Each training example is rendered with the same chat templates used at
inference (`enmed.data.formatting`), then trained response-only so the loss is
computed on the assistant turn.
"""
from __future__ import annotations

from pathlib import Path

from datasets import Dataset, concatenate_datasets

from enmed.data.formatting import build_prompt
from enmed.data.loaders import load_task
from enmed.training.model_utils import attach_lora, load_backbone, setup_chat_template
from enmed.utils.io import get_logger, set_seed, snapshot_config

log = get_logger(__name__)


def _to_text(tokenizer, task: str, example: dict) -> str:
    """Render a training example into a single chat-formatted string."""
    messages = build_prompt(task, example, shots=[])
    messages.append({"role": "assistant", "content": str(example["answer"])})
    return tokenizer.apply_chat_template(messages, tokenize=False)


def build_mixture(tokenizer, cfg: dict) -> Dataset:
    """Assemble the supervised training mixture from the configured tasks."""
    parts = []
    for task in cfg["tasks_in_mix"]:
        rows = load_task(task, cfg, split="train")
        texts = [{"text": _to_text(tokenizer, task, ex)} for ex in rows]
        parts.append(Dataset.from_list(texts))
        log.info("Task %s contributes %d examples", task, len(texts))
    mixture = concatenate_datasets(parts).shuffle(seed=cfg.get("seed", 12181531))
    log.info("Training mixture: %d examples", len(mixture))
    return mixture


def run_finetune(cfg: dict) -> str:
    """Fine-tune a LoRA adapter and return the saved path."""
    set_seed(cfg.get("train_seed", 3407))
    out_dir = Path(cfg["output_root"]) / cfg["output_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_config(cfg, out_dir)

    # Load the DAPT backbone if available, else the base backbone.
    dapt_path = Path(cfg["output_root"]) / cfg.get("base_adapter", "")
    if dapt_path.exists():
        cfg = {**cfg, "backbone": str(dapt_path)}
        log.info("Building on DAPT checkpoint at %s", dapt_path)

    model, tokenizer = load_backbone(cfg)
    tokenizer = setup_chat_template(tokenizer)
    model = attach_lora(model, cfg)

    mixture = build_mixture(tokenizer, cfg)

    from trl import SFTConfig, SFTTrainer
    from unsloth.chat_templates import train_on_responses_only

    t = cfg["train"]
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=mixture,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=t["per_device_train_batch_size"],
            gradient_accumulation_steps=t["gradient_accumulation_steps"],
            warmup_ratio=t["warmup_ratio"],
            num_train_epochs=t["num_train_epochs"],
            learning_rate=t["learning_rate"],
            logging_steps=t.get("logging_steps", 1),
            optim=t.get("optim", "adamw_8bit"),
            weight_decay=t.get("weight_decay", 0.0),
            lr_scheduler_type=t.get("lr_scheduler_type", "cosine"),
            seed=cfg.get("train_seed", 3407),
            output_dir=str(out_dir),
            report_to="wandb" if cfg.get("use_wandb") else "none",
        ),
    )

    # Compute loss on the assistant response only.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    log.info("Starting fine-tune %s → %s", cfg["output_name"], out_dir)
    trainer.train()
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    log.info("Fine-tune complete. Saved to %s", out_dir)
    return str(out_dir)
