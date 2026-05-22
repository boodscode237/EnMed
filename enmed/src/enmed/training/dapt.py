"""Stage 02 — Domain-Adaptive Continual Pre-Training (DAPT).

Continual pre-training of the Qwen3-14B backbone on the Mannion et al. (2026)
French health corpus. No task supervision is used here; the objective is plain
causal-LM continuation over in-domain French clinical/biomedical text. The
resulting checkpoint (EnMed-DAPT) is the substrate for all downstream adapters.
"""
from __future__ import annotations

from pathlib import Path

from enmed.data.loaders import load_dapt_corpus
from enmed.training.model_utils import attach_lora, load_backbone, setup_chat_template
from enmed.utils.io import get_logger, set_seed, snapshot_config

log = get_logger(__name__)


def run_dapt(cfg: dict) -> str:
    """Run DAPT and return the path to the saved adapter/checkpoint."""
    set_seed(cfg.get("train_seed", 3407))
    out_dir = Path(cfg["output_root"]) / cfg["output_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_config(cfg, out_dir)

    model, tokenizer = load_backbone(cfg)
    tokenizer = setup_chat_template(tokenizer)
    model = attach_lora(model, cfg)

    corpus = load_dapt_corpus(cfg)
    text_field = cfg["dapt_corpus"].get("text_field", "text")

    from trl import SFTConfig, SFTTrainer

    t = cfg["train"]
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=corpus,
        args=SFTConfig(
            dataset_text_field=text_field,
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
            save_steps=t.get("save_steps", 200),
            save_total_limit=t.get("save_total_limit", 1),
            output_dir=str(out_dir),
            report_to="wandb" if cfg.get("use_wandb") else "none",
        ),
    )

    log.info("Starting DAPT → %s", out_dir)
    trainer.train()
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    log.info("DAPT complete. Saved to %s", out_dir)
    return str(out_dir)
