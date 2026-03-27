"""
train.py — VetQwen QLoRA fine-tuning script

Fine-tunes Qwen2.5-3B-Instruct with 4-bit QLoRA
using SFTTrainer from the trl library.

Usage:
    python scripts/train.py --config configs/train_default.yaml
    python scripts/train.py --config configs/train_default.yaml --lora-r 32 --run-name vetqwen_r32_full
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(path: str) -> dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(cfg: dict):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # type: ignore

    model_name = cfg["model"]["name"]
    q_cfg = cfg["quantization"]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=q_cfg["load_in_4bit"],
        bnb_4bit_quant_type=q_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, q_cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=q_cfg["bnb_4bit_use_double_quant"],
    )

    log.info(f"Loading model: {model_name} (4-bit QLoRA)")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=cfg["model"].get("trust_remote_code", True),
        revision=cfg["model"].get("revision", "main"),
    )
    model.config.use_cache = False  # Required for gradient checkpointing

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=cfg["model"].get("trust_remote_code", True),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    log.info(f"Model loaded. Parameters: {model.num_parameters() / 1e9:.2f}B")
    return model, tokenizer


# ---------------------------------------------------------------------------
# LoRA setup
# ---------------------------------------------------------------------------


def apply_lora(model, cfg: dict):
    from peft import LoraConfig, get_peft_model  # type: ignore

    l_cfg = cfg["lora"]

    # Enable gradient checkpointing and input gradients manually —
    # prepare_model_for_kbit_training upcasts params to float32 which OOMs on 8GB GPUs.
    # We replicate its essential behavior without the upcast.
    if cfg["training"].get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=l_cfg["r"],
        lora_alpha=l_cfg["lora_alpha"],
        lora_dropout=l_cfg["lora_dropout"],
        bias=l_cfg["bias"],
        task_type=l_cfg["task_type"],
        target_modules=l_cfg["target_modules"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, lora_config


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_datasets(cfg: dict):
    from datasets import load_dataset  # type: ignore

    d_cfg = cfg["data"]
    fraction = d_cfg.get("dataset_fraction", 1.0)

    def _load(path: str, split_name: str):
        ds = load_dataset("json", data_files=path, split="train")
        if fraction < 1.0:
            total = len(ds)
            n = int(total * fraction)
            ds = ds.select(range(n))
            log.info(f"  {split_name}: using {n}/{total} samples (fraction={fraction})")
        else:
            log.info(f"  {split_name}: {len(ds)} samples")
        return ds

    train_ds = _load(d_cfg["train_file"], "train")
    val_ds = _load(d_cfg["val_file"], "val")
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def build_training_args(cfg: dict):
    from trl import SFTConfig  # type: ignore

    t_cfg = cfg["training"]

    return SFTConfig(
        output_dir=t_cfg["output_dir"],
        num_train_epochs=t_cfg["num_train_epochs"],
        per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=t_cfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
        gradient_checkpointing=t_cfg["gradient_checkpointing"],
        optim=t_cfg["optim"],
        learning_rate=t_cfg["learning_rate"],
        lr_scheduler_type=t_cfg["lr_scheduler_type"],
        warmup_ratio=t_cfg["warmup_ratio"],
        max_seq_length=cfg["data"]["max_seq_length"],
        bf16=t_cfg.get("bf16", True),
        fp16=t_cfg.get("fp16", False),
        logging_steps=t_cfg["logging_steps"],
        eval_strategy=t_cfg["eval_strategy"],
        eval_steps=t_cfg["eval_steps"],
        save_strategy=t_cfg["save_strategy"],
        save_steps=t_cfg.get("save_steps", t_cfg.get("eval_steps", 50)),
        save_total_limit=t_cfg.get("save_total_limit", 2),
        load_best_model_at_end=t_cfg["load_best_model_at_end"],
        metric_for_best_model=t_cfg.get("metric_for_best_model", "eval_loss"),
        report_to=t_cfg.get("report_to", "tensorboard"),
        run_name=t_cfg.get("run_name", "vetqwen"),
    )


def train(cfg: dict) -> None:
    from trl import SFTTrainer  # type: ignore

    # Load model
    model, tokenizer = load_model_and_tokenizer(cfg)
    model, lora_config = apply_lora(model, cfg)

    # Load data
    log.info("Loading datasets ...")
    train_ds, val_ds = load_datasets(cfg)

    # Build trainer
    training_args = build_training_args(cfg)

    # Convert ChatML messages to a single string for SFTTrainer
    def formatting_func(example):
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_config,
        args=training_args,
        formatting_func=formatting_func,
    )

    log.info("Starting training ...")
    trainer.train()

    # Save LoRA adapter
    adapter_dir = Path("adapter")
    adapter_dir.mkdir(exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    log.info(f"LoRA adapter saved to {adapter_dir}/")
    log.info("Training complete. Adapter and tokenizer saved.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="VetQwen QLoRA fine-tuning")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument("--lora-r", type=int, default=None, help="Override LoRA rank r")
    parser.add_argument("--run-name", type=str, default=None, help="Override run name")
    parser.add_argument(
        "--dataset-fraction",
        type=float,
        default=None,
        help="Override dataset fraction (0-1)",
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=None, help="Override max_seq_length"
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Override model name/path"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Apply CLI overrides
    if args.lora_r is not None:
        cfg["lora"]["r"] = args.lora_r
    if args.run_name is not None:
        cfg["training"]["run_name"] = args.run_name
    if args.dataset_fraction is not None:
        cfg["data"]["dataset_fraction"] = args.dataset_fraction
    if args.max_seq_length is not None:
        cfg["data"]["max_seq_length"] = args.max_seq_length
    if args.model is not None:
        cfg["model"]["name"] = args.model

    log.info(f"Config loaded from {args.config}")
    log.info(
        f"Run: {cfg['training'].get('run_name')} | LoRA r={cfg['lora']['r']} | "
        f"model={cfg['model']['name']}"
    )

    train(cfg)


if __name__ == "__main__":
    main()
