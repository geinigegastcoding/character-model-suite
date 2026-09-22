from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tomllib


def load_config(path: str | Path) -> dict:
    with Path(path).open("rb") as handle:
        config = tomllib.load(handle)
    if config.get("stage") not in {"text", "coding"}:
        raise ValueError("LLM trainer requires stage='text' or stage='coding'")
    for key in ("stage", "model_id", "data_path", "output_dir", "target_modules"):
        if not config.get(key):
            raise ValueError(f"LLM config is missing {key}")
    return config


def dry_run_summary(config: dict) -> dict:
    return {
        "stage": config["stage"],
        "model_id": config["model_id"],
        "method": config.get("method", "qlora"),
        "data_path": config["data_path"],
        "output_dir": config["output_dir"],
        "target_modules": config["target_modules"],
        "lora_rank": config.get("lora_rank", 16),
        "max_seq_length": config.get("max_seq_length", 4096),
        "gpu": config.get("gpu", "L40S"),
    }


def _format_messages(tokenizer, messages: list[dict]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def train(config: dict) -> None:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    data_path = Path(config["data_path"])
    if not data_path.is_file():
        raise RuntimeError(
            f"Training manifest not found: {data_path}. Upload it to the Modal Volume before starting this stage."
        )
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_dataset("json", data_files=str(data_path), split="train")
    if "split" in raw.column_names:
        train_rows = raw.filter(lambda item: item.get("split", "train") == "train")
    else:
        train_rows = raw
    if len(train_rows) == 0:
        raise RuntimeError("The manifest has no rows with split='train'.")

    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_seq_length = int(config.get("max_seq_length", 4096))

    def tokenize_row(row: dict) -> dict:
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("Each training row must contain a non-empty messages list")
        encoded = tokenizer(
            _format_messages(tokenizer, messages),
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    tokenized = train_rows.map(tokenize_row, remove_columns=train_rows.column_names)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        quantization_config=quantization,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=int(config.get("lora_rank", 16)),
        lora_alpha=int(config.get("lora_alpha", 32)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        target_modules=config["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.config.use_cache = False

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(config.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 16)),
        learning_rate=float(config.get("learning_rate", 2e-4)),
        num_train_epochs=float(config.get("num_train_epochs", 3)),
        logging_steps=int(config.get("logging_steps", 5)),
        save_steps=int(config.get("save_steps", 100)),
        save_total_limit=int(config.get("save_total_limit", 2)),
        gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
        bf16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
        seed=int(config.get("seed", 42)),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    metadata = {
        "stage": config["stage"],
        "model_id": config["model_id"],
        "gpu": os.environ.get("MODAL_GPU", os.environ.get("SUITE_GPU", "unknown")),
        "config": config,
    }
    (output_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a text or coding character adapter with 4-bit QLoRA.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.dry_run:
        print(json.dumps(dry_run_summary(config), indent=2))
        return 0
    train(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
