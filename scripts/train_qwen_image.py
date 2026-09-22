from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tomllib


DEFAULT_TRAINER = "/opt/diffusers/examples/dreambooth/train_dreambooth_lora_qwen_image.py"
VALUE_FLAGS = {
    "pretrained_model_name_or_path": "model_id",
    "instance_data_dir": "instance_data_dir",
    "instance_prompt": "instance_prompt",
    "output_dir": "output_dir",
    "resolution": "resolution",
    "mixed_precision": "mixed_precision",
    "train_batch_size": "train_batch_size",
    "gradient_accumulation_steps": "gradient_accumulation_steps",
    "learning_rate": "learning_rate",
    "max_train_steps": "max_train_steps",
    "rank": "rank",
    "lora_alpha": "lora_alpha",
    "seed": "seed",
    "validation_prompt": "validation_prompt",
}
BOOLEAN_FLAGS = {
    "use_8bit_adam": "use_8bit_adam",
    "gradient_checkpointing": "gradient_checkpointing",
    "offload": "offload",
    "cache_latents": "cache_latents",
    "skip_final_inference": "skip_final_inference",
    "allow_tf32": "allow_tf32",
}


def load_config(path: str | Path) -> dict:
    with Path(path).open("rb") as handle:
        config = tomllib.load(handle)
    if config.get("stage") != "image":
        raise ValueError("Qwen Image trainer requires stage='image'")
    for key in ("model_id", "instance_data_dir", "instance_prompt", "output_dir"):
        if not config.get(key):
            raise ValueError(f"image config is missing {key}")
    return config


def build_command(config: dict, trainer: str = DEFAULT_TRAINER) -> list[str]:
    command = ["accelerate", "launch", trainer]
    for flag, key in VALUE_FLAGS.items():
        if key in config and config[key] is not None:
            command.extend([f"--{flag}", str(config[key])])
    for flag, key in BOOLEAN_FLAGS.items():
        if config.get(key, False):
            command.append(f"--{flag}")
    hub_repo = os.environ.get("HF_REPO_ID")
    if hub_repo:
        command.extend(["--push_to_hub", "--hub_model_id", hub_repo])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the pinned Qwen Image DreamBooth-LoRA trainer.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--trainer", default=DEFAULT_TRAINER)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    command = build_command(config, args.trainer)
    print(" ".join(command))
    if args.dry_run:
        return 0

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)
    metadata = {
        "model_id": config["model_id"],
        "trainer": args.trainer,
        "gpu": os.environ.get("MODAL_GPU", os.environ.get("SUITE_GPU", "unknown")),
        "config": str(args.config),
        "command": command,
    }
    (output_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
