from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tomllib


DEFAULT_TRAINER = "/opt/finetrainers/train.py"


def load_config(path: str | Path) -> dict:
    with Path(path).open("rb") as handle:
        config = tomllib.load(handle)
    if config.get("stage") != "video":
        raise ValueError("Wan trainer requires stage='video'")
    for key in ("model_id", "model_name", "trainer", "dataset_config", "data_root", "output_dir"):
        if not config.get(key):
            raise ValueError(f"video config is missing {key}")
    return config


def _add_value(command: list[str], flag: str, config: dict, key: str) -> None:
    if key in config and config[key] is not None:
        command.extend([f"--{flag}", str(config[key])])


def _add_bool(command: list[str], flag: str, config: dict, key: str) -> None:
    if config.get(key, False):
        command.append(f"--{flag}")


def build_command(config: dict, trainer: str | None = None) -> list[str]:
    command = [
        "accelerate",
        "launch",
        "--num_processes",
        str(config.get("num_processes", 1)),
        trainer or config["trainer"] or DEFAULT_TRAINER,
    ]

    value_flags = (
        ("parallel_backend", "parallel_backend"),
        ("pp_degree", "pp_degree"),
        ("dp_degree", "dp_degree"),
        ("dp_shards", "dp_shards"),
        ("cp_degree", "cp_degree"),
        ("tp_degree", "tp_degree"),
        ("model_name", "model_name"),
        ("pretrained_model_name_or_path", "model_id"),
        ("dataset_config", "dataset_config"),
        ("dataset_shuffle_buffer_size", "dataset_shuffle_buffer_size"),
        ("precomputation_items", "precomputation_items"),
        ("dataloader_num_workers", "dataloader_num_workers"),
        ("flow_weighting_scheme", "flow_weighting_scheme"),
        ("training_type", "training_type"),
        ("seed", "seed"),
        ("batch_size", "batch_size"),
        ("train_steps", "train_steps"),
        ("rank", "rank"),
        ("lora_alpha", "lora_alpha"),
        ("target_modules", "target_modules"),
        ("gradient_accumulation_steps", "gradient_accumulation_steps"),
        ("checkpointing_steps", "checkpointing_steps"),
        ("checkpointing_limit", "checkpointing_limit"),
        ("optimizer", "optimizer"),
        ("lr", "lr"),
        ("lr_scheduler", "lr_scheduler"),
        ("lr_warmup_steps", "lr_warmup_steps"),
        ("lr_num_cycles", "lr_num_cycles"),
        ("beta1", "beta1"),
        ("beta2", "beta2"),
        ("weight_decay", "weight_decay"),
        ("epsilon", "epsilon"),
        ("max_grad_norm", "max_grad_norm"),
        ("validation_dataset_file", "validation_dataset_file"),
        ("validation_steps", "validation_steps"),
        ("output_dir", "output_dir"),
        ("tracker_name", "tracker_name"),
        ("report_to", "report_to"),
    )
    for flag, key in value_flags:
        _add_value(command, flag, config, key)

    for flag, key in (
        ("enable_precomputation", "enable_precomputation"),
        ("precomputation_once", "precomputation_once"),
        ("gradient_checkpointing", "gradient_checkpointing"),
        ("enable_slicing", "enable_slicing"),
        ("enable_tiling", "enable_tiling"),
        ("enable_model_cpu_offload", "enable_model_cpu_offload"),
    ):
        _add_bool(command, flag, config, key)
    return command


def _has_video_caption_pair(data_root: Path) -> bool:
    extensions = {".mp4", ".webm", ".mov", ".mkv"}
    return any(
        path.suffix.lower() in extensions and path.with_suffix(".txt").is_file()
        for path in data_root.iterdir()
        if path.is_file()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the finetrainers Wan video LoRA recipe.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--trainer")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    command = build_command(config, args.trainer)
    print(" ".join(command))
    if args.dry_run:
        return 0

    data_root = Path(config["data_root"])
    if not data_root.is_dir() or not _has_video_caption_pair(data_root):
        raise RuntimeError(
            f"No video/.txt caption pairs found in {data_root}. Upload clips and matching captions first with "
            "`modal volume put character-model-suite-data data/raw/video /data/raw/video/`."
        )
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd="/opt/finetrainers", check=True, env={**os.environ, "WANDB_MODE": "offline"})
    metadata = {
        "model_id": config["model_id"],
        "trainer": command[4],
        "gpu": os.environ.get("MODAL_GPU", os.environ.get("SUITE_GPU", "unknown")),
        "config": str(args.config),
        "command": command,
    }
    (output_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
