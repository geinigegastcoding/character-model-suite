from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tomllib


def load_config(path: str | Path) -> dict:
    with Path(path).open("rb") as handle:
        config = tomllib.load(handle)
    if config.get("stage") != "audio":
        raise ValueError("Qwen3-TTS trainer requires stage='audio'")
    for key in (
        "model_id",
        "trainer_repo",
        "raw_jsonl",
        "prepared_jsonl",
        "output_dir",
        "tokenizer_model_path",
        "device",
        "speaker_name",
    ):
        if not config.get(key):
            raise ValueError(f"audio config is missing {key}")
    return config


def build_commands(config: dict) -> list[list[str]]:
    # Modal runs Linux even when this dry-run is generated on Windows.
    repo = PurePosixPath(config["trainer_repo"])
    finetuning = repo / "finetuning"
    return [
        [
            "python",
            str(finetuning / "prepare_data.py"),
            "--device",
            str(config["device"]),
            "--tokenizer_model_path",
            str(config["tokenizer_model_path"]),
            "--input_jsonl",
            str(config["raw_jsonl"]),
            "--output_jsonl",
            str(config["prepared_jsonl"]),
        ],
        [
            "python",
            str(finetuning / "sft_12hz.py"),
            "--init_model_path",
            str(config["model_id"]),
            "--output_model_path",
            str(config["output_dir"]),
            "--train_jsonl",
            str(config["prepared_jsonl"]),
            "--batch_size",
            str(config.get("batch_size", 2)),
            "--lr",
            str(config.get("learning_rate", 2e-5)),
            "--num_epochs",
            str(config.get("num_epochs", 3)),
            "--speaker_name",
            str(config["speaker_name"]),
        ],
    ]


def _patch_attention_implementation(script: Path, implementation: str) -> None:
    """Use eager attention when flash-attn is not installed in the Modal image."""
    source = script.read_text(encoding="utf-8")
    marker = 'attn_implementation="flash_attention_2"'
    replacement = f'attn_implementation="{implementation}"'
    if marker not in source and replacement not in source:
        raise RuntimeError(f"Qwen3-TTS script changed; expected attention marker was not found in {script}")
    if marker in source:
        script.write_text(source.replace(marker, replacement), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the official Qwen3-TTS prepare-data and SFT stages.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    commands = build_commands(config)
    for command in commands:
        print(" ".join(command))
    if args.dry_run:
        return 0

    raw_jsonl = Path(config["raw_jsonl"])
    if not raw_jsonl.is_file():
        raise RuntimeError(
            f"Audio manifest not found: {raw_jsonl}. Upload it with `modal volume put "
            "character-model-suite-data data/manifests/audio.train.jsonl /data/manifests/audio.train.jsonl`."
        )
    Path(config["prepared_jsonl"]).parent.mkdir(parents=True, exist_ok=True)
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
    sft_script = Path(config["trainer_repo"]) / "finetuning" / "sft_12hz.py"
    _patch_attention_implementation(sft_script, str(config.get("attn_implementation", "eager")))
    for command in commands:
        subprocess.run(command, cwd=str(sft_script.parent), check=True)

    metadata = {
        "model_id": config["model_id"],
        "trainer": str(sft_script),
        "gpu": os.environ.get("MODAL_GPU", os.environ.get("SUITE_GPU", "unknown")),
        "config": str(args.config),
        "commands": commands,
    }
    output_dir = Path(config["output_dir"])
    (output_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
