from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_llm import dry_run_summary, load_config as load_llm_config  # noqa: E402
from train_qwen_tts import build_commands as build_tts_commands  # noqa: E402
from train_wan import build_command as build_wan_command, load_config as load_wan_config  # noqa: E402


def test_wan_command_has_single_gpu_parallelism_and_precompute() -> None:
    config = load_wan_config(PROJECT_ROOT / "configs/stages/video_wan21.toml")
    command = build_wan_command(config)
    assert command[:4] == ["accelerate", "launch", "--num_processes", "1"]
    assert "--model_name" in command
    assert command[command.index("--model_name") + 1] == "wan"
    assert "--training_type" in command
    assert "--enable_precomputation" in command
    assert "--precomputation_once" in command


def test_tts_dry_run_keeps_modal_paths_posix_on_windows() -> None:
    config = {
        "trainer_repo": "/opt/Qwen3-TTS",
        "raw_jsonl": "/mnt/suite/data/manifests/audio.train.jsonl",
        "prepared_jsonl": "/mnt/suite/data/processed/audio/train_with_codes.jsonl",
        "device": "cuda:0",
        "tokenizer_model_path": "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "output_dir": "/mnt/suite/adapters/audio/qwen3-tts",
        "speaker_name": "character",
    }
    commands = build_tts_commands(config)
    assert commands[0][1] == "/opt/Qwen3-TTS/finetuning/prepare_data.py"
    assert all("\\" not in argument for command in commands for argument in command)


def test_llm_summary_preserves_separate_coding_stage() -> None:
    config = load_llm_config(PROJECT_ROOT / "configs/stages/coding_qwen3.toml")
    summary = dry_run_summary(config)
    assert summary["stage"] == "coding"
    assert summary["model_id"] == "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    assert summary["target_modules"] == "all-linear"
