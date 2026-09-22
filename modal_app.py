from __future__ import annotations

import os
from pathlib import Path
import subprocess

import modal


ROOT = Path(__file__).parent
VOLUME_NAME = "character-model-suite-data"
HF_SECRET_NAME = "character-model-suite-hf"
IMAGE_GPU = "A100-80GB"
VIDEO_GPU = "A100-80GB"
AUDIO_GPU = "L40S"
TEXT_GPU = "L40S"
CODING_GPU = ["L40S", "A100-80GB"]
GPU = os.environ.get("SUITE_GPU", IMAGE_GPU)
MOUNT = "/mnt/suite"
DIFFUSERS_REF = "8b3c707ebd3ec4881f4190cf42931da07eaf3b65"
FINETRAINERS_REF = "v0.2.0"
QWEN3_TTS_REF = "022e286b98fbec7e1e916cb940cdf532cd9f488e"

app = modal.App("character-model-suite")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .run_commands("git clone --depth 1 https://github.com/huggingface/diffusers.git /opt/diffusers")
    .pip_install(
        "torch",
        "torchvision",
        "accelerate",
        "bitsandbytes",
        f"diffusers @ git+https://github.com/huggingface/diffusers.git@{DIFFUSERS_REF}",
        "transformers",
        "peft",
        "datasets",
        "safetensors",
        "huggingface_hub",
        "ftfy",
        "tensorboard",
        "Jinja2",
    )
    .env({"HF_HOME": f"{MOUNT}/cache/huggingface", "SUITE_GPU": GPU})
    .add_local_dir(
        ROOT,
        remote_path="/root/project",
        copy=True,
        ignore=[".git", ".venv", "__pycache__", ".pytest_cache", "data/raw", "data/processed", "adapters", "runs"],
    )
)

video_image = (
    base_image
    .run_commands(
        f"git clone --depth 1 --branch {FINETRAINERS_REF} "
        "https://github.com/huggingface/finetrainers.git /opt/finetrainers"
    )
    .pip_install(
        "decord",
        "torchdata",
        "torchao",
        "kornia",
        "pandas",
        "imageio-ffmpeg",
        "hf_transfer",
        "sentencepiece",
        "wandb",
    )
)

audio_image = (
    base_image
    .run_commands(
        "git clone --depth 1 https://github.com/QwenLM/Qwen3-TTS.git /opt/Qwen3-TTS "
        f"&& git -C /opt/Qwen3-TTS checkout {QWEN3_TTS_REF}"
    )
    .pip_install(
        f"qwen-tts @ git+https://github.com/QwenLM/Qwen3-TTS.git@{QWEN3_TTS_REF}",
        "librosa",
        "soundfile",
        "torchaudio",
    )
)


@app.function(image=base_image, volumes={MOUNT: volume}, timeout=10 * 60)
def smoke() -> str:
    """Check that the Modal image, project files, and persistent Volume are usable."""
    volume.reload()
    Path(MOUNT).mkdir(parents=True, exist_ok=True)
    marker = Path(MOUNT) / "smoke.txt"
    marker.write_text("character-model-suite ok\n", encoding="utf-8")
    volume.commit()
    return f"ok: {marker}"


@app.function(
    image=base_image,
    gpu=GPU,
    volumes={MOUNT: volume},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    timeout=24 * 60 * 60,
)
def train_image() -> str:
    """Run the official Diffusers Qwen Image DreamBooth-LoRA example."""
    volume.reload()
    config = "/root/project/configs/stages/image_qwen_image.toml"
    data_dir = Path(MOUNT) / "data" / "raw" / "image"
    data_dir.mkdir(parents=True, exist_ok=True)
    if not any(path.is_file() for path in data_dir.iterdir()):
        raise RuntimeError(
            "No image data found in the Modal Volume. Upload it first with "
            "`modal volume put character-model-suite-data data/raw/image /data/raw/image/`."
        )
    command = [
        "python",
        "/root/project/scripts/train_qwen_image.py",
        "--config",
        config,
        "--trainer",
        "/opt/diffusers/examples/dreambooth/train_dreambooth_lora_qwen_image.py",
    ]
    print(f"using config metadata: {config}")
    subprocess.run(command, cwd="/root/project", check=True)
    volume.commit()
    return "image adapter written to /mnt/suite/adapters/image/qwen-image"


@app.function(
    image=video_image,
    gpu=VIDEO_GPU,
    volumes={MOUNT: volume},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    timeout=24 * 60 * 60,
)
def train_video() -> str:
    """Run the finetrainers Wan2.1 video LoRA recipe."""
    volume.reload()
    command = [
        "python",
        "/root/project/scripts/train_wan.py",
        "--config",
        "/root/project/configs/stages/video_wan21.toml",
    ]
    subprocess.run(command, cwd="/root/project", check=True)
    volume.commit()
    return "video adapter written to /mnt/suite/adapters/video/wan21"


@app.function(
    image=audio_image,
    gpu=AUDIO_GPU,
    volumes={MOUNT: volume},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    timeout=24 * 60 * 60,
)
def train_audio() -> str:
    """Run the official Qwen3-TTS single-speaker preparation and SFT recipe."""
    volume.reload()
    command = [
        "python",
        "/root/project/scripts/train_qwen_tts.py",
        "--config",
        "/root/project/configs/stages/audio_qwen3_tts.toml",
    ]
    subprocess.run(command, cwd="/root/project", check=True)
    volume.commit()
    return "audio adapter/checkpoints written to /mnt/suite/adapters/audio/qwen3-tts"


@app.function(
    image=base_image,
    gpu=TEXT_GPU,
    volumes={MOUNT: volume},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    timeout=24 * 60 * 60,
)
def train_text() -> str:
    """Train the Qwen3 conversation adapter with 4-bit QLoRA."""
    volume.reload()
    command = [
        "python",
        "/root/project/scripts/train_llm.py",
        "--config",
        "/root/project/configs/stages/text_qwen3.toml",
    ]
    subprocess.run(command, cwd="/root/project", check=True)
    volume.commit()
    return "text adapter written to /mnt/suite/adapters/text/qwen3-8b"


@app.function(
    image=base_image,
    gpu=CODING_GPU,
    volumes={MOUNT: volume},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    timeout=24 * 60 * 60,
)
def train_coding() -> str:
    """Train the separate Qwen3-Coder adapter with 4-bit QLoRA."""
    volume.reload()
    command = [
        "python",
        "/root/project/scripts/train_llm.py",
        "--config",
        "/root/project/configs/stages/coding_qwen3.toml",
    ]
    subprocess.run(command, cwd="/root/project", check=True)
    volume.commit()
    return "coding adapter written to /mnt/suite/adapters/coding/qwen3-coder-30b-a3b"
