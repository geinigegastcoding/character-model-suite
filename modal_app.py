from __future__ import annotations

import os
from pathlib import Path
import subprocess

import modal


ROOT = Path(__file__).parent
VOLUME_NAME = "character-model-suite-data"
HF_SECRET_NAME = "character-model-suite-hf"
GPU = os.environ.get("SUITE_GPU", "A100-80GB")
MOUNT = "/mnt/suite"

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
        "diffusers @ git+https://github.com/huggingface/diffusers.git",
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
    .add_local_dir(ROOT, remote_path="/root/project", copy=True)
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
