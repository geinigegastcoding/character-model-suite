from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GPUProfile:
    name: str
    vram_gib: int
    dollars_per_hour: float
    best_for: str
    qwen_image_training: bool


# Modal prices verified on 2026-09-22 from https://modal.com/pricing.
GPU_PROFILES = {
    "L40S": GPUProfile("L40S", 48, 0.000542 * 3600, "text/audio LoRA, inference, cheap iterations", False),
    "A100-40GB": GPUProfile("A100-40GB", 40, 0.000583 * 3600, "smaller training jobs", False),
    "A100-80GB": GPUProfile("A100-80GB", 80, 0.000694 * 3600, "Qwen Image LoRA with VRAM headroom", True),
    "H100": GPUProfile("H100", 80, 0.001097 * 3600, "time-critical training and large models", True),
}


def recommend(stage: str) -> GPUProfile:
    """Choose the cheapest profile that is a sensible starting point for a stage."""
    if stage == "image":
        return GPU_PROFILES["A100-80GB"]
    if stage in {"text", "coding", "audio"}:
        return GPU_PROFILES["L40S"]
    if stage == "video":
        return GPU_PROFILES["A100-80GB"]
    raise ValueError(f"unsupported stage: {stage}")

