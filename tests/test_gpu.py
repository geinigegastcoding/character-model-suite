from suite.gpu import GPU_PROFILES, recommend


def test_image_recommendation_has_vram_headroom() -> None:
    gpu = recommend("image")
    assert gpu.name == "A100-80GB"
    assert gpu.qwen_image_training is True


def test_small_lora_stages_use_lower_cost_gpu() -> None:
    assert recommend("text") is GPU_PROFILES["L40S"]
    assert recommend("audio") is GPU_PROFILES["L40S"]

