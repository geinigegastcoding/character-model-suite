from suite.gpu import GPU_PROFILES, recommend


if __name__ == "__main__":
    for stage in ("image", "video", "audio", "text", "coding"):
        gpu = recommend(stage)
        print(f"{stage:7} -> {gpu.name:10} {gpu.vram_gib:>2} GiB ${gpu.dollars_per_hour:.2f}/h — {gpu.best_for}")
    print("\nCheapest realistic first image-training choice: A100-80GB")
    print("Use L40S for text/audio LoRA and inference; benchmark H100 only when wall-clock time matters.")

