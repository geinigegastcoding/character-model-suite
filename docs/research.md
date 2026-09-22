# Research snapshot

Research date: 2026-09-22

## The naming issue

The phrase “Qwen 3.1 7B image model” is ambiguous. I could not verify an official Qwen model with that exact combination. The official model families separate the jobs:

| Job | First candidate | Why | Status |
|---|---|---|---|
| Image generation/editing | `Qwen/Qwen-Image` | Official Qwen image foundation model; the Qwen repo describes it as a 20B MMDiT model. | First trainer |
| Image/video understanding | `Qwen/Qwen3-VL-8B-Instruct` | Vision-language model for understanding, captioning, and evaluation; not a text-to-image generator. | Optional |
| Video generation | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | Small enough to iterate on first; Diffusers documents LoRA loading for Wan2.1. | Planned |
| Voice/audio | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Official single-speaker fine-tuning recipe with audio codes. | Planned |
| Character text | `Qwen/Qwen3-8B` | Separate causal language model; PEFT/QLoRA is the appropriate adaptation path. | Planned |
| Coding character | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | A code-specialized Qwen family; requires a larger GPU budget than a 7B model. | Planned |

The suite treats this as a model matrix, not one universal LoRA. An adapter is tied to a base-model architecture and checkpoint revision. Cross-modal consistency comes from shared character metadata, captions, transcripts, evaluation prompts, and naming—not from loading one adapter into every model.

## Training conclusions

### Image

Hugging Face Diffusers provides a dedicated Qwen Image DreamBooth-LoRA example. The documented path uses `Qwen/Qwen-Image`, bf16, 1024px resolution, a small batch, gradient accumulation, optional 8-bit Adam, validation prompts, and optional Hub upload. The project keeps those settings in `configs/stages/image_qwen_image.toml` and runs the first job through Modal.

The first run should be an identity proof, not a “maximum quality” run: a curated set of varied, consented images; fixed validation prompts; a held-out validation image set; and a short run that can be compared against the base model. Only then should rank, steps, resolution, or target layers be tuned.

### Video

Wan2.1 is architecturally separate from Qwen Image. Diffusers documents LoRA loading for Wan2.1, but a production training workflow still needs a confirmed trainer, frame sampling policy, and a motion/identity evaluation set. The repository therefore starts with the 1.3B checkpoint as a budget-conscious candidate and does not pretend the video stage is runnable yet.

### Audio

Qwen3-TTS is not a generic text LoRA drop-in. The official fine-tuning instructions currently describe single-speaker fine-tuning and require JSONL records containing `audio`, `text`, and `ref_audio`, followed by `audio_codes` preprocessing. The audio stage must therefore preserve transcript quality, sample rate, silence policy, consent, and speaker reference handling. Do not copy the image QLoRA assumptions into this stage.

### Text and coding

For text and coding, PEFT is the common adapter layer. QLoRA is useful when the frozen base is loaded in low precision. PEFT's documented QLoRA-style configuration uses `target_modules="all-linear"`; the actual target modules and rank still need an A/B experiment on the chosen checkpoint. Keep conversation style and coding style in separate adapters and test for capability regression on a held-out set.

## Modal design

- Profiles select the Modal account.
- Environments isolate dev/staging/prod namespaces.
- A Volume stores raw training data, cached models, adapters, and checkpoints.
- Secrets inject `HF_TOKEN` and optional experiment-tracking credentials.
- GPU choice is an explicit environment variable; the default is `A100-80GB` for the 20B image trainer, not a claim that every stage needs that GPU.

This is deliberately one Modal app with separate functions. It keeps startup commands identical across accounts while letting the operator activate the correct profile and environment before each run.

## Sources

- [Qwen Image repository](https://github.com/QwenLM/Qwen-Image)
- [Qwen3-VL repository](https://github.com/QwenLM/Qwen3-VL)
- [Diffusers Qwen Image DreamBooth-LoRA guide](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_qwen.md)
- [Diffusers Qwen Image training script](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_qwen_image.py)
- [Diffusers Wan pipeline and LoRA notes](https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/wan.md)
- [Qwen3-TTS fine-tuning instructions](https://github.com/QwenLM/Qwen3-TTS/blob/main/finetuning/README.md)
- [Qwen3-Coder collection](https://huggingface.co/collections/Qwen/qwen3-coder)
- [Hugging Face PEFT LoRA reference](https://huggingface.co/docs/peft/en/package_reference/lora)
- [Modal profiles](https://modal.com/docs/cli/latest/profile)
- [Modal environments](https://modal.com/docs/guide/environments)
- [Modal GPU acceleration](https://modal.com/docs/guide/gpu)
- [Modal Secrets](https://modal.com/docs/guide/secrets)
- [Modal Volumes](https://modal.com/docs/guide/volumes)
