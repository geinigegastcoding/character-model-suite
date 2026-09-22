# Character Model Suite

An open-source workspace for training separate, composable LoRA/QLoRA adapters for one character across image, video, audio, text, and coding models.

The first implementation target is image training on Modal. The repository keeps every modality's data, base model, trainer, and adapter separate so that a character identity can be reused without pretending that incompatible architectures share one LoRA file.

## Important model correction

The requested “Qwen 3.1 7B image model” does not map to a verified official model ID. The closest official pieces are:

- `Qwen/Qwen-Image`: a 20B MMDiT image generation/editing foundation model.
- `Qwen/Qwen3-VL-8B-Instruct`: an image/video understanding model, not an image generator.
- `Qwen/Qwen3-8B`: a text model; it is not an image model.

This suite therefore starts with `Qwen/Qwen-Image` for image generation. If you meant a different Hugging Face model, change the ID in `configs/models.toml` and the stage config before training.

## Quick start

Windows PowerShell:

```powershell
uv venv
uv pip install -e ".[dev]"
uv run python scripts/validate_registry.py
uv run pytest
```

Copy `.env.example` to `.env` only for local, non-secret defaults. Put tokens in Modal Secrets for cloud jobs.

## Modal accounts and fast startup

Modal profiles are the account switch; Modal environments are the project namespace. Use one profile per account and one environment per lifecycle stage:

```powershell
modal token new --activate
modal profile list
modal profile activate <account-profile>
modal environment create dev
modal config set-environment dev
modal secret create character-model-suite-hf HF_TOKEN=hf_...
modal volume create character-model-suite-data
modal volume put character-model-suite-data data/raw/image /data/raw/image/
modal run --env=dev modal_app.py::smoke
```

The first GPU run builds the image and caches dependencies. Later runs reuse the Modal image and keep checkpoints in the persistent Volume. Do not put tokens in Git or in a command that will be copied into shell history.

Upload large/private datasets with `modal volume put`; the trainer fails loudly when the mounted image directory is empty.

## Dataset workflow

1. Put private source data in `data/raw/<modality>/`.
2. Create a JSONL manifest under `data/manifests/`.
3. Run the validator before any GPU job.
4. Start with a small, held-out validation split and fixed prompts.
5. Save adapter metadata, base-model revision, dataset hash, and seed next to every run.

Example:

```powershell
uv run python scripts/validate_dataset.py `
  --modality image `
  --manifest data/manifests/image.example.jsonl
```

The example manifest intentionally points at files that are not committed. Replace the paths with data you have the right to use.

## Training entrypoints

```powershell
# CPU-side configuration and volume check
modal run --env=dev modal_app.py::smoke

# First GPU trainer: Qwen Image DreamBooth-LoRA (A100-80GB by default)
modal run --env=dev modal_app.py::train_image
```

`train_image` follows the official Diffusers Qwen Image DreamBooth-LoRA example. It is deliberately the only runnable trainer in this first slice; video, audio, text, and coding have pinned research/config placeholders until their data contracts and hardware budgets are confirmed.

GPU choice is explicit: use `A100-80GB` for the first Qwen Image training run, `L40S` for the cheaper 7–14B text/audio LoRA stages, and only switch to `H100` after a short benchmark shows that its shorter wall-clock time beats its higher hourly rate. The current price snapshot and reasoning are in [`docs/gpu-choice.md`](docs/gpu-choice.md).

## Publishing an adapter to Hugging Face

After a successful run, publish only the adapter directory and its model card:

```powershell
$env:HF_TOKEN = "hf_..."
uv run python scripts/publish_hf.py `
  --repo-id geinigegastcoding/character-model-suite-image-lora `
  --adapter-dir adapters/image/qwen-image
```

The script creates a public model repo only when explicitly run. Base-model weights and raw character data are never copied into this Git repository or the adapter repo by default.

## Repository map

```text
configs/       model registry and per-stage training settings
data/          local-only raw data and public manifest examples
docs/          research, dataset contract, and roadmap
scripts/       validation and Hugging Face publishing helpers
src/suite/     small stdlib validators
modal_app.py   Modal image, Volume, Secret, and training entrypoints
tests/         intent-focused checks for the validators
```

See [`docs/research.md`](docs/research.md) for the evidence and trade-offs behind the model matrix.
