# Roadmap

## Phase 0 — foundation (this repository)

- [x] Separate model registry and stage configs.
- [x] Fail-loud manifest and registry validators.
- [x] Modal profile/environment/Secret/Volume workflow.
- [x] Hugging Face adapter publishing helper.
- [x] Research the model-family mismatch and document it.

## Phase 1 — image identity proof

- [ ] Decide the character name/trigger wording and rights metadata.
- [ ] Collect and caption a small, varied image set.
- [ ] Run the Modal smoke job.
- [ ] Run a short Qwen Image LoRA job and save metadata.
- [ ] Compare base vs adapter on fixed validation prompts.
- [ ] Publish only after the model card, license, and privacy review are complete.

## Phase 2 — video

- [x] Confirm the initial Wan checkpoint and wire a maintained LoRA trainer.
- [x] Add a Modal video function with an A100-80GB default.
- [ ] Define frame sampling, clip length, motion buckets, and evaluation.
- [ ] Train a separate adapter; never merge the image adapter into it.

## Phase 3 — audio

- [ ] Obtain explicit voice consent and clean transcripts.
- [x] Wire Qwen3-TTS `audio_codes` preparation and official SFT script.
- [x] Add a Modal audio function with an L40S default.
- [ ] Compare base/finetuned voice on held-out sentences and noise conditions.

## Phase 4 — text and coding

- [x] Wire separate conversational and coding QLoRA runners.
- [x] Add L40S/A100-80GB Modal GPU routing.
- [ ] Establish regression tests before QLoRA.
- [ ] Tune rank/target modules with a small controlled matrix.
- [ ] Publish separate adapters and model cards.
