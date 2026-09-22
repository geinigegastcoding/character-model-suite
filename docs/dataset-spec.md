# Dataset contract

Raw data is private by default. Only manifests, metadata, and generated adapters should be published, and only after rights and consent are checked.

## Shared rules

- Every item has a stable source path and a `split` (`train`, `validation`, or `test`).
- Never put secrets, private URLs, or personal identifiers in captions or transcripts.
- Keep the original source outside Git and record a hash in the run metadata.
- Do not mix validation items into the training folder by accident.
- Record permission, provenance, and license in a sidecar metadata file.
- Preserve a fixed seed and fixed validation prompts for comparisons.

## Image and video

```json
{"file":"data/raw/image/001.jpg","caption":"a front-facing portrait of <CHARACTER_NAME>","split":"train","subject_id":"character"}
```

Captions should describe what is visible. They should not rely on an opaque trigger token that carries no meaning in the base model. Include varied lighting, framing, clothing, background, and expressions; hold out genuinely different examples for validation.

## Audio

```json
{"audio":"data/raw/audio/utt0001.wav","text":"Transcript with punctuation.","ref_audio":"data/raw/audio/ref.wav","split":"train"}
```

Keep the transcript exact. Record sample rate, channels, clipping checks, noise policy, and speaker consent. The Qwen3-TTS preparation step adds `audio_codes` before training.

## Text and coding

```json
{"messages":[{"role":"user","content":"What would you do?"},{"role":"assistant","content":"A character-consistent answer."}],"split":"train"}
```

Use realistic multi-turn examples and a held-out evaluation set. For coding, keep executable tests and repository context where licensing allows; evaluate both code correctness and the character's communication style.

