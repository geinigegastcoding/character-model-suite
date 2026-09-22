from __future__ import annotations

from pathlib import Path
import tomllib

STAGES = {"image", "vision", "video", "audio", "text", "coding"}
REQUIRED_MODEL_FIELDS = {"stage", "role", "model_id", "trainer", "status"}


def load_registry(path: str | Path) -> dict:
    """Load and validate the model registry, raising on incomplete entries."""
    registry_path = Path(path)
    with registry_path.open("rb") as handle:
        data = tomllib.load(handle)

    models = data.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("registry must contain at least one [[models]] entry")

    seen_stages: set[str] = set()
    errors: list[str] = []
    for index, model in enumerate(models, start=1):
        missing = REQUIRED_MODEL_FIELDS - model.keys()
        if missing:
            errors.append(f"models[{index}] missing: {', '.join(sorted(missing))}")
        stage = model.get("stage")
        if stage not in STAGES:
            errors.append(f"models[{index}] has unsupported stage: {stage!r}")
        elif stage in seen_stages:
            errors.append(f"duplicate model stage: {stage}")
        else:
            seen_stages.add(stage)
        if not isinstance(model.get("model_id"), str) or "/" not in model.get("model_id", ""):
            errors.append(f"models[{index}] model_id must look like 'owner/name'")

    if errors:
        raise ValueError("registry validation failed:\n- " + "\n- ".join(errors))
    return data

