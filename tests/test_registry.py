from pathlib import Path

import pytest

from suite.registry import load_registry


def test_registry_covers_distinct_model_stages() -> None:
    registry = load_registry(Path(__file__).parents[1] / "configs" / "models.toml")
    stages = {model["stage"] for model in registry["models"]}
    assert {"image", "video", "audio", "text", "coding"} <= stages


def test_registry_rejects_duplicate_stages(tmp_path: Path) -> None:
    path = tmp_path / "models.toml"
    path.write_text(
        '[[models]]\nstage="image"\nrole="generation"\nmodel_id="x/y"\ntrainer="x"\nstatus="planned"\n'
        '[[models]]\nstage="image"\nrole="generation"\nmodel_id="x/z"\ntrainer="x"\nstatus="planned"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate model stage"):
        load_registry(path)

