import json
from pathlib import Path

from suite.manifest import validate_manifest


def test_manifest_requires_real_source_files(tmp_path: Path) -> None:
    (tmp_path / "data" / "raw" / "image").mkdir(parents=True)
    source = tmp_path / "data" / "raw" / "image" / "portrait.jpg"
    source.write_bytes(b"not-a-real-image-but-a-present-source")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"file": "data/raw/image/portrait.jpg", "caption": "character", "split": "train"}) + "\n",
        encoding="utf-8",
    )
    assert validate_manifest(manifest, "image", tmp_path) == []


def test_manifest_reports_missing_files_and_bad_splits(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"file": "missing.jpg", "caption": "character", "split": "holdout"}) + "\n",
        encoding="utf-8",
    )
    errors = validate_manifest(manifest, "image", tmp_path)
    assert any("missing file" in error for error in errors)
    assert any("invalid split" in error for error in errors)

