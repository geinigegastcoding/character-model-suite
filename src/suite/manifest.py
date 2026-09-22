from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

REQUIRED_FIELDS = {
    "image": {"file", "caption", "split"},
    "video": {"file", "caption", "split"},
    "audio": {"audio", "text", "ref_audio"},
    "text": {"messages", "split"},
    "coding": {"messages", "split"},
}


def validate_manifest(path: str | Path, modality: str, root: str | Path) -> list[str]:
    """Return all manifest errors; callers must fail loudly when the list is non-empty."""
    if modality not in REQUIRED_FIELDS:
        raise ValueError(f"unsupported modality: {modality}")

    manifest_path = Path(path)
    root_path = Path(root)
    errors: list[str] = []
    seen_files: set[str] = set()
    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                item = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON ({exc.msg})")
                continue
            if not isinstance(item, dict):
                errors.append(f"line {line_number}: item must be an object")
                continue

            missing = REQUIRED_FIELDS[modality] - item.keys()
            if missing:
                errors.append(f"line {line_number}: missing {', '.join(sorted(missing))}")

            split = item.get("split")
            if split is not None and split not in {"train", "validation", "test"}:
                errors.append(f"line {line_number}: invalid split {split!r}")

            for field in _path_fields(modality):
                value = item.get(field)
                if not isinstance(value, str) or not value:
                    continue
                if value in seen_files and field in {"file", "audio"}:
                    errors.append(f"line {line_number}: duplicate source path {value!r}")
                seen_files.add(value)
                if not (root_path / value).is_file():
                    errors.append(f"line {line_number}: missing file {value!r}")

            if modality in {"text", "coding"} and not _valid_messages(item.get("messages")):
                errors.append(f"line {line_number}: messages must be a non-empty role/content list")
    return errors


def _path_fields(modality: str) -> Iterable[str]:
    if modality == "audio":
        return ("audio", "ref_audio")
    if modality in {"image", "video"}:
        return ("file",)
    return ()


def _valid_messages(messages: object) -> bool:
    return bool(
        isinstance(messages, list)
        and messages
        and all(
            isinstance(message, dict)
            and isinstance(message.get("role"), str)
            and isinstance(message.get("content"), str)
            and message["content"]
            for message in messages
        )
    )

