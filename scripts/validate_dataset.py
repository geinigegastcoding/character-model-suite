from __future__ import annotations

import argparse
from pathlib import Path

from suite.manifest import validate_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a Character Model Suite JSONL manifest.")
    parser.add_argument("--modality", required=True, choices=["image", "video", "audio", "text", "coding"])
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[1])
    args = parser.parse_args()
    errors = validate_manifest(args.manifest, args.modality, args.root)
    if errors:
        print("manifest validation failed:")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print(f"manifest ok: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

