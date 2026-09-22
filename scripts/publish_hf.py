from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish one adapter directory to a public Hugging Face model repo.")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--private", action="store_true", help="Keep the Hub repo private.")
    parser.add_argument("--commit-message", default="Upload character adapter")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required; use `hf auth login` or a Modal Secret instead.")
    if not args.adapter_dir.is_dir():
        raise SystemExit(f"adapter directory does not exist: {args.adapter_dir}")

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(args.adapter_dir),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.commit_message,
    )
    visibility = "private" if args.private else "public"
    print(f"published {args.adapter_dir} to {args.repo_id} ({visibility})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

