from pathlib import Path

from suite.registry import load_registry


if __name__ == "__main__":
    registry = Path(__file__).parents[1] / "configs" / "models.toml"
    data = load_registry(registry)
    print(f"registry ok: {len(data['models'])} model stages")

