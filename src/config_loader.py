import json
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _ROOT / "config.json"


def load_config() -> dict:
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {_CONFIG_PATH}")

    with _CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_api_key(path: str) -> str:
    p = _ROOT / path
    if not p.exists():
        raise FileNotFoundError(f"API key file not found: {p}")
    return p.read_text(encoding="utf-8").strip()
