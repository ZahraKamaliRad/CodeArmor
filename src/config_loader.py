import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _ROOT / "config.json"

_CONFIG_CACHE: dict | None = None


def load_config() -> dict:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        if not _CONFIG_PATH.exists():
            raise FileNotFoundError(f"Config file not found: {_CONFIG_PATH}")
        with _CONFIG_PATH.open("r", encoding="utf-8") as f:
            _CONFIG_CACHE = json.load(f)
    return _CONFIG_CACHE


def override_config(updates: dict) -> None:
    cfg = load_config()

    def merge(dst: dict, src: dict):
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                merge(dst[k], v)
            else:
                dst[k] = v

    merge(cfg, updates)


def load_api_key(path: str) -> str:
    p = _ROOT / path
    if not p.exists():
        raise FileNotFoundError(f"API key file not found: {p}")
    return p.read_text(encoding="utf-8").strip()