from pathlib import Path
from typing import Iterable, Dict, Any, List
import orjson

def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("rb") as f:
        for line in f:
            if not line.strip():
                continue
            yield orjson.loads(line)

def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for row in rows:
            f.write(orjson.dumps(row))
            f.write(b"\n")

def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

