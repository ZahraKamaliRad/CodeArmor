from __future__ import annotations
from pathlib import Path
import json

TOKEN_STATS = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
}

def add_token_usage(
    prompt_tokens: int,
    completion_tokens: int,
    log_file: str | Path | None = None,
    extra: dict | None = None,
):
    total = prompt_tokens + completion_tokens
    TOKEN_STATS["prompt_tokens"] += prompt_tokens
    TOKEN_STATS["completion_tokens"] += completion_tokens
    TOKEN_STATS["total_tokens"] += total

    record = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total,
    }
    if extra:
        record.update(extra)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record

def get_token_stats() -> dict:
    return dict(TOKEN_STATS)
