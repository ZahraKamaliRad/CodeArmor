from __future__ import annotations
from pathlib import Path
import json
import threading


TOKEN_STATS = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
}

LLM_STATS = {
    "llm_time": 0.0,
    "api_calls": 0,
}

TOOL_STATS = {
    "bandit_time": 0.0,
    "semgrep_time": 0.0,
}

_LOCK = threading.Lock()


def add_token_usage(prompt_tokens: int, completion_tokens: int,
                   log_file: str | Path | None = None,
                   extra: dict | None = None):

    total = prompt_tokens + completion_tokens

    with _LOCK:
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


def record_llm_time(seconds: float):
    with _LOCK:
        LLM_STATS["llm_time"] += seconds


def increment_api_calls(count: int = 1):
    with _LOCK:
        LLM_STATS["api_calls"] += count


def record_tool_time(tool: str, seconds: float):
    with _LOCK:
        if tool == "bandit":
            TOOL_STATS["bandit_time"] += seconds
        elif tool == "semgrep":
            TOOL_STATS["semgrep_time"] += seconds


def get_token_stats() -> dict:
    with _LOCK:
        return dict(TOKEN_STATS)


def get_llm_stats() -> dict:
    with _LOCK:
        return dict(LLM_STATS)


def get_tool_stats() -> dict:
    with _LOCK:
        return dict(TOOL_STATS)


def reset_token_stats():
    with _LOCK:
        TOKEN_STATS["prompt_tokens"] = 0
        TOKEN_STATS["completion_tokens"] = 0
        TOKEN_STATS["total_tokens"] = 0


def reset_llm_stats():
    with _LOCK:
        LLM_STATS["llm_time"] = 0.0
        LLM_STATS["api_calls"] = 0


def reset_tool_stats():
    with _LOCK:
        TOOL_STATS["bandit_time"] = 0.0
        TOOL_STATS["semgrep_time"] = 0.0