from __future__ import annotations
import subprocess
from pathlib import Path


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=str(repo_root),
        stderr=subprocess.DEVNULL,
    ).decode().strip()


def get_commit(repo_root: Path) -> str | None:
    try:
        return _git(repo_root, "rev-parse", "HEAD")
    except Exception:
        return None


