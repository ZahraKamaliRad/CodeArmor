import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

Issue = Dict[str, Any]

def which_or_raise(cmd: str) -> str:
    if shutil.which(cmd) is None:
        raise RuntimeError(f"Required tool '{cmd}' not found in PATH.")
    return cmd

def count_loc(text: str) -> int:
    return sum(1 for ln in text.splitlines() if ln.strip())

def sev_norm(v: Optional[str]) -> str:
    if not v:
        return "LOW"
    v = v.upper()
    if v in ("CRITICAL", "ERROR", "HIGH"):
        return "HIGH"
    if v in ("MEDIUM", "MODERATE", "WARNING"):
        return "MEDIUM"
    return "LOW"

def lang_key(language: str, path: Optional[Path] = None) -> str:
    lang = (language or "").lower()
    if lang.startswith("py") or (path and path.suffix.lower() == ".py"):
        return "python"
    return "other"

def analyze_path(path: Path, language: str) -> List[Issue]:
    if lang_key(language, path) != "python":
        raise RuntimeError("Bandit supports only Python.")
    which_or_raise("bandit")
    proc = subprocess.run(
        ["bandit", "-f", "json", "-q", str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"Bandit failed: {proc.stderr}")
    data = json.loads(proc.stdout or "{}")
    out: List[Issue] = []
    for it in data.get("results", []) or []:
        out.append({
            "tool": "bandit",
            "rule_id": it.get("test_id"),
            "severity": sev_norm(it.get("issue_severity")),
            "message": it.get("issue_text"),
            "line": int(it.get("line_number") or 0),
            "filename": it.get("filename") or str(path),
            "cwe": (it.get("issue_cwe") or {}).get("id"),
        })
    return out

def analyze_code(code: str, language: str, tmpname: str = "snippet") -> Tuple[List[Issue], int]:
    if lang_key(language) != "python":
        raise RuntimeError("Bandit supports only Python.")
    with tempfile.TemporaryDirectory() as td:
        fpath = Path(td) / f"{tmpname}.py"
        fpath.write_text(code, encoding="utf-8")
        loc = count_loc(code)
        issues = analyze_path(fpath, language)
        return issues, loc
