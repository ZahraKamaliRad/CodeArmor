import json
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Tuple

Issue = Dict[str, Any]

def _which_or_raise(cmd: str):
    if shutil.which(cmd) is None:
        raise RuntimeError(f"Required tool '{cmd}' not found in PATH.")
    return cmd

def run_bandit(py_file: Path) -> List[Issue]:
    _which_or_raise("bandit")
    proc = subprocess.run(
        ["bandit", "-f", "json", "-q", str(py_file)],
        capture_output=True, text=True
    )
    if proc.returncode not in (0, 1):  
        raise RuntimeError(f"Bandit failed: {proc.stderr}")
    data = json.loads(proc.stdout or "{}")
    issues = []
    for it in data.get("results", []):
        issues.append({
            "tool": "bandit",
            "rule_id": it.get("test_id"),
            "severity": it.get("issue_severity"),
            "message": it.get("issue_text"),
            "line": it.get("line_number"),
        })
    return issues

def run_semgrep_c(c_file: Path) -> List[Issue]:
    _which_or_raise("semgrep")
    proc = subprocess.run(
        ["semgrep", "--config", "p/cwe-top-25", "--json", str(c_file)],
        capture_output=True, text=True
    )
    if proc.returncode not in (0, 1):  
        raise RuntimeError(f"Semgrep failed: {proc.stderr}")
    data = json.loads(proc.stdout or "{}")
    issues = []
    for r in data.get("results", []):
        extra = r.get("extra", {})
        span = r.get("start", {}) or r.get("location", {}).get("start", {})
        issues.append({
            "tool": "semgrep",
            "rule_id": extra.get("rule_id"),
            "severity": (extra.get("severity") or "").upper(),
            "message": extra.get("message"),
            "line": span.get("line"),
        })
    return issues

def analyze_code(code: str, language: str, tmpname: str = "snippet") -> Tuple[List[Issue], int]:
    suffix = ".py" if language.lower().startswith("py") else ".c"
    with tempfile.TemporaryDirectory() as td:
        fpath = Path(td) / f"{tmpname}{suffix}"
        fpath.write_text(code, encoding="utf-8")
        loc = sum(1 for ln in code.splitlines() if ln.strip())

        if suffix == ".py":
            issues = run_bandit(fpath)
        else:
            issues = run_semgrep_c(fpath)

        return issues, loc
