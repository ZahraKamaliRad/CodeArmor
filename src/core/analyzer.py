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

def severity_rank(sev: str) -> int:
    sev = (sev or "").upper()
    if sev == "HIGH":
        return 3
    if sev == "MEDIUM":
        return 2
    return 1

def lang_key(language: str, path: Optional[Path] = None) -> str:
    lang = (language or "").strip().lower()
    if lang in ("python", "py", "python3"):
        return "python"
    if lang in ("c", "c99", "c11"):
        return "c"
    if path is not None:
        suf = path.suffix.lower()
        if suf == ".py":
            return "python"
        if suf == ".c":
            return "c"
    return "other"

def extension_for_language(language: str) -> str:
    lk = lang_key(language)
    if lk == "python":
        return ".py"
    if lk == "c":
        return ".c"
    return ".txt"

def normalized_rule_key(issue: Issue) -> str:
    cwe = issue.get("cwe")
    if cwe:
        s = str(cwe)
        return s if s.upper().startswith("CWE-") else f"CWE-{s}"
    rid = (issue.get("rule_id") or "").strip()
    if not rid:
        return "RULE-UNKNOWN"
    parts = rid.split(".")
    if len(parts) >= 3:
        return ".".join(parts[:3])
    return parts[0]

def analyze_bandit(path: Path, language: str) -> List[Issue]:
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

def semgrep_language(language: str) -> str:
    lk = lang_key(language)
    if lk in ("c", "python"):
        return "auto"
    return "auto"

def analyze_semgrep(path: Path, language: str) -> List[Issue]:
    lk = lang_key(language, path)
    if lk not in ("python", "c"):
        raise RuntimeError("Semgrep supports only Python and C in this project.")
    which_or_raise("semgrep")
    cfg = semgrep_language(language)
    proc = subprocess.run(
        ["semgrep", "--config", cfg, "--json", str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if not proc.stdout.strip():
        raise RuntimeError(f"Semgrep failed:\n{proc.stderr}")
    data = json.loads(proc.stdout)
    out: List[Issue] = []
    for r in data.get("results", []) or []:
        extra = r.get("extra") or {}
        start = r.get("start") or {}
        out.append({
            "tool": "semgrep",
            "rule_id": r.get("check_id"),
            "severity": sev_norm(extra.get("severity")),
            "message": extra.get("message") or (r.get("message") if isinstance(r.get("message"), str) else None),
            "line": int(start.get("line") or 0),
            "filename": r.get("path") or str(path),
            "cwe": None,
        })
    return out

def deduplicate_issues(issues: List[Issue], stable_filename: Optional[str] = None) -> List[Issue]:
    uniq: Dict[Tuple[str, int], Issue] = {}
    for iss in issues:
        fname = iss.get("filename") or ""
        if stable_filename:
            fname = stable_filename
        line = int(iss.get("line") or 0)
        rule_key = normalized_rule_key(iss)
        k = (fname, line)
        if k not in uniq:
            merged = dict(iss)
            merged["filename"] = fname
            merged["tools"] = [iss.get("tool")] if iss.get("tool") else []
            merged["rule_ids"] = [iss.get("rule_id")] if iss.get("rule_id") else []
            merged["rule_key"] = rule_key
            uniq[k] = merged
        else:
            cur = uniq[k]
            t = iss.get("tool")
            if t and t not in cur["tools"]:
                cur["tools"].append(t)
            rid = iss.get("rule_id")
            if rid and rid not in cur["rule_ids"]:
                cur["rule_ids"].append(rid)
            if severity_rank(iss.get("severity")) > severity_rank(cur.get("severity")):
                cur["severity"] = iss.get("severity")
            if not cur.get("message") and iss.get("message"):
                cur["message"] = iss.get("message")
    return list(uniq.values())

def analyze_code(code: str,language: str,tmpname: str = "snippet",tools: Optional[List[str]] = None) -> Tuple[List[Issue], int, Dict[str, Any]]:
    lk = lang_key(language)
    if lk not in ("python", "c"):
        raise RuntimeError("This analyzer currently supports only Python and C.")
    if tools is None:
        tools = ["bandit", "semgrep"] if lk == "python" else ["semgrep"]
    selected: List[str] = []
    for t in tools:
        tl = (t or "").strip().lower()
        if tl == "bandit" and lk != "python":
            continue
        if tl in ("bandit", "semgrep"):
            selected.append(tl)
        else:
            raise ValueError(f"Unknown tool: {t}")
    ext = extension_for_language(language)
    stable_filename = f"{tmpname}{ext}"
    with tempfile.TemporaryDirectory() as td:
        fpath = Path(td) / stable_filename
        fpath.write_text(code, encoding="utf-8", errors="replace")
        loc = count_loc(code)
        per_tool: Dict[str, List[Issue]] = {}
        all_issues: List[Issue] = []
        for tool in selected:
            if tool == "bandit":
                issues = analyze_bandit(fpath, language)
            elif tool == "semgrep":
                issues = analyze_semgrep(fpath, language)
            else:
                raise ValueError(f"Unknown tool: {tool}")
            for iss in issues:
                iss["filename"] = stable_filename
            per_tool[tool] = issues
            all_issues.extend(issues)
        unique_issues = deduplicate_issues(all_issues, stable_filename=stable_filename)
        summary = {
            "language": lk,
            "tools": selected,
            "tool_issue_counts": {k: len(v) for k, v in per_tool.items()},
            "total_tool_issues": sum(len(v) for v in per_tool.values()),
            "unique_vulnerabilities": len(unique_issues),
            "secure": len(unique_issues) == 0,
        }
        return unique_issues, loc, summary
