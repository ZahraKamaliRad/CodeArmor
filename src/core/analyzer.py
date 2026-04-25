import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dotenv import load_dotenv
import os 



ENABLE_SEMGREP = True
Issue = Dict[str, Any]

load_dotenv()

rules_path = os.getenv("SEMGREP_RULES_PATH")

if not rules_path:
    raise RuntimeError("SEMGREP_RULES_PATH is not set in environment.")

SEMGREP_RULES = Path(rules_path)

def which_or_raise(cmd: str) -> str:
    if shutil.which(cmd) is None:
        raise RuntimeError(f"Required tool '{cmd}' not found in PATH.")
    return cmd


def lang_key(language: str) -> str:
    return (language or "").strip().lower()

def extension_for_language(language: str) -> str:
    lk = lang_key(language)
    if lk in ("py", "python"):
        return ".py"
    if lk == "c":
        return ".c"
    return ".txt"

def count_loc(code: str) -> int:
    return sum(1 for ln in (code or "").splitlines() if ln.strip())

def normalize_severity(v: Optional[str]) -> str:
    if not v:
        return "LOW"
    v = str(v).strip().upper()
    if v in ("CRITICAL", "ERROR"):
        return "HIGH"
    if v == "WARNING":
        return "MEDIUM"
    if v == "INFO":
        return "LOW"
    if v in ("HIGH", "MEDIUM", "LOW"):
        return v
    if "HIGH" in v:
        return "HIGH"
    if "MED" in v or "WARN" in v:
        return "MEDIUM"
    return "LOW"

def to_cwe_key(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    if s.isdigit():
        return f"CWE-{s}"
    m = re.search(r"(CWE-\d+)", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None

def extract_cwe_keys(cwe_field: Any) -> List[str]:
    keys: List[str] = []
    if cwe_field is None:
        return keys
    if isinstance(cwe_field, list):
        for item in cwe_field:
            k = to_cwe_key(item)
            if k:
                keys.append(k)
        return keys
    k = to_cwe_key(cwe_field)
    if k:
        keys.append(k)
    return keys

def run_bandit(fpath: Path) -> List[Issue]:
    which_or_raise("bandit")
    p = subprocess.run(
        ["bandit", "-q", "-f", "json", str(fpath)],capture_output=True,text=True)
    out = (p.stdout or "").strip()
    if not out:
        return []
    try:
        data = json.loads(out)
    except Exception:
        return []
    issues: List[Issue] = []
    for r in data.get("results") or []:
        cwe_id = None
        issue_cwe = r.get("issue_cwe")
        if isinstance(issue_cwe, dict):
            cwe_id = issue_cwe.get("id")
        issues.append(
            {
                "tool": "bandit",
                "rule_id": r.get("test_id"),
                "cwe": cwe_id,
                "severity": normalize_severity(r.get("issue_severity")),
                "message": r.get("issue_text"),
                "line": int(r.get("line_number") or 0)
            }
        )
    return issues

def semgrep_extract_cwe(meta: Dict[str, Any]) -> Optional[Union[str, List[Any]]]:
    cwe = meta.get("cwe")
    if isinstance(cwe, list) and cwe:
        return cwe
    if isinstance(cwe, str) and cwe.strip():
        return cwe.strip()
    return None


def run_semgrep(fpath: Path) -> List[Issue]:
    which_or_raise("semgrep")

    cmd = ["semgrep", "scan","--quiet","--metrics=off","--config", str(SEMGREP_RULES),
    "--json","--include", fpath.name,str(fpath.parent)]
    
    p = subprocess.run(cmd, capture_output=True, text=True, errors="ignore")

    if p.returncode not in (0, 1):
        return []

    out = (p.stdout or "").strip()
    if not out:
        return []

    json_start = out.find("{")
    if json_start == -1:
        return []

    try:
        data = json.loads(out[json_start:])
    except:
        return []

    issues = []
    for r in data.get("results") or []:
        extra = r.get("extra") or {}
        meta = extra.get("metadata") or {}

        issues.append({
            "tool": "semgrep",
            "rule_id": r.get("check_id"),
            "cwe": semgrep_extract_cwe(meta),
            "severity": normalize_severity(extra.get("severity") or meta.get("severity")),
            "message": extra.get("message") or "",
            "line": int((r.get("start") or {}).get("line") or 0)
        })

    return issues

def analyze_code(code: str,language: str,tools: Optional[List[str]] = None,
    tmpname: str = "snippet",) -> Tuple[List[Issue], int, Dict[str, Any]]:
    
    selected = [t.strip().lower() for t in (tools or ["bandit", "semgrep"]) if t and t.strip()]
    ext = extension_for_language(language)
    stable_filename = f"{tmpname}{ext}"

    with tempfile.TemporaryDirectory() as td:
        fpath = Path(td) / stable_filename
        fpath.write_text(code or "", encoding="utf-8", errors="replace")

        loc = count_loc(code or "")
        per_tool: Dict[str, List[Issue]] = {}
        all_issues: List[Issue] = []

        for tool in selected:

            print(f"[analysis] Running {tool} on {stable_filename} ...")

            try:
                if tool == "bandit":
                    issues = run_bandit(fpath)

                elif tool == "semgrep":
                    issues = run_semgrep(fpath)

                else:
                    raise ValueError(f"Unknown tool: {tool}")

                print(f"[analysis] {tool} analysis SUCCESS - issues found: {len(issues)}")

            except Exception as e:
                print(f"[analysis] {tool} analysis FAILED: {e}")
                issues = []

            for iss in issues:
                iss["filename"] = stable_filename

            per_tool[tool] = issues
            all_issues.extend(issues)

        severity_counts: Dict[str, int] = {}
        cwe_counts: Dict[str, int] = {}

        for iss in all_issues:
            sev = (iss.get("severity") or "LOW").upper()
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
            for cwe_key in extract_cwe_keys(iss.get("cwe")):
                cwe_counts[cwe_key] = cwe_counts.get(cwe_key, 0) + 1

        summary: Dict[str, Any] = {
            "language": lang_key(language),
            "tools": selected,
            "tool_issue_counts": {k: len(v) for k, v in per_tool.items()},
            "total_tool_issues": sum(len(v) for v in per_tool.values()),
            "severity_counts": severity_counts,
            "cwe_counts": cwe_counts,
            "deduplicated": False,
            "secure": len(all_issues) == 0
        }

        return all_issues, loc, summary


def analyze_code_split(code: str, language: str, tmpname: str = "snippet") -> Dict[str, Any]:
    bandit_issues, loc, bandit_summary = analyze_code(code, language, tools=["bandit"], tmpname=tmpname)

    if ENABLE_SEMGREP:
        semgrep_issues, _, semgrep_summary = analyze_code(code, language, tools=["semgrep"], tmpname=tmpname)
        semgrep_secure = bool(semgrep_summary.get("secure"))
    else:
        semgrep_issues = []
        semgrep_summary = {
            "secure": True,
            "severity_counts": {},
            "cwe_counts": {}
        }
        semgrep_secure = True
    return {
        "loc": int(loc),
        "bandit_result": {
            "issues": bandit_issues,
            "summary": bandit_summary,
            "secure": bool(bandit_summary.get("secure"))
        },
        "semgrep_result": {
            "issues": semgrep_issues,
            "summary": semgrep_summary,
            "secure": semgrep_secure
        }
    }
