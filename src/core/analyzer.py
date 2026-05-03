import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dotenv import load_dotenv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from datetime import datetime
from ..utils.usage_stats import record_tool_time



ENABLE_BANDIT = True
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
    p = subprocess.run(["bandit", "-q", "-f", "json", str(fpath)],
        capture_output=True,text=True)
    
    out = (p.stdout or "").strip()
    if not out:
        return []
    try:
        data = json.loads(out)
    except:
        return []
    issues: List[Issue] = []
    for r in data.get("results") or []:
        cwe_id = None
        issue_cwe = r.get("issue_cwe")
        if isinstance(issue_cwe, dict):
            cwe_id = issue_cwe.get("id")

        issues.append({
            "tool": "bandit",
            "rule_id": r.get("test_id"),
            "cwe": cwe_id,
            "severity": normalize_severity(r.get("issue_severity")),
            "message": r.get("issue_text"),
            "line": int(r.get("line_number") or 0)
        })
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

    cmd = [
        "semgrep", "scan",
        "--quiet",
        "--metrics=off",
        "--config", str(SEMGREP_RULES),
        "--json",
        "--include", fpath.name,
        str(fpath.parent)
    ]

    p = subprocess.run(cmd, capture_output=True, text=True, errors="ignore")

    if p.returncode not in (0, 1):
        return []

    out = (p.stdout or "").strip()
    if not out:
        return []

    idx = out.find("{")
    if idx == -1:
        return []

    try:
        data = json.loads(out[idx:])
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
    tmpname: str = "snippet") -> Tuple[List[Issue], int, Dict[str, Any]]:

    selected = [
        t.strip().lower()
        for t in (tools or [])
        if t and t.strip()
    ]

    ext = extension_for_language(language)
    stable_filename = f"{tmpname}{ext}"

    with tempfile.TemporaryDirectory() as td:
        fpath = Path(td) / stable_filename
        fpath.write_text(code or "", encoding="utf-8")

        loc = count_loc(code or "")
        per_tool: Dict[str, List[Issue]] = {}
        all_issues: List[Issue] = []

        def run_tool_wrapper(tool: str):
            start = time.time()
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[analysis] Running {tool} ... (start {ts})")

            try:
                if tool == "bandit":
                    issues = run_bandit(fpath)
                elif tool == "semgrep":
                    issues = run_semgrep(fpath)
                else:
                    issues = []

                duration = time.time() - start
                record_tool_time(tool , duration)

                print(f"[analysis] {tool} finished in {duration:.2f}s | issues={len(issues)}")

            except Exception as e:

                duration = time.time() - start
                record_tool_time(tool, duration)
                print(f"[analysis] {tool} FAILED after {duration:.2f}s | error={e}")
                issues = []

            for iss in issues:
                iss["filename"] = stable_filename

            return tool, issues

        with ThreadPoolExecutor(max_workers=len(selected)) as ex:
            futures = [ex.submit(run_tool_wrapper, t) for t in selected]

            for f in as_completed(futures):
                tool, issues = f.result()
                per_tool[tool] = issues
                all_issues.extend(issues)

        severity_counts: Dict[str, int] = {}
        cwe_counts: Dict[str, int] = {}
        for iss in all_issues:
            sev = (iss.get("severity") or "LOW").upper()
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

            for c in extract_cwe_keys(iss.get("cwe")):
                cwe_counts[c] = cwe_counts.get(c, 0) + 1

        summary = {
            "language": lang_key(language),
            "tools": selected,
            "tool_issue_counts": {k: len(v) for k, v in per_tool.items()},
            "total_tool_issues": sum(len(v) for v in per_tool.values()),
            "severity_counts": severity_counts,
            "cwe_counts": cwe_counts,
            "secure": len(all_issues) == 0
        }

        return all_issues, loc, summary



def analyze_code_split(code: str, language: str, tmpname: str = "snippet") -> Dict[str, Any]:

    tools: List[str] = []
    if ENABLE_BANDIT:
        tools.append("bandit")
    if ENABLE_SEMGREP:
        tools.append("semgrep")

    if not tools:
        raise ValueError("No security tools enabled")

    all_issues, loc, _ = analyze_code(code, language, tools=tools, tmpname=tmpname)

    def build_summary(tool_name: str, issues: List[Issue]) -> Dict[str, Any]:
        severity_counts: Dict[str, int] = {}
        cwe_counts: Dict[str, int] = {}

        for iss in issues:
            sev = (iss.get("severity") or "LOW").upper()
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
            for c in extract_cwe_keys(iss.get("cwe")):
                cwe_counts[c] = cwe_counts.get(c, 0) + 1

        return {
            
            "language": lang_key(language),
            "tools": [tool_name],
            "tool_issue_counts": {tool_name: len(issues)},
            "total_tool_issues": len(issues),
            "severity_counts": severity_counts,
            "cwe_counts": cwe_counts,
            "secure": len(issues) == 0 }

    result: Dict[str, Any] = {"loc": int(loc)}

    for tool in tools:

        tool_issues = [i for i in all_issues if i.get("tool") == tool]

        if not tool_issues and tool not in tools:
            continue

        result[f"{tool}_result"] = {
            "issues": tool_issues,
            "summary": build_summary(tool, tool_issues),
            "secure": len(tool_issues) == 0
        }


    return result
