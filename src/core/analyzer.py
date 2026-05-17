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
import threading




ENABLE_BANDIT = True
ENABLE_SEMGREP = True

Issue = Dict[str, Any]

load_dotenv()

rules_path = os.getenv("SEMGREP_RULES_PATH")
if not rules_path:
    raise RuntimeError("SEMGREP_RULES_PATH is not set in environment.")

SEMGREP_RULES = Path(rules_path)

def build_tool_result(issues, language, tool_name):
    return {
        "issues": issues,
        "summary": build_summary(tool_name, issues, language),
        "secure": len(issues) == 0
    }


def run_with_spinner(cmd: List[str], label: str) -> subprocess.CompletedProcess:
    stop_event = threading.Event()

    def spinner():
        chars = ["|", "/", "-", "\\"]
        i = 0
        while not stop_event.is_set():
            print(f"\r{label} ... {chars[i % len(chars)]}", end="", flush=True)
            i += 1
            time.sleep(0.15)
        print(f"\r{label} ... done{' ' * 20}")

    t = threading.Thread(target=spinner, daemon=True)
    t.start()

    try:
        p = subprocess.run(cmd, capture_output=True, text=True, errors="ignore")
    finally:
        stop_event.set()
        t.join()

    return p

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

def build_summary(tool_name: str, issues: List[Issue], language: str) -> Dict[str, Any]:
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
        "secure": len(issues) == 0
    }

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



# def analyze_code(code: str,language: str,tools: Optional[List[str]] = None,
#     tmpname: str = "snippet") -> Tuple[List[Issue], int, Dict[str, Any]]:

#     selected = [
#         t.strip().lower()
#         for t in (tools or [])
#         if t and t.strip()
#     ]

#     ext = extension_for_language(language)
#     stable_filename = f"{tmpname}{ext}"

#     with tempfile.TemporaryDirectory() as td:
#         fpath = Path(td) / stable_filename
#         fpath.write_text(code or "", encoding="utf-8")

#         loc = count_loc(code or "")
#         per_tool: Dict[str, List[Issue]] = {}
#         all_issues: List[Issue] = []

#         def run_tool_wrapper(tool: str):
#             start = time.time()
#             ts = datetime.now().strftime("%H:%M:%S")
#             print(f"[analysis] Running {tool} ... (start {ts})")

#             try:
#                 if tool == "bandit":
#                     issues = run_bandit(fpath)
#                 elif tool == "semgrep":
#                     issues = run_semgrep(fpath)
#                 else:
#                     issues = []

#                 duration = time.time() - start
#                 record_tool_time(tool , duration)

#                 print(f"[analysis] {tool} finished in {duration:.2f}s | issues={len(issues)}")

#             except Exception as e:

#                 duration = time.time() - start
#                 record_tool_time(tool, duration)
#                 print(f"[analysis] {tool} FAILED after {duration:.2f}s | error={e}")
#                 issues = []

#             for iss in issues:
#                 iss["filename"] = stable_filename

#             return tool, issues

#         with ThreadPoolExecutor(max_workers=len(selected)) as ex:
#             futures = [ex.submit(run_tool_wrapper, t) for t in selected]

#             for f in as_completed(futures):
#                 tool, issues = f.result()
#                 per_tool[tool] = issues
#                 all_issues.extend(issues)

#         severity_counts: Dict[str, int] = {}
#         cwe_counts: Dict[str, int] = {}
#         for iss in all_issues:
#             sev = (iss.get("severity") or "LOW").upper()
#             severity_counts[sev] = severity_counts.get(sev, 0) + 1

#             for c in extract_cwe_keys(iss.get("cwe")):
#                 cwe_counts[c] = cwe_counts.get(c, 0) + 1

#         summary = {
#             "language": lang_key(language),
#             "tools": selected,
#             "tool_issue_counts": {k: len(v) for k, v in per_tool.items()},
#             "total_tool_issues": sum(len(v) for v in per_tool.values()),
#             "severity_counts": severity_counts,
#             "cwe_counts": cwe_counts,
#             "secure": len(all_issues) == 0
#         }

#         return all_issues, loc, summary



# def analyze_code_split(code: str, language: str, tmpname: str = "snippet") -> Dict[str, Any]:

#     tools: List[str] = []
#     if ENABLE_BANDIT:
#         tools.append("bandit")
#     if ENABLE_SEMGREP:
#         tools.append("semgrep")

#     if not tools:
#         raise ValueError("No security tools enabled")

#     all_issues, loc, _ = analyze_code(code, language, tools=tools, tmpname=tmpname)

#     def build_summary(tool_name: str, issues: List[Issue]) -> Dict[str, Any]:
#         severity_counts: Dict[str, int] = {}
#         cwe_counts: Dict[str, int] = {}

#         for iss in issues:
#             sev = (iss.get("severity") or "LOW").upper()
#             severity_counts[sev] = severity_counts.get(sev, 0) + 1
#             for c in extract_cwe_keys(iss.get("cwe")):
#                 cwe_counts[c] = cwe_counts.get(c, 0) + 1

#         return {
            
#             "language": lang_key(language),
#             "tools": [tool_name],
#             "tool_issue_counts": {tool_name: len(issues)},
#             "total_tool_issues": len(issues),
#             "severity_counts": severity_counts,
#             "cwe_counts": cwe_counts,
#             "secure": len(issues) == 0 }

#     result: Dict[str, Any] = {"loc": int(loc)}

#     for tool in tools:

#         tool_issues = [i for i in all_issues if i.get("tool") == tool]

#         if not tool_issues and tool not in tools:
#             continue

#         result[f"{tool}_result"] = build_tool_result(
#         tool_issues, language, tool)


#     return result



def deduplicate_issues(issues: List[Issue]) -> List[Issue]:
    seen = set()
    unique = []

    for i in issues:
        key = (
            i.get("tool"),
            i.get("rule_id"),
            i.get("filename"),
            i.get("line"),
            i.get("message"),
        )

        if key not in seen:
            seen.add(key)
            unique.append(i)

    return unique
################## Remove repetitive cwe issue in semgrep #######################
from typing import List, Dict, Any, Optional, Union

def deduplicate_issues(issues: List[Dict[str, Any]], mode: str = "strict") -> List[Dict[str, Any]]:
    seen = set()
    unique = []

    for issue in issues:
        tool = issue.get("tool")
        filename = issue.get("filename")
        line = issue.get("line")

        if mode == "cwe_line_merge" and tool == "semgrep":
            cwe = issue.get("cwe")
            if isinstance(cwe, list) and cwe:
                cwe_key = tuple(sorted(cwe)) 
            elif isinstance(cwe, str) and cwe.strip():
                cwe_key = cwe.strip()
            else:
                cwe_key = (issue.get("rule_id"), issue.get("message"))

            key = (tool, filename, line, cwe_key)

            if key not in seen:
                seen.add(key)
                unique.append(issue)
            
        else: 
            key = (
                tool,
                issue.get("rule_id"),
                filename,
                line,
                issue.get("message"),
            )
            if key not in seen:
                seen.add(key)
                unique.append(issue)

    return unique
################## Remove repetitive cwe issue in semgrep #######################
################## Add real vuln in semgrep analysis ############################
SECURITY_RULE_PATTERNS = ["security","injection","sql","xss","ssti","ssrf",
    "csrf","deserialization","command","exec","eval","path-traversal","traversal",
    "open-redirect","redirect","crypto","hash","weak-random","random","jwt",
    "auth","authentication","authorization","secret","hardcoded","token",
    "password","pickle","xml","xxe","ldap","rce","dos","overflow","taint","unsafe",]

def is_real_security_issue(rule_id: Optional[str],message: str,metadata: Dict[str, Any],
    cwe: Optional[Union[str, List[str]]]) -> bool:

    if cwe:
        return True
    
    text = " ".join([str(rule_id or ""),str(message or "")]).lower()

    category = str(metadata.get("category") or "").lower()

    confidence = str(metadata.get("confidence") or "").lower()

    impact = str(metadata.get("impact") or "").lower()

    if category == "security":
        return True

    for pattern in SECURITY_RULE_PATTERNS:
        if pattern in text:
            return True

    meta_text = " ".join([category,confidence,impact]).lower()

    for pattern in SECURITY_RULE_PATTERNS:
        if pattern in meta_text:
            return True

    return False
################## Add real vuln in semgrep analysis ############################
def analyze_jsonl_batch(jsonl_path: str) -> None:
    jsonl_file = Path(jsonl_path)

    if not jsonl_file.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    print("\nStarting batch security analysis ...")

    records: List[Dict[str, Any]] = []

    with jsonl_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        print("No records found.")
        return

    output_path = Path(jsonl_path)
    scan_dir = output_path.parent / f"{output_path.stem}__batch_scan"
    if scan_dir.exists():
        shutil.rmtree(scan_dir)

    scan_dir.mkdir(parents=True, exist_ok=True)
    print(f"Preparing {len(records)} files for batch scanning...")
    file_map: Dict[str, int] = {}
    language_map: Dict[str, str] = {}

    for idx, rec in enumerate(records):
        code = rec.get("code") or ""
        language = rec.get("language") or "python"
        ext = extension_for_language(language)
        fname = f"snippet_{idx}{ext}"
        fpath = scan_dir / fname
        fpath.write_text(code, encoding="utf-8")
        file_map[fname] = idx
        language_map[fname] = language

    bandit_results: Dict[str, List[Issue]] = {}
    semgrep_results: Dict[str, List[Issue]] = {}

    if ENABLE_BANDIT:
        which_or_raise("bandit")
        start = time.time()
        cmd = [
            "bandit",
            "-r",
            str(scan_dir),
            "-f",
            "json",
            "-q",
        ]

        p = run_with_spinner(cmd, "Running Bandit on batch")
        duration = time.time() - start
        record_tool_time("bandit", duration)
        print(f"Bandit finished with return code {p.returncode} in {duration:.2f}s")
        if p.returncode not in (0, 1):
            print(f"Bandit stderr:\n{p.stderr}")
        out = (p.stdout or "").strip()
        if out:
            try:
                data = json.loads(out)
                for r in data.get("results") or []:
                    filename = Path(r.get("filename") or "").name
                    cwe_id = None
                    issue_cwe = r.get("issue_cwe")
                    if isinstance(issue_cwe, dict):
                        cwe_id = issue_cwe.get("id")
                    issue = {
                        "tool": "bandit",
                        "rule_id": r.get("test_id"),
                        "cwe": cwe_id,
                        "severity": normalize_severity(r.get("issue_severity")),
                        "message": r.get("issue_text"),
                        "line": int(r.get("line_number") or 0),
                        "filename": filename,
                    }
                    bandit_results.setdefault(filename, []).append(issue)
                for fname, issues in bandit_results.items():
                    bandit_results[fname] = deduplicate_issues(issues)
            except Exception as e:
                print(f"Failed to parse Bandit JSON output: {e}")

    if ENABLE_SEMGREP:
        which_or_raise("semgrep")
        start = time.time()
        cmd = [
            "semgrep",
            "scan",
            "--jobs","0",
            "--quiet",
            "--metrics=off",
            "--no-git-ignore",
            "--config",
            str(SEMGREP_RULES),
            "--json",
            str(scan_dir),
        ]


        p = run_with_spinner(cmd, "Running Semgrep on batch")
        duration = time.time() - start
        record_tool_time("semgrep", duration)
        print(f"Semgrep finished with return code {p.returncode} in {duration:.2f}s")

        if p.returncode not in (0, 1):
            print(f"Semgrep stderr:\n{p.stderr}")

        out = (p.stdout or "").strip()

        if out and p.returncode in (0, 1):
            idx = out.find("{")
            if idx != -1:
                try:
                    data = json.loads(out[idx:])
                    for r in data.get("results") or []:
                        extra = r.get("extra") or {}
                        meta = extra.get("metadata") or {}

                        cwe_value = semgrep_extract_cwe(meta)
                        filename = Path(r.get("path") or "").name
                        rule_id = r.get("check_id")
                        message = extra.get("message") or ""

                        if not is_real_security_issue(rule_id=rule_id,message=message,metadata=meta,cwe=cwe_value):
                            continue
                        issue = {
                            "tool": "semgrep",
                            "rule_id": rule_id,
                            "cwe": cwe_value,
                            "severity": normalize_severity(extra.get("severity") or meta.get("severity")),
                            "message": message,
                            "line": int((r.get("start") or {}).get("line") or 0),
                            "filename": filename,
                        }
                        semgrep_results.setdefault(filename, []).append(issue)

                    for fname, issues in semgrep_results.items():
                        #semgrep_results[fname] = deduplicate_issues(issues)
                        semgrep_results[fname] = deduplicate_issues(issues, mode="cwe_line_merge")
                except Exception as e:
                    print(f"Failed to parse Semgrep JSON output: {e}")
            else:
                print("Semgrep output did not contain JSON object.")
        else:
            print("Semgrep returned no output.")
    print("\nAttaching results to records...")

    for fname, rec_idx in file_map.items():
        language = language_map.get(fname, "python")

        bandit_issues = bandit_results.get(fname, [])
        semgrep_issues = semgrep_results.get(fname, [])

        records[rec_idx]["loc"] = count_loc(records[rec_idx].get("code") or "")

        records[rec_idx]["bandit_result"] = build_tool_result(
        bandit_issues, language, "bandit")
        
        records[rec_idx]["semgrep_result"] = build_tool_result(
        semgrep_issues, language, "semgrep"
        )
    print("Writing results back to JSONL...")
    
    with jsonl_file.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("Batch analysis completed.")
