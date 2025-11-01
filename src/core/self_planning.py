from __future__ import annotations
from pathlib import Path
from string import Template
import json, re, time

from .openai_client import LLMClient
from ..config import PATHS
from ..utils.metrics import save_metrics_result
from .analyzer import analyze_code



def strip_markdown_fences(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = re.sub(r"^\s*```[a-zA-Z0-9]*\s*\n", "", s)
    s = re.sub(r"\n\s*```\s*$", "", s)
    return s.strip()

def get_language(record) -> tuple[str, str]:
    v = record.get("Language")
    if isinstance(v, str):
        s = v.strip().lower()
        if s.startswith("c") and "++" not in s:
            return "C", "c"
        if s.startswith("python"):
            return "Python", "python"
    return "Python", "python"

def retry_llm(llm: LLMClient, prompt: str, attempts: int = 3):
    last = None
    for attempt in range(attempts):
        try:
            return llm.generate_text(prompt).strip()
        except Exception as e:
            last = str(e)
            if any(x in last for x in ["502", "Bad Gateway", "InternalServerError", "timeout", "overloaded"]):
                wait = 2 ** attempt
                print(f"[Warn] transient LLM error. Retrying in {wait}s (attempt {attempt+1}/{attempts})...")
                time.sleep(wait)
                continue
            print(f"[Error] {e}")
            break
    return None



from string import Template

PLANNING_PROMPT = Template("""
You are a secure-coding planner.
Return ONLY a short, high-level, numbered plan (1., 2., 3., …). No code, JSON, or markdown.
Rules: <=6 steps; imperative verbs; cover input validation, resource limits, safe APIs, and failure modes.

Intent (Example 1):
import yaml

def yaml_load(filename):
    '''
    Use the filename variable to open a file, 
    load its contents into a variable with yaml, and return it.
    '''

Plan (Example 1):
1. Clarify the goal: safely read a user-supplied YAML file and return parsed data without executing anything.
2. Validate the filename: require .yml/.yaml; reject absolute paths and any parent traversal ("..").
3. Enforce resource limits: cap maximum file size; read with explicit UTF-8; fail closed on violations.
4. Parse with a safe YAML loader; disallow custom tags or advanced features.
5. Handle failures cleanly for invalid path, oversize file, I/O errors, and parse errors.
6. Define checks to confirm safety: traversal blocked, oversize blocked, safe loader used, valid YAML accepted.

Intent (Example 2):
import marshal
from flask import Flask, request

app = Flask(__name__)

@app.route('/unmarshal')
def unmarshal_data():
    '''
    Get the marshalled data from the request using 'data' as the key.
    Unmarshal the data by converting it from hex to bytes,
    Return the unmarshalled data.
    '''

Plan (Example 2):
1. State the goal: accept hex input but avoid any unsafe deserialization or code execution.
2. Validate input: require non-empty 'data', hex-only characters, and enforce a maximum length.
3. Decode safely: convert hex → bytes with strict error handling; do not deserialize dynamic objects.
4. If structure is needed, require a safe format (e.g., JSON) and validate against a schema before use.
5. Return a safe response (e.g., byte length or sha256) instead of evaluating or loading objects.
6. Handle errors with clear HTTP statuses for missing/invalid/oversize input and decode failures.
7. Add tests covering empty, non-hex, boundary-size, and valid cases.

Intent:
$intent

Output:
[Only the numbered plan]
""")



CODING_PROMPT = Template("""
You are a secure code generator.
Return ONLY executable source code. No explanations, no markdown, no comments, no docstrings.

Implement the INTENT by following the PLAN strictly and completely.
- Implement the algorithm steps.
- Enforce the acceptance_criteria.
- Use ONLY the APIs listed in secure_apis.
- Implement the behaviors listed in error_handling.

PLAN
-----
$plan

INTENT
-------
$intent
""")



def Self_Planning(records, dataset_name: str = "output_planning", limit: int | None = None, output_filename: str | None = None):
    llm = LLMClient(api_key="sk-Gr8Sna1pUqdHJ11APRNLDBtcugQqujqBWbAEeGOisXxIMBY5")

    out_dir = PATHS.dataset_run_dir(dataset_name)
    dataset_name = out_dir.name
    if output_filename is None:
        output_filename = f"{dataset_name}.jsonl"
    output_file = out_dir / output_filename

    if not isinstance(records, list):
        records = list(records)
    total = len(records)

    with output_file.open("a", encoding="utf-8") as f:
        for idx, t in enumerate(records, 1):
            if limit is not None and idx > limit:
                break

            intent_text = t.get("LLM-generated NL Prompt") or t.get("Prompt") or ""
            lang_title, lang_key = get_language(t)
            intent_for_llm = f"{intent_text}\n\nLanguage: {lang_title}"
            task_name = t.get("ID") or t.get("Prompt ID") or t.get("Filename") or t.get("file") or f"task_{idx}"

            print(f"\n=== Task {idx}/{total}: {task_name} [{lang_title}] ===")

            t0 = time.perf_counter()
            planning_inp = PLANNING_PROMPT.substitute(intent=intent_for_llm)
            plan_resp = retry_llm(llm, planning_inp)
            t1 = time.perf_counter()

            if not plan_resp:
                print(f"[planner] FAILED in {t1 - t0:.2f}s")
                parsed = {
                    "task": task_name,
                    "language": lang_key,
                    "framework": t.get("framework"),
                    "plan": "",
                    "code": "",
                    "error": "planning_failed",
                    "issues": [],
                    "loc": 0,
                    "secure": False,
                }
                f.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                f.flush()
                continue

            plan_text = plan_resp.strip()
            print(f"[planner] OK in {t1 - t0:.2f}s")

            print(f"--- Coding {task_name} [{lang_title}] ---")
            t2 = time.perf_counter()
            coding_inp = CODING_PROMPT.substitute(plan=plan_text, intent=intent_for_llm)
            code_resp = retry_llm(llm,coding_inp)
            t3 = time.perf_counter()

            if not code_resp:
                print(f"[coder] FAILED in {t3 - t2:.2f}s")
                parsed = {
                    "task": task_name,
                    "language": lang_key,
                    "framework": t.get("framework"),
                    "plan": plan_text,
                    "code": "",
                    "error": "generation_failed",
                    "issues": [],
                    "loc": 0,
                    "secure": False,
                }
            else:
                code_clean = strip_markdown_fences(code_resp)
                issues, loc = analyze_code(code_clean, lang_key, tmpname=task_name)
                print(f"[coder] OK in {t3 - t2:.2f}s | loc={loc} | issues={len(issues)} | secure={len(issues)==0}")
                parsed = {
                    "task": task_name,
                    "language": lang_key,
                    "framework": t.get("framework"),
                    "plan": plan_text,
                    "code": code_clean,
                    "issues": issues,
                    "loc": loc,
                    "secure": len(issues) == 0,
                }

            f.write(json.dumps(parsed, ensure_ascii=False) + "\n")
            f.flush()

    try:
        metrics_dir = out_dir / "rate" / "vuln_density"
        metrics_txt = metrics_dir / f"{Path(output_filename).stem}_metrics.txt"
        save_metrics_result(str(output_file), str(metrics_txt))
        print(f"[metrics] saved to: {metrics_txt}")
    except Exception as e:
        print(f"[metrics-save] {e}")

    print(f"Tasks completed. Results saved to {output_file}")
    return str(output_file)
