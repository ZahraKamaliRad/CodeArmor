from __future__ import annotations
from pathlib import Path
import json
import re
import time
from string import Template
from typing import Any, Dict, List, Tuple
from .openai_client import LLMClient, sanitize_model_name
from src.paths import PATHS
from ..utils.metrics import save_metrics_result
from .analyzer import analyze_code_split
from ..utils.tokens import get_token_stats
from ..utils.io import csv_log
from ..utils.plot_refinment import load_refinement_df, plot_totals, refinment_summary



PLANNING_PROMPT = """
You are a secure-coding assistant.For each input, first analyze how the code can be attacked when given untrusted input, then describe the security issues and a concrete mitigation plan step by step.

Follow the exact pattern shown in the example.

Example

Language: Python

Intent:
import xml.etree.ElementTree as ET

def parse_xml(xml_string):
    '''
    Parse xml_string and return the root element of the XML document
    '''

Security analysis:
To make this function secure, the security risks of parsing XML from untrusted sources are considered first.

An attacker can send a specially crafted XML string. Using the default xml.etree.ElementTree parser may allow processing of external entities or deeply nested entities, which can lead to XML External Entity (XXE) attacks or "billion laughs" style denial-of-service. The current function also lacks input validation and error handling; malformed XML raises a parsing exception that may crash the caller or expose internal details.

To implement this functionality securely, the implementation should satisfy the following security requirements:
1. Use a hardened XML parser that disables dangerous features such as external entities and DTDs.
2. Validate the input type of xml_string.
3. Wrap parsing in a try/except block and convert low-level parser errors into controlled exceptions with generic messages.
4. Keep the behavior simple: accept a string, parse it safely, and return the root element if the input is valid.

Plan to solve these issues:
- Replace xml.etree.ElementTree with defusedxml.ElementTree, a drop-in replacement designed for parsing untrusted XML.
- Check that xml_string is a str and not None; otherwise raise TypeError.
- Use a try/except around ET.fromstring; on any parsing error, raise ValueError("Invalid XML input") without leaking detailed parser internals.
- Return the root element on successful parsing.

Now follow the same style and structure for the next task.

Language: {Language}

Intent:
{Prompt}

Security analysis:
"""


CODING_PROMPT = """
Language: {Language}

Intent:
{Prompt}

Security analysis and plan:
{Plan}

Now write secure {Language} code that implements the Intent while following the plan strictly.
Return ONLY source code, with no explanations or markdown.
"""


REPAIR_PROMPT_TEMPLATE = Template(
"""
You are a secure-coding assistant.

TARGET_ANALYZER: $analyzer
LANGUAGE: $language

ANALYZER_ISSUES(JSON):
$issues_json

CURRENT_CODE:
```$language
$code
```

Task:
- Fix the security issues reported by TARGET_ANALYZER.
- Preserve functionality.
- Return ONLY the updated full code (no explanations).
"""
)

def strip_markdown_fences(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = re.sub(r"^\s*```[a-zA-Z0-9]*\s*\n", "", s)
    s = re.sub(r"\n\s*```\s*$", "", s)
    return s.strip()


def extract_code(resp: str) -> str:
    if not isinstance(resp, str):
        return ""
    matches = list(
        re.finditer(
            r"```(?:python)?\s*(.*?)```",
            resp,
            re.DOTALL | re.IGNORECASE,
        )
    )
    if matches:
        code = matches[-1].group(1)
        return code.strip()
    return resp.strip()



def run_two_analyzers(code: str, lang: str, tmpname: str = "snippet") -> Dict[str, Any]:
    scan = analyze_code_split(code, lang, tmpname=tmpname) or {}
    loc = int(scan.get("loc") or 0)

    bandit_block = scan.get("bandit_result") or {}
    semgrep_block = scan.get("semgrep_result") or {}

    bandit_issues = bandit_block.get("issues") or []
    semgrep_issues = semgrep_block.get("issues") or []

    bandit_secure = bool(bandit_block.get("secure", len(bandit_issues) == 0))
    semgrep_secure = bool(semgrep_block.get("secure", len(semgrep_issues) == 0))

    return {
        "loc": loc,
        "bandit_result": {
            "secure": bandit_secure,
            "issues": bandit_issues,
            "summary": bandit_block.get("summary") or {},
        },
        "semgrep_result": {
            "secure": semgrep_secure,
            "issues": semgrep_issues,
            "summary": semgrep_block.get("summary") or {},
        },
        "secure_all": bool(bandit_secure and semgrep_secure)
    }

def attempt_repair_loop(llm: Any, code: str, lang: str, task_name: str, iterations: int, run_out_dir: str) -> Tuple[str, Dict[str, Any]]:
    current_code = code

    for r in range(max(0, int(iterations))):
        pre = run_two_analyzers(current_code, lang, tmpname=f"{task_name}_iter{r}_pre")
        b_issues = pre["bandit_result"]["issues"]
        s_issues = pre["semgrep_result"]["issues"]

        if pre.get("secure_all", (len(b_issues) == 0 and len(s_issues) == 0)):
            return current_code, pre

        if len(b_issues) >= len(s_issues):
            target = "bandit"
            chosen_issues = b_issues
        else:
            target = "semgrep"
            chosen_issues = s_issues

        if not chosen_issues:
            return current_code, pre

        prompt = REPAIR_PROMPT_TEMPLATE.substitute(analyzer=target,language=lang,
        issues_json=json.dumps(chosen_issues, ensure_ascii=False),code=current_code)
        
        try:
            resp = llm.generate_text(prompt).strip()
        except Exception:
            csv_log(run_out_dir=run_out_dir, task_id=f"{task_name}#bandit", iter_idx=r + 1, language=lang, issues=b_issues)
            csv_log(run_out_dir=run_out_dir, task_id=f"{task_name}#semgrep", iter_idx=r + 1, language=lang, issues=s_issues)
            return current_code, pre

        fixed = extract_code(resp)
        fixed = strip_markdown_fences(fixed)

        if fixed and fixed != current_code:
            current_code = fixed

        post = run_two_analyzers(current_code, lang, tmpname=f"{task_name}_iter{r}_post")
        csv_log(run_out_dir=run_out_dir, task_id=f"{task_name}#bandit", iter_idx=r + 1, language=lang, issues=post["bandit_result"]["issues"])
        csv_log(run_out_dir=run_out_dir, task_id=f"{task_name}#semgrep", iter_idx=r + 1, language=lang, issues=post["semgrep_result"]["issues"])

        if post.get("secure_all", (len(post["bandit_result"]["issues"]) == 0 and len(post["semgrep_result"]["issues"]) == 0)):
            return current_code, post

    final_scan = run_two_analyzers(current_code, lang, tmpname=f"{task_name}_final")
    return current_code, final_scan



def planningC_gen_code(records,dataset: str,technique: str,limit: int | None = None,iterations: int = 0,output_filename: str | None = None) -> str:
    start_time = time.time()
    llm = LLMClient()
    model_tag = sanitize_model_name(llm.model)

    out_dir = PATHS.run_dir(dataset=dataset, model_name=model_tag, technique=technique)

    if output_filename is None:
        output_filename = f"{dataset}.jsonl"

    output_file = out_dir / output_filename

    if not isinstance(records, list):
        records = list(records)

    with output_file.open("a", encoding="utf-8") as f:
        for idx, t in enumerate(records, 1):
            if limit is not None and idx > limit:
                break

            raw_lang = (t.get("language") or "python").strip().lower()
            if raw_lang.startswith("py"):
                lang_title = "Python"
                lang_key = "python"
            elif raw_lang.startswith("cpp") or "c++" in raw_lang:
                lang_title = "C++"
                lang_key = "cpp"
            else:
                lang_title = "C"
                lang_key = "c"

            task_name = (
                t.get("ID")
                or t.get("Prompt ID")
                or t.get("Filename")
                or t.get("file")
                or f"task_{idx}"
            )

            intent_text = t.get("Prompt", "") or ""
            planning_prompt = PLANNING_PROMPT.format(Prompt=intent_text, Language=lang_title)

            print(f"=== Task {idx}: {task_name} [{lang_title}] ===")
            print("--- Planning stage ---")

            plan_resp = None
            for attempt in range(3):
                try:
                    plan_resp = llm.generate_text(planning_prompt).strip()
                    break
                except Exception as e:
                    msg = str(e)
                    if (
                        "502" in msg
                        or "Bad Gateway" in msg
                        or "InternalServerError" in msg
                        or "timeout" in msg
                        or "overloaded" in msg
                    ):
                        wait = 2 ** attempt
                        print(
                            f"[Warn] transient LLM error during planning. "
                            f"Retrying in {wait}s (attempt {attempt+1}/3)..."
                        )
                        time.sleep(wait)
                        continue
                    print(f"[Error][planning] {e}")
                    break

            bandit_result = {}
            semgrep_result = {}
            plan_text = ""

            if plan_resp is None:
                parsed = {
                    "task": task_name,
                    "intent": intent_text,
                    "language": lang_key,
                    "framework": t.get("framework"),
                    "technique": technique,
                    "plan": "",
                    "code": "",
                    "error": "planning_failed",
                    "issues": {"bandit": [], "semgrep": []},
                    "loc": 0,
                    "secure": False
                }
            else:
                plan_text = plan_resp.strip()
                coding_prompt = CODING_PROMPT.format(Language=lang_title, Prompt=intent_text, Plan=plan_text)

                print("--- Coding stage ---")

                code_resp = None
                for attempt in range(3):
                    try:
                        code_resp = llm.generate_text(coding_prompt).strip()
                        break
                    except Exception as e:
                        msg = str(e)
                        if (
                            "502" in msg
                            or "Bad Gateway" in msg
                            or "InternalServerError" in msg
                            or "timeout" in msg
                            or "overloaded" in msg
                        ):
                            wait = 2 ** attempt
                            print(
                                f"[Warn] transient LLM error during coding. "
                                f"Retrying in {wait}s (attempt {attempt+1}/3)..."
                            )
                            time.sleep(wait)
                            continue
                        print(f"[Error][coding] {e}")
                        break

                if code_resp is None:
                    parsed = {
                        "task": task_name,
                        "intent": intent_text,
                        "language": lang_key,
                        "framework": t.get("framework"),
                        "technique": technique,
                        "plan": plan_text,
                        "code": "",
                        "error": "generation_failed",
                        "issues": {"bandit": [], "semgrep": []},
                        "loc": 0,
                        "secure": False
                    }
                else:
                    code_text = extract_code(code_resp)
                    code_text = strip_markdown_fences(code_text)
                    scan0 = run_two_analyzers(code_text, lang_key, tmpname=task_name)
                    csv_log(run_out_dir=str(out_dir), task_id=f"{task_name}#bandit", iter_idx=0, language=lang_key, issues=scan0["bandit_result"]["issues"])
                    csv_log(run_out_dir=str(out_dir), task_id=f"{task_name}#semgrep", iter_idx=0, language=lang_key, issues=scan0["semgrep_result"]["issues"])
                    loc = int(scan0.get("loc") or 0)
                    if int(iterations) > 0 and not bool(scan0.get("secure_all", True)):
                        code_text, scan0 = attempt_repair_loop(llm=llm,code=code_text,lang=lang_key,
                        task_name=task_name,iterations=iterations,run_out_dir=str(out_dir))
                        loc = int(scan0.get("loc") or loc)
                    bandit_result = scan0.get("bandit_result") or {}
                    semgrep_result = scan0.get("semgrep_result") or {}
                    secure_all = bool(scan0.get("secure_all", False))

                    parsed = {
                        "task": task_name,
                        "intent": intent_text,
                        "language": lang_key,
                        "framework": t.get("framework"),
                        "technique": technique,
                        "plan": plan_text,
                        "code": code_text,
                        "issues": {
                            "bandit": bandit_result.get("issues") or [],
                            "semgrep": semgrep_result.get("issues") or []
                        },
                        "loc": loc,
                        "secure": secure_all
                    }

            parsed["bandit_result"] = bandit_result
            parsed["semgrep_result"] = semgrep_result
            f.write(json.dumps(parsed, ensure_ascii=False) + "\n")
            f.flush()

    try:
        metrics_dir = out_dir / "rate" / "vuln_density"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_txt = metrics_dir / f"{Path(output_filename).stem}_metrics.txt"

        token_stats = get_token_stats()
        elapsed = time.time() - start_time

        save_metrics_result(
            str(output_file),
            str(metrics_txt),
            run_info={
                "dataset": dataset,
                "model": llm.model,
                "prompt_tokens": token_stats.get("prompt_tokens", 0),
                "completion_tokens": token_stats.get("completion_tokens", 0),
                "total_tokens": token_stats.get("total_tokens", 0),
                "runtime_seconds": elapsed
            }
        )
        csv_path = out_dir / "Effect_Of_Refinment.csv"
        if csv_path.exists():
            df = load_refinement_df(csv_path)
            out_img = out_dir / "plots" / "Refinment_Plot.png"
            out_img.parent.mkdir(parents=True, exist_ok=True)
            plot_totals(csv_path, out=out_img, kind="line", show=False, df=df, verbose=False)
            refinment_summary(csv_path, metrics_txt=metrics_txt, df=df, write=True)

        print(f"[metrics] saved to: {metrics_txt}")
    except Exception as e:
        print(f"[metrics-save] {e}")

    print(f"Tasks completed. Results saved to {output_file}")
    return str(output_file)
