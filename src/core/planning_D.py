from __future__ import annotations
from pathlib import Path
import json
import re
import time

from .openai_client import LLMClient, sanitize_model_name
from src.paths import PATHS
from ..utils.metrics import save_metrics_result
from .analyzer import analyze_code
from ..utils.tokens import get_token_stats


PROMPT_TEMPLATE = """
You are a secure-coding assistant. For each input, first analyze how the code can be attacked when given untrusted input, then describe the security issues and a concrete mitigation plan step by step, and finally write secure code.

Follow the style of the example below for the analysis and plan, but DO NOT include the example in your answer.

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

Now do the same for the given task.

Language: {Language}

Intent:
{Prompt}

Your tasks:
1. Write a detailed Security analysis and step-by-step Plan to make the implementation secure.
2. Then write secure {Language} code that implements the Intent while following the plan strictly.

Output instructions (IMPORTANT):
- You may write analysis text normally.
- At the VERY END of your response, include a valid JSON object between the markers below.

===BEGIN_JSON===
{{
  "plan": "<put the full security analysis and plan here, as plain text>",
  "code": "<put ONLY the final {Language} source code here, as plain text, no markdown fences>"
}}
===END_JSON===
"""



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


def extract_json_block(text: str) -> str:
    if not isinstance(text, str):
        return ""

    start = text.find("===BEGIN_JSON===")
    end = text.find("===END_JSON===")

    if start == -1 or end == -1 or end <= start:
        return ""

    return text[start + len("===BEGIN_JSON==="):end].strip()



def planningD_gen_code(records,dataset: str,technique: str,limit: int | None = None,
    output_filename: str | None = None) -> str:
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
            unified_prompt = PROMPT_TEMPLATE.format(Prompt=intent_text, Language=lang_title)

            print(f"=== Task {idx}: {task_name} [{lang_title}] ===")
            print("--- Planning + Coding (single call) ---")

            resp = None
            for attempt in range(3):
                try:
                    resp = llm.generate_text(unified_prompt).strip()
                    break
                except Exception as e:
                    msg = str(e)
                    if any(x in msg for x in ["502", "Bad Gateway", "InternalServerError", "timeout", "overloaded"]):
                        wait = 2 ** attempt
                        print(f"[Warn] transient LLM error. Retrying in {wait}s (attempt {attempt+1}/3)...")
                        time.sleep(wait)
                        continue
                    print(f"[Error][unified_call] {e}")
                    break

            if resp is None:
                parsed = {
                    "task": task_name,
                    "language": lang_key,
                    "framework": t.get("framework"),
                    "technique": technique,
                    "plan": "",
                    "code": "",
                    "error": "generation_failed",
                    "issues": [],
                    "loc": 0,
                    "secure": False,
                }
            else:
                plan_text = ""
                code_text = ""
                error_tag = None

                blob = extract_json_block(resp)
                try:
                    data = json.loads(blob)
                    plan_text = (data.get("plan") or "").strip()
                    code_raw = data.get("code") or ""
                    code_raw = strip_markdown_fences(code_raw)
                    code_text = extract_code(code_raw)
                except Exception:
                    error_tag = "json_parse_failed"
                    plan_text = ""
                    code_text = ""

                if not code_text:
                    parsed = {
                        "task": task_name,
                        "language": lang_key,
                        "framework": t.get("framework"),
                        "technique": technique,
                        "plan": plan_text,
                        "code": code_text,
                        "error": error_tag or "empty_code",
                        "issues": [],
                        "loc": 0,
                        "secure": False,
                    }
                else:
                    issues, loc = analyze_code(code_text, lang_key, tmpname=task_name)
                    parsed = {
                        "task": task_name,
                        "language": lang_key,
                        "framework": t.get("framework"),
                        "technique": technique,
                        "plan": plan_text,
                        "code": code_text,
                        "issues": issues,
                        "loc": loc,
                        "secure": len(issues) == 0,
                    }

            parsed["bandit_result"] = {"secure": parsed["secure"], "issues": parsed["issues"]}
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
                "runtime_seconds": elapsed,
            },
        )

        print(f"[metrics] saved to: {metrics_txt}")
    except Exception as e:
        print(f"[metrics-save] {e}")

    print(f"Tasks completed. Results saved to {output_file}")
    return str(output_file)
