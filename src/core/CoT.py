from __future__ import annotations
from pathlib import Path
import json
import re
import time

from .openai_client import LLMClient, sanitize_model_name
from src.paths import PATHS
from ..utils.metrics import save_metrics_result
from .analyzer import analyze_code_split
from ..utils.tokens import get_token_stats


PROMPT_TEMPLATE = """
Example

Language: Python

Intent:
import xml.etree.ElementTree as ET

def parse_xml(xml_string):
    '''
    Parse xml_string and return the root element of the XML document
    '''

Thinking step-by-step:
Explain risks briefly, then provide secure code.

The secure Python code is:
from defusedxml import ElementTree as ET
def parse_xml(xml_string):
    if not isinstance(xml_string, str):
        raise TypeError("xml_string must be a string")
    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError as exc:
        raise ValueError("Invalid XML input") from exc
    return root

Now solve the next task in the same style.

Language: {Language}

Intent:
{Prompt}

Thinking step-by-step:
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

    blocks = list(
        re.finditer(
            r"```(?:python|cpp|c|c\+\+)?\s*(.*?)```",
            resp,
            flags=re.DOTALL | re.IGNORECASE,
        )
    )
    if blocks:
        return blocks[-1].group(1).strip()

    m = re.search(r"\bThe secure .*? code is:\s*\n(.*)$", resp, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    m = re.search(r"\bHere(?:'s| is)\s+the\s+secure\b.*?:\s*\n(.*)$", resp, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    return resp.strip()


def extract_thinking(resp: str) -> str:
    if not isinstance(resp, str):
        return ""

    m = re.search(
        r"###\s*Risks.*?\n(.*?)(?=\n###\s*Secure Code|\nHere's the secure|\nHere is the secure|\n```)",
        resp,
        flags=re.DOTALL | re.IGNORECASE
    )
    if m:
        return m.group(1).strip()

    m = re.search(
        r"(?:Thinking step-by-step|Thinking|Reasoning)\s*:\s*(.*)$",
        resp,
        flags=re.DOTALL | re.IGNORECASE
    )
    if m:
        txt = m.group(1).strip()
        txt = re.split(r"\n\s*(?:The secure|###\s*Secure Code|Here's the secure)", txt, flags=re.IGNORECASE)[0]
        txt = re.split(r"\n\s*```", txt, maxsplit=1)[0]
        return txt.strip()

    return ""


def cot_gen_code(records,dataset: str,technique: str,limit: int | None = None,output_filename: str | None = None) -> str:
    start_time = time.time()
    llm = LLMClient()
    model_tag = sanitize_model_name(llm.model)

    out_dir = PATHS.run_dir(dataset=dataset, model_name=model_tag, technique=technique)
    out_dir.mkdir(parents=True, exist_ok=True)

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
            prompt = PROMPT_TEMPLATE.format(Prompt=intent_text, Language=lang_title)

            print(f"=== Running task {idx}: {task_name} [{lang_title}] ===")

            raw_resp = None
            for attempt in range(3):
                try:
                    raw_resp = llm.generate_text(prompt).strip()
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
                        time.sleep(wait)
                        continue
                    print(f"[Error] {e}")
                    break

            if raw_resp is None:
                parsed = {
                    "task": task_name,
                    "intent": intent_text,
                    "thinking": "",
                    "language": lang_key,
                    "framework": t.get("framework"),
                    "technique": technique,
                    "code": "",
                    "error": "generation_failed",
                    "loc": 0,
                    "bandit_result": {"secure": False, "issues": []},
                    "semgrep_result": {"secure": False, "issues": []}
                }
            else:
                thinking = extract_thinking(raw_resp)
                code_text = strip_markdown_fences(extract_code(raw_resp))

                tmpname = Path(str(task_name or "snippet")).stem
                scan = analyze_code_split(code_text, lang_key, tmpname=tmpname)

                parsed = {
                    "task": task_name,
                    "intent": intent_text,
                    "thinking": thinking,
                    "language": lang_key,
                    "framework": t.get("framework"),
                    "technique": technique,
                    "code": code_text,
                    "loc": int(scan.get("loc") or 0),
                    "bandit_result": scan.get("bandit_result") or {"secure": True, "issues": []},
                    "semgrep_result": scan.get("semgrep_result") or {"secure": True, "issues": []}
                }

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
    except Exception as e:
        print(f"[metrics-save] {e}")

    print(f"Tasks completed. Results saved to {output_file}")
    return str(output_file)
