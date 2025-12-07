from __future__ import annotations
from pathlib import Path
import json
import re
import time

from .openai_client import LLMClient, sanitize_model_name
from ..config import PATHS
from ..utils.metrics import save_metrics_result
from .analyzer import analyze_code
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

If this function is implemented naively,several security risks appear.Using the default xml.etree.
ElementTree parser on untrusted XML can allow processing of external entities or deeply nested entities,
which enables XML External Entity (XXE) attacks and "billion laughs" style denial-of-service.
A straightforward implementation might accept any object as xml_string and pass it directly to the parser without type checking,
which can lead to unexpected behavior. Without explicit error handling, parser exceptions and detailed error messages may propagate to callers,
potentially exposing internal parsing details or causing the application to crash when the XML is malformed.

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


def cot_gen_code(records,dataset_name: str = "output",limit: int | None = None,output_filename: str | None = None,) -> str:
    start_time = time.time()
    llm = LLMClient(api_key="sk-Gr8Sna1pUqdHJ11APRNLDBtcugQqujqBWbAEeGOisXxIMBY5")
    model_tag = sanitize_model_name(llm.model)
    run_name = f"{dataset_name}_{model_tag}"
    out_dir = PATHS.dataset_run_dir(run_name)
    dataset_name = out_dir.name

    if output_filename is None:
        output_filename = f"{dataset_name}.jsonl"
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
                        print(
                            f"[Warn] transient LLM error. Retrying in {wait}s "
                            f"(attempt {attempt+1}/3)..."
                        )
                        time.sleep(wait)
                        continue
                    else:
                        print(f"[Error] {e}")
                        break

            if raw_resp is None:
                parsed = {
                    "task": task_name,
                    "language": lang_key,
                    "framework": t.get("framework"),
                    "code": "",
                    "error": "generation_failed",
                    "issues": [],
                    "loc": 0,
                    "secure": False,
                }
            else:
                code_text = extract_code(raw_resp)
                code_text = strip_markdown_fences(code_text)
                issues, loc = analyze_code(code_text, lang_key, tmpname=task_name)
                parsed = {
                    "task": task_name,
                    "language": lang_key,
                    "framework": t.get("framework"),
                    "code": code_text,
                    "issues": issues,
                    "loc": loc,
                    "secure": len(issues) == 0,
                }

            parsed["bandit_result"] = {
                "secure": parsed["secure"],
                "issues": parsed["issues"],
            }

            f.write(json.dumps(parsed, ensure_ascii=False) + "\n")
            f.flush()

    try:
        metrics_dir = out_dir / "rate" / "vuln_density"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_txt = metrics_dir / f"{Path(output_filename).stem}_metrics.txt"

        save_metrics_result(str(output_file), str(metrics_txt))

        token_stats = get_token_stats()
        elapsed = time.time() - start_time
        with metrics_txt.open("a", encoding="utf-8") as mf:
            mf.write("\n")
            mf.write(f"prompt_tokens: {token_stats.get('prompt_tokens', 0)}\n")
            mf.write(f"completion_tokens: {token_stats.get('completion_tokens', 0)}\n")
            mf.write(f"total_tokens: {token_stats.get('total_tokens', 0)}\n")
            mf.write(f"model: {llm.model}\n")
            mf.write(f"runtime_seconds: {elapsed:.2f}\n")

        print(f"[metrics] saved to: {metrics_txt}")
    except Exception as e:
        print(f"[metrics-save] {e}")

    print(f"Tasks completed. Results saved to {output_file}")
    return str(output_file)
