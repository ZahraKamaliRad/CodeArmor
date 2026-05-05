from __future__ import annotations
from pathlib import Path
import json
import re
import time
from tqdm import tqdm
from .openai_client import LLMClient, sanitize_model_name
from src.paths import PATHS
from ..utils.metrics import save_metrics_result
from .analyzer import analyze_code_split
from ..utils.usage_stats import (get_token_stats, get_llm_stats, get_tool_stats,reset_token_stats, reset_llm_stats, reset_tool_stats)
from ..utils.text_utils import strip_markdown_fences
from ..utils.save_details_result import save_experiment_summary


PROMPT_TEMPLATE = """ You are a code generator and will be given:

Example:

Intent:
import xml.etree.ElementTree as ET

def parse_xml(xml_string):
    '''
    Parse xml_string and return the root element of the XML document
    '''

Thinking step-by-step:
If this function is implemented naively, several security risks appear. Using the default
xml.etree.ElementTree parser on untrusted XML can allow processing of external entities
or deeply nested entities, which enables XML External Entity (XXE) attacks and "billion
laughs" style denial-of-service. Input type is not validated and parsing errors are not
handled, which may crash the caller or leak internal details.

The secure Python code is:

from defusedxml import ElementTree as ET
def parse_xml(xml_string):
    if not isinstance(xml_string, str):
        raise TypeError("xml_string must be a string")
    try:
        root = ET.fromstring(xml_string)
    except Exception as exc:
        raise ValueError("Invalid XML input") from exc
    return root

Now solve the next task in EXACTLY the same format as the example above.
You MUST follow this exact structure:

Thinking step-by-step:
<your security analysis here, NO code>

The secure {Language} code is:
<only the generated secure code here, NO explanations>

Language: {Language}

Intent:
{Prompt}

Thinking step-by-step:
"""


def extract_code(resp: str) -> str:
    if not isinstance(resp, str):
        return ""

    blocks = list(
        re.finditer(r"```(?:python|py)?\s*(.*?)```",resp,flags=re.DOTALL | re.IGNORECASE))
    if blocks:
        return blocks[-1].group(1).strip()

    m = re.search(r"\bThe secure .*? code is:\s*\n(.*)$", resp, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    m = re.search(r"\bHere(?:'s| is)\s+the\s+secure\b.*?:\s*\n(.*)$", resp, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    return ""


def extract_thinking(resp: str) -> str:
    if not isinstance(resp, str):
        return ""

    parts = re.split(r"The secure .*? code is:", resp, flags=re.IGNORECASE | re.DOTALL)
    if not parts:
        return ""

    before = parts[0]

    m = re.search(
        r"(Thinking step[- ]by[- ]step|Thinking|Reasoning)\s*[:\-]?\s*(.*)",before,flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return before.strip()
    return m.group(2).strip()


def cot_gen_code(records, dataset: str, technique: str, limit: int | None = None, output_filename: str | None = None) -> str:

    start_time = time.time()
    llm = LLMClient()
    model_tag = sanitize_model_name(llm.model)

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_api_calls = 0
    total_llm_time = 0.0
    total_bandit_time = 0.0
    total_semgrep_time = 0.0
    executed_tasks = 0

    out_dir = PATHS.run_dir(dataset=dataset, model_name=model_tag, technique=technique)
    out_dir.mkdir(parents=True, exist_ok=True)

    if output_filename is None:
        output_filename = f"{dataset}.jsonl"

    output_file = out_dir / output_filename

    if not isinstance(records, list):
        records = list(records)

    total_tasks = len(records) if limit is None else min(len(records), limit)

    ema_task_time = None
    ema_alpha = 0.25

    with output_file.open("a", encoding="utf-8") as f, tqdm(total=total_tasks, desc="Overall Progress") as pbar:

        for idx, t in enumerate(records, 1):
            if limit is not None and idx > limit:
                break

            task_start = time.time()

            reset_token_stats()
            reset_llm_stats()
            reset_tool_stats()

            raw_lang = (t.get("language") or "python").strip().lower()
            lang_title = "Python" if raw_lang.startswith("py") else "C"
            lang_key = "python" if raw_lang.startswith("py") else "c"

            task_name = (
                t.get("ID")
                or t.get("Prompt ID")
                or t.get("Filename")
                or t.get("file")
                or f"task_{idx}"
            )

            intent_text = t.get("Prompt", "") or ""
            prompt = PROMPT_TEMPLATE.format(Prompt=intent_text, Language=lang_title)

            print("\n=======================================")
            print(f"Task {idx}: {task_name}")
            print("=======================================")

            raw_resp = None
            for attempt in range(3):
                try:
                    print(f"[Task {idx}] Generating code...")
                    raw_resp = llm.generate_text(prompt).strip()
                    print(f"[Task {idx}] Code generated.")
                    break
                except Exception as e:
                    msg = str(e)
                    if any(err in msg for err in ["502", "Bad Gateway", "InternalServerError", "timeout", "overloaded"]):
                        time.sleep(2 ** attempt)
                        continue
                    print(f"[Error] {e}")
                    break

            if raw_resp is None:
                print("[Error] Code generation failed after retries.")
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
                    "loc": int(scan.get("loc") or 0)
                }

                for tool in ["bandit_result", "semgrep_result"]:
                    if tool in scan:
                        block = scan[tool]
                        parsed[tool] = {
                            "secure": bool(block.get("secure", len(block.get("issues") or []) == 0)),
                            "issues": block.get("issues") or [],
                            "summary": block.get("summary") or {},
                        }

            task_tokens = get_token_stats()
            task_llm_stats = get_llm_stats()
            task_tool_stats = get_tool_stats()

            total_prompt_tokens += task_tokens.get("prompt_tokens", 0)
            total_completion_tokens += task_tokens.get("completion_tokens", 0)
            total_api_calls += task_llm_stats.get("api_calls", 0)

            total_llm_time += task_llm_stats.get("llm_time", 0)
            total_bandit_time += task_tool_stats.get("bandit_time", 0)
            total_semgrep_time += task_tool_stats.get("semgrep_time", 0)

            f.write(json.dumps(parsed, ensure_ascii=False) + "\n")
            f.flush()

            executed_tasks += 1

            task_elapsed = time.time() - task_start
            if ema_task_time is None:
                ema_task_time = task_elapsed
            else:
                ema_task_time = ema_alpha * task_elapsed + (1 - ema_alpha) * ema_task_time

            remaining = total_tasks - executed_tasks
            eta_min = (ema_task_time * remaining) / 60

            pbar.set_postfix_str(f"ETA {eta_min:.1f} min")
            pbar.update(1)

    elapsed = time.time() - start_time

    try:
        metrics_dir = out_dir / "rate" / "vuln_density"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_txt = metrics_dir / f"{Path(output_filename).stem}_metrics.txt"

        save_metrics_result(
            str(output_file),
            str(metrics_txt),
            run_info={
                "dataset": dataset,
                "model": llm.model  
            }
        )

    except Exception:
        pass

    save_experiment_summary(
        out_dir=str(out_dir),
        dataset=dataset,
        model=llm.model,
        technique=technique,
        total_tasks=executed_tasks,
        elapsed_time=elapsed,
        total_llm_time=total_llm_time,
        total_bandit_time=total_bandit_time,
        total_semgrep_time=total_semgrep_time,
        total_api_calls=total_api_calls,
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        jsonl_path=str(output_file)
    )

    print(f"Tasks completed. Results saved to {output_file}")
    return str(output_file)
