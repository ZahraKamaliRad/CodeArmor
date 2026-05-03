from __future__ import annotations
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Tuple
from .openai_client import LLMClient, sanitize_model_name
from src.paths import PATHS
from ..utils.metrics import save_metrics_result
from .analyzer import analyze_code_split
from ..utils.io import csv_log
from ..utils.plot_refinment import load_refinement_df, plot_totals
from ..utils.text_utils import strip_markdown_fences
from ..utils.usage_stats import (get_token_stats,get_llm_stats,get_tool_stats,reset_token_stats,reset_llm_stats,reset_tool_stats)
from ..utils.save_details_result import save_experiment_summary

PLANNING_PROMPT = """
You are a secure-coding assistant.

Follow this sequence:

1. Analyze how the code can be attacked when given untrusted input.
2. Describe the security issues.
3. Provide a concrete mitigation plan step by step.

Follow the structure shown below.

Example:

Language: Python

Intent:
import xml.etree.ElementTree as ET

def parse_xml(xml_string):
'''
Parse xml_string and return the root element of the XML document
'''

Security analysis:
When parsing XML from untrusted sources, attackers may supply malicious XML.
Using the default xml.etree.ElementTree parser can allow external entity
processing or deeply nested entities, leading to XML External Entity (XXE)
attacks or denial-of-service.

The function also lacks input validation and proper error handling, which may
lead to crashes or leakage of internal details.

To be secure, the implementation should:

1. Use a hardened XML parser that disables unsafe features.
2. Validate the input type.
3. Handle parsing errors safely.
4. Avoid exposing internal error details.

Plan:

* Replace the default XML parser with a secure alternative.
* Validate that the input is a non-null string.
* Wrap parsing in error handling and return a controlled error.
* Return the root element only if parsing succeeds.

Now follow the same structure for the task below.

Language: {Language}

Intent:
{Prompt}

Security analysis:
[Write security analysis here]

Plan:
[Write ONLY step-by-step mitigation steps. Do NOT include any code.]

"""

CODING_PROMPT = """
You are a secure-coding assistant.

Language: {Language}

Intent:
{Prompt}

Security analysis and plan:
{Plan}

Write secure {Language} code that implements the intent while strictly following the plan.

Return ONLY the source code. Do not include explanations or markdown.
"""

REVIEW_TEMPLATE = """You are a security reviewer.
List 3-4 main security issues in separate lines.

For each issue, include:
- vulnerability type
- exact location (line number or specific code fragment)
- short fix suggestion

Be concise. No explanations. No markdown.
Code to review:
Code: {CODE} """

IMPROVE_TEMPLATE = """
You are a secure coding assistant.
Based on the critique:
{CRIT}
Improve the following code:
{CODE}
Rules: 
- Modify only the parts related to the listed issues.
- Do NOT rewrite or restructure unrelated code.
- Do NOT add new features or unnecessary complexity.
- Do NOT introduce new vulnerabilities.
- If an issue is unclear, leave that part unchanged.

Return only raw source code. No explanations. No markdown. """


def extract_plan(plan_resp: str) -> str:
    if not isinstance(plan_resp, str):
        return ""

    plan = re.sub(r"```[\s\S]*?```", "", plan_resp)
    return plan.strip()

def extract_raw_code(resp: str) -> str:
    if not isinstance(resp, str):
        return ""
    resp = resp.strip()
    m = re.search(r"```(?:[\w]+)?\s*(.*?)```", resp, re.DOTALL)
    if m:
        return m.group(1).strip()
    return strip_markdown_fences(resp)



def run_two_analyzers(code: str, lang: str, tmpname: str = "snippet") -> Dict[str, Any]:
    scan = analyze_code_split(code, lang, tmpname=tmpname) or {}

    result: Dict[str, Any] = {"loc": int(scan.get("loc") or 0)}

    bandit_secure = True
    semgrep_secure = True

    if "bandit_result" in scan:
        block = scan["bandit_result"]
        issues = block.get("issues") or []
        bandit_secure = bool(block.get("secure", len(issues) == 0))
        result["bandit_result"] = {
            "secure": bandit_secure,
            "issues": issues,
            "summary": block.get("summary") or {},
        }

    if "semgrep_result" in scan:
        block = scan["semgrep_result"]
        issues = block.get("issues") or []
        semgrep_secure = bool(block.get("secure", len(issues) == 0))
        result["semgrep_result"] = {
            "secure": semgrep_secure,
            "issues": issues,
            "summary": block.get("summary") or {},
        }

    result["secure"] = bool(bandit_secure and semgrep_secure)
    return result


def normalize_language(raw: str) -> Tuple[str, str]:
    raw = (raw or "python").strip().lower()
    if raw.startswith("py"):
        return "Python", "python"
    return raw.capitalize(), raw


def llm_call_with_retry(llm: LLMClient, prompt: str, stage: str = "", retries: int = 3):
    resp = None
    for attempt in range(retries):
        try:
            resp = llm.generate_text(prompt).strip()
            break
        except Exception as e:
            msg = str(e)
            if (
                "502" in msg
                or "Bad Gateway" in msg
                or "timeout" in msg
                or "InternalServerError" in msg
                or "overloaded" in msg
            ):
                time.sleep(2**attempt)
                continue
            print(f"[{stage}] {e}")
            break
    return resp


def build_record(task_name: str,intent_text: str,lang_key: str,framework,
    technique: str,plan: str,code: str,scan: Dict[str, Any] | None = None,error: str | None = None,
    initial_code: str | None = None,code_before_last: str | None = None,critique_before_last: str | None = None) -> Dict[str, Any]:
    scan = scan or {}
    record: Dict[str, Any] = {
        "task": task_name,
        "intent": intent_text,
        "language": lang_key,
        "framework": framework,
        "technique": technique,
        "plan": plan,
        "initial_code": initial_code,
        "code_before_last": code_before_last,
        "critique_before_last": critique_before_last,
        "final_code": code,
        "code": code,
        "loc": int(scan.get("loc") or 0),
        "secure": bool(scan.get("secure", False)),
        "issues": {},
    }

    if "bandit_result" in scan:
        block = scan["bandit_result"]
        record["bandit_result"] = block
        record["issues"]["bandit"] = block.get("issues") or []

    if "semgrep_result" in scan:
        block = scan["semgrep_result"]
        record["semgrep_result"] = block
        record["issues"]["semgrep"] = block.get("issues") or []

    if error:
        record["error"] = error

    return record


def write_record(f, record: Dict[str, Any]) -> None:
    f.write(json.dumps(record, ensure_ascii=False) + "\n")
    f.flush()


def refinment_loop(llm: LLMClient,code: str,lang: str,task_name: str,iterations: int,run_out_dir: str):
    current_code = code
    code_before_last = None
    critique_before_last = None

    for i in range(1, iterations + 1):
        print(f"[Refinement] Iteration {i}/{iterations} ({task_name})")

        print("[Step] Generating critique...")
        review_prompt = REVIEW_TEMPLATE.format(CODE=current_code)
        critique = llm_call_with_retry(llm, review_prompt, stage="review")

        if critique is None:
            print("[Step] Critique failed. Stopping refinement.")
            break

        print("[Improve] Improving code...")
        improve_prompt = IMPROVE_TEMPLATE.format(CRIT=critique, CODE=current_code)
        improved_raw = llm_call_with_retry(llm, improve_prompt, stage="improve")

        if improved_raw is None:
            print("[Improve] Improve failed. Stopping refinement.")
            break

        improved_code = extract_raw_code(improved_raw)

        code_before_last = current_code
        critique_before_last = critique

        print("[Step] Running analyzers after improvement...")
        scan = run_two_analyzers(improved_code, lang, tmpname=task_name)

        if "bandit_result" in scan:
            csv_log(
                run_out_dir=run_out_dir,
                task_id=f"{task_name}#bandit",
                iter_idx=i,
                language=lang,
                issues=(scan["bandit_result"].get("issues") or []),
            )

        if "semgrep_result" in scan:
            csv_log(
                run_out_dir=run_out_dir,
                task_id=f"{task_name}#semgrep",
                iter_idx=i,
                language=lang,
                issues=(scan["semgrep_result"].get("issues") or []),
            )

        current_code = improved_code

    return current_code, scan, code_before_last, critique_before_last


def planning_few_shot_gen_code(records,dataset: str,technique: str,limit: int | None = None,
    iterations: int = 0,output_filename: str | None = None,) -> str:

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

    with output_file.open("a", encoding="utf-8") as f:
        for idx, t in enumerate(records, 1):

            reset_token_stats()
            reset_llm_stats()
            reset_tool_stats()

            if limit is not None and idx > limit:
                break

            lang_title, lang_key = normalize_language(t.get("language", ""))

            task_name = (
                t.get("ID")
                or t.get("Prompt ID")
                or t.get("Filename")
                or t.get("file")
                or f"task_{idx}"
            )
            intent_text = t.get("Prompt", "") or ""

            print("\n======================================")
            print(f"Task {idx}: {task_name}")
            print("=======================================")

            print("[Planning] Generating security plan...")
            planning_prompt = PLANNING_PROMPT.format(Prompt=intent_text, Language=lang_title)
            plan_resp = llm_call_with_retry(llm, planning_prompt, stage="planning")

            if plan_resp is None:
                parsed = build_record(
                    task_name=task_name,
                    intent_text=intent_text,
                    lang_key=lang_key,
                    framework=t.get("framework"),
                    technique=technique,
                    plan="",
                    code="",
                    error="planning_failed",
                )
                write_record(f, parsed)

                task_tokens = get_token_stats() or {}
                task_llm_stats = get_llm_stats() or {}
                task_tool_stats = get_tool_stats() or {}

                total_prompt_tokens += task_tokens.get("prompt_tokens", 0)
                total_completion_tokens += task_tokens.get("completion_tokens", 0)
                total_api_calls += task_llm_stats.get("api_calls", 0)
                total_llm_time += task_llm_stats.get("llm_time", 0)
                total_bandit_time += task_tool_stats.get("bandit_time", 0)
                total_semgrep_time += task_tool_stats.get("semgrep_time", 0)
                executed_tasks += 1

                continue

            plan_text = extract_plan(plan_resp)
            print("[Planning] Plan generated.")

            print("[Coding] Generating secure code...")
            coding_prompt = CODING_PROMPT.format(Language=lang_title, Prompt=intent_text, Plan=plan_text)
            code_resp = llm_call_with_retry(llm, coding_prompt, stage="coding")

            if code_resp is None:
                parsed = build_record(
                    task_name=task_name,
                    intent_text=intent_text,
                    lang_key=lang_key,
                    framework=t.get("framework"),
                    technique=technique,
                    plan=plan_text,
                    code="",
                    error="generation_failed",
                )
                write_record(f, parsed)

                task_tokens = get_token_stats() or {}
                task_llm_stats = get_llm_stats() or {}
                task_tool_stats = get_tool_stats() or {}

                total_prompt_tokens += task_tokens.get("prompt_tokens", 0)
                total_completion_tokens += task_tokens.get("completion_tokens", 0)
                total_api_calls += task_llm_stats.get("api_calls", 0)
                total_llm_time += task_llm_stats.get("llm_time", 0)
                total_bandit_time += task_tool_stats.get("bandit_time", 0)
                total_semgrep_time += task_tool_stats.get("semgrep_time", 0)
                executed_tasks += 1

                continue

            code_text = extract_raw_code(code_resp)
            initial_code = code_text
            print("[Coding] Code generated.")

            print("[Step] Running analyzers (iter=0)...")
            scan = run_two_analyzers(code_text, lang_key, tmpname=task_name)

            if "bandit_result" in scan:
                csv_log(
                    run_out_dir=str(out_dir),
                    task_id=f"{task_name}#bandit",
                    iter_idx=0,
                    language=lang_key,
                    issues=(scan["bandit_result"].get("issues") or []),
                )

            if "semgrep_result" in scan:
                csv_log(
                    run_out_dir=str(out_dir),
                    task_id=f"{task_name}#semgrep",
                    iter_idx=0,
                    language=lang_key,
                    issues=(scan["semgrep_result"].get("issues") or []),
                )

            code_before_last = None
            critique_before_last = None

            if int(iterations) > 0:
                print(f"[Refinement] Starting refinement loop (iterations={iterations})...")
                code_text, scan, code_before_last, critique_before_last = refinment_loop(
                    llm=llm,code=code_text,lang=lang_key,task_name=task_name,
                    iterations=iterations,run_out_dir=str(out_dir))
                print("[Refinement] Refinement loop finished.")

            parsed = build_record(
                task_name=task_name,
                intent_text=intent_text,
                lang_key=lang_key,
                framework=t.get("framework"),
                technique=technique,
                plan=plan_text,
                code=code_text,
                scan=scan,
                initial_code=initial_code,
                code_before_last=code_before_last,
                critique_before_last=critique_before_last
            )
            write_record(f, parsed)

            task_tokens = get_token_stats() or {}
            task_llm_stats = get_llm_stats() or {}
            task_tool_stats = get_tool_stats() or {}
            total_prompt_tokens += task_tokens.get("prompt_tokens", 0)
            total_completion_tokens += task_tokens.get("completion_tokens", 0)
            total_api_calls += task_llm_stats.get("api_calls", 0)
            total_llm_time += task_llm_stats.get("llm_time", 0)
            total_bandit_time += task_tool_stats.get("bandit_time", 0)
            total_semgrep_time += task_tool_stats.get("semgrep_time", 0)
            executed_tasks += 1

    elapsed = time.time() - start_time

    try:
        metrics_dir = out_dir / "rate" / "vuln_density"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        metrics_txt = metrics_dir / f"{output_file.stem}_metrics.txt"

        save_metrics_result(
            str(output_file),
            str(metrics_txt),
            run_info={
                "dataset": dataset,
                "model": llm.model
            }
        )

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

        csv_path = out_dir / "Effect_Of_Refinment.csv"
        if csv_path.exists():
            df = load_refinement_df(csv_path)
            out_img = out_dir / "plots" / "Refinment_Plot.png"
            out_img.parent.mkdir(parents=True, exist_ok=True)
            plot_totals(csv_path, out=out_img, kind="line", show=False, df=df, verbose=False)

    except Exception:
        pass

    return str(output_file)
