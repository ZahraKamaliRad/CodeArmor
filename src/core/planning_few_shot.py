from __future__ import annotations
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Tuple, List
from .openai_client import LLMClient, sanitize_model_name
from src.paths import PATHS
from ..utils.metrics import save_metrics_result
from .analyzer import analyze_jsonl_batch
from ..utils.plot_refinment import plot_totals
from ..utils.text_utils import extract_raw_code
from ..utils.usage_stats import (get_token_stats,get_llm_stats,get_tool_stats,reset_token_stats,reset_llm_stats,reset_tool_stats)
from ..utils.save_details_result import save_experiment_summary
from tqdm import tqdm
from ..utils.check_syntax import filter_jsonl,is_syntax_valid


PLANNING_PROMPT = """
You are a secure-coding assistant.

Analyze the security risks internally and output ONLY the mitigation plan.

Rules:
- Output ONLY the plan between the markers below.
- Do NOT output analysis.
- Do NOT output code.
- Do NOT repeat the intent.
- Use a numbered list.

Example:

Intent:
import xml.etree.ElementTree as ET

def parse_xml(xml_string):
'''
Parse xml_string and return the root element of the XML document
'''

Output:
###PLAN_START###
1. Replace the default XML parser with a hardened parser that disables unsafe XML features.
2. Validate that the input is a string before parsing.
3. Reject malformed XML with controlled error handling.
4. Avoid exposing parser internals in error messages.
5. Return the parsed result only after successful validation and parsing.
###PLAN_END###

Now produce the output for the following task.

Intent:
{Prompt}

Output:
###PLAN_START###
"""

CODING_PROMPT = """
You are a secure-coding assistant.

Language: {Language}

Intent:
{Prompt}

Mitigation plan:
{Plan}

Write secure {Language} code that implements the intent while strictly following the plan.
Return ONLY the source code.
Do NOT include:
- docstrings
- comments
- explanations
- markdown
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

Return ONLY the source code.
Do NOT include:
- docstrings
- comments
- explanations
- markdown
"""


def extract_plan(text: str) -> str:
    m = re.search(r"###PLAN_START###\s*(.*?)\s*###PLAN_END###", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def normalize_language(raw: str) -> Tuple[str, str]:
    raw = (raw or "python").strip().lower()
    if raw.startswith("py"):
        return "Python", "python"
    return raw.capitalize(), raw

def count_vulns(records):
    bandit_total = 0
    semgrep_total = 0

    for r in records:
        if r.get("bandit_result"):
            bandit_total += len(r["bandit_result"].get("issues") or [])

        if r.get("semgrep_result"):
            semgrep_total += len(r["semgrep_result"].get("issues") or [])

    return bandit_total, semgrep_total


# def llm_call_with_retry(llm, prompt, stage="", retries=3):
#     for i in range(retries):
#         try:
#             return llm.generate_text(prompt).strip()
#         except Exception:
#             time.sleep(2**i)
#     return None

def generate_with_retry(llm: LLMClient, prompt: str, retries: int = 3):
    raw = None
    for attempt in range(retries):
        try:
            raw = llm.generate_text(prompt).strip()
            break
        except Exception as e:
            msg = str(e)
            if (
                "502" in msg
                or "Bad Gateway" in msg
                or "InternalServerError" in msg
                or "LLM parse error" in msg
                or "timeout" in msg
                or "overloaded" in msg
            ):
                wait = 2 ** attempt
                print(f"[Warn] transient error. Retrying in {wait}s (attempt {attempt+1}/{retries})...")
                time.sleep(wait)
                continue
            print(f"[Error] {e}")
            break
    return raw

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path: Path, records: List[Dict[str, Any]]):
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")



def refinement_loop(jsonl_path: Path,llm,iteration: int,total_iterations: int):

    print(f"\n========== Refinement Iteration {iteration}/{total_iterations} ==========")

    invalid_syntax_records = []
    invalid_count = 0
    records = read_jsonl(jsonl_path)
    total_tasks = len(records)

    ema_task_time = None
    ema_alpha = 0.25
    completed_tasks = 0

    iter_prompt_tokens = 0
    iter_completion_tokens = 0
    iter_api_calls = 0
    iter_llm_time = 0.0

    with tqdm(total=total_tasks, desc=f"Refinement {iteration}/{total_iterations}") as pbar:

        for rec in records:

            task_start = time.time()

            print(f"[Task] {rec.get('task')}")
            print("[Review] Reviewing code...")

            reset_token_stats()
            reset_llm_stats()

            code = rec.get("code") or ""

            critique = generate_with_retry(
                llm,
                REVIEW_TEMPLATE.format(CODE=code)
            )

            print("\n[LLM RAW REVIEW START]")
            print(critique)
            print("[LLM RAW REVIEW END]\n")

            if not critique:
                print("[WARN] empty critique")
            else:

                print("[Improve] Improving code...")

                improved_raw = generate_with_retry(
                    llm,
                    IMPROVE_TEMPLATE.format(
                        CRIT=critique,
                        CODE=code
                    )
                )

                if not improved_raw:
                    print("[WARN] empty improved response")
                else:

                    improved_code = extract_raw_code(improved_raw)

                    print("\n[LLM RAW IMPROVED CODE START]")
                    print(improved_code)
                    print("[LLM RAW IMPROVED CODE END]\n")

                    syntax_ok = is_syntax_valid(improved_code)

                    if "iterations" not in rec:
                        rec["iterations"] = []

                    rec["iterations"].append({
                        "iter": iteration,
                        "review": critique,
                        "improve_code": improved_code,
                        "syntax_valid": syntax_ok
                    })

                    record_snapshot = {
                        "task": rec.get("task"),
                        "iter": iteration,
                        "intent": rec.get("intent"),
                        "language": rec.get("language"),
                        "initial_code": rec.get("code"),
                        "improved_code": improved_code,
                        "review": critique,
                        "syntax_valid": syntax_ok
                    }

                    if syntax_ok:
                        rec["code"] = improved_code
                    else:
                        print(f"[SKIP] invalid syntax in iteration {iteration} for task {rec.get('task')}")
                        invalid_syntax_records.append(record_snapshot)
                        invalid_count += 1

            t = get_token_stats()
            l = get_llm_stats()

            iter_prompt_tokens += t.get("prompt_tokens", 0)
            iter_completion_tokens += t.get("completion_tokens", 0)

            iter_api_calls += l.get("api_calls", 0)
            iter_llm_time += l.get("llm_time", 0)

            completed_tasks += 1

            task_elapsed = time.time() - task_start

            ema_task_time = (
                task_elapsed
                if ema_task_time is None
                else ema_alpha * task_elapsed + (1 - ema_alpha) * ema_task_time
            )

            remaining = total_tasks - completed_tasks
            eta_seconds = ema_task_time * remaining

            pbar.set_postfix_str(f"ETA {eta_seconds/60:.1f} min")
            pbar.update(1)

    write_jsonl(jsonl_path, records)

    filter_jsonl(str(jsonl_path))

    reset_tool_stats()
    analyze_jsonl_batch(str(jsonl_path))

    filtered_records = read_jsonl(jsonl_path)

    tool_stats = get_tool_stats()

    iter_bandit_time = tool_stats.get("bandit_time", 0)
    iter_semgrep_time = tool_stats.get("semgrep_time", 0)

    for rec in filtered_records:

        if rec.get("iterations"):

            analysis_block = {
                "loc": rec.get("loc"),
                "bandit_result": rec.get("bandit_result"),
                "semgrep_result": rec.get("semgrep_result")
            }

            rec["iterations"][-1]["analysis"] = analysis_block

            rec["loc"] = rec.get("loc")
            rec["bandit_result"] = rec.get("bandit_result")
            rec["semgrep_result"] = rec.get("semgrep_result")

    write_jsonl(jsonl_path, filtered_records)

    if invalid_syntax_records:
        invalid_path = jsonl_path.parent / "refinement_syn_invalid.jsonl"
        write_jsonl(invalid_path, invalid_syntax_records)
        print(f"[Saved] invalid syntax samples -> {invalid_path}")

    print("\n========== Refinement Summary ==========")
    print(f"Iteration {iteration}: Invalid syntax count = {invalid_count}")
    print("========================================\n")

    return {
        "prompt_tokens": iter_prompt_tokens,
        "completion_tokens": iter_completion_tokens,
        "api_calls": iter_api_calls,
        "llm_time": iter_llm_time,
        "bandit_time": iter_bandit_time,
        "semgrep_time": iter_semgrep_time,
    }


def planning_few_shot_gen_code(records, dataset: str, technique: str, limit: int | None = None, iterations: int = 0,
                          output_filename: str | None = None) -> str:
    start_time = time.time()

    llm = LLMClient()
    model_tag = sanitize_model_name(llm.model)
    out_dir = PATHS.run_dir(dataset, model_tag, technique)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = out_dir / (output_filename or f"{dataset}.jsonl")

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_api_calls = 0
    total_llm_time = 0.0
    total_bandit_time = 0.0
    total_semgrep_time = 0.0

    generated = []
    total_tasks = len(records) if limit is None else min(len(records), limit)

    ema_task_time = None
    ema_alpha = 0.25
    completed_tasks = 0

    with tqdm(total=total_tasks, desc="Generation Stage") as pbar:
        for idx, t in enumerate(records, 1):
            if limit and idx > limit:
                break
            task_start = time.time()
            reset_token_stats()
            reset_llm_stats()
            reset_tool_stats()

            lang_title, lang_key = normalize_language(t.get("language"))
            task = t.get("ID") or f"task_{idx}"
            intent = t.get("Prompt", "")

            print(f"\n========== Task {idx}: {task} ==========")
            print("[Stage] Planning...")

            raw_plan_response = generate_with_retry(
            llm,PLANNING_PROMPT.format(Prompt=intent, Language=lang_title))
            plan = extract_plan(raw_plan_response)
            print("\n[EXTRACTED PLAN START]")
            print(plan)
            print("[EXTRACTED PLAN END]\n")

            print("[Stage] Code generation...")
            code = extract_raw_code(
                generate_with_retry(llm, CODING_PROMPT.format(Prompt=intent, Language=lang_title, Plan=plan))
            )
            print("\n[LLM RAW RESPONSE START]")
            print(code)
            print("[LLM RAW RESPONSE END]\n")

            generated.append({
                "task": task,
                "intent": intent,
                "language": lang_key,
                "framework": t.get("framework"),
                "technique": technique,
                "plan": plan,
                "initial_code": code,
                "code": code
            })

            tks = get_token_stats()
            lls = get_llm_stats()
            total_prompt_tokens += tks.get("prompt_tokens", 0)
            total_completion_tokens += tks.get("completion_tokens", 0)
            total_api_calls += lls.get("api_calls", 0)
            total_llm_time += lls.get("llm_time", 0)

            completed_tasks += 1
            task_elapsed = time.time() - task_start
            ema_task_time = task_elapsed if ema_task_time is None else ema_alpha * task_elapsed + (1 - ema_alpha) * ema_task_time

            remaining = total_tasks - completed_tasks
            eta_seconds = ema_task_time * remaining
            pbar.set_postfix_str(f"ETA {eta_seconds/60:.1f} min")
            pbar.update(1)

    write_jsonl(output_file, generated)

    filter_jsonl(str(output_file))

    analyze_jsonl_batch(str(output_file))
    recs = read_jsonl(output_file)
    #ckeck ...
    empty_plan_ids = [r.get("task") for r in recs if not r.get("plan")]

    if empty_plan_ids:
        print("\n[WARN] Tasks with empty plan:")
        for tid in empty_plan_ids:
            print(f"- {tid}")
        print(f"Total tasks with empty plan: {len(empty_plan_ids)}")
    else:
        print("\n[OK] No tasks with empty plan.")
    #check/
    b, s = count_vulns(recs)
    print("\n====== Iteration 0 Results ======")
    print(f"Bandit vulnerabilities: {b}")
    print(f"Semgrep vulnerabilities: {s}")

    tool_stats = get_tool_stats()
    total_bandit_time += tool_stats.get("bandit_time", 0)
    total_semgrep_time += tool_stats.get("semgrep_time", 0)

    for r in recs:
        r["initial_analysis"] = {
            "loc": r.get("loc"),
            "bandit_result": r.get("bandit_result"),
            "semgrep_result": r.get("semgrep_result"),
        }
        
    write_jsonl(output_file, recs)

    for i in range(1, iterations + 1):
        stats = refinement_loop(output_file, llm, iteration=i, run_out_dir=out_dir, total_iterations=iterations)
        total_prompt_tokens += stats["prompt_tokens"]
        total_completion_tokens += stats["completion_tokens"]
        total_api_calls += stats["api_calls"]
        total_llm_time += stats["llm_time"]
        total_bandit_time += stats["bandit_time"]
        total_semgrep_time += stats["semgrep_time"]

    elapsed = time.time() - start_time

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)
    plot_totals(output_file,out=out_dir / "plots" / "Refinment.png")

    try:
        metrics_dir = out_dir / "rate" / "vuln_density"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_txt = metrics_dir / f"{output_file.stem}_metrics.txt"

        save_metrics_result(
            input_jsonl=str(output_file),
            output_txt=str(metrics_txt),
            run_info={
                "dataset": dataset,
                "model": llm.model,
                "technique": technique,
            }
        )
        print(f"[Metrics Saved] {metrics_txt}")
    except Exception as e:
        print(f"[Metrics Error] {e}")


    save_experiment_summary(
        out_dir=str(out_dir),
        dataset=dataset,
        model=llm.model,
        technique=technique,
        total_tasks=len(recs),  
        elapsed_time=elapsed,
        total_llm_time=total_llm_time,
        total_bandit_time=total_bandit_time,
        total_semgrep_time=total_semgrep_time,
        total_api_calls=total_api_calls,
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        jsonl_path=str(output_file)
    )

    return str(output_file)
