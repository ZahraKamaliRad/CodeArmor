from __future__ import annotations
import json
import re
import time
from pathlib import Path
from tqdm import tqdm
from typing import Any, Dict, Tuple, List

from .openai_client import LLMClient, sanitize_model_name
from src.paths import PATHS
from ..utils.metrics import save_metrics_result
from .analyzer import analyze_jsonl_batch
from ..utils.io import csv_log
from ..utils.plot_refinment import load_refinement_df, plot_totals
from ..utils.text_utils import  extract_raw_code
from ..utils.usage_stats import (
    get_token_stats, get_llm_stats, get_tool_stats,
    reset_token_stats, reset_llm_stats, reset_tool_stats
)
from ..utils.save_details_result import save_experiment_summary
from ..utils.check_syntax import filter_jsonl,is_syntax_valid



PLANNING_PROMPT= """
You are a secure-coding assistant.

Language: {Language}

Produce a concise security mitigation plan for the given task.

Rules:
- Analyze security risks internally, but do NOT output the analysis.
- Output ONLY the mitigation plan between these exact markers:
###PLAN_START###
###PLAN_END###
- Do NOT output code.
- Do NOT repeat or summarize the intent.
- Use a numbered list.
- Keep the plan concise, specific, and actionable.
- Focus only on security-relevant mitigations for {Language}.
- Do NOT include any extra text outside the markers.
- Do NOT use markdown code fences.

Intent:
{Prompt}
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
Review the following answer and find security problems with it 
List 3-4 main security issues in separate lines.
For each issue:
- vulnerability type
- location
- fix suggestion

Code:
{CODE}
"""

IMPROVE_TEMPLATE = """You are a secure coding assistant.

Based on the critique:
{CRIT}

Improve the following code:
{CODE}

Rules:
- Only fix listed issues
- Do not refactor unrelated parts
- Do not add features
- Keep the code concise and balanced in length; avoid producing overly long or verbose code

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


def llm_call_with_retry(llm, prompt, stage="", retries=3):
    for i in range(retries):
        try:
            return llm.generate_text(prompt).strip()
        except Exception:
            time.sleep(2**i)
    return None



def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path: Path, records: List[Dict[str, Any]]):
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")



def refinement_batch(jsonl_path: Path, llm, iteration: int, run_out_dir: Path, total_iterations: int):
    print(f"\n========== Refinement Iteration {iteration}/{total_iterations} ==========")
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
            print("Reviewing...")

            reset_token_stats()
            reset_llm_stats()

            code = rec.get("code") or ""

            critique = llm_call_with_retry(llm, REVIEW_TEMPLATE.format(CODE=code))

            if not critique:
                completed_tasks += 1
                pbar.update(1)
                continue

            print("Improving code...")

            improved_raw = llm_call_with_retry(
                llm,
                IMPROVE_TEMPLATE.format(CRIT=critique, CODE=code)
            )

            if not improved_raw:
                completed_tasks += 1
                pbar.update(1)
                continue

            improved_code = extract_raw_code(improved_raw)

            syntax_ok = is_syntax_valid(improved_code)

            if "iterations" not in rec:
                rec["iterations"] = []

            rec["iterations"].append({
                "iter": iteration,
                "review": critique,
                "improve_code": improved_code,
                "syntax_valid": syntax_ok
            })

            if syntax_ok:
                rec["code"] = improved_code
            else:
                print(f"[SKIP] invalid code in iteration {iteration} for task {rec.get('task')}")

            t = get_token_stats()
            l = get_llm_stats()
            iter_prompt_tokens += t.get("prompt_tokens", 0)
            iter_completion_tokens += t.get("completion_tokens", 0)
            iter_api_calls += l.get("api_calls", 0)
            iter_llm_time += l.get("llm_time", 0)

            completed_tasks += 1

            task_elapsed = time.time() - task_start
            ema_task_time = task_elapsed if ema_task_time is None else (
                ema_alpha * task_elapsed + (1 - ema_alpha) * ema_task_time
            )

            remaining = total_tasks - completed_tasks
            eta_seconds = ema_task_time * remaining
            pbar.set_postfix_str(f"ETA {eta_seconds/60:.1f} min")
            pbar.update(1)

    
    write_jsonl(jsonl_path, records)

   
    reset_tool_stats()
    filter_jsonl(str(jsonl_path))   

   
    analyze_jsonl_batch(str(jsonl_path))
    filtered_records = read_jsonl(jsonl_path)

    
    analyzed_by_task = {r["task"]: r for r in filtered_records}

    tool_stats = get_tool_stats()
    iter_bandit_time = tool_stats.get("bandit_time", 0)
    iter_semgrep_time = tool_stats.get("semgrep_time", 0)

    for rec in records:
        task = rec.get("task")
        analyzed = analyzed_by_task.get(task)

        if analyzed:
            rec["iterations"][-1]["analysis"] = {
                "loc": analyzed.get("loc"),
                "bandit_result": analyzed.get("bandit_result"),
                "semgrep_result": analyzed.get("semgrep_result")
            }

            if "bandit_result" in analyzed:
                csv_log(run_out_dir, f"{task}#bandit", iteration, rec.get("language"), analyzed["bandit_result"].get("issues"))
            if "semgrep_result" in analyzed:
                csv_log(run_out_dir, f"{task}#semgrep", iteration, rec.get("language"), analyzed["semgrep_result"].get("issues"))

    write_jsonl(jsonl_path, records)

    return {
        "prompt_tokens": iter_prompt_tokens,
        "completion_tokens": iter_completion_tokens,
        "api_calls": iter_api_calls,
        "llm_time": iter_llm_time,
        "bandit_time": iter_bandit_time,
        "semgrep_time": iter_semgrep_time,
    }


def planning_rci_gen_code(records, dataset: str, technique: str, limit: int | None = None, iterations: int = 0,
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

            raw_plan_response = llm_call_with_retry(
            llm,PLANNING_PROMPT.format(Prompt=intent, Language=lang_title))
            plan = extract_plan(raw_plan_response)
            print("\n[LLM RAW PLAN RESPONSE START]")
            print(raw_plan_response)
            print("[LLM RAW PLAN RESPONSE END]\n")

            print("\n[EXTRACTED PLAN START]")
            print(plan)
            print("[EXTRACTED PLAN END]\n")

            code = extract_raw_code(
                llm_call_with_retry(llm, CODING_PROMPT.format(Prompt=intent, Language=lang_title, Plan=plan))
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
        if "bandit_result" in r:
            csv_log(out_dir, f"{r['task']}#bandit", 0, r["language"], r["bandit_result"].get("issues"))
        if "semgrep_result" in r:
            csv_log(out_dir, f"{r['task']}#semgrep", 0, r["language"], r["semgrep_result"].get("issues"))

    write_jsonl(output_file, recs)

    for i in range(1, iterations + 1):
        stats = refinement_batch(output_file, llm, iteration=i, run_out_dir=out_dir, total_iterations=iterations)
        total_prompt_tokens += stats["prompt_tokens"]
        total_completion_tokens += stats["completion_tokens"]
        total_api_calls += stats["api_calls"]
        total_llm_time += stats["llm_time"]
        total_bandit_time += stats["bandit_time"]
        total_semgrep_time += stats["semgrep_time"]

    elapsed = time.time() - start_time
    csv_path = out_dir / "Effect_Of_Refinment.csv"

    if csv_path.exists():
        df = load_refinement_df(csv_path)
        plot_totals(csv_path, out=out_dir / "plots" / "Refinment.png", df=df)
    else:
        print("[WARN] CSV not found for plotting:", csv_path)

    save_metrics_result(
        str(output_file),
        str(out_dir / "metrics.txt"),
        run_info={"dataset": dataset, "model": llm.model}
    )

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
