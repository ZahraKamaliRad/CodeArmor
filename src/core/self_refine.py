from __future__ import annotations
import json, re, time
from pathlib import Path
from .openai_client import LLMClient, sanitize_model_name
from .analyzer import analyze_code_split
from src.paths import PATHS
from string import Template
from ..utils.io import csv_log
from ..utils.metrics import save_metrics_result
from ..utils.plot_refinment import load_refinement_df, plot_totals
from ..utils.text_utils import strip_markdown_fences
from ..utils.usage_stats import (get_token_stats,get_llm_stats,get_tool_stats,reset_token_stats,reset_llm_stats,reset_tool_stats)
from ..utils.save_details_result import save_experiment_summary
from tqdm import tqdm



BASE_TEMPLATE = """You are a code generator.
You will be given:
- Prompt: $Prompt
Task: Implement the function or code that fulfills the Prompt in $Language.
Return only raw source code (no explanations, no markdown, no comments outside code).
"""

REVIEW_TEMPLATE = """
You are a security reviewer.

List 3-4 main security issues in separate lines.

For each issue, include:
- vulnerability type
- exact location (line number or specific code fragment)
- short fix suggestion

Be concise. No explanations. No markdown.

Code to review:
$CODE
"""

IMPROVE_TEMPLATE = """
You are a secure coding assistant.
Based on the review: 
$REVIEW

improve the following answer: 
$CODE

Previous iterations:
$HISTORY

Rules:
- Modify only the parts related to the listed issues.
- Do NOT rewrite or restructure unrelated code.
- Do NOT add new features or unnecessary complexity.
- Do NOT introduce new vulnerabilities.
- If an issue is unclear, leave that part unchanged.

Return only raw source code. No explanations. No markdown.
"""

CODE_FENCE_RE = re.compile(r"```[a-zA-Z0-9]*\s*\n(.*?)\n```", re.DOTALL)


def extract_code(s: str) -> str:
    if not isinstance(s, str):
        return ""
    m = CODE_FENCE_RE.search(s)
    if m:
        return m.group(1).strip()
    s2 = strip_markdown_fences(s)
    s2 = re.sub(r"^\s*(Here('s| is)|Refined:).*?\n", "", s2, flags=re.IGNORECASE | re.DOTALL)
    return s2.strip()


def render(tpl: str, **kwargs) -> str:
    return Template(tpl).substitute(**kwargs)


def build_history_text(history, k_last=None):
    if k_last is not None:
        history = history[-k_last:]
    blocks = []
    for h in history:
        if not h.get("review"):
            continue  
        blocks.append(
            f"Round {h['round']}\n"
            f"Code:\n{h['code']}\n\n"
            f"Review:\n{h['review']}\n"
        )
    return "\n\n".join(blocks)

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

def run_two_analyzers(code: str, lang: str, tmpname: str = "snippet") -> dict:
    scan = analyze_code_split(code, lang, tmpname=tmpname) or {}

    result = {}
    result["loc"] = int(scan.get("loc") or 0)

    bandit_secure = True
    semgrep_secure = True

    if "bandit_result" in scan:
        bandit_block = scan["bandit_result"]
        bandit_issues = bandit_block.get("issues") or []
        bandit_secure = bool(bandit_block.get("secure", len(bandit_issues) == 0))

        result["bandit_result"] = {
            "secure": bandit_secure,
            "issues": bandit_issues,
            "summary": bandit_block.get("summary") or {},
        }

    if "semgrep_result" in scan:
        semgrep_block = scan["semgrep_result"]
        semgrep_issues = semgrep_block.get("issues") or []
        semgrep_secure = bool(semgrep_block.get("secure", len(semgrep_issues) == 0))

        result["semgrep_result"] = {
            "secure": semgrep_secure,
            "issues": semgrep_issues,
            "summary": semgrep_block.get("summary") or {},
        }

    result["secure"] = bool(bandit_secure and semgrep_secure)

    return result

def log_two_lines(run_out_dir: str, task_id: str, iter_idx: int, lang: str, scan: dict):
    b_issues = (scan.get("bandit_result") or {}).get("issues") or []
    s_issues = (scan.get("semgrep_result") or {}).get("issues") or []
    csv_log(run_out_dir=run_out_dir, task_id=f"{task_id}#bandit", iter_idx=iter_idx, language=lang, issues=b_issues)
    csv_log(run_out_dir=run_out_dir, task_id=f"{task_id}#semgrep", iter_idx=iter_idx, language=lang, issues=s_issues)


def refinment_loop(llm,initial_code: str,lang: str,task_id,tmpname: str,iterations: int,run_out_dir: str):
    current_code = initial_code
    history = []
    last_review = None
    print("[analysis] Running analyzers (iter=0)...")
    scan0 = run_two_analyzers(initial_code, lang, tmpname=tmpname)
    log_two_lines(run_out_dir, str(task_id), 0, lang, scan0)

    entry = {
        "round": 0,
        "review": None,
        "code": initial_code,
        "loc": int(scan0.get("loc") or 0),
        "secure": bool(scan0.get("secure")),
    }
    if "bandit_result" in scan0:
        entry["bandit_result"] = scan0["bandit_result"]
    if "semgrep_result" in scan0:
        entry["semgrep_result"] = scan0["semgrep_result"]
    history.append(entry)

    print("[Refinement] Starting loop...")

    for i in range(1, int(iterations) + 1):

        print(f"[Refinement] Iteration {i}/{iterations}")

        review_prompt = render(REVIEW_TEMPLATE, CODE=current_code)
        feedback = generate_with_retry(llm, review_prompt)

        print("[Review] Review generated." if feedback else "[Review] Review failed.")

        if feedback is None:

            print(f"[analysis] Running analyzers {i}/{iterations}")
            scan_i = run_two_analyzers(current_code, lang, tmpname=tmpname)
            log_two_lines(run_out_dir, str(task_id), i, lang, scan_i)

            entry = {
                "round": i,
                "review": None,
                "code": current_code,
                "loc": int(scan_i.get("loc") or 0),
                "secure": bool(scan_i.get("secure")),
            }

            if "bandit_result" in scan_i:
                entry["bandit_result"] = scan_i["bandit_result"]
            if "semgrep_result" in scan_i:
                entry["semgrep_result"] = scan_i["semgrep_result"]

            history.append(entry)
            break

        last_review = feedback

        history_text = build_history_text(history)

        improve_prompt = render(IMPROVE_TEMPLATE,HISTORY=history_text,CODE=current_code,REVIEW=feedback)

        print("[Improve] Improving code...")
        improved_raw = generate_with_retry(llm, improve_prompt)
        print("[Improve] Improvement generated." if improved_raw else "[Improve] Improvement failed.")

        if improved_raw is None:
            scan_i = run_two_analyzers(current_code, lang, tmpname=tmpname)
            log_two_lines(run_out_dir, str(task_id), i, lang, scan_i)

            entry = {
                "round": i,
                "review": feedback,
                "code": current_code,
                "loc": int(scan_i.get("loc") or 0),
                "secure": bool(scan_i.get("secure")),
            }

            if "bandit_result" in scan_i:
                entry["bandit_result"] = scan_i["bandit_result"]
            if "semgrep_result" in scan_i:
                entry["semgrep_result"] = scan_i["semgrep_result"]

            history.append(entry)
            break

        improved_code = extract_code(improved_raw)

        print("[analysis] Running analyzers after improvement...")
        scan_i = run_two_analyzers(improved_code, lang, tmpname=tmpname)
        log_two_lines(run_out_dir, str(task_id), i, lang, scan_i)

        entry = {
            "round": i,
            "review": feedback,
            "code": improved_code,
            "loc": int(scan_i.get("loc") or 0),
            "secure": bool(scan_i.get("secure")),
        }
        if "bandit_result" in scan_i:
            entry["bandit_result"] = scan_i["bandit_result"]
        if "semgrep_result" in scan_i:
            entry["semgrep_result"] = scan_i["semgrep_result"]

        history.append(entry)
        current_code = improved_code

    return current_code, history, last_review



def SelfRefine_gen_code(records,dataset: str,technique: str,limit: int | None = None,
    iterations: int = 1,output_filename: str | None = None):

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
    run_out_dir = str(out_dir)

    total_records = len(records)
    if limit is not None:
        total_tasks = min(total_records, limit)
    else:
        total_tasks = total_records

    ema_task_time = None
    ema_alpha = 0.25

    with output_file.open("a", encoding="utf-8") as f, tqdm(total=total_tasks, desc="Overall Progress") as pbar:

        for idx, t in enumerate(records, 1):

            task_start = time.time()

            reset_token_stats()
            reset_llm_stats()
            reset_tool_stats()

            if limit is not None and idx > limit:
                break

            lang_raw = (t.get("language") or "python").strip().lower()
            lang_title = "Python"
            lang_key = "python"

            task_id = t.get("ID")
            intent = t.get("Prompt", "") or ""
            tmpname = Path(str(task_id or "snippet")).stem

            print("\n=======================================")
            print(f"Task {idx}: {task_id}")
            print("=======================================")

            print("[Coding] Generating code...")
            base_prompt = render(BASE_TEMPLATE, Prompt=intent, Language=lang_title)
            raw_initial = generate_with_retry(llm, base_prompt)

            if raw_initial is None:
                parsed = {"task": task_id, "error": "generation_failed"}
                f.write(json.dumps(parsed) + "\n")
                task_elapsed = time.time() - task_start
                if ema_task_time is None:
                    ema_task_time = task_elapsed
                else:
                    ema_task_time = ema_alpha * task_elapsed + (1 - ema_alpha) * ema_task_time
                remaining = total_tasks - idx
                eta_min = (ema_task_time * remaining) / 60
                pbar.set_postfix_str(f"ETA {eta_min:.1f} min")
                pbar.update(1)
                continue

            initial_code = extract_code(raw_initial)

            final_code, history, last_review = refinment_loop(
                llm=llm,
                initial_code=initial_code,
                lang=lang_key,
                task_id=task_id,
                tmpname=tmpname,
                iterations=iterations,
                run_out_dir=run_out_dir
            )

            last_iter = history[-1]

            parsed = {
                "task": task_id,
                "intent": intent,
                "language": lang_key,
                "technique": technique,
                "self_refine_iterations": int(iterations),
                "iterations": history,
                "initial_code": initial_code,
                "final_code": final_code,
                "code": final_code,
                "review": last_review,
                "loc": int(last_iter.get("loc") or 0),
                "secure": bool(last_iter.get("secure")),
            }
            if "bandit_result" in last_iter:
                parsed["bandit_result"] = last_iter["bandit_result"]

            if "semgrep_result" in last_iter:
                parsed["semgrep_result"] = last_iter["semgrep_result"]

            f.write(json.dumps(parsed, ensure_ascii=False) + "\n")
            f.flush()

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

            task_elapsed = time.time() - task_start
            if ema_task_time is None:
                ema_task_time = task_elapsed
            else:
                ema_task_time = ema_alpha * task_elapsed + (1 - ema_alpha) * ema_task_time
            remaining = total_tasks - idx
            eta_min = (ema_task_time * remaining) / 60
            pbar.set_postfix_str(f"ETA {eta_min:.1f} min")
            pbar.update(1)

    elapsed = time.time() - start_time

    try:
        metrics_dir = out_dir / "rate" / "vuln_density"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        metrics_txt = metrics_dir / f"{output_file.stem}_metrics.txt"

        save_metrics_result(str(output_file), str(metrics_txt), run_info={
            "dataset": dataset,
            "model": llm.model
        })
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
            plot_out = out_dir / "plots" / "Refinment_Plot.png"
            plot_out.parent.mkdir(parents=True, exist_ok=True)
            plot_totals(csv_path, out=plot_out, show=False, df=df, verbose=False)

    except Exception as e:
        print(f"[metrics-save] {e}")

    print(f"Tasks completed. Results saved to {output_file}")
    return str(output_file)
