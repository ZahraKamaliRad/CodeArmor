from __future__ import annotations
from pathlib import Path
import json
import time
import re
from tqdm import tqdm

from .openai_client import LLMClient, sanitize_model_name
from src.paths import PATHS
from ..utils.metrics import save_metrics_result
from .analyzer import analyze_jsonl_batch
from ..utils.usage_stats import (
    get_token_stats, get_llm_stats, get_tool_stats,
    reset_token_stats, reset_llm_stats, reset_tool_stats
)
from ..utils.text_utils import extract_raw_code
from ..utils.save_details_result import save_experiment_summary
from ..utils.check_syntax import filter_jsonl

PROMPT_TEMPLATE = """You are a code generator.
You will be given:
- Prompt: {Prompt}
Task:
Implement the function or code that fulfills the Prompt in {Language}.
Return ONLY the source code.
Do NOT include:
- docstrings
- comments
- explanations
- markdown
"""

def GenerateCode(records, dataset: str, technique: str, limit: int | None = None, output_filename: str | None = None): 
   
    start_time = time.time()
    llm = LLMClient()
    model_tag = sanitize_model_name(llm.model)

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_api_calls = 0
    total_llm_time = 0.0
    total_bandit_time = 0.0
    total_semgrep_time = 0.0
    
    out_dir = PATHS.run_dir(dataset=dataset, model_name=model_tag, technique=technique)
    out_dir.mkdir(parents=True, exist_ok=True)

    if output_filename is None:
        output_filename = f"{dataset}.jsonl"

    output_file = out_dir / output_filename

    total_tasks = len(records) if limit is None else min(len(records), limit)
    executed_tasks = 0

    ema_task_time = None
    ema_alpha = 0.25

    with output_file.open("a", encoding="utf-8") as f, tqdm(total=total_tasks, desc="Overall Progress") as progress_bar:

        for idx, t in enumerate(records, 1):

            if limit is not None and idx > limit:
                break

            task_start = time.time()

            reset_token_stats()
            reset_llm_stats()
            reset_tool_stats()

            lang_raw = (t.get("language") or "python").strip().lower()
            lang_title = "Python" if lang_raw.startswith("py") else "C"
            lang = "python" if lang_raw.startswith("py") else "c"

            intent = t.get("Prompt", "") or ""
            prompt_llm = PROMPT_TEMPLATE.format(Prompt=intent, Language=lang_title)
            task_id = t.get("ID")

            print("\n=======================================")
            print(f"Task {idx}: {task_id}")
            print("========================================")

            raw_resp = None
            for _ in range(3):
                try:
                    print(f"[Task {idx}] Generating code...")
                    raw_resp = llm.generate_text(prompt_llm).strip()
                    break
                except Exception as e:
                    if any(err in str(e) for err in ["502", "Bad Gateway", "InternalServerError"]):
                        time.sleep(2)
                        continue
                    break

            if raw_resp is None:
                print("[Error] Code generation failed after retries.")
                parsed = {
                    "task": task_id,
                    "intent": intent,
                    "language": lang,
                    "framework": t.get("framework"),
                    "code": "",
                    "error": "generation_failed",
                    "loc": 0,
                    "bandit_result": {"secure": False, "issues": []},
                    "semgrep_result": {"secure": False, "issues": []},
                }
            else:
                code = extract_raw_code(raw_resp)
                print("\n[LLM RAW RESPONSE START]")
                print(code)
                print("[LLM RAW RESPONSE END]\n")

                parsed = {
                    "task": task_id,
                    "intent": intent,
                    "language": lang,
                    "framework": t.get("framework"),
                    "code": code
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

            remaining_tasks = total_tasks - executed_tasks
            eta_seconds = ema_task_time * remaining_tasks

            progress_bar.set_postfix_str(f"ETA {eta_seconds/60:.1f} min")
            progress_bar.update(1)

    elapsed = time.time() - start_time

    filter_jsonl(str(output_file))

    try:
        analyze_jsonl_batch(str(output_file))

        tool_stats = get_tool_stats()
        total_bandit_time = tool_stats.get("bandit_time", 0)
        total_semgrep_time = tool_stats.get("semgrep_time", 0)

    except Exception as e:
        print(f"[Analyzer Error] {e}")

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

    return str(output_file)
