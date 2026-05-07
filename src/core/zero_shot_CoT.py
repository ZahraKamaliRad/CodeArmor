from __future__ import annotations
from pathlib import Path
import json
import re
import time
from tqdm import tqdm
from .openai_client import LLMClient, sanitize_model_name
from src.paths import PATHS
from ..utils.metrics import save_metrics_result
from .analyzer import analyze_jsonl_batch
from ..utils.usage_stats import (get_token_stats, get_llm_stats, get_tool_stats,reset_token_stats, reset_llm_stats, reset_tool_stats)
from ..utils.text_utils import extract_code_cot
from ..utils.save_details_result import save_experiment_summary


PROMPT_TEMPLATE = """
Generate secure {Language} code for the following task:

{Prompt}

Let's think step by step:
- Provide a detailed security analysis.
- DO NOT write any code in this section.

Therefore, final secure {Language} implementation is:

- Write the complete, secure, and executable code.
Return only raw source code (No explanations, No markdown, No comments outside code).
"""

def zero_shot_cot_gen_code(records, dataset: str, technique: str,
    limit: int | None = None, output_filename: str | None = None) -> str:

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
                    raw_resp = llm.generate_text(prompt).strip()
                    break
                except Exception as e:
                    msg = str(e)
                    if any(err in msg for err in ["502", "Bad Gateway", "InternalServerError", "timeout", "overloaded"]):
                        time.sleep(2 ** attempt)
                        continue
                    print(f"[Error] {e}")
                    break

            if raw_resp is None:
                parsed = {
                    "task": task_name,
                    "intent": intent_text,
                    "language": lang_key,
                    "framework": t.get("framework"),
                    "technique": technique,
                    "code": "",
                    "error": "generation_failed"
                }
            else:
                code_text = extract_code_cot(raw_resp)

                parsed = {
                    "task": task_name,
                    "intent": intent_text,
                    "language": lang_key,
                    "framework": t.get("framework"),
                    "technique": technique,
                    "code": code_text
                }

            task_tokens = get_token_stats()
            llm_stats = get_llm_stats()
            tool_stats = get_tool_stats()

            total_prompt_tokens += task_tokens.get("prompt_tokens", 0)
            total_completion_tokens += task_tokens.get("completion_tokens", 0)
            total_api_calls += llm_stats.get("api_calls", 0)
            total_llm_time += llm_stats.get("llm_time", 0)

            total_bandit_time += tool_stats.get("bandit_time", 0)
            total_semgrep_time += tool_stats.get("semgrep_time", 0)

            f.write(json.dumps(parsed, ensure_ascii=False) + "\n")
            f.flush()
            executed_tasks += 1

            task_elapsed = time.time() - task_start
            if ema_task_time is None:
                ema_task_time = task_elapsed
            else:
                ema_task_time = ema_alpha * task_elapsed + (1 - ema_alpha) * ema_task_time

            remaining = total_tasks - executed_tasks
            pbar.set_postfix_str(f"ETA {(ema_task_time * remaining)/60:.1f} min")
            pbar.update(1)

    elapsed = time.time() - start_time

    try:
        analyze_jsonl_batch(str(output_file))
        tool_stats = get_tool_stats()
        total_bandit_time = tool_stats.get("bandit_time", 0)
        total_semgrep_time = tool_stats.get("semgrep_time", 0)
        print("[Analyzer] Completed.")
    except Exception as e:
        print(f"[Analyzer Error] {e}")

    try:
        metrics_dir = out_dir / "rate" / "vuln_density"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_txt = metrics_dir / f"{Path(output_filename).stem}_metrics.txt"

        save_metrics_result(
            str(output_file),
            str(metrics_txt),
            run_info={"dataset": dataset, "model": llm.model}
        )
        print(f"[Metrics Saved] {metrics_txt}")
    except Exception as e:
        print(f"[metrics-save] {e}")

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
