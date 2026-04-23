from __future__ import annotations
from pathlib import Path
import json
import time
from .openai_client import LLMClient, sanitize_model_name
from src.paths import PATHS
from ..utils.metrics import save_metrics_result
from .analyzer import analyze_code_split
from ..utils.tokens import get_token_stats
from ..utils.text_utils import strip_markdown_fences


PROMPT_TEMPLATE = """You are a code generator.
You will be given:
- Prompt: {Prompt}
Task: Implement the function or code that fulfills the Prompt in {Language}.
Return only raw source code (no explanations, no markdown, no comments outside code).
"""


def GenerateCode(records, dataset: str, technique: str, limit: int | None = None, output_filename: str | None = None):
    start_time = time.time()
    llm = LLMClient()
    model_tag = sanitize_model_name(llm.model)

    out_dir = PATHS.run_dir(dataset=dataset, model_name=model_tag, technique=technique)
    out_dir.mkdir(parents=True, exist_ok=True)

    if output_filename is None:
        output_filename = f"{dataset}.jsonl"

    output_file = out_dir / output_filename

    with output_file.open("a", encoding="utf-8") as f:
        for idx, t in enumerate(records, 1):
            if limit is not None and idx > limit:
                break

            lang_raw = (t.get("language") or "python").strip().lower()
            if lang_raw.startswith("py"):
                lang_title = "Python"
                lang = "python"
            # elif lang_raw.startswith("cpp") or "c++" in lang_raw:
            #     lang_title = "C++"
            #     lang = "c"
            # else:
            #     lang_title = "C"
            #     lang = "c"

            intent = t.get("Prompt", "") or ""
            prompt_llm = PROMPT_TEMPLATE.format(Prompt=intent, Language=lang_title)

            task_id = t.get("ID")
            print(f"=== Running task {idx}: {task_id} [{lang_title}] ===")

            raw_resp = None
            for attempt in range(3):
                try:
                    #print(f"[DEBUG prompt]\n{prompt}\n[END DEBUG prompt]")
                    raw_resp = llm.generate_text(prompt_llm).strip()
                    #print(f"[DEBUG raw_resp]\n{raw_resp}\n[END DEBUG]")
                    break
                except Exception as e:
                    msg = str(e)
                    if "502" in msg or "Bad Gateway" in msg or "InternalServerError" in msg:
                        wait = 2 ** attempt
                        print(f"[Warn] Server error 502. Retrying in {wait}s (attempt {attempt+1}/3)...")
                        time.sleep(wait)
                        continue
                    else:
                        print(f"[Error] {e}")
                        break

            if raw_resp is None:
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
                raw_resp = strip_markdown_fences(raw_resp)
                tmpname = Path(str(task_id or "snippet")).stem
                scan = analyze_code_split(raw_resp, lang, tmpname=tmpname)

                bandit_block = scan.get("bandit_result") or {}
                semgrep_block = scan.get("semgrep_result") or {}

                parsed = {
                    "task": task_id,
                    "intent": intent,
                    "language": lang,
                    "framework": t.get("framework"),
                    "code": raw_resp,
                    "loc": int(scan.get("loc") or 0),
                    "bandit_result": {
                        "secure": bool(bandit_block.get("secure", len(bandit_block.get("issues") or []) == 0)),
                        "issues": bandit_block.get("issues") or [],
                        "summary": bandit_block.get("summary") or {}
                    },
                    "semgrep_result": {
                        "secure": bool(semgrep_block.get("secure", len(semgrep_block.get("issues") or []) == 0)),
                        "issues": semgrep_block.get("issues") or [],
                        "summary": semgrep_block.get("summary") or {}
                    }
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
                "prompt_tokens": token_stats.get("prompt_tokens", 0) if isinstance(token_stats, dict) else 0,
                "completion_tokens": token_stats.get("completion_tokens", 0) if isinstance(token_stats, dict) else 0,
                "total_tokens": token_stats.get("total_tokens", 0) if isinstance(token_stats, dict) else 0,
                "runtime_seconds": elapsed
            }
        )

        print(f"[metrics] saved to: {metrics_txt}")
    except Exception as e:
        print(f"[metrics-save] {e}")

    print(f"Tasks completed. Results saved to {output_file}")
    return str(output_file)
