from __future__ import annotations
from pathlib import Path
import json, re, time

from .openai_client import LLMClient, sanitize_model_name
from src.paths import PATHS
from ..utils.metrics import save_metrics_result
from .analyzer import analyze_code
from ..utils.tokens import get_token_stats


PROMPT_TEMPLATE = """You are a code generator.
You will be given:
- Prompt: {Prompt}
Task: Implement the function or code that fulfills the Prompt in {Language}.
Return only raw source code (no explanations, no markdown, no comments outside code).
"""


def strip_markdown_fences(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = re.sub(r"^\s*```[a-zA-Z0-9]*\s*\n", "", s)
    s = re.sub(r"\n\s*```\s*$", "", s)
    return s.strip()


def GenerateCode(records,dataset: str, technique: str, limit: int | None = None,output_filename: str | None = None):
    start_time = time.time()
    llm = LLMClient()
    model_tag = sanitize_model_name(llm.model)

    out_dir = PATHS.run_dir(dataset=dataset, model_name=model_tag, technique=technique)

    if output_filename is None:
        output_filename = f"{dataset}.jsonl"

    output_file = out_dir / output_filename

    with output_file.open("a", encoding="utf-8") as f:
        for idx, t in enumerate(records, 1):
            if limit is not None and idx > limit:
                break

            lang = (t.get("language") or "python").strip().lower()
            if lang.startswith("py"):
                lang_title = "Python"
                lang = "python"
            elif lang.startswith("cpp") or "c++" in lang:
                lang_title = "C++"
                lang = "c"
            else:
                lang_title = "C"
                lang = "c"


            prompt = PROMPT_TEMPLATE.format(Prompt=t.get("Prompt", ""), Language=lang_title)
            print(f"=== Running task {idx}: {t.get('ID')} [{lang_title}] ===")

            raw_resp = None
            for attempt in range(3):
                try:
                    raw_resp = llm.generate_text(prompt).strip()
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
                    "task": t.get("ID"),
                    "language": lang,
                    "framework": t.get("framework"),
                    "code": "",
                    "error": "generation_failed",
                    "issues": [],
                    "loc": 0,
                    "secure": False,
                }
            else:
                raw_resp = strip_markdown_fences(raw_resp)
                tmpname = Path(t.get("ID") or "snippet").stem
                issues, loc, summary = analyze_code(raw_resp, lang, tmpname=tmpname)

                parsed = {
                    "task": t.get("ID"),
                    "language": lang,
                    "framework": t.get("framework"),
                    "code": raw_resp,
                    "issues": issues,
                    "loc": loc,
                    "secure": summary["secure"],
                    "unique_vulnerabilities": summary["unique_vulnerabilities"],
                    "tool_issue_counts": summary["tool_issue_counts"],
                    "tools": summary["tools"]
                }

                #parsed["analysis_result"] = {"secure": parsed["secure"], "issues": parsed["issues"]}



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
                "runtime_seconds": elapsed,
            },
        )

        print(f"[metrics] saved to: {metrics_txt}")
    except Exception as e:
        print(f"[metrics-save] {e}")

    print(f"Tasks completed. Results saved to {output_file}")
    return str(output_file)
