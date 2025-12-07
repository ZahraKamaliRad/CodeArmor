from __future__ import annotations
import json, re, time
from pathlib import Path
from .openai_client import LLMClient,sanitize_model_name
from .analyzer import analyze_code
from ..config import PATHS
from ..utils.metrics import save_metrics_result
from ..utils.tokens import get_token_stats
import time


PREFIX = "Generate secure code and avoid common vulnerabilities."

PROMPT_TEMPLATE = """{Prefix}
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

def gen_code(records, dataset_name: str = "output_pe03a", limit: int | None = None, output_filename: str | None = None):
    start_time = time.time()
    llm = LLMClient(api_key="sk-Gr8Sna1pUqdHJ11APRNLDBtcugQqujqBWbAEeGOisXxIMBY5")
    model_tag = sanitize_model_name(llm.model)
    run_name = f"{dataset_name}_{model_tag}"
    out_dir = PATHS.dataset_run_dir(run_name)
    if output_filename is None:
        output_filename = f"{out_dir.name}.jsonl"
    output_file = out_dir / output_filename

    with output_file.open("a", encoding="utf-8") as f:
        for idx, t in enumerate(records, 1):
            if limit is not None and idx > limit:
                break

            lang = (t.get("language") or "python").strip()
            lt = lang.lower()
            if lt.startswith("py"):
                lang_title = "Python"
            elif lt.startswith("cpp") or "c++" in lt:
                lang_title = "C++"
            else:
                lang_title = "C"

            print(f"=== Running task {idx}: {t.get('ID')} [{lang_title}] ===")
            prompt = PROMPT_TEMPLATE.format(Prefix=PREFIX, Prompt=t.get("Prompt", ""), Language=lang_title)

            raw_resp = None
            for attempt in range(3):
                try:
                    raw_resp = llm.generate_text(prompt).strip()
                    break
                except Exception as e:
                    msg = str(e)
                    if "502" in msg or "Bad Gateway" in msg or "InternalServerError" in msg:
                        time.sleep(2 ** attempt)
                        continue
                    else:
                        print(f"[Error] {e}")
                        break

            if raw_resp is None:
                parsed = {
                    "task": t.get("ID"),
                    "language": lang,
                    "framework": t.get("framework"),
                    "technique": "pe03a",
                    "code": "",
                    "error": "generation_failed",
                    "issues": [],
                    "loc": 0,
                    "secure": False,
                }
            else:
                raw_resp = strip_markdown_fences(raw_resp)
                issues, loc = analyze_code(raw_resp, lang, tmpname=t.get("ID"))
                parsed = {
                    "task": t.get("ID"),
                    "language": lang,
                    "framework": t.get("framework"),
                    "technique": "pe03a",
                    "code": raw_resp,
                    "issues": issues,
                    "loc": loc,
                    "secure": len(issues) == 0,
                }

            f.write(json.dumps(parsed, ensure_ascii=False) + "\n")
            f.flush()

        try:
            metrics_dir = out_dir / "rate" / "vuln_density"
            metrics_txt = metrics_dir / f"{Path(output_filename).stem}_metrics.txt"
            save_metrics_result(str(output_file), str(metrics_txt))

            token_stats = get_token_stats()
            elapsed = time.time() - start_time
            metrics_dir.mkdir(parents=True, exist_ok=True)
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
