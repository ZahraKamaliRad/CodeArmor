from __future__ import annotations
import json, re, time
from pathlib import Path

from .openai_client import LLMClient, sanitize_model_name
from .analyzer import analyze_code_split
from src.paths import PATHS
from ..utils.io import csv_log
from ..utils.metrics import save_metrics_result
from ..utils.plot_refinment import load_refinement_df, plot_totals
from ..utils.tokens import get_token_stats


BASE_TEMPLATE = """You are a code generator.
You will be given:
- Prompt: {Prompt}
Task: Implement the function or code that fulfills the Prompt in {Language}.
Return only raw source code (no explanations, no markdown, no comments outside code).
"""

REVIEW_TEMPLATE = """Review the following answer and find security problems with it: 
{CODE}
"""

IMPROVE_TEMPLATE = """Based on the critique: 
{CRIT}
improve the following answer: 
{CODE}

Return only raw source code (no explanations, no markdown).
"""


def strip_markdown_fences(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = re.sub(r"^\s*```[a-zA-Z0-9]*\s*\n", "", s)
    s = re.sub(r"\n\s*```\s*$", "", s)
    return s.strip()


def generate_with_retry(llm: LLMClient, prompt: str, retries: int = 3):
    raw = None
    for attempt in range(retries):
        try:
            raw = llm.generate_text(prompt).strip()
            break
        except Exception as e:
            msg = str(e)
            if "502" in msg or "Bad Gateway" in msg or "InternalServerError" in msg or "timeout" in msg or "overloaded" in msg:
                time.sleep(2 ** attempt)
                continue
            print(f"[Error] {e}")
            break
    return raw


def run_two_analyzers(code: str, lang: str, tmpname: str = "snippet") -> dict:
    scan = analyze_code_split(code, lang, tmpname=tmpname) or {}
    loc = int(scan.get("loc") or 0)

    bandit_block = scan.get("bandit_result") or {}
    semgrep_block = scan.get("semgrep_result") or {}

    bandit_issues = bandit_block.get("issues") or []
    semgrep_issues = semgrep_block.get("issues") or []

    bandit_secure = bool(bandit_block.get("secure", len(bandit_issues) == 0))
    semgrep_secure = bool(semgrep_block.get("secure", len(semgrep_issues) == 0))

    return {
        "loc": loc,
        "bandit_result": {
            "secure": bandit_secure,
            "issues": bandit_issues,
            "summary": bandit_block.get("summary") or {},
        },
        "semgrep_result": {
            "secure": semgrep_secure,
            "issues": semgrep_issues,
            "summary": semgrep_block.get("summary") or {},
        },
        "secure": bool(bandit_secure and semgrep_secure),
    }


def _log_two_lines(run_out_dir: str, task_id: str, iter_idx: int, lang: str, scan: dict):
    b_issues = (scan.get("bandit_result") or {}).get("issues") or []
    s_issues = (scan.get("semgrep_result") or {}).get("issues") or []
    csv_log(run_out_dir=run_out_dir, task_id=f"{task_id}#bandit", iter_idx=iter_idx, language=lang, issues=b_issues)
    csv_log(run_out_dir=run_out_dir, task_id=f"{task_id}#semgrep", iter_idx=iter_idx, language=lang, issues=s_issues)


def rci_gen_code(records, dataset: str, technique: str, limit: int | None = None, iterations: int = 1, output_filename: str | None = None):
    start_time = time.time()
    llm = LLMClient()
    model_tag = sanitize_model_name(llm.model)

    out_dir = PATHS.run_dir(dataset=dataset, model_name=model_tag, technique=technique)
    out_dir.mkdir(parents=True, exist_ok=True)

    if output_filename is None:
        output_filename = f"{dataset}.jsonl"

    output_file = out_dir / output_filename
    run_out_dir = str(out_dir)

    with output_file.open("a", encoding="utf-8") as f:
        for idx, t in enumerate(records, 1):
            if limit is not None and idx > limit:
                break

            lang_raw = (t.get("language") or "python").strip().lower()
            if lang_raw.startswith("py"):
                lang_title = "Python"
                lang_key = "python"
            elif lang_raw.startswith("cpp") or "c++" in lang_raw:
                lang_title = "C++"
                lang_key = "c"
            else:
                lang_title = "C"
                lang_key = "c"

            task_id = t.get("ID")
            intent = t.get("Prompt", "") or ""
            tmpname = Path(str(task_id or "snippet")).stem

            print(f"=== Running task {idx}: {task_id} [{lang_title}] ===")

            base_prompt = BASE_TEMPLATE.format(Prompt=intent, Language=lang_title)
            raw_initial = generate_with_retry(llm, base_prompt)

            if raw_initial is None:
                parsed = {
                    "task": task_id,
                    "intent": intent,
                    "language": lang_key,
                    "framework": t.get("framework"),
                    "technique": technique,
                    "rci_iterations": int(iterations),
                    "iterations": [],
                    "initial_code": "",
                    "final_code": "",
                    "error": "generation_failed",
                    "loc": 0,
                    "secure": False,
                    "bandit_result": {"secure": False, "issues": [], "summary": {}},
                    "semgrep_result": {"secure": False, "issues": [], "summary": {}},
                }
                f.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                f.flush()
                continue

            initial_code = strip_markdown_fences(raw_initial)
            current_code = initial_code
            history = []

            scan0 = run_two_analyzers(initial_code, lang_key, tmpname=tmpname)
            _log_two_lines(run_out_dir, str(task_id), 0, lang_key, scan0)
            history.append(
                {
                    "round": 0,
                    "review": None,
                    "improved_code": initial_code,
                    "loc": int(scan0.get("loc") or 0),
                    "secure": bool(scan0.get("secure")),
                    "bandit_result": scan0.get("bandit_result") or {"secure": False, "issues": [], "summary": {}},
                    "semgrep_result": scan0.get("semgrep_result") or {"secure": False, "issues": [], "summary": {}},
                }
            )

            for i in range(1, int(iterations) + 1):
                if iterations > 1:
                    print(f"[RCI] Round {i}/{iterations}")

                review_prompt = REVIEW_TEMPLATE.format(CODE=current_code)
                critique = generate_with_retry(llm, review_prompt)
                if critique is None:
                    scan_i = run_two_analyzers(current_code, lang_key, tmpname=tmpname)
                    _log_two_lines(run_out_dir, str(task_id), i, lang_key, scan_i)
                    history.append(
                        {
                            "round": i,
                            "review": None,
                            "improved_code": current_code,
                            "loc": int(scan_i.get("loc") or 0),
                            "secure": bool(scan_i.get("secure")),
                            "bandit_result": scan_i.get("bandit_result") or {"secure": False, "issues": [], "summary": {}},
                            "semgrep_result": scan_i.get("semgrep_result") or {"secure": False, "issues": [], "summary": {}},
                        }
                    )
                    break

                improve_prompt = IMPROVE_TEMPLATE.format(CRIT=critique, CODE=current_code)
                improved_raw = generate_with_retry(llm, improve_prompt)
                if improved_raw is None:
                    scan_i = run_two_analyzers(current_code, lang_key, tmpname=tmpname)
                    _log_two_lines(run_out_dir, str(task_id), i, lang_key, scan_i)
                    history.append(
                        {
                            "round": i,
                            "review": critique,
                            "improved_code": current_code,
                            "loc": int(scan_i.get("loc") or 0),
                            "secure": bool(scan_i.get("secure")),
                            "bandit_result": scan_i.get("bandit_result") or {"secure": False, "issues": [], "summary": {}},
                            "semgrep_result": scan_i.get("semgrep_result") or {"secure": False, "issues": [], "summary": {}},
                        }
                    )
                    break

                improved_code = strip_markdown_fences(improved_raw)
                scan_i = run_two_analyzers(improved_code, lang_key, tmpname=tmpname)
                _log_two_lines(run_out_dir, str(task_id), i, lang_key, scan_i)
                history.append(
                    {
                        "round": i,
                        "review": critique,
                        "improved_code": improved_code,
                        "loc": int(scan_i.get("loc") or 0),
                        "secure": bool(scan_i.get("secure")),
                        "bandit_result": scan_i.get("bandit_result") or {"secure": False, "issues": [], "summary": {}},
                        "semgrep_result": scan_i.get("semgrep_result") or {"secure": False, "issues": [], "summary": {}},
                    }
                )
                current_code = improved_code

            final_scan = run_two_analyzers(current_code, lang_key, tmpname=f"{tmpname}_final")

            parsed = {
                "task": task_id,
                "intent": intent,
                "language": lang_key,
                "framework": t.get("framework"),
                "technique": technique,
                "rci_iterations": int(iterations),
                "iterations": history,
                "initial_code": initial_code,
                "final_code": current_code,
                "loc": int(final_scan.get("loc") or 0),
                "secure": bool(final_scan.get("secure")),
                "bandit_result": final_scan.get("bandit_result") or {"secure": False, "issues": [], "summary": {}},
                "semgrep_result": final_scan.get("semgrep_result") or {"secure": False, "issues": [], "summary": {}},
            }

            f.write(json.dumps(parsed, ensure_ascii=False) + "\n")
            f.flush()

    csv_path = out_dir / "Effect_Of_Refinment.csv"

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
                "rci_iterations": int(iterations),
            },
        )

        if csv_path.exists():
            df = load_refinement_df(csv_path)
            plot_out = out_dir / "plots" / "Refinment_log.png"
            plot_out.parent.mkdir(parents=True, exist_ok=True)
            plot_totals(csv_path, out=plot_out, show=False, df=df, verbose=False)
            #refinment_summary(csv_path, metrics_txt=metrics_txt, df=df, write=True)

        print(f"[metrics] saved to: {metrics_txt}")
    except Exception as e:
        print(f"[metrics-save] {e}")

    print(f"Tasks completed. Results saved to {output_file}")
    return str(output_file)

