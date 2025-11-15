from __future__ import annotations
import json, re, time
from pathlib import Path

from .openai_client import LLMClient,sanitize_model_name
from .analyzer import analyze_code
from ..config import PATHS
from ..utils.io import csv_log
from ..utils.metrics import save_metrics_result
from ..utils.plot_refinment import load_refinement_df, plot_totals, refinment_summary
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
            if "502" in msg or "Bad Gateway" in msg or "InternalServerError" in msg:
                time.sleep(2 ** attempt)
                continue
            else:
                print(f"[Error] {e}")
                break
    return raw


def rci_tecniqu(records,dataset_name: str = "output_rci",limit: int | None = None,
    iterations: int = 1,output_filename: str | None = None,):
    start_time = time.time()
    llm = LLMClient(api_key="sk-Gr8Sna1pUqdHJ11APRNLDBtcugQqujqBWbAEeGOisXxIMBY5")
    model_tag = sanitize_model_name(llm.model)
    run_name = f"{dataset_name}_{model_tag}"
    out_dir = PATHS.dataset_run_dir(run_name)

    if output_filename is None:
        output_filename = f"{out_dir.name}.jsonl"
    output_file = out_dir / output_filename
    run_out_dir = str(out_dir)

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

            task_id = t.get("ID")
            dataset_prompt = t.get("Prompt", "")

            print(f"=== Running task {idx}: {task_id} [{lang_title}] ===")

            base_prompt = BASE_TEMPLATE.format(Prompt=dataset_prompt, Language=lang_title)
            raw_initial = generate_with_retry(llm, base_prompt)

            if raw_initial is None:
                parsed = {
                    "task": task_id,
                    "language": lang,
                    "framework": t.get("framework"),
                    "technique": f"rci_iter{iterations}",
                    "iterations": [],
                    "initial_code": "",
                    "final_code": "",
                    "error": "generation_failed",
                    "issues": [],
                    "loc": 0,
                    "secure": False,
                }
                f.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                f.flush()
                continue

            initial_code = strip_markdown_fences(raw_initial)
            current_code = initial_code
            history = []

            issues0, loc0 = analyze_code(initial_code, lang, tmpname=task_id)
            csv_log(run_out_dir, task_id, 0, lang, issues0)
            history.append(
                {
                    "round": 0,
                    "review": None,
                    "improved_code": initial_code,
                    "issues": issues0,
                    "loc": loc0,
                }
            )

            for i in range(1, max(1, iterations) + 1):
                if iterations > 1:
                    print(f"[RCI] Round {i}/{iterations}")

                review_prompt = REVIEW_TEMPLATE.format(CODE=current_code)
                critique = generate_with_retry(llm, review_prompt)
                if critique is None:
                    issues_i, loc_i = analyze_code(current_code, lang, tmpname=task_id)
                    csv_log(run_out_dir, task_id, i, lang, issues_i)
                    history.append(
                        {
                            "round": i,
                            "review": None,
                            "improved_code": current_code,
                            "issues": issues_i,
                            "loc": loc_i,
                        }
                    )
                    break

                improve_prompt = IMPROVE_TEMPLATE.format(CRIT=critique, CODE=current_code)
                improved_raw = generate_with_retry(llm, improve_prompt)
                if improved_raw is None:
                    issues_i, loc_i = analyze_code(current_code, lang, tmpname=task_id)
                    csv_log(run_out_dir, task_id, i, lang, issues_i)
                    history.append(
                        {
                            "round": i,
                            "review": critique,
                            "improved_code": current_code,
                            "issues": issues_i,
                            "loc": loc_i,
                        }
                    )
                    break

                improved_code = strip_markdown_fences(improved_raw)
                issues_i, loc_i = analyze_code(improved_code, lang, tmpname=task_id)
                csv_log(run_out_dir, task_id, i, lang, issues_i)
                history.append(
                    {
                        "round": i,
                        "review": critique,
                        "improved_code": improved_code,
                        "issues": issues_i,
                        "loc": loc_i,
                    }
                )
                current_code = improved_code

            issues, loc = analyze_code(current_code, lang, tmpname=task_id)

            parsed = {
                "task": task_id,
                "language": lang,
                "framework": t.get("framework"),
                "technique": f"rci_iter{iterations}",
                "iterations": history,
                "initial_code": initial_code,
                "final_code": current_code,
                "issues": issues,
                "loc": loc,
                "secure": len(issues) == 0,
            }

            f.write(json.dumps(parsed, ensure_ascii=False) + "\n")
            f.flush()

        try:
            csv_path = out_dir / "Effect_Of_Refinment.csv"
            metrics_dir = out_dir / "rate" / "vuln_density"
            metrics_txt = metrics_dir / f"{Path(output_file).stem}_metrics.txt"

            save_metrics_result(str(output_file), str(metrics_txt))

            if csv_path.exists():
                df = load_refinement_df(csv_path)
                plot_out = out_dir / "plots" / "Refinment_log.png"
                plot_totals(csv_path, out=plot_out, show=False, df=df, verbose=False)
                refinment_summary(csv_path, metrics_txt=metrics_txt, df=df, write=True)

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
