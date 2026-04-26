from __future__ import annotations
import json
import re
import time
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Tuple
from .openai_client import LLMClient, sanitize_model_name
from src.paths import PATHS
from ..utils.metrics import save_metrics_result
from .analyzer import analyze_code_split
from ..utils.tokens import get_token_stats
from ..utils.io import csv_log
from ..utils.plot_refinment import load_refinement_df, plot_totals
from ..utils.text_utils import strip_markdown_fences


PLANNING_PROMPT = """
You are a secure coding assistant.

Task:
Analyze the security risks of the given code intent and produce a mitigation plan.

Language: {Language}

Intent:
{Prompt}

Output format:

Security analysis:
- Describe how the code could be attacked when handling untrusted input.
- Identify concrete vulnerabilities and risks.

Plan:
- Provide a step-by-step mitigation plan.
- Focus on secure design decisions.
- Do NOT include any code.
"""

CODING_PROMPT = """
You are a secure-coding assistant.

Language: {Language}

Intent:
{Prompt}

Security analysis and plan:
{Plan}

Write secure {Language} code that implements the intent while strictly following the plan.

Return ONLY the source code. Do not include explanations or markdown.

"""
REPAIR_PROMPT_TEMPLATE = Template(
"""
You are a secure-coding assistant.

TARGET_ANALYZER: $analyzer
LANGUAGE: $language

ANALYZER_ISSUES (JSON):
$issues_json

CURRENT_CODE:

$code

Task:

Fix all reported security issues.
Preserve existing functionality.

Return ONLY the full updated code (no explanations).
"""
)

def extract_code(resp: str) -> str:
    if not isinstance(resp, str):
        return ""

    pattern = r"```(?:python)?\s*(.*?)```"
    matches = list(re.finditer(pattern, resp, re.DOTALL | re.IGNORECASE))
    if matches:
        return matches[-1].group(1).strip()
    return resp.strip()


def extract_plan(plan_resp: str) -> str:
    if not isinstance(plan_resp, str):
        return ""
    for marker in ["Security analysis:", "Plan to solve these issues:"]:
        idx = plan_resp.find(marker)
        plan = plan_resp[idx:].strip() if idx != -1 else plan_resp.strip()
        plan = re.sub(r'```[\s\S]*?```', '', plan).strip()
        plan = re.sub(
            r'\n*(Here is|Below is|The following is|This implementation)[^\n]*\n?',
            '',plan,flags=re.IGNORECASE).strip()
        return plan


def run_two_analyzers(code: str, lang: str, tmpname: str = "snippet") -> Dict[str, Any]:
    scan = analyze_code_split(code, lang, tmpname=tmpname) or {}

    result: Dict[str, Any] = {
        "loc": int(scan.get("loc") or 0)
    }

    bandit_secure = True
    semgrep_secure = True

    if "bandit_result" in scan:
        bandit_block = scan["bandit_result"]
        bandit_issues = bandit_block.get("issues") or []
        bandit_secure = bool(bandit_block.get("secure", len(bandit_issues) == 0))

        result["bandit_result"] = {
            "secure": bandit_secure,
            "issues": bandit_issues,
            "summary": bandit_block.get("summary") or {}
        }

    if "semgrep_result" in scan:
        semgrep_block = scan["semgrep_result"]
        semgrep_issues = semgrep_block.get("issues") or []
        semgrep_secure = bool(semgrep_block.get("secure", len(semgrep_issues) == 0))

        result["semgrep_result"] = {
            "secure": semgrep_secure,
            "issues": semgrep_issues,
            "summary": semgrep_block.get("summary") or {}
        }

    result["secure_all"] = bool(bandit_secure and semgrep_secure)

    return result

def attempt_repair_loop(llm: Any, code: str, lang: str, task_name: str,
                        iterations: int, run_out_dir: str) -> Tuple[str, Dict[str, Any]]:
    current_code = code

    for r in range(max(0, int(iterations))):
        print(f"[repair] >>> Entering iteration {r+1}/{iterations}")

        pre = run_two_analyzers(current_code, lang, tmpname=f"{task_name}_iter{r}_pre")
        b_issues = (pre.get("bandit_result") or {}).get("issues") or []
        s_issues = (pre.get("semgrep_result") or {}).get("issues") or []

        #print(f"[repair] bandit={len(b_issues)} issues, semgrep={len(s_issues)} issues")

        if pre.get("secure_all", (len(b_issues) == 0 and len(s_issues) == 0)):
            print("[repair] No issues found — exiting repair loop.")
            return current_code, pre

        if len(b_issues) >= len(s_issues):
            target = "bandit"
            chosen_issues = b_issues
        else:
            target = "semgrep"
            chosen_issues = s_issues

        print(f"[repair] Using analyzer: {target}")

        if not chosen_issues:
            print("[repair] No issues selected, exiting repair loop.")
            return current_code, pre

        prompt = REPAIR_PROMPT_TEMPLATE.substitute(analyzer=target,language=lang,
            issues_json=json.dumps(chosen_issues, ensure_ascii=False),code=current_code)

        try:
            resp = llm.generate_text(prompt).strip()
        except Exception:
            csv_log(run_out_dir=run_out_dir, task_id=f"{task_name}#bandit",
                    iter_idx=r + 1, language=lang, issues=b_issues)
            csv_log(run_out_dir=run_out_dir, task_id=f"{task_name}#semgrep",
                    iter_idx=r + 1, language=lang, issues=s_issues)
            return current_code, pre

        fixed = strip_markdown_fences(extract_code(resp))
        if fixed and fixed != current_code:
            current_code = fixed

        post = run_two_analyzers(current_code, lang, tmpname=f"{task_name}_iter{r}_post")

        csv_log(
            run_out_dir=run_out_dir,
            task_id=f"{task_name}#bandit",
            iter_idx=r + 1,
            language=lang,
            issues=(post.get("bandit_result") or {}).get("issues") or []
        )

        csv_log(
            run_out_dir=run_out_dir,
            task_id=f"{task_name}#semgrep",
            iter_idx=r + 1,
            language=lang,
            issues=(post.get("semgrep_result") or {}).get("issues") or []
        )

        if post.get(
            "secure_all",
            (
                len((post.get("bandit_result") or {}).get("issues") or []) == 0
                and len((post.get("semgrep_result") or {}).get("issues") or []) == 0
            )
            ):
            return current_code, post


    final_scan = run_two_analyzers(current_code, lang, tmpname=f"{task_name}_final")
    return current_code, final_scan


def normalize_language(raw: str) -> Tuple[str, str]:
    raw = (raw or "python").strip().lower()
    if raw.startswith("py"):
        return "Python", "python"


TRANSIENT_KEYWORDS = ("502", "Bad Gateway", "InternalServerError", "timeout", "overloaded")


def llm_call_with_retry(llm: Any, prompt: str, stage: str) -> str | None:
    for attempt in range(3):
        try:
            return llm.generate_text(prompt).strip()
        except Exception as exc:
            msg = str(exc)
            if any(kw in msg for kw in TRANSIENT_KEYWORDS):
                wait = 2 ** attempt
                print(
                    f"[Warn] transient LLM error during {stage}. "
                    f"Retrying in {wait}s (attempt {attempt + 1}/3)..."
                )
                time.sleep(wait)
                continue
            print(f"[Error][{stage}] {exc}")
            break
    return None


def planning_gen_code(records, dataset: str, technique: str,
                      limit: int | None = None, iterations: int = 0,
                      output_filename: str | None = None) -> str:

    start_time = time.time()
    llm = LLMClient()
    model_tag = sanitize_model_name(llm.model)
    out_dir = PATHS.run_dir(dataset=dataset, model_name=model_tag, technique=technique)

    if output_filename is None:
        output_filename = f"{dataset}.jsonl"

    output_file = out_dir / output_filename

    if not isinstance(records, list):
        records = list(records)

    with output_file.open("a", encoding="utf-8") as f:
        for idx, t in enumerate(records, 1):
            if limit is not None and idx > limit:
                break

            lang_title, lang_key = normalize_language(t.get("language", ""))
            task_name = (
                t.get("ID")
                or t.get("Prompt ID")
                or t.get("Filename")
                or t.get("file")
                or f"task_{idx}"
            )
            intent_text = t.get("Prompt", "") or ""

            print(f"=== Task {idx}: {task_name} [{lang_title}] ===")
            print("--- Planning stage ---")

            planning_prompt = PLANNING_PROMPT.format(
                Prompt=intent_text,
                Language=lang_title
            )
            plan_resp = llm_call_with_retry(llm, planning_prompt, stage="planning")

            if plan_resp is None:
                parsed = build_record(
                    task_name=task_name,
                    intent_text=intent_text,
                    lang_key=lang_key,
                    framework=t.get("framework"),
                    technique=technique,
                    plan="",
                    code="",
                    error="planning_failed"
                )
                write_record(f, parsed)
                continue

            plan_text = extract_plan(plan_resp)

            print("--- Coding stage ---")
            coding_prompt = CODING_PROMPT.format(
                Language=lang_title,
                Prompt=intent_text,
                Plan=plan_text
            )
            code_resp = llm_call_with_retry(llm, coding_prompt, stage="coding")

            if code_resp is None:
                parsed = build_record(
                    task_name=task_name,
                    intent_text=intent_text,
                    lang_key=lang_key,
                    framework=t.get("framework"),
                    technique=technique,
                    plan=plan_text,
                    code="",
                    error="generation_failed"
                )
                write_record(f, parsed)
                continue

            code_text = strip_markdown_fences(extract_code(code_resp))

            scan = run_two_analyzers(code_text, lang_key, tmpname=task_name)
            csv_log(
                run_out_dir=str(out_dir),
                task_id=f"{task_name}#bandit",
                iter_idx=0,
                language=lang_key,
                issues=(scan.get("bandit_result") or {}).get("issues") or [],
            )

            csv_log(
                run_out_dir=str(out_dir),
                task_id=f"{task_name}#semgrep",
                iter_idx=0,
                language=lang_key,
                issues=(scan.get("semgrep_result") or {}).get("issues") or [],
            )


            if int(iterations) > 0 and not scan.get("secure_all", True):
                code_text, scan = attempt_repair_loop(
                    llm=llm,
                    code=code_text,
                    lang=lang_key,
                    task_name=task_name,
                    iterations=iterations,
                    run_out_dir=str(out_dir)
                )

            parsed = build_record(
                task_name=task_name,
                intent_text=intent_text,
                lang_key=lang_key,
                framework=t.get("framework"),
                technique=technique,
                plan=plan_text,
                code=code_text,
                scan=scan
            )
            write_record(f, parsed)

    save_run_metrics(llm=llm,out_dir=out_dir,output_file=output_file,dataset=dataset,
        start_time=start_time)

    print(f"Tasks completed. Results saved to {output_file}")
    return str(output_file)


def build_record(task_name: str, intent_text: str, lang_key: str, framework,
                 technique: str, plan: str, code: str,
                 scan: Dict[str, Any] | None = None,
                 error: str | None = None) -> Dict[str, Any]:

    scan = scan or {}

    record: Dict[str, Any] = {
        "task": task_name,
        "intent": intent_text,
        "language": lang_key,
        "framework": framework,
        "technique": technique,
        "plan": plan,
        "code": code,
        "loc": int(scan.get("loc") or 0),
        "secure": bool(scan.get("secure_all", False)),
        "issues": {}
    }

    if "bandit_result" in scan:
        bandit_block = scan["bandit_result"]
        record["bandit_result"] = bandit_block
        record["issues"]["bandit"] = bandit_block.get("issues") or []

    if "semgrep_result" in scan:
        semgrep_block = scan["semgrep_result"]
        record["semgrep_result"] = semgrep_block
        record["issues"]["semgrep"] = semgrep_block.get("issues") or []

    if error:
        record["error"] = error

    return record


def write_record(f, record: Dict[str, Any]) -> None:
    f.write(json.dumps(record, ensure_ascii=False) + "\n")
    f.flush()


def save_run_metrics(llm: Any, out_dir: Path, output_file: Path,
                     dataset: str, start_time: float) -> None:
    try:
        metrics_dir = out_dir / "rate" / "vuln_density"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_txt = metrics_dir / f"{output_file.stem}_metrics.txt"

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
                "runtime_seconds": elapsed
            }
        )

        csv_path = out_dir / "Effect_Of_Refinment.csv"
        if csv_path.exists():
            df = load_refinement_df(csv_path)
            out_img = out_dir / "plots" / "Refinment_Plot.png"
            out_img.parent.mkdir(parents=True, exist_ok=True)
            plot_totals(csv_path, out=out_img, kind="line", show=False,
                        df=df, verbose=False)

        print(f"[metrics] saved to: {metrics_txt}")

    except Exception as exc:
        print(f"[metrics-save] {exc}")
