from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json, re, time
from typing import Any, Dict, List, Tuple
from string import Template

from .openai_client import LLMClient, sanitize_model_name
from .analyzer import analyze_code_split
from src.paths import PATHS
from ..utils.io import csv_log
from ..utils.plot_refinment import load_refinement_df, plot_totals, refinment_summary
from ..utils.metrics import save_metrics_result
from ..utils.tokens import get_token_stats


PLANNING_TEMPLATE = Template("""
You are a secure-coding planner.
Return ONLY a valid JSON object and nothing else (no markdown, no prose, no code).

Fields required:
cwe, summary, threats, policy, algorithm, secure_apis, error_handling, acceptance_criteria.

Rules:
- summary: short (<= 2 lines, no implementation details)
- threats: concise list (3–5 key risks)
- policy: object with limits/allowlists/paths/extensions as needed
- algorithm: list of short, numbered, sequential steps (1., 2., 3., ...) describing what to do conceptually, not how
- secure_apis: list of safe APIs or functions (e.g., yaml.safe_load, urllib.parse.urlparse, parameterized SQL)
- error_handling: list of "Exception -> behavior"
- acceptance_criteria: list of concrete checks that must hold true
- If language is not specified in input, assume "python".

Input Prompt:
$problem
""")


CODING_TEMPLATE = (
    """You are a secure code generator.
You will be given a Prompt containing PLAN(JSON) and TASK.
Strictly follow PLAN(JSON) when writing code:
- Implement the algorithm steps.
- Enforce the acceptance_criteria.
- Use ONLY the APIs listed in secure_apis.
- Implement the behaviors listed in error_handling.
Return only raw source code for the target language (no explanations, no markdown, no comments outside code).

Prompt:
{Prompt}
"""
)


REPAIR_PROMPT_TEMPLATE = Template(
    """
You are a SECURITY PATCH GENERATOR. Return ONLY the fixed source code (plain text). No explanations.

CONTEXT (read-only, do not echo back):
- SECURITY_GUIDANCE:
$guide
- TARGET_ANALYZER: $analyzer
- ANALYZER_ISSUES(JSON): $bandit_json

ORIGINAL CODE:
<code>
$code
</code>

GOAL:
- Fix the reported vulnerabilities.
- Keep changes minimal and preserve original behavior and public interface.
- Prefer secure APIs per SECURITY_GUIDANCE when applicable.
- Enforce acceptance_criteria hinted in SECURITY_GUIDANCE when relevant.
- Do not add heavy new dependencies; small stdlib imports are OK if necessary.

OUTPUT:
[full fixed source code only]
"""
)

REQUIRED_KEYS = [
    "cwe", "summary", "threats", "policy", "algorithm",
    "secure_apis", "error_handling", "acceptance_criteria"
]


def strip_markdown_fences(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = re.sub(r"^\s*```[a-zA-Z0-9]*\s*\n", "", s)
    s = re.sub(r"\n\s*```\s*$", "", s)
    return s.strip()


def extract_first_json(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"^\s*```[\w-]*\s*|\s*```\s*$", "", text.strip(), flags=re.DOTALL)
    stack = 0
    start = None
    for i, ch in enumerate(text):
        if ch == '{':
            if stack == 0:
                start = i
            stack += 1
        elif ch == '}':
            stack -= 1
            if stack == 0 and start is not None:
                return text[start:i+1]
    return text.strip()


def as_list(x) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def as_dict(x) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def parse_and_validate_plan(raw: str) -> Tuple[Dict[str, Any], List[str]]:
    warns: List[str] = []
    blob = extract_first_json(raw)
    try:
        obj = json.loads(blob)
    except Exception as e:
        return {}, [f"Invalid JSON from model: {e}"]

    for k in REQUIRED_KEYS:
        if k not in obj:
            warns.append(f"Missing key: {k}")
            if k in ("threats", "algorithm", "secure_apis", "error_handling", "acceptance_criteria"):
                obj[k] = []
            elif k == "policy":
                obj[k] = {}
            else:
                obj[k] = ""

    obj["threats"] = as_list(obj.get("threats"))
    obj["algorithm"] = as_list(obj.get("algorithm"))
    obj["secure_apis"] = as_list(obj.get("secure_apis"))
    obj["error_handling"] = as_list(obj.get("error_handling"))
    obj["acceptance_criteria"] = as_list(obj.get("acceptance_criteria"))
    obj["policy"] = as_dict(obj.get("policy"))

    cleaned = []
    for step in obj["algorithm"]:
        s = str(step)
        if re.search(r"\b(import|def|class|try:|except|#include|console\.log|print\()", s, flags=re.I):
            warns.append("Algorithm contained code-like content; removed a line.")
            continue
        cleaned.append(s.strip())
    obj["algorithm"] = cleaned

    obj["summary"] = str(obj.get("summary", ""))[:200].split('\n')[0].strip()
    return obj, warns


def build_planning_prompt(problem: str) -> str:
    return PLANNING_TEMPLATE.substitute(problem=problem)


def minimal_plan(plan_obj: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "cwe": plan_obj.get("cwe", ""),
        "summary": plan_obj.get("summary", ""),
        "secure_apis": plan_obj.get("secure_apis", []),
        "error_handling": plan_obj.get("error_handling", []),
        "acceptance_criteria": plan_obj.get("acceptance_criteria", []),
        "policy": {},
    }


def summarize_for_coder(plan_obj: Dict[str, Any]) -> str:
    lines = []
    sa = plan_obj.get("secure_apis") or []
    if sa:
        lines.append("secure_apis: " + ", ".join(map(str, sa)))
    acc = plan_obj.get("acceptance_criteria") or []
    if acc:
        lines.append("acceptance_criteria: " + " | ".join(map(str, acc[:6])))
    eh = plan_obj.get("error_handling") or []
    if eh:
        lines.append("error_handling: " + " | ".join(map(str, eh[:4])))
    return "SECURITY_GUIDANCE:\n" + "\n".join("- " + ln for ln in lines) if lines else "SECURITY_GUIDANCE:\n- (none)"


def strip_md(s: str) -> str:
    return strip_markdown_fences(s)



def _tag_issues(issues: List[Any], analyzer_name: str) -> List[Dict[str, Any]]:
    tagged: List[Dict[str, Any]] = []
    for it in issues or []:
        if isinstance(it, dict):
            d = dict(it)
            d["_analyzer"] = analyzer_name
            tagged.append(d)
        else:
            tagged.append({"_analyzer": analyzer_name, "raw": it})
    return tagged


def merge_issues(bandit_issues: List[Any], semgrep_issues: List[Any]) -> List[Dict[str, Any]]:
    return _tag_issues(bandit_issues, "bandit") + _tag_issues(semgrep_issues, "semgrep")


def run_two_analyzers(code: str, lang: str, tmpname: str = "snippet") -> Dict[str, Any]:
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

def attempt_repair_loop(llm: "LLMClient",code: str,issues: List[Dict[str, Any]],plan_obj: Dict[str, Any],lang: str,
    task_id: str,iterations: int = 0,history: list | None = None,run_out_dir: str = "") -> Tuple[str, List[Dict[str, Any]], bool, int]:
    current_code = code
    rounds = 0
    success = False

    if history is None:
        history = []
    guide = summarize_for_coder(minimal_plan(plan_obj))
    scan0 = run_two_analyzers(current_code, lang, tmpname=str(task_id))
    history.append(
        {
            "iter": 0,
            "chosen": None,
            "bandit_count": len(scan0["bandit_result"]["issues"]),
            "semgrep_count": len(scan0["semgrep_result"]["issues"]),
        }
    )

    if scan0["secure"]:
        print(f"[{task_id}] No issues found (both analyzers clean). Skipping repair loop.")
        return current_code, [], success, rounds

    print(
        f"[{task_id}] Starting repair loop. bandit={len(scan0['bandit_result']['issues'])}, "
        f"semgrep={len(scan0['semgrep_result']['issues'])}"
    )

    for r in range(1, max(0, int(iterations)) + 1):
        rounds = r

        pre = run_two_analyzers(current_code, lang, tmpname=f"{task_id}_iter{r}_pre")
        b_issues = pre["bandit_result"]["issues"] or []
        s_issues = pre["semgrep_result"]["issues"] or []

        if len(b_issues) == 0 and len(s_issues) == 0:
            print(f"[{task_id}] All issues fixed at iteration {r-1}.")
            success = True
            return current_code, [], success, rounds

        if len(b_issues) >= len(s_issues):
            chosen = "bandit"
            chosen_issues = b_issues
        else:
            chosen = "semgrep"
            chosen_issues = s_issues

        print(f"[{task_id}] Iteration={r} chosen_analyzer={chosen} (bandit={len(b_issues)}, semgrep={len(s_issues)})")

        prompt = REPAIR_PROMPT_TEMPLATE.substitute(
            guide=guide,
            analyzer=chosen,
            code=current_code,
            bandit_json=json.dumps(chosen_issues, ensure_ascii=False),
        )

        try:
            fixed_raw = llm.generate_text(prompt).strip()
        except Exception as e:
            print(f"[{task_id}] [Error] LLM generation failed: {e}")
            break

        fixed_code = strip_md(fixed_raw)

        if fixed_code == current_code:
            csv_log(run_out_dir=run_out_dir, task_id=f"{task_id}#bandit", iter_idx=r, language=lang, issues=pre["bandit_result"]["issues"])
            csv_log(run_out_dir=run_out_dir, task_id=f"{task_id}#semgrep", iter_idx=r, language=lang, issues=pre["semgrep_result"]["issues"])
            history.append(
                {
                    "iter": r,
                    "chosen": chosen,
                    "bandit_count": len(b_issues),
                    "semgrep_count": len(s_issues),
                    "note": "no_changes",
                }
            )
            print(f"[{task_id}] No changes detected. Breaking repair loop.")
            break

        post = run_two_analyzers(fixed_code, lang, tmpname=f"{task_id}_iter{r}_post")

       
        csv_log(run_out_dir=run_out_dir, task_id=f"{task_id}#bandit", iter_idx=r, language=lang, issues=post["bandit_result"]["issues"])
        csv_log(run_out_dir=run_out_dir, task_id=f"{task_id}#semgrep", iter_idx=r, language=lang, issues=post["semgrep_result"]["issues"])

        history.append(
            {
                "iter": r,
                "chosen": chosen,
                "bandit_count": len(post["bandit_result"]["issues"]),
                "semgrep_count": len(post["semgrep_result"]["issues"]),
            }
        )

        current_code = fixed_code

        if post["secure"]:
            print(f"[{task_id}] Both analyzers clean at iteration {r}.")
            success = True
            return current_code, [], success, rounds
        else:
            print(
                f"[{task_id}] Remaining after iteration {r}: "
                f"bandit={len(post['bandit_result']['issues'])}, semgrep={len(post['semgrep_result']['issues'])}"
            )

    final_scan = run_two_analyzers(current_code, lang, tmpname=f"{task_id}_final")
    print(
        f"[{task_id}] Repair loop finished after {rounds} rounds. Remaining: "
        f"bandit={len(final_scan['bandit_result']['issues'])}, semgrep={len(final_scan['semgrep_result']['issues'])}"
    )
    return current_code, [], success, rounds

def run_framework(records,dataset: str,technique: str,limit: int | None = None,
    save_plans: bool = True,iterations: int = 0,output_filename: str | None = None):
    start_time = time.time()
    llm = LLMClient()
    model_tag = sanitize_model_name(llm.model)

    out_dir = PATHS.run_dir(dataset=dataset, model_name=model_tag, technique=technique)

    if output_filename is None:
        output_filename = f"{dataset}.jsonl"

    output_file = out_dir / output_filename
    plans_file = out_dir / f"{Path(output_filename).stem}_plans.jsonl"

    pfile = plans_file.open("a", encoding="utf-8") if save_plans else None
    try:
        with output_file.open("a", encoding="utf-8") as outf:
            for idx, t in enumerate(records, 1):
                if limit is not None and idx > limit:
                    break

                lang = (t.get("language") or "python").strip()
                lang_title = "Python" if lang.lower().startswith("py") else "C" if lang.lower().startswith("c") else lang
                task_id = t.get("ID")
                dataset_prompt = t.get("Prompt", "")

                print(f"=== Planning task {idx}: {task_id} [{lang_title}] ===")

                plan_prompt = build_planning_prompt(dataset_prompt)
                try:
                    plan_raw = llm.generate_text(plan_prompt).strip()
                except Exception as e:
                    print(f"[Error] planning failed: {e}")
                    plan_raw = "{}"

                plan_obj, warns = parse_and_validate_plan(plan_raw)
                for w in warns:
                    print(f"[Warn] {task_id}: {w}")

                if pfile:
                    pfile.write(json.dumps({"task": task_id, "plan_raw": plan_raw, "plan": plan_obj}, ensure_ascii=False) + "\n")
                    pfile.flush()

                guide_for_coder = summarize_for_coder(minimal_plan(plan_obj))
                bundle = {
                    "plan": minimal_plan(plan_obj),
                    "guide": guide_for_coder,
                    "task": dataset_prompt,
                    "language": lang,
                }
                final_prompt = CODING_TEMPLATE.format(Prompt=json.dumps(bundle, ensure_ascii=False))

                print(f"=== Coding task {idx}: {task_id} ===")
                raw_resp: str | None = None
                for attempt in range(3):
                    try:
                        raw_resp = llm.generate_text(final_prompt).strip()
                        break
                    except Exception as e:
                        msg = str(e)
                        if "502" in msg or "Bad Gateway" in msg or "InternalServerError" in msg or "timeout" in msg:
                            wait = 2 ** attempt
                            print(f"[Warn] transient error. Retrying in {wait}s (attempt {attempt+1}/3)...")
                            time.sleep(wait)
                            continue
                        print(f"[Error] {e}")
                        break

                if raw_resp is None:
                    parsed = {
                        "task": task_id,
                        "intent" : dataset_prompt,
                        "language": lang,
                        "framework": t.get("framework"),
                        "technique": technique,
                        "code": "",
                        "error": "generation_failed",
                        "issues": [],
                        "loc": 0,
                        "secure": False,
                        "plan": minimal_plan(plan_obj),
                        "repair": {"success": False, "rounds": 0},
                        "bandit_result": {"secure": False, "issues": [], "summary": {}},
                        "semgrep_result": {"secure": False, "issues": [], "summary": {}},
                    }
                else:
                    code = strip_markdown_fences(raw_resp)
                    scan0 = run_two_analyzers(code, lang, tmpname=task_id)
                    csv_log(run_out_dir=str(out_dir), task_id=f"{task_id}#bandit", iter_idx=0, language=lang, issues=scan0["bandit_result"]["issues"])
                    csv_log(run_out_dir=str(out_dir), task_id=f"{task_id}#semgrep", iter_idx=0, language=lang, issues=scan0["semgrep_result"]["issues"])

                    final_code = code
                    final_scan = scan0
                    repair_summary = {"success": False, "rounds": 0}

                    if (not scan0["secure"]) and iterations > 0:
                        fixed_code, _issues_after, success, rounds = attempt_repair_loop(
                            llm=llm,
                            code=code,
                            issues=[],
                            plan_obj=plan_obj,
                            lang=lang,
                            task_id=task_id,
                            iterations=int(iterations),
                            run_out_dir=str(out_dir)
                        )
                        final_code = fixed_code
                        final_scan = run_two_analyzers(final_code, lang, tmpname=f"{task_id}_final")
                        repair_summary = {"success": bool(success), "rounds": int(rounds)}
                    parsed = {
                        "task": task_id,
                        "intent": dataset_prompt,
                        "language": lang,
                        "framework": t.get("framework"),
                        "technique": technique,
                        "code": final_code,
                        "issues": [],
                        "loc": int(final_scan.get("loc") or 0),
                        "secure": bool(final_scan.get("secure")),
                        "bandit_result": final_scan.get("bandit_result") or {"secure": False, "issues": [], "summary": {}},
                        "semgrep_result": final_scan.get("semgrep_result") or {"secure": False, "issues": [], "summary": {}},
                        "plan": minimal_plan(plan_obj),
                        "repair": repair_summary,
                    }

                outf.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                outf.flush()
    finally:
        if pfile:
            pfile.close()

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
                "iterations": int(iterations),
                "save_plans": bool(save_plans)
            }
        )

        if csv_path.exists():
            df = load_refinement_df(csv_path)
            out_img = out_dir / "plots" / "Refinment_Plot.png"
            out_img.parent.mkdir(parents=True, exist_ok=True)
            plot_totals(csv_path, out=out_img, kind="line", show=False, df=df, verbose=False)
            refinment_summary(csv_path, metrics_txt=metrics_txt, df=df, write=True)

        print(f"[metrics] saved to: {metrics_txt}")
    except Exception as e:
        print(f"[metrics-save] {e}")