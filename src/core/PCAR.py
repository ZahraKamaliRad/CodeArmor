from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json, re, time
from typing import Any, Dict, List, Tuple
from string import Template

from .openai_client import LLMClient
from .analyzer import analyze_code
from ..config import PATHS
from ..utils.io import csv_log
from ..utils.plot_refinment import load_refinement_df, plot_totals, refinment_summary
from ..utils.metrics import save_metrics_result




PLANNING_TEMPLATE = Template(
    """You are a secure-coding planner.
Return ONLY a valid JSON object and nothing else (no markdown, no prose, no code).

Fields required: cwe, summary, threats, policy, algorithm, secure_apis, error_handling, acceptance_criteria.
- summary: short (<= 2 lines)
- threats: list of strings
- policy: object with relevant limits/allowlists/paths/extensions (as needed for the task)
- algorithm: list of short actionable steps (no code)
- secure_apis: list of API names to use (e.g., yaml.safe_load, urllib.parse.urlparse, parameterized SQL, etc.)
- error_handling: lines like "Exception -> behavior"
- acceptance_criteria: concrete checks that MUST be enforced

Input Prompt:
$problem
"""
)


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
- SECURITY_GUIDANCE:\n$guide
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
            if k in ("threats","algorithm","secure_apis","error_handling","acceptance_criteria"):
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

    obj["summary"] = str(obj.get("summary",""))[:200].split('\n')[0].strip()
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

def attempt_repair_loop(llm: "LLMClient",code: str,issues: List[Dict[str, Any]],plan_obj: Dict[str, Any],
    lang: str,task_id: str,max_rounds: int = 1,history: list | None = None,run_out_dir: str = "") -> Tuple[str, List[Dict[str, Any]], bool, int]:
    current_code = code
    current_issues = issues
    rounds = 0
    success = False
    if history is None:
        history = [{"iter": 0, "issues": issues}]
    guide = summarize_for_coder(minimal_plan(plan_obj))

    if len(current_issues) == 0:
        print(f"[{task_id}] No issues found. Skipping repair loop.")
        return current_code, current_issues, success, rounds
    
    print(f"[{task_id}] Starting repair loop with {len(current_issues)} issues...")

    for r in range(1, max_rounds + 1):
        rounds = r
        print(f"[{task_id}] Repair iteration = {r}")

        prompt = REPAIR_PROMPT_TEMPLATE.substitute(
            guide=guide,
            code=current_code,
            bandit_json=json.dumps(current_issues, ensure_ascii=False),
        )

        try:
            fixed_raw = llm.generate_text(prompt).strip()
        except Exception as e:
            print(f"[{task_id}] [Error] LLM repair generation failed: {e}")
            break

        fixed_code = strip_md(fixed_raw)

        if fixed_code == current_code:
            csv_log(run_out_dir=run_out_dir, task_id=str(task_id),
                           iter_idx=r, language=lang, issues=current_issues)
            history.append({"iter": r, "issues": current_issues})
            print(f"[{task_id}] No changes detected. Breaking repair loop.")
            break

        new_issues, _ = analyze_code(fixed_code, lang, tmpname=f"{task_id}_repair_r{r}")
        csv_log(run_out_dir=run_out_dir, task_id=str(task_id),
                       iter_idx=r, language=lang, issues=new_issues)
        history.append({"iter": r, "issues": new_issues})
        current_code, current_issues = fixed_code, new_issues

        if len(new_issues) == 0:
            print(f"[{task_id}] All issues fixed at iteration {r}.")
            success = True
            return current_code, current_issues, success, rounds
        else:
            print(f"[{task_id}] Remaining issues after iteration {r}: {len(new_issues)}")

    print(f"[{task_id}] Repair loop finished after {rounds} rounds. Remaining issues: {len(current_issues)}")
    return current_code, current_issues, success, rounds

def run_framework(records, dataset_name: str = "output", limit: int | None = None, save_plans: bool = True, max_repair_rounds: int = 1, output_filename: str | None = None):
    llm = LLMClient(api_key="")
    out_dir = PATHS.dataset_run_dir(dataset_name)
    if output_filename is None:
        output_filename = f"{out_dir.name}.jsonl"
    output_file = out_dir / output_filename
    plans_file = out_dir / f"{Path(output_filename).stem}_plans.jsonl"
    with output_file.open("a", encoding="utf-8") as outf:
        pfile = plans_file.open("a", encoding="utf-8") if save_plans else None

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
                    else:
                        print(f"[Error] {e}")
                        break

            if raw_resp is None:
                parsed = {
                    "task": task_id,
                    "language": lang,
                    "framework": t.get("framework"),
                    "code": "",
                    "error": "generation_failed",
                    "issues": [],
                    "loc": 0,
                    "secure": False,
                    "plan": minimal_plan(plan_obj),
                    "repair": {"success": False, "rounds": 0},
                }
            else:
                code = strip_markdown_fences(raw_resp)
                issues, loc = analyze_code(code, lang, tmpname=task_id)
                csv_log(
                    run_out_dir=str(out_dir),
                    task_id=str(task_id),
                    iter_idx=0,
                    language=lang,
                    issues=issues
                )
                final_code, final_issues, final_loc = code, issues, loc
                repair_summary = {"success": False, "rounds": 0}
                if len(issues) > 0 and max_repair_rounds > 0:
                    fixed_code, issues_after, success, rounds = attempt_repair_loop(
                        llm=llm,
                        code=code,
                        issues=issues,
                        plan_obj=plan_obj,
                        lang=lang,
                        task_id=task_id,
                        max_rounds=max_repair_rounds,
                        run_out_dir=str(out_dir)
                    )
                    final_code, final_issues = fixed_code, issues_after
                    _, final_loc = analyze_code(final_code, lang, tmpname=f"{task_id}_final")
                    repair_summary = {"success": bool(success), "rounds": int(rounds)}

                parsed = {
                    "task": task_id,
                    "language": lang,
                    "framework": t.get("framework"),
                    "code": final_code,
                    "issues": final_issues,
                    "loc": final_loc,
                    "secure": len(final_issues) == 0,
                    "plan": minimal_plan(plan_obj),
                    "repair": repair_summary
                }

            parsed["bandit_result"] = {"secure": parsed["secure"], "issues": parsed["issues"]}

            outf.write(json.dumps(parsed, ensure_ascii=False) + "\n")
            outf.flush()

        if pfile:
            pfile.close()


    csv_path = out_dir / "Effect_Of_Refinment.csv"
    try:
        metrics_dir = out_dir / "rate" / "vuln_density"
        metrics_txt = metrics_dir / f"{Path(output_filename).stem}_metrics.txt"

        save_metrics_result(str(output_file), str(metrics_txt))

        if csv_path.exists():
            df = load_refinement_df(csv_path)
            out_img = out_dir / "plots" / "Refinment_Plot.png"
            plot_totals(csv_path, out=out_img, kind="line", show=False, df=df, verbose=False)
            refinment_summary(csv_path, metrics_txt=metrics_txt, df=df, write=True)

        print(f"[metrics] saved to: {metrics_txt}")
    except Exception as e:
        print(f"[metrics-save] {e}")

    print(f"Tasks completed. Results saved to {output_file}")
    return str(output_file)
