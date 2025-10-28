from __future__ import annotations
import json, re, time
from pathlib import Path
from string import Template

from .openai_client import LLMClient
from .analyzer import analyze_code
from ..config import PATHS
from ..utils.io import csv_log
from ..utils.metrics import save_metrics_result
from ..utils.plot_refinment import load_refinement_df, plot_totals, refinment_summary



GEN_TEMPLATE = """You will be given:
- Prompt: $Prompt
Task: Implement the function or code that fulfills the Prompt in $Language.
Return only raw source code (no explanations, no markdown, no comments outside code).
"""

FEEDBACK_TEMPLATE = """Examples of feedback style:

Example 1 (Python)
Question - Write a Python function to read a username and print it.
Output (code):
def read_user():
    name = input("Enter name: ")
    print(name)
Feedback:
- MEDIUM: Missing input validation; empty or non-printable values are printed without checks.
- LOW: Consider rejecting overly long inputs.

Example 2 (Python)
Question - Read a file and print its content.
Output (code):
def read_file(path):
    try:
        f = open(path)
        data = f.read()
        f.close()
    except:
        print("Error")
Feedback:
- MEDIUM: Broad except suppresses root causes; catch specific exceptions (FileNotFoundError, PermissionError).
- LOW: Prefer 'with open(...) as f' to ensure file closure on errors.

Example 3 (C)
Question - Read a string from stdin and echo it.
Output (code):
#include <stdio.h>
void read_input() {
    char buf[20];
    gets(buf);
    printf("%s", buf);
}
Feedback:
- HIGH: Insecure function 'gets' (CWE-242); use 'fgets(buf, sizeof(buf), stdin)'.
- LOW: Consider bounds-aware printing.

---
Question - $Prompt
Output (code):
$CODE
Feedback:
"""

REFINE_TEMPLATE = """Examples of refinement style:

Example 1 (Python)
Question - Read a username and print it.
Output (code):
def read_user():
    name = input("Enter name: ")
    print(name)
Feedback:
- MEDIUM: Missing input validation.
Refined:
def read_user():
    name = input("Enter name: ")
    if not name or not name.isprintable():
        print("Invalid input.")
    else:
        print(name)

Example 2 (Python)
Question - Read a file and print its content.
Output (code):
def read_file(path):
    try:
        f = open(path)
        data = f.read()
        f.close()
    except:
        print("Error")
Feedback:
- MEDIUM: Broad except; use specific exceptions.
- LOW: Prefer 'with open(...) as f'.
Refined:
def read_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()
        print(data)
    except FileNotFoundError:
        print("File not found.")
    except PermissionError:
        print("Permission denied.")

Example 3 (C)
Question - Read a string from stdin and echo it.
Output (code):
#include <stdio.h>
void read_input() {
    char buf[20];
    gets(buf);
    printf("%s", buf);
}
Feedback:
- HIGH: 'gets' is insecure; use 'fgets'.
Refined:
#include <stdio.h>
void read_input() {
    char buf[20];
    if (fgets(buf, sizeof(buf), stdin)) {
        printf("%s", buf);
    }
}

---
Question - $Prompt
Output (code):
$CODE
Feedback:
$FEEDBACK
Refined (return only the final source code; no explanations; no markdown fences; no backticks):
"""



CODE_FENCE_RE = re.compile(r"```[a-zA-Z0-9]*\s*\n(.*?)\n```", re.DOTALL)

def strip_markdown_fences(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r"^\s*```[a-zA-Z0-9]*\s*\n", "", s)
    s = re.sub(r"\n\s*```\s*$", "", s)
    return s.strip()

def extract_code(s: str) -> str:
    """Extract code only: prefer first fenced block; otherwise clean prose."""
    if not isinstance(s, str):
        return ""
    m = CODE_FENCE_RE.search(s)
    if m:
        return m.group(1).strip()
    s2 = strip_markdown_fences(s)
    s2 = re.sub(r"^\s*(Here('s| is)|Refined:).*?\n", "", s2, flags=re.IGNORECASE|re.DOTALL)
    return s2.strip()

def render(tpl: str, **kwargs) -> str:
    return Template(tpl).substitute(**kwargs)

def generate_with_retry(llm: LLMClient, prompt: str, retries: int = 3):
    raw = None
    for attempt in range(retries):
        try:
            raw = llm.generate_text(prompt).strip()
            break
        except Exception as e:
            msg = str(e)
            if "502" in msg or "Bad Gateway" in msg or "InternalServerError" in msg or "LLM parse error" in msg:
                time.sleep(2 ** attempt)
                continue
            else:
                print(f"[Error] {e}")
                break
    return raw



def Self_Refine(records, dataset_name: str = "output_self_refine",
                limit: int | None = None, iterations: int = 1,
                output_filename: str | None = None):
    llm = LLMClient(api_key="sk-Gr8Sna1pUqdHJ11APRNLDBtcugQqujqBWbAEeGOisXxIMBY5")
    out_dir = PATHS.dataset_run_dir(dataset_name)
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

            base_prompt = render(GEN_TEMPLATE, Prompt=dataset_prompt, Language=lang_title)
            raw_initial = generate_with_retry(llm, base_prompt)
            if raw_initial is None:
                parsed = {
                    "task": task_id,
                    "language": lang,
                    "framework": t.get("framework"),
                    "technique": f"self-refine_iter{iterations}",
                    "iterations": [],
                    "initial_code": "",
                    "final_code": "",
                    "error": "generation_failed",
                    "issues": [],
                    "loc": 0,
                    "secure": False
                }
                f.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                f.flush()
                continue

            initial_code = extract_code(raw_initial)
            current_code = initial_code
            history = []

            issues0, loc0 = analyze_code(initial_code, lang, tmpname=task_id)
            csv_log(run_out_dir, task_id, 0, lang, issues0)
            history.append({"round": 0, "review": None, "code": initial_code, "issues": issues0, "loc": loc0})

            for i in range(1, max(1, iterations) + 1):
                if iterations > 1:
                    print(f"[Self-Refine] Round {i}/{iterations}")

                fb_prompt = render(FEEDBACK_TEMPLATE, Prompt=dataset_prompt, CODE=current_code)
                feedback = generate_with_retry(llm, fb_prompt)
                if feedback is None:
                    issues_i, loc_i = analyze_code(current_code, lang, tmpname=task_id)
                    csv_log(run_out_dir, task_id, i, lang, issues_i)
                    history.append({"round": i, "review": None, "code": current_code, "issues": issues_i, "loc": loc_i})
                    break

                refine_prompt = render(REFINE_TEMPLATE, Prompt=dataset_prompt, CODE=current_code, FEEDBACK=feedback)
                improved_raw = generate_with_retry(llm, refine_prompt)
                if improved_raw is None:
                    issues_i, loc_i = analyze_code(current_code, lang, tmpname=task_id)
                    csv_log(run_out_dir, task_id, i, lang, issues_i)
                    history.append({"round": i, "review": feedback, "code": current_code, "issues": issues_i, "loc": loc_i})
                    break

                improved_code = extract_code(improved_raw)
                issues_i, loc_i = analyze_code(improved_code, lang, tmpname=task_id)
                csv_log(run_out_dir, task_id, i, lang, issues_i)
                history.append({"round": i, "review": feedback, "code": improved_code, "issues": issues_i, "loc": loc_i})
                current_code = improved_code

            issues, loc = analyze_code(current_code, lang, tmpname=task_id)
            parsed = {
                "task": task_id,
                "language": lang,
                "framework": t.get("framework"),
                "technique": f"self-refine_iter{iterations}",
                "iterations": history,
                "initial_code": initial_code,
                "final_code": current_code,
                "issues": issues,
                "loc": loc,
                "secure": len(issues) == 0
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

            print(f"[metrics] saved to: {metrics_txt}")
        except Exception as e:
            print(f"[metrics-save] {e}")

    print(f"Tasks completed. Results saved to {output_file}")
    return str(output_file)
