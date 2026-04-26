from __future__ import annotations

import json
import re
import time
from pathlib import Path
from string import Template

from .openai_client import LLMClient, sanitize_model_name
from .analyzer import analyze_code_split
from src.paths import PATHS
from ..utils.io import csv_log
from ..utils.metrics import save_metrics_result
from ..utils.plot_refinment import load_refinement_df, plot_totals
from ..utils.tokens import get_token_stats
from ..utils.text_utils import strip_markdown_fences



BASE_TEMPLATE = """You are a code generator.
You will be given:
- Prompt: $Prompt
Task: Implement the function or code that fulfills the Prompt in $Language.
Return only raw source code (no explanations, no markdown, no comments outside code).
"""

REVIEW_TEMPLATE = """Example of review generated code:
Intent:
import xml.etree.ElementTree as ET

def parse_xml(xml_string):
    '''
    Parse xml_string and return the root element of the XML document
    '''
Generated code:
import xml.etree.ElementTree as ET

def parse_xml(xml_string):
    return ET.fromstring(xml_string)

Review generated code (list 4-5 main security issues, be concise):

1) Using xml.etree.ElementTree to parse untrusted XML data is vulnerable to XML attacks.
 Replace with defusedxml.
2) No input validation - xml_string could be None or non-string type.
3) No error handling for malformed XML.

Now review this code and list 4-5 main security issues concisely:
Intent: $Prompt
Generated code:
$CODE
"""

IMPROVE_TEMPLATE = """Example of improve generated code:
Intent:
import xml.etree.ElementTree as ET
def parse_xml(xml_string):
    '''
    Parse xml_string and return the root element of the XML document
    '''

Generated code:
import xml.etree.ElementTree as ET

def parse_xml(xml_string):
    return ET.fromstring(xml_string)

Review generated code:

1) Using xml.etree.ElementTree to parse untrusted XML data is vulnerable to XML attacks.
 Replace with defusedxml.
2) No input validation - xml_string could be None or non-string type.
3) No error handling for malformed XML.

Improved code:

from defusedxml import ElementTree as ET

def parse_xml(xml_string):
    if not isinstance(xml_string, str):
        raise ValueError("Input must be a string")
    try:
        return ET.fromstring(xml_string)
    except Exception as e:
        raise ValueError(f"Invalid XML: {e}")


Now improve the following code based on the review:
Intent: $Prompt
Generated code:
$CODE

Review:
$REVIEW

Improved code:"""

CODE_FENCE_RE = re.compile(r"```[a-zA-Z0-9]*\s*\n(.*?)\n```", re.DOTALL)


def extract_code(s: str) -> str:
    if not isinstance(s, str):
        return ""
    m = CODE_FENCE_RE.search(s)
    if m:
        return m.group(1).strip()
    s2 = strip_markdown_fences(s)
    s2 = re.sub(r"^\s*(Here('s| is)|Refined:).*?\n", "", s2, flags=re.IGNORECASE | re.DOTALL)
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
            if (
                "502" in msg
                or "Bad Gateway" in msg
                or "InternalServerError" in msg
                or "LLM parse error" in msg
                or "timeout" in msg
                or "overloaded" in msg
            ):
                wait = 2 ** attempt
                print(f"[Warn] transient error. Retrying in {wait}s (attempt {attempt+1}/{retries})...")
                time.sleep(wait)
                continue
            print(f"[Error] {e}")
            break
    return raw


def run_two_analyzers(code: str, lang: str, tmpname: str = "snippet") -> dict:
    scan = analyze_code_split(code, lang, tmpname=tmpname) or {}

    result = {}
    result["loc"] = int(scan.get("loc") or 0)

    bandit_secure = True
    semgrep_secure = True

    if "bandit_result" in scan:
        bandit_block = scan["bandit_result"]
        bandit_issues = bandit_block.get("issues") or []
        bandit_secure = bool(bandit_block.get("secure", len(bandit_issues) == 0))

        result["bandit_result"] = {
            "secure": bandit_secure,
            "issues": bandit_issues,
            "summary": bandit_block.get("summary") or {},
        }

    if "semgrep_result" in scan:
        semgrep_block = scan["semgrep_result"]
        semgrep_issues = semgrep_block.get("issues") or []
        semgrep_secure = bool(semgrep_block.get("secure", len(semgrep_issues) == 0))

        result["semgrep_result"] = {
            "secure": semgrep_secure,
            "issues": semgrep_issues,
            "summary": semgrep_block.get("summary") or {},
        }

    result["secure"] = bool(bandit_secure and semgrep_secure)

    return result

def log_two_lines(run_out_dir: str, task_id: str, iter_idx: int, lang: str, scan: dict):
    b_issues = (scan.get("bandit_result") or {}).get("issues") or []
    s_issues = (scan.get("semgrep_result") or {}).get("issues") or []
    csv_log(run_out_dir=run_out_dir, task_id=f"{task_id}#bandit", iter_idx=iter_idx, language=lang, issues=b_issues)
    csv_log(run_out_dir=run_out_dir, task_id=f"{task_id}#semgrep", iter_idx=iter_idx, language=lang, issues=s_issues)


def SelfRefine_gen_code(records,dataset: str,technique: str,
    limit: int | None = None,iterations: int = 1,output_filename: str | None = None):
    iterations = int(iterations or 0)
    if iterations <= 0:
        iterations = 1
    if "_iter" not in technique:
        technique = f"{technique}_iter{iterations}"

    start_time = time.time()
    llm = LLMClient()
    model_tag = sanitize_model_name(llm.model)

    out_dir = PATHS.run_dir(dataset=dataset, model_name=model_tag, technique=technique)

    if output_filename is None:
        output_filename = f"{dataset}.jsonl"

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
                lang_key = "python"
            # elif lt.startswith("cpp") or "c++" in lt:
            #     lang_title = "C++"
            #     lang_key = "cpp"
            # else:
            #     lang_title = "C"
            #     lang_key = "c"

            task_id = t.get("ID")
            intent = t.get("Prompt", "") or ""
            dataset_prompt = intent
            tmpname = Path(str(task_id or "snippet")).stem

            print(f"=== Running task {idx}: {task_id} [{lang_title}] ===")

            base_prompt = render(BASE_TEMPLATE, Prompt=dataset_prompt, Language=lang_title)
            raw_initial = generate_with_retry(llm, base_prompt)
            if raw_initial is None:
                parsed = {
                    "task": task_id,
                    "intent": intent,
                    "language": lang_key,
                    "framework": t.get("framework"),
                    "technique": technique,
                    "self_refine_iterations": int(iterations),
                    "iterations": [],
                    "initial_code": "",
                    "final_code": "",
                    "review": None,
                    "error": "generation_failed",
                    "loc": 0,
                    "secure": False,
                }
                f.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                f.flush()
                continue

            initial_code = extract_code(raw_initial)
            current_code = initial_code
            history = []
            last_review = None

            scan0 = run_two_analyzers(initial_code, lang_key, tmpname=tmpname)
            log_two_lines(run_out_dir, str(task_id), 0, lang_key, scan0)

            entry = {
                "round": 0,
                "review": None,
                "code": initial_code,
                "loc": int(scan0.get("loc") or 0),
                "secure": bool(scan0.get("secure")),
            }

            if "bandit_result" in scan0:
                entry["bandit_result"] = scan0["bandit_result"]

            if "semgrep_result" in scan0:
                entry["semgrep_result"] = scan0["semgrep_result"]

            history.append(entry)


            for i in range(1, max(1, int(iterations)) + 1):

                if iterations > 1:
                    print(f"[Self-Refine] >>> Entering refinement round {i}/{iterations}")

                fb_prompt = render(REVIEW_TEMPLATE, Prompt=dataset_prompt, CODE=current_code)
                feedback = generate_with_retry(llm, fb_prompt)

                if feedback is None:
                    print("[Self-Refine] Review failed, using previous code.")

                    scan_i = run_two_analyzers(current_code, lang_key, tmpname=tmpname)

                    b = len((scan_i.get("bandit_result") or {}).get("issues") or [])
                    s = len((scan_i.get("semgrep_result") or {}).get("issues") or [])
                    print(f"[Self-Refine] bandit={b} semgrep={s}")

                    log_two_lines(run_out_dir, str(task_id), i, lang_key, scan_i)

                    entry = {
                        "round": i,
                        "review": None,
                        "code": current_code,
                        "loc": int(scan_i.get("loc") or 0),
                        "secure": bool(scan_i.get("secure")),
                    }

                    if "bandit_result" in scan_i:
                        entry["bandit_result"] = scan_i["bandit_result"]

                    if "semgrep_result" in scan_i:
                        entry["semgrep_result"] = scan_i["semgrep_result"]

                    history.append(entry)
                    break


                print("[Self-Refine] Review generated.")
                last_review = feedback


                refine_prompt = render(IMPROVE_TEMPLATE,Prompt=dataset_prompt,CODE=current_code,REVIEW=feedback)
                improved_raw = generate_with_retry(llm, refine_prompt)

                if improved_raw is None:
                    print("[Self-Refine] Improvement failed, using previous code.")

                    scan_i = run_two_analyzers(current_code, lang_key, tmpname=tmpname)

                    b = len((scan_i.get("bandit_result") or {}).get("issues") or [])
                    s = len((scan_i.get("semgrep_result") or {}).get("issues") or [])
                    #print(f"[Self-Refine] bandit={b} semgrep={s}")

                    log_two_lines(run_out_dir, str(task_id), i, lang_key, scan_i)

                    entry = {
                        "round": i,
                        "review": feedback,
                        "code": current_code,
                        "loc": int(scan_i.get("loc") or 0),
                        "secure": bool(scan_i.get("secure")),
                    }

                    if "bandit_result" in scan_i:
                        entry["bandit_result"] = scan_i["bandit_result"]

                    if "semgrep_result" in scan_i:
                        entry["semgrep_result"] = scan_i["semgrep_result"]

                    history.append(entry)
                    break


                print("[Self-Refine] Improvement generated.")

                improved_code = extract_code(improved_raw)

                scan_i = run_two_analyzers(improved_code, lang_key, tmpname=tmpname)

                b = len((scan_i.get("bandit_result") or {}).get("issues") or [])
                s = len((scan_i.get("semgrep_result") or {}).get("issues") or [])
                #print(f"[Self-Refine] bandit={b} semgrep={s}")

                log_two_lines(run_out_dir, str(task_id), i, lang_key, scan_i)

                entry = {
                    "round": i,
                    "review": feedback,
                    "code": improved_code,
                    "loc": int(scan_i.get("loc") or 0),
                    "secure": bool(scan_i.get("secure")),
                }

                if "bandit_result" in scan_i:
                    entry["bandit_result"] = scan_i["bandit_result"]

                if "semgrep_result" in scan_i:
                    entry["semgrep_result"] = scan_i["semgrep_result"]

                history.append(entry)

                current_code = improved_code

            last_iter = history[-1]
            parsed = {
                "task": task_id,
                "intent": intent,
                "language": lang_key,
                "framework": t.get("framework"),
                "technique": technique,
                "self_refine_iterations": int(iterations),
                "iterations": history,
                "initial_code": initial_code,
                "final_code": current_code,
                "review": last_review,
                "loc": int(last_iter.get("loc") or 0),
                "secure": bool(last_iter.get("secure")),
            }

            if "bandit_result" in last_iter:
                parsed["bandit_result"] = last_iter["bandit_result"]

            if "semgrep_result" in last_iter:
                parsed["semgrep_result"] = last_iter["semgrep_result"]

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
                "self_refine_iterations": int(iterations)
            }
        )

        if csv_path.exists():
            df = load_refinement_df(csv_path)
            out_img = out_dir / "plots" / "Refinment_Plot.png"
            out_img.parent.mkdir(parents=True, exist_ok=True)
            plot_totals(csv_path, out=out_img, kind="line", show=False, df=df, verbose=False)
            #refinment_summary(csv_path, metrics_txt=metrics_txt, df=df, write=True)

        print(f"[metrics] saved to: {metrics_txt}")
    except Exception as e:
        print(f"[metrics-save] {e}")

    print(f"Tasks completed. Results saved to {output_file}")
    return str(output_file)
