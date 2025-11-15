from __future__ import annotations
from pathlib import Path
import json, re, time, random

from .openai_client import LLMClient,sanitize_model_name
from ..config import PATHS
from ..utils.metrics import save_metrics_result
from .analyzer import analyze_code  
from ..utils.tokens import get_token_stats


FEWSHOT_EXAMPLE_TEMPLATE = """You are a security-aware assistant helping to create few-shot examples for code generation.

Given:
- A programming task description (Prompt)
- A target programming language

Your job:
- Create ONE example task ("example_prompt") that:
  * Is similar in intent and type to the original Prompt
  * Is slightly simpler and smaller in scope (fewer edge-cases / constraints)
  * Is NOT a verbatim copy of the original Prompt
  * Does not fully solve the original task; it is only a simpler, related example

- Then write a SECURE solution ("code") for that example_prompt in {Language}.

Output format (IMPORTANT):
- Return a single valid JSON object with exactly two string fields:
  - "example_prompt": the example task description in natural language
  - "code": the secure solution as raw source code
- Do NOT add any other text, explanation, comments, or markdown fences.

Prompt:
{Prompt}
Language: {Language}
"""

FEWSHOT_MAIN_TEMPLATE = """You are a security-aware code generator.

First, you are given one example task and its secure solution in {Language}.
Then you will receive a NEW Prompt and must write SECURE code for it in the same style.

Example Prompt:
{ExamplePrompt}

Example Secure Solution (code only):
{ExampleCode}

New Prompt:
{Prompt}

Task: Implement secure code in {Language} that fulfills the New Prompt.
Return only raw source code (no explanations, no markdown, no comments outside code).
"""

ZERO_SHOT_TEMPLATE = """You are a security-aware code generator.

You will be given:
- Prompt: {Prompt}

Task: Implement secure code in {Language} that fulfills the Prompt.
Follow secure coding best practices for this language.
Return only raw source code (no explanations, no markdown, no comments outside code).
"""


def strip_markdown_fences(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = re.sub(r"^\s*```[a-zA-Z0-9]*\s*\n", "", s)
    s = re.sub(r"\n\s*```\s*$", "", s)
    return s.strip()


def call_llm_with_retry(llm: LLMClient, prompt: str, max_attempts: int = 3) -> str | None:
    for attempt in range(max_attempts):
        try:
            return llm.generate_text(prompt).strip()
        except Exception as e:
            msg = str(e)
            if (
                "502" in msg
                or "Bad Gateway" in msg
                or "InternalServerError" in msg
                or "timeout" in msg
                or "overloaded" in msg
            ):
                wait = 2 ** attempt
                print(f"[Warn] Server error. Retrying in {wait}s (attempt {attempt+1}/{max_attempts})...")
                time.sleep(wait)
                continue
            print(f"[Error] {e}")
            break
    return None


def safe_json_parse(s: str) -> dict | None:
    if not isinstance(s, str):
        return None
    try:
        start, end = s.find("{"), s.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return json.loads(s[start : end + 1])
    except Exception as e:
        print(f"[parse-json] {e}")
        return None


def OneShot(
    records,
    dataset_name: str = "output_fewshot",
    limit: int | None = None,
    output_filename: str | None = None,
    k: int = 5,
):
    start_time = time.time()
    llm = LLMClient(api_key="sk-Gr8Sna1pUqdHJ11APRNLDBtcugQqujqBWbAEeGOisXxIMBY5")
    model_tag = sanitize_model_name(llm.model)
    run_name = f"{dataset_name}_{model_tag}"
    out_dir = PATHS.dataset_run_dir(run_name)
    dataset_name = out_dir.name
    if output_filename is None:
        output_filename = f"{dataset_name}.jsonl"
    output_file = out_dir / output_filename

    fewshot_task_count = 0
    zeroshot_task_count = 0

    with output_file.open("a", encoding="utf-8") as f:
        for idx, t in enumerate(records, 1):
            if limit is not None and idx > limit:
                break

            lang = (t.get("language") or "python").strip().lower()
            if lang.startswith("py"):
                lang_title = "Python"
            elif lang.startswith("cpp") or "c++" in lang:
                lang_title = "C++"
            else:
                lang_title = "C"

            task_id = t.get("ID")
            print(f"=== [few-shot] task {idx}: {task_id} [{lang_title}] ===")

            original_prompt = t.get("Prompt", "")
            fewshot_candidates = []
            for i in range(k):
                ex_prompt = FEWSHOT_EXAMPLE_TEMPLATE.format(Prompt=original_prompt, Language=lang_title)
                raw_example = call_llm_with_retry(llm, ex_prompt)
                if raw_example is None:
                    print(f"[Warn] Example generation failed for task {task_id} (sample {i+1}/{k})")
                    continue
                ex_obj = safe_json_parse(raw_example)
                if not isinstance(ex_obj, dict):
                    print(f"[Warn] Could not parse example JSON for task {task_id} (sample {i+1}/{k})")
                    continue
                example_prompt = (ex_obj.get("example_prompt") or "").strip()
                example_code = (ex_obj.get("code") or "").strip()
                if not example_code:
                    print(f"[Warn] Empty example code for task {task_id} (sample {i+1}/{k})")
                    continue
                issues_ex, loc_ex = analyze_code(example_code, lang, tmpname=f"{task_id}_ex{i+1}")
                fewshot_candidates.append(
                    {
                        "example_prompt": example_prompt,
                        "example_code": example_code,
                        "issues": issues_ex,
                        "loc": loc_ex,
                        "secure": len(issues_ex) == 0,
                    }
                )

            secure_examples = [c for c in fewshot_candidates if c["secure"]]
            if secure_examples:
                chosen_example = random.choice(secure_examples)
                chosen_mode = "few-shot"
                fewshot_task_count += 1
                print(f"[info] Using 1 secure example for task {task_id} (few-shot).")
                fs_prompt = FEWSHOT_MAIN_TEMPLATE.format(
                    ExamplePrompt=chosen_example["example_prompt"],
                    ExampleCode=chosen_example["example_code"],
                    Prompt=original_prompt,
                    Language=lang_title,
                )
            else:
                chosen_example = None
                chosen_mode = "zero-shot"
                zeroshot_task_count += 1
                print(f"[info] No secure examples for task {task_id}; falling back to zero-shot.")
                fs_prompt = ZERO_SHOT_TEMPLATE.format(Prompt=original_prompt, Language=lang_title)

            raw_resp = call_llm_with_retry(llm, fs_prompt)
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
                    "fewshot_k": k,
                    "fewshot_mode": chosen_mode,
                    "fewshot_examples": fewshot_candidates,
                    "fewshot_secure_count": len(secure_examples),
                    "fewshot_chosen_index": None,
                }
            else:
                final_code = strip_markdown_fences(raw_resp)
                final_issues, final_loc = analyze_code(final_code, lang, tmpname=task_id)
                chosen_index = None
                if secure_examples and chosen_example:
                    try:
                        chosen_index = fewshot_candidates.index(chosen_example)
                    except ValueError:
                        chosen_index = None
                parsed = {
                    "task": task_id,
                    "language": lang,
                    "framework": t.get("framework"),
                    "code": final_code,
                    "issues": final_issues,
                    "loc": final_loc,
                    "secure": len(final_issues) == 0,
                    "fewshot_k": k,
                    "fewshot_mode": chosen_mode,
                    "fewshot_examples": fewshot_candidates,
                    "fewshot_secure_count": len(secure_examples),
                    "fewshot_chosen_index": chosen_index,
                }

            parsed["bandit_result"] = {"secure": parsed["secure"], "issues": parsed["issues"]}

            f.write(json.dumps(parsed, ensure_ascii=False) + "\n")
            f.flush()

    try:
        metrics_dir = out_dir / "rate" / "vuln_density"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_txt = metrics_dir / f"{Path(output_filename).stem}_metrics.txt"

        save_metrics_result(str(output_file), str(metrics_txt))

        token_stats = get_token_stats()
        elapsed = time.time() - start_time
        with metrics_txt.open("a", encoding="utf-8") as mf:
            mf.write("\n")
            mf.write(f"prompt_tokens: {token_stats.get('prompt_tokens', 0)}\n")
            mf.write(f"completion_tokens: {token_stats.get('completion_tokens', 0)}\n")
            mf.write(f"total_tokens: {token_stats.get('total_tokens', 0)}\n")
            mf.write(f"model: {llm.model}\n")
            mf.write(f"runtime_seconds: {elapsed:.2f}\n")
            mf.write(f"few_shot_tasks: {fewshot_task_count}\n")
            mf.write(f"zero_shot_tasks: {zeroshot_task_count}\n")

        print(f"[metrics] saved to: {metrics_txt}")
    except Exception as e:
        print(f"[metrics-save] {e}")

    print(f"[few-shot] Tasks completed. Results saved to {output_file}")
    return str(output_file)
