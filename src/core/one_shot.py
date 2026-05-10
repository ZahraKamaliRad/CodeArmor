from __future__ import annotations
from pathlib import Path
import json
import re
import time
import numpy as np
import os
from tqdm import tqdm
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from .openai_client import LLMClient, sanitize_model_name
from src.paths import PATHS
from .analyzer import analyze_jsonl_batch
from ..utils.text_utils import extract_raw_code
from ..utils.usage_stats import (get_token_stats, get_llm_stats, get_tool_stats,reset_token_stats, reset_llm_stats, reset_tool_stats)
from ..utils.save_details_result import save_experiment_summary
from ..utils.metrics import save_metrics_result
from ..utils.check_syntax import filter_jsonl


load_dotenv()
RETRIEVAL_DATASET_PATH = Path(os.getenv("RETRIEVAL_DATASET_PATH"))
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH")

embedding_model = None
retrieval_index = {}
SIM_LOWER, SIM_UPPER = 0.30, 0.80

ONE_SHOT_PROMPT_TEMPLATE = """
Example Prompt:
{Example_Prompt}

Reference Secure Code (illustrative only — do NOT reuse its libraries,
patterns, structure, or solution specifics):
{Example_Secure_Code}

The above example demonstrates only the OUTPUT FORMAT and the general idea
of producing secure code. Your task is to:
- Follow the same structure of output
- BUT generate a solution fully adapted to the new task and language
- Avoid copying or mimicking specific implementation details from the example
- Use only what is appropriate for the new problem

Now, write the secure code for this task:

Task: {Prompt}
Language: {Language}

Code:
Return ONLY the source code.
Do NOT include:
- docstrings
- comments
- explanations
- markdown
"""
def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)
    return embedding_model

def load_retrieval_dataset(path: Path) -> list[dict]:
    examples = []
    if not path or not path.exists():
        return examples
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    examples.append(json.loads(line))
                except Exception:
                    pass
    return examples

def extract_cwe(task_id: str) -> str | None:
    match = re.search(r"cwe[-_]?(\d+)", str(task_id), re.IGNORECASE)
    return match.group(1) if match else None

def build_retrieval_index(retrieval_data: list[dict]):
    global retrieval_index
    if retrieval_index:
        return retrieval_index

    model = get_embedding_model()
    groups = {}
    for ex in retrieval_data:
        cwe = ex.get("cwe")
        if cwe:
            groups.setdefault(cwe, []).append(ex)

    for cwe, items in groups.items():
        prompts = [it["prompt"] for it in items]
        embs = model.encode(prompts, normalize_embeddings=True)
        retrieval_index[cwe] = {"examples": items, "embeddings": embs}

    return retrieval_index

def retrieval_example(query_prompt: str, target_cwe: str | None, exclude_id: str):
    model = get_embedding_model()
    query_emb = model.encode([query_prompt], normalize_embeddings=True)

    if target_cwe and target_cwe in retrieval_index:
        idx_data = retrieval_index[target_cwe]
        sims = cosine_similarity(query_emb, idx_data["embeddings"])[0]
        valid_cwe_matches = [
            (idx_data["examples"][i], float(s))
            for i, s in enumerate(sims)
            if idx_data["examples"][i]["id"] != exclude_id and SIM_LOWER < s < SIM_UPPER
        ]
        if valid_cwe_matches:
            example, sim = min(valid_cwe_matches, key=lambda x: x[1])
            return example, sim

    all_ex, all_embs = [], []
    for d in retrieval_index.values():
        all_ex.extend(d["examples"])
        all_embs.extend(d["embeddings"])

    if all_embs:
        sims_global = cosine_similarity(query_emb, np.array(all_embs))[0]
        valid_global_matches = [
            (all_ex[i], float(s))
            for i, s in enumerate(sims_global)
            if all_ex[i]["id"] != exclude_id and SIM_LOWER < s < SIM_UPPER
        ]
        if valid_global_matches:
            example, sim = max(valid_global_matches, key=lambda x: x[1])
            return example, sim

    return None, None

def build_prompt(example, intent, lang_title):
    return ONE_SHOT_PROMPT_TEMPLATE.format(Example_Prompt=example.get("prompt", ""),
        Example_Secure_Code=example.get("secure_code", ""),Prompt=intent,Language=lang_title)

def generate_with_retry(llm, prompt, retries=3):
    for attempt in range(retries):
        try:
            return llm.generate_text(prompt).strip()
        except Exception as e:
            if "502" in str(e):
                time.sleep(2 ** attempt)
            else:
                break
    return None

def build_parsed_result(task_id, intent, lang, framework, example, similarity_score, code, scan=None):
    res = {
        "task": task_id,
        "intent": intent,
        "language": lang,
        "framework": framework,
        "retrieval_example_id": example["id"] if example else None,
        "retrieval_similarity": similarity_score,
        "retrieval_example_prompt": example.get("prompt") if example else None,
        "code": code,
        "loc": 0,
        "bandit_result": {"secure": False, "issues": []},
        "semgrep_result": {"secure": False, "issues": []},
    }

    if scan:
        res["loc"] = int(scan.get("loc") or 0)
        res["bandit_result"] = scan.get("bandit_result", res["bandit_result"])
        res["semgrep_result"] = scan.get("semgrep_result", res["semgrep_result"])

    return res

def OneShot_gen_code(records, dataset: str, technique: str, limit: int | None = None):
    start_time = time.time()
    llm = LLMClient()
    model_tag = sanitize_model_name(llm.model)

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_api_calls = 0
    total_llm_time = 0.0
    total_bandit_time = 0.0
    total_semgrep_time = 0.0
    executed_tasks = 0
    failed_tasks = 0

    out_dir = PATHS.run_dir(dataset=dataset, model_name=model_tag, technique=technique)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_file = out_dir / f"{dataset}.jsonl"
    failed_file = out_dir / "failed_tasks.jsonl"

    retrieval_data = load_retrieval_dataset(RETRIEVAL_DATASET_PATH)
    if retrieval_data:
        build_retrieval_index(retrieval_data)

    if not isinstance(records, list):
        records = list(records)
    total_tasks = len(records) if limit is None else min(len(records), limit)

    ema_task_time = None
    ema_alpha = 0.25

    with output_file.open("a", encoding="utf-8") as f, failed_file.open("a", encoding="utf-8") as ff, tqdm(total=total_tasks, desc="Overall Progress") as pbar:

        for idx, t in enumerate(records, 1):
            if limit is not None and idx > limit:
                break

            task_start = time.time()

            reset_token_stats()
            reset_llm_stats()
            reset_tool_stats()

            task_id = t.get("ID") or t.get("id", "")
            intent = t.get("Prompt") or t.get("prompt") or ""
            lang = (t.get("language") or "python").lower()
            lang_title = "Python" if lang.startswith("py") else lang.capitalize()
            framework = t.get("framework")
            cwe = extract_cwe(task_id)

            example, sim = (None, None)
            if retrieval_data:
                example, sim = retrieval_example(intent, cwe, task_id)

            if example is None:
                failed_tasks += 1
                ff.write(json.dumps(t, ensure_ascii=False) + "\n")
                ff.flush()
                print(f"[{idx}] skipped (no example)")
                pbar.update(1)
                continue

            print("\n=======================================")
            print(f"Task {idx}: {task_id}")
            print("========================================")

            prompt_llm = build_prompt(example, intent, lang_title)

            print(f"[Task {idx}] Generating code...")
            raw_resp = generate_with_retry(llm, prompt_llm)
            #print(f"[Task {idx}] Code generated.")
            print("\n[LLM RAW RESPONSE START]")
            print(raw_resp)
            print("[LLM RAW RESPONSE END]\n")
            
            code = extract_raw_code(raw_resp) if raw_resp else ""

            parsed = build_parsed_result(task_id, intent, lang, framework, example, sim, code, scan=None)

            f.write(json.dumps(parsed, ensure_ascii=False) + "\n")
            f.flush()

            stats_tok = get_token_stats()
            stats_llm = get_llm_stats()
            stats_tool = get_tool_stats()

            total_prompt_tokens += stats_tok.get("prompt_tokens", 0)
            total_completion_tokens += stats_tok.get("completion_tokens", 0)
            total_api_calls += stats_llm.get("api_calls", 0)
            total_llm_time += stats_llm.get("llm_time", 0.0)

            total_bandit_time += stats_tool.get("bandit_time", 0.0)
            total_semgrep_time += stats_tool.get("semgrep_time", 0.0)

            executed_tasks += 1

            task_elapsed = time.time() - task_start
            if ema_task_time is None:
                ema_task_time = task_elapsed
            else:
                ema_task_time = ema_alpha * task_elapsed + (1 - ema_alpha) * ema_task_time

            remaining = total_tasks - idx
            pbar.set_postfix_str(f"ETA {(ema_task_time * remaining)/60:.1f} min")
            pbar.update(1)

    elapsed = time.time() - start_time
    filter_jsonl(str(output_file))

    try:
        analyze_jsonl_batch(str(output_file))

        tool_stats = get_tool_stats()
        total_bandit_time = tool_stats.get("bandit_time", 0)
        total_semgrep_time = tool_stats.get("semgrep_time", 0)

        print("[Analyzer] Batch analysis completed.")
    except Exception as e:
        print(f"[Analyzer Error] {e}")

    try:
        metrics_dir = out_dir / "rate" / "vuln_density"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_txt = metrics_dir / f"{output_file.stem}_metrics.txt"
        save_metrics_result(
            str(output_file),
            str(metrics_txt),
            run_info={
                "dataset": dataset,
                "model": llm.model
            }
        )
        print(f"[Metrics Saved] {metrics_txt}")
    except Exception as e:
        print(f"[metrics-save] Error: {e}")

    save_experiment_summary(
        out_dir=str(out_dir),
        dataset=dataset,
        model=llm.model,
        technique=technique,
        total_tasks=executed_tasks,
        elapsed_time=elapsed,
        total_llm_time=total_llm_time,
        total_bandit_time=total_bandit_time,
        total_semgrep_time=total_semgrep_time,
        total_api_calls=total_api_calls,
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        jsonl_path=str(output_file),
    )
    return str(out_dir)
