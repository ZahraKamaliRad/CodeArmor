from __future__ import annotations
from pathlib import Path
import json
import re
import time
import random
import numpy as np
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .openai_client import LLMClient, sanitize_model_name
from src.paths import PATHS
from ..utils.metrics import save_metrics_result
from .analyzer import analyze_code_split
from ..utils.tokens import get_token_stats
from ..utils.text_utils import strip_markdown_fences


# RETRIEVAL_DATASET_PATH = Path(r"E:\secure_codegenration\src\retrieval\data\retrieval_examples.jsonl")
# EMBEDDING_MODEL_PATH = r"E:\all-MiniLM-L6-v2"
load_dotenv()

RETRIEVAL_DATASET_PATH = Path(os.getenv("RETRIEVAL_DATASET_PATH"))
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH")

embedding_model = None
retrieval_index = {}


ONE_SHOT_PROMPT_TEMPLATE = """You are a secure code generator.

Below is an example of a security-related coding task and its secure implementation:

### Example Task:
{Example_Prompt}

### Example Secure Code:
{Example_Secure_Code}

Now complete the following task. Return only raw source code (no explanations, no markdown, no comments outside code).

### Task:
{Prompt}

Language: {Language}
"""

ZERO_SHOT_PROMPT_TEMPLATE = """You are a secure code generator.

Complete the following task. Return only raw source code (no explanations, no markdown, no comments outside code).

### Task:
{Prompt}

Language: {Language}
"""


def generate_with_retry(llm, prompt, retries=3):
    for attempt in range(retries):
        try:
            return llm.generate_text(prompt).strip()
        except Exception as e:
            msg = str(e)
            if "502" in msg or "Bad Gateway" in msg:
                time.sleep(2 ** attempt)
            else:
                break
    return None


def build_prompt(example, intent, lang_title):
    if example:
        return (
            ONE_SHOT_PROMPT_TEMPLATE.format(
                Example_Prompt=example.get("prompt", ""),
                Example_Secure_Code=example.get("secure_code", ""),
                Prompt=intent,
                Language=lang_title
            ),
            "one_shot"
        )

    return (
        ZERO_SHOT_PROMPT_TEMPLATE.format(
            Prompt=intent,
            Language=lang_title
        ),
        "zero_shot"
    )


def build_parsed_result(task_id, intent, lang, framework, retrieval_mode,
                        retrieval_mode_label, example, similarity_score,
                        code, scan=None):

    if scan is None:
        return {
            "task": task_id,
            "intent": intent,
            "language": lang,
            "framework": framework,
            "retrieval_strategy": retrieval_mode,
            "retrieval_mode": retrieval_mode_label,
            "retrieval_example_id": example["id"] if example else None,
            "retrieval_similarity": similarity_score,
            "code": "",
            "loc": 0,
            "bandit_result": {"secure": False, "issues": []},
            "semgrep_result": {"secure": False, "issues": []},
        }

    get_scan = scan.get
    bandit = get_scan("bandit_result")
    semgrep = get_scan("semgrep_result")

    return {
        "task": task_id,
        "intent": intent,
        "language": lang,
        "framework": framework,
        "retrieval_strategy": retrieval_mode,
        "retrieval_mode": retrieval_mode_label,
        "retrieval_example_id": example["id"] if example else None,
        "retrieval_similarity": similarity_score,
        "code": code,
        "loc": int(get_scan("loc") or 0),
        "bandit_result": bandit,
        "semgrep_result": semgrep
    }


def load_retrieval_dataset(path: Path) -> list[dict]:
    examples = []
    if not path.exists():
        return examples

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return examples


def extract_cwe(task_id: str) -> str | None:
    match = re.search(r"cwe[-_]?(\d+)", task_id, re.IGNORECASE)
    return match.group(1) if match else None


def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)
    return embedding_model


def build_retrieval_index(retrieval_data: list[dict]):
    global retrieval_index
    if retrieval_index:
        return retrieval_index

    model = get_embedding_model()
    groups = {}
    prompts = {}

    for ex in retrieval_data:
        cwe = ex.get("cwe")
        if not cwe:
            continue
        groups.setdefault(cwe, []).append(ex)
        prompts.setdefault(cwe, []).append(ex["prompt"])

    for cwe, prompt_list in prompts.items():
        embeddings = np.array(model.encode(prompt_list, normalize_embeddings=True))
        retrieval_index[cwe] = {
            "examples": groups[cwe],
            "embeddings": embeddings
        }

    return retrieval_index


def random_example(retrieval_data: list[dict], cwe: str, exclude_id: str):
    candidates = [
        ex for ex in retrieval_data
        if ex.get("cwe") == cwe and ex.get("id") != exclude_id
    ]
    return random.choice(candidates) if candidates else None


def similar_example(cwe: str, query_prompt: str, exclude_id: str):
    index = retrieval_index.get(cwe)
    if not index:
        return None, None

    examples = index["examples"]
    embeddings = index["embeddings"]

    model = get_embedding_model()
    query_emb = model.encode([query_prompt], normalize_embeddings=True)

    sims = cosine_similarity(query_emb, embeddings)[0]

    best_idx = None
    best_score = -1

    for i, sim in enumerate(sims):
        if examples[i]["id"] == exclude_id:
            continue
        if sim > best_score:
            best_score = float(sim)
            best_idx = i

    if best_idx is None:
        return None, None

    return examples[best_idx], best_score


def OneShot_gen_code(records, dataset: str, technique: str, limit: int | None = None,
                    retrieval_mode: str = "similarity"):

    start_time = time.time()

    llm = LLMClient()
    model_tag = sanitize_model_name(llm.model)

    base_dir = PATHS.run_dir(dataset=dataset, model_name=model_tag, technique=technique)
    base_dir.mkdir(parents=True, exist_ok=True)

    one_dir = base_dir / "one_shot"
    zero_dir = base_dir / "zero_shot"
    one_dir.mkdir(parents=True, exist_ok=True)
    zero_dir.mkdir(parents=True, exist_ok=True)

    one_jsonl = one_dir / "one_shot.jsonl"
    zero_jsonl = zero_dir / "zero_shot.jsonl"

    retrieval_data = load_retrieval_dataset(RETRIEVAL_DATASET_PATH)

    if retrieval_mode == "similarity":
        build_retrieval_index(retrieval_data)

    print(f"Retrieval strategy: {retrieval_mode}")
    print(f"Total tasks: {len(records)}")

    one_f = one_jsonl.open("a", encoding="utf-8")
    zero_f = zero_jsonl.open("a", encoding="utf-8")

    for idx, t in enumerate(records, 1):
        if limit is not None and idx > limit:
            break

        lang_raw = (t.get("language") or "python").strip().lower()
        if lang_raw.startswith("py"):
            lang_title, lang = "Python", "python"

        intent = t.get("Prompt", "") or ""
        task_id = t.get("ID") or t.get("id", "")

        print(f"[DATASET LINE {idx}] id={task_id}")

        cwe = extract_cwe(task_id)

        example = None
        similarity_score = None

        if cwe and retrieval_data:
            if retrieval_mode == "random":
                example = random_example(retrieval_data, cwe, task_id)
            elif retrieval_mode == "similarity":
                example, similarity_score = similar_example(cwe, intent, task_id)

        prompt_llm, retrieval_mode_label = build_prompt(example, intent, lang_title)

        raw_resp = generate_with_retry(llm, prompt_llm)

        if raw_resp is None:
            parsed = build_parsed_result(
                task_id, intent, lang, t.get("framework"),
                retrieval_mode, retrieval_mode_label,
                example, similarity_score,
                code=""
            )
        else:
            raw_resp = strip_markdown_fences(raw_resp)
            tmpname = Path(str(task_id or "snippet")).stem

            scan = analyze_code_split(raw_resp, lang, tmpname=tmpname)
            mode = retrieval_mode_label

            bandit_count = len((scan.get("bandit_result") or {}).get("issues") or [])
            semgrep_count = len((scan.get("semgrep_result") or {}).get("issues") or [])
            print(f"[{mode}] task={task_id} bandit={bandit_count} semgrep={semgrep_count}")

            parsed = build_parsed_result(
                task_id, intent, lang, t.get("framework"),
                retrieval_mode, retrieval_mode_label,
                example, similarity_score,
                code=raw_resp,
                scan=scan
            )

        use_f = one_f if retrieval_mode_label == "one_shot" else zero_f
        use_f.write(json.dumps(parsed, ensure_ascii=False) + "\n")
        use_f.flush()

    one_f.close()
    zero_f.close()

    try:
        one_m_dir = one_dir / "rate" / "vuln_density"
        zero_m_dir = zero_dir / "rate" / "vuln_density"
        one_m_dir.mkdir(parents=True, exist_ok=True)
        zero_m_dir.mkdir(parents=True, exist_ok=True)

        one_m_file = one_m_dir / "one_shot.txt"
        zero_m_file = zero_m_dir / "zero_shot.txt"

        elapsed = time.time() - start_time
        token_stats = get_token_stats()

        save_metrics_result(
            str(one_jsonl),
            str(one_m_file),
            run_info={
                "dataset": dataset,
                "model": llm.model,
                "prompt_tokens": token_stats.get("prompt_tokens", 0),
                "completion_tokens": token_stats.get("completion_tokens", 0),
                "total_tokens": token_stats.get("total_tokens", 0),
                "runtime_seconds": elapsed,
                "retrieval_strategy": retrieval_mode
            }
        )

        save_metrics_result(
            str(zero_jsonl),
            str(zero_m_file),
            run_info={
                "dataset": dataset,
                "model": llm.model,
                "prompt_tokens": token_stats.get("prompt_tokens", 0),
                "completion_tokens": token_stats.get("completion_tokens", 0),
                "total_tokens": token_stats.get("total_tokens", 0),
                "runtime_seconds": elapsed,
                "retrieval_strategy": retrieval_mode
            }
        )

    except Exception:
        pass

    return str(base_dir)
