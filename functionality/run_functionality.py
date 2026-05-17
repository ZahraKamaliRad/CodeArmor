#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import importlib
import json
import inspect

import src.core.analyzer as analyzer
import src.utils.metrics as metrics
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _noop_analyzer(*args, **kwargs):
    return None

def _noop_metrics(*args, **kwargs):
    return None

analyzer.analyze_jsonl_batch = _noop_analyzer
metrics.save_metrics_result = _noop_metrics

from src.data.base import get_dataset, to_direct_records
from src.core.direct import GenerateCode
from src.core.rci import rci_gen_code
from src.core.self_refine import self_refine_gen_code
from src.core.prefix import gen_code
from src.core.one_shot import OneShot_gen_code
from src.core.Few_shot_CoT import few_shot_cot_gen_code
from src.core.planning_few_shot import planning_few_shot_gen_code
from src.core.persona import persona_gen_code
from src.core.planning_zero_shot import planning_rci_gen_code
from src.core.zero_shot_CoT import zero_shot_cot_gen_code
from src.paths import PATHS


def dataset_registered(name: str):
    module_map = {
        "securityeval": "src.data.securityeval",
        "llmseceval": "src.data.llmseceval",
        "sallm": "src.data.sallm",
        "humaneval": "src.data.humaneval"
    }
    mod = module_map.get(name.lower())
    if not mod:
        raise KeyError(f"Unknown dataset: {name}")
    importlib.import_module(mod)


def load_records(dataset_name: str, filename: str):
    fpath = Path(filename)
    if not fpath.is_absolute():
        fpath = PATHS.datasets / filename
    data = get_dataset(dataset_name, fpath)
    records = to_direct_records(data)
    return records


def enrich_records(records):
    enriched = []
    for r in records:
        meta = r.get("metadata") or {}
        enriched.append({
            **r,
            "test_case": r.get("test") or meta.get("test"),
            "entry_point": r.get("entry_point") or meta.get("entry_point"),
        })
    return enriched

def attach_tests_to_output(records, output_path):
    record_map = {r.get("task_id"): r for r in records}

    lines = []
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            task_id = obj.get("task_id")
            if task_id in record_map:
                obj["test_case"] = record_map[task_id].get("test_case")
                obj["entry_point"] = record_map[task_id].get("entry_point")
            lines.append(obj)

    with open(output_path, "w", encoding="utf-8") as f:
        for obj in lines:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


DISPATCH = {
    "direct": GenerateCode,
    "rci": rci_gen_code,
    "self_refine": self_refine_gen_code,
    "prefix": gen_code,
    "one_shot": OneShot_gen_code,
    "few_shot_cot": few_shot_cot_gen_code,
    "planning_few_shot": planning_few_shot_gen_code,
    "persona": persona_gen_code,
    "planning_zero_shot": planning_rci_gen_code,
    "zero_shot_cot": zero_shot_cot_gen_code,
}


def smart_call(func, kwargs):
    sig = inspect.signature(func)
    needed = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return func(**needed)


def run(mode, dataset, file, limit=None, iterations=None, provider=None):
    dataset_registered(dataset)
    records = load_records(dataset, file)
    records = enrich_records(records)

    fn = DISPATCH[mode]
    output_path = OUTPUT_DIR / "output_functionality.jsonl"

    print(f"\n=== FUNCTIONALITY MODE | {mode} ===\n")

    kwargs = {
        "records": records,
        "dataset": dataset,
        "technique": "functionality",
        "limit": limit,

        "output_file": str(output_path),
        "output_filename": str(output_path),
        "output_path": str(output_path),

        "enable_security_analysis": False,
        "iterations": iterations,
        "provider": provider,
    }

    if mode in ["rci", "self_refine", "planning_zero_shot"] and kwargs["iterations"] is None:
        kwargs["iterations"] = 1

    smart_call(fn, kwargs)

    if not output_path.exists():
        raise FileNotFoundError(f"Generator did not create output file: {output_path}")
    
    attach_tests_to_output(records, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run functionality pipeline on all generators")
    parser.add_argument("--mode", required=True, choices=list(DISPATCH.keys()))
    parser.add_argument("--dataset", required=True, choices=["securityeval", "llmseceval", "sallm", "humaneval"])
    parser.add_argument("--file", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--provider", choices=["api", "local"], required=True)

    args = parser.parse_args()
    run(args.mode, args.dataset, args.file, args.limit, args.iterations, args.provider)
