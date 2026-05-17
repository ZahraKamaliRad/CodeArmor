import argparse
import inspect
import importlib
from pathlib import Path
from src.paths import PATHS
from src.data.base import get_dataset, to_direct_records
from src.core.direct import GenerateCode
from src.core.prefix import gen_code
from src.core.rci import rci_gen_code
from src.core.self_refine import self_refine_gen_code
from src.core.one_shot import OneShot_gen_code
from src.core.Few_shot_CoT import few_shot_cot_gen_code
from src.core.planning_few_shot import planning_few_shot_gen_code
from src.core.persona import persona_gen_code
from src.config_loader import override_config
from src.core.planning_zero_shot import planning_rci_gen_code
from src.core.zero_shot_CoT import zero_shot_cot_gen_code



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


def abs_dataset_path(filename: str) -> Path:
    p = Path(filename)
    if not p.is_absolute():
        p = PATHS.datasets / filename
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {p}")
    return p


def tech(name: str, iterations: int):
    return f"{name}_iter{iterations}" if iterations > 0 else name



def load_records(dataset: str, filename: str):
    dataset_registered(dataset)
    fpath = abs_dataset_path(filename)
    data = get_dataset(dataset, fpath)
    records = to_direct_records(data)
    output_filename = Path(filename).name
    return records, output_filename


def run_direct(dataset: str, file: str, limit: int):
    records, out = load_records(dataset, file)
    GenerateCode(records, dataset, "direct", limit=limit, output_filename=out)


def run_naive_secure(dataset: str, file: str, limit: int):
    records, out = load_records(dataset, file)
    gen_code(records, dataset, "prefix", limit=limit, output_filename=out)


def run_rci(dataset: str, file: str, limit: int, iterations: int=1):
    records, out = load_records(dataset, file)
    rci_gen_code(records, dataset, tech("rci", iterations),
                 limit=limit, iterations=iterations, output_filename=out)


def run_self_refine(dataset: str, file: str, limit: int, iterations: int=1):
    records, out = load_records(dataset, file)
    self_refine_gen_code(records, dataset, tech("self_refine", iterations),
                        limit=limit, iterations=iterations, output_filename=out)


def run_planning_few_shot(dataset: str, file: str, limit: int, iterations: int=0):
    records, out = load_records(dataset, file)
    planning_few_shot_gen_code(records, dataset, tech("planning_few_shot", iterations),
                          limit=limit, iterations=iterations, output_filename=out)


def run_one_shot(dataset: str, file: str, limit: int):
    records, _ = load_records(dataset, file)
    OneShot_gen_code(records, dataset, "one_shot", limit=limit)


def run_cot(dataset: str, file: str, limit: int):
    records, out = load_records(dataset, file)
    few_shot_cot_gen_code(records, dataset, "few_shot_cot", limit=limit, output_filename=out)

def run_zero_shot_cot(dataset: str, file: str, limit: int):
    records, out = load_records(dataset, file)
    zero_shot_cot_gen_code(records, dataset, "zero_shot_cot", limit=limit, output_filename=out)


def run_persona(dataset: str, file: str, limit: int):
    records, out = load_records(dataset, file)
    persona_gen_code(records, dataset, "persona", limit=limit, output_filename=out)
    

def run_planning_rci(dataset: str, file: str, limit: int, iterations: int=0):
    records, out = load_records(dataset, file)
    planning_rci_gen_code(records, dataset, tech("planning_zero_shot", iterations),
                          limit=limit, iterations=iterations, output_filename=out)



dispatch = {
    "direct": run_direct,
    "prefix": run_naive_secure,
    "rci": run_rci,
    "self_refine": run_self_refine,
    "planning_few_shot": run_planning_few_shot,
    "one_shot": run_one_shot,
    "few_shot_cot": run_cot,
    "persona": run_persona,
    "planning_zero_shot": run_planning_rci,
    "zero_shot_cot": run_zero_shot_cot
}


def smart_call(func, args_dict):
    sig = inspect.signature(func)
    needed = {
        name: args_dict[name]
        for name in sig.parameters
        if name in args_dict
    }
    return func(**needed)


def main():
    parser = argparse.ArgumentParser(description="Secure CodeGen CLI")

    parser.add_argument("--mode", required=True,
                        choices=list(dispatch.keys()))
    parser.add_argument("--dataset", required=True,
                        choices=["securityeval", "llmseceval", "sallm", "humaneval"])
    parser.add_argument("--file", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--provider", choices=["api", "local"], required=True)

    args = parser.parse_args()

    override_config({"llm": {"provider": args.provider}})

    args_dict = vars(args)

    func = dispatch[args.mode]

    if args.iterations is None:
        sig = inspect.signature(func)
        if "iterations" in sig.parameters:
            default_iter = sig.parameters["iterations"].default
            if default_iter is not inspect._empty:
                args.iterations = default_iter

    args_dict = vars(args)
    smart_call(func, args_dict)


if __name__ == "__main__":
    main()

# python main.py --mode direct --dataset securityeval --file SecurityEval.jsonl
#  python main.py --mode direct --dataset sallm --file SALLM.jsonl

