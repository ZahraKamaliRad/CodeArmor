import argparse
import importlib
from pathlib import Path

from src.config import PATHS
from src.data.base import get_dataset, to_direct_records
from src.core.direct import GenerateCode
from src.core.PCAR import run_framework
from src.core.prefix import pe03a
from src.core.rci import rci_tecniqu
from src.core.self_refine import Self_Refine

def dataset_registered(name: str):
    module_map = {
        "securityeval": "src.data.securityeval",
        "llmseceval": "src.data.llmseceval",
        "sallm": "src.data.sallm",
    }
    mod = module_map.get(name.lower())
    if not mod:
        raise KeyError(f"Unknown dataset: {name}. Expected one of {list(module_map)}")
    importlib.import_module(mod)

def abs_dataset_path(filename: str) -> Path:
    fpath = Path(filename)
    if not fpath.is_absolute():
        fpath = PATHS.datasets / filename

    if not fpath.exists():
        raise FileNotFoundError(f"Dataset file not found: {fpath}")
    return fpath

def run_direct(dataset: str, filename: str, limit: int):
    dataset_registered(dataset)
    fpath = abs_dataset_path(filename)
    ds = get_dataset(dataset, fpath)
    records = to_direct_records(ds)
    output_filename = Path(filename).name
    GenerateCode(records, dataset_name=f"{dataset}_direct", limit=limit, output_filename=output_filename)

def run_planning(dataset: str, filename: str, limit: int, iterations: int):
    dataset_registered(dataset)
    fpath = abs_dataset_path(filename)
    ds = get_dataset(dataset, fpath)
    records = to_direct_records(ds)
    output_filename = Path(filename).name
    run_framework(
        records,
        dataset_name=f"{dataset}_planning_iter{iterations}",
        limit=limit,
        save_plans=True,
        iterations=max(0, iterations),
        output_filename=output_filename,
    )

def run_pe03a(dataset: str, filename: str, limit: int):
    dataset_registered(dataset)
    fpath = abs_dataset_path(filename)
    ds = get_dataset(dataset, fpath)
    records = to_direct_records(ds)
    output_filename = Path(filename).name
    pe03a(records, dataset_name=f"{dataset}_prefix", limit=limit, output_filename=output_filename)

def run_rci(dataset: str, filename: str, limit: int, iterations: int):
    dataset_registered(dataset)
    fpath = abs_dataset_path(filename)
    ds = get_dataset(dataset, fpath)
    records = to_direct_records(ds)
    output_filename = Path(filename).name
    rci_tecniqu(
        records,
        dataset_name=f"{dataset}_rci_iter{iterations}",
        limit=limit,
        iterations=iterations,
        output_filename=output_filename,
    )

def run_self_refine(dataset: str, filename: str, limit: int, iterations: int):
    dataset_registered(dataset)
    fpath = abs_dataset_path(filename)
    ds = get_dataset(dataset, fpath)
    records = to_direct_records(ds)
    output_filename = Path(filename).name
    Self_Refine(
        records,
        dataset_name=f"{dataset}_selfrefine_iter{iterations}",
        limit=limit,
        iterations=iterations,
        output_filename=output_filename,
    )



def main():
    parser = argparse.ArgumentParser(description="Secure CodeGen CLI (Bandit-only)")
    parser.add_argument("--mode", choices=["direct", "planning", "prefix", "rci","self_refine"], required=True)
    parser.add_argument("--dataset", choices=["securityeval", "llmseceval", "sallm"], required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=0)
    args = parser.parse_args()

    if args.mode == "direct":
        run_direct(args.dataset, args.file, args.limit)
    elif args.mode == "planning":
        run_planning(args.dataset, args.file, args.limit, args.iterations)
    elif args.mode == "prefix":
        run_pe03a(args.dataset, args.file, args.limit)
    elif args.mode == "rci":
        run_rci(args.dataset, args.file, args.limit, args.iterations)
    elif args.mode == "self_refine":
        run_self_refine(args.dataset, args.file, args.limit, args.iterations)


if __name__ == "__main__":
    main()

# python main.py --mode direct --dataset securityeval --file securityeval.jsonl
# python main.py --mode prefix --dataset securityeval --file SecurityEval.jsonl
# python main.py --mode rci --dataset securityeval --file SecurityEval.jsonl --iterations 5 --limit 5
#  python main.py --mode planning --dataset securityeval --file SecurityEval.jsonl --iterations 3 --limit 3

