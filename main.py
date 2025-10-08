import argparse
import importlib
from pathlib import Path

from src.config import PATHS
from src.data.base import get_dataset, to_direct_records
from src.core.direct import GenerateCode
from src.core.PCAR import run_framework 
from src.core.prefix import pe03a 
from src.core.rci import rci_tecniqu

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

def run_direct(dataset: str, filename: str, limit: int):
    dataset_registered(dataset)

    fpath = Path(filename)
    if not fpath.is_absolute():
        fpath = PATHS.datasets / filename

    if not fpath.exists():
        raise FileNotFoundError(f"Dataset file not found: {fpath}")

    ds = get_dataset(dataset, fpath)
    records = to_direct_records(ds)
    GenerateCode(records, dataset_name=f"{dataset}_direct", limit=limit)

def run_planning(dataset: str, filename: str, limit: int, repairs: int):
    dataset_registered(dataset)

    fpath = Path(filename)
    if not fpath.is_absolute():
        fpath = PATHS.datasets / filename
    if not fpath.exists():
        raise FileNotFoundError(f"Dataset file not found: {fpath}")

    ds = get_dataset(dataset, fpath)
    records = to_direct_records(ds)
    run_framework(
        records,
        dataset_name=f"{dataset}_planning",
        limit=limit,
        save_plans=True,
        max_repair_rounds=max(0, repairs)  
    )

def run_pe03a(dataset: str, filename: str, limit: int):
    dataset_registered(dataset)
    fpath = Path(filename)
    if not fpath.is_absolute():
        fpath = PATHS.datasets / filename
    if not fpath.exists():
        raise FileNotFoundError(f"Dataset file not found: {fpath}")
    ds = get_dataset(dataset, fpath)
    records = to_direct_records(ds)
    pe03a(records, dataset_name=f"{dataset}_pe03a", limit=limit)

def run_rci(dataset: str, filename: str, limit: int, iterations: int):
    dataset_registered(dataset)
    fpath = Path(filename)
    if not fpath.is_absolute():
        fpath = PATHS.datasets / filename
    if not fpath.exists():
        raise FileNotFoundError(f"Dataset file not found: {fpath}")
    ds = get_dataset(dataset, fpath)
    records = to_direct_records(ds)
    rci_tecniqu(records, dataset_name=f"{dataset}_rci_iter{iterations}", limit=limit, iterations=iterations)

def main():
    parser = argparse.ArgumentParser(description="Secure CodeGen CLI")
    parser.add_argument("--mode", choices=["direct", "planning", "pe03a", "rci"], required=True)
    parser.add_argument("--dataset", choices=["securityeval", "llmseceval", "sallm"], required=True)
    parser.add_argument("--file", required=True, help="dataset file name inside datasets/")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=1, help="RCI review/improve rounds (only for --mode rci)")
    parser.add_argument("--repairs", type=int, default=1, help="max repair rounds for planning mode (default: 1)")
    args = parser.parse_args()

    if args.mode == "direct":
        run_direct(args.dataset, args.file, args.limit)
    elif args.mode == "planning":
        run_planning(args.dataset, args.file, args.limit, args.repairs)
    elif args.mode == "pe03a":
        run_pe03a(args.dataset, args.file, args.limit)
    elif args.mode == "rci":
        run_rci(args.dataset, args.file, args.limit, args.iterations)

if __name__ == "__main__":
    main()

# Examples:
# python main.py --mode direct   --dataset securityeval --file SecurityEval.jsonl
# python main.py --mode planning --dataset securityeval --file SecurityEval.jsonl --repairs 1
# python main.py --mode planning --dataset sallm --file SALLM.jsonl --repairs 1
# python main.py --mode planning --dataset llmseceval  --file LLMSecEval.
# python main.py --mode pe03a --dataset securityeval --file SecurityEval.jsonl --limit 10
# python main.py --mode rci --dataset securityeval --file SecurityEval.jsonl --limit 10 --iterations 3
# python main.py --mode rci --dataset sallm --file SALLM.jsonl --iterations 1
