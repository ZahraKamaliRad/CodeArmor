import argparse
import importlib
from pathlib import Path

from src.config import PATHS
from src.data.base import get_dataset, to_direct_records
from src.core.direct import GenerateCode
from src.core.planning import GeneratePlanAndCode   

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

def run_planning(dataset: str, filename: str, limit: int):
    dataset_registered(dataset)

    fpath = Path(filename)
    if not fpath.is_absolute():
        fpath = PATHS.datasets / filename

    if not fpath.exists():
        raise FileNotFoundError(f"Dataset file not found: {fpath}")

    ds = get_dataset(dataset, fpath)
    records = to_direct_records(ds)
    GeneratePlanAndCode(records, dataset_name=f"{dataset}_planning", limit=limit, save_plans=True)

def main():
    parser = argparse.ArgumentParser(description="Secure CodeGen CLI")
    parser.add_argument("--mode", choices=["direct", "planning"], required=True)  # ← تغییر لیست
    parser.add_argument("--dataset", choices=["securityeval", "llmseceval", "sallm"], required=True)
    parser.add_argument("--file", required=True, help="dataset file name inside datasets/")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.mode == "direct":
        run_direct(args.dataset, args.file, args.limit)
    elif args.mode == "planning":
        run_planning(args.dataset, args.file, args.limit)

if __name__ == "__main__":
    main()

# Examples:
# python main.py --mode direct   --dataset securityeval --file SecurityEval.jsonl
# python main.py --mode planning --dataset securityeval --file SecurityEval.jsonl
# python main.py --mode planning --dataset sallm --file SALLM.jsonl
# python main.py --mode planning --dataset llmseceval  --file LLMSecEval.jsonl
