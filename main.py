import argparse
import importlib
from pathlib import Path

from src.config import PATHS
from src.data.base import get_dataset, to_direct_records
from src.core.direct import GenerateCode
from src.core.planning_A import run_framework
from src.core.naive_secure import gen_code
from src.core.rci import rci_gen_code
from src.core.self_refine import SelfRefine_gen_code
from src.core.planning_B import PlanningB_gen_code
from src.core.one_shot import OneShot_gen_code
from src.core.CoT import cot_gen_code
from src.core.planning_C import planningC_gen_code
from src.core.planning_cot import coding

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


def run_planning_A(dataset: str, filename: str, limit: int, iterations: int):
    dataset_registered(dataset)
    fpath = abs_dataset_path(filename)
    ds = get_dataset(dataset, fpath)
    records = to_direct_records(ds)
    output_filename = Path(filename).name
    run_framework(records,dataset_name=f"{dataset}_planning_A{iterations}",
        limit=limit,save_plans=True,
        iterations=max(0, iterations),output_filename=output_filename,
    )


def run_nive_secure(dataset: str, filename: str, limit: int):
    dataset_registered(dataset)
    fpath = abs_dataset_path(filename)
    ds = get_dataset(dataset, fpath)
    records = to_direct_records(ds)
    output_filename = Path(filename).name
    gen_code(records, dataset_name=f"{dataset}_nive_secure", limit=limit, output_filename=output_filename)


def run_rci(dataset: str, filename: str, limit: int, iterations: int):
    dataset_registered(dataset)
    fpath = abs_dataset_path(filename)
    ds = get_dataset(dataset, fpath)
    records = to_direct_records(ds)
    output_filename = Path(filename).name
    rci_gen_code(
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
    SelfRefine_gen_code(records,dataset_name=f"{dataset}_selfrefine_iter{iterations}",
        limit=limit,iterations=iterations,output_filename=output_filename)


def run_planning_B(dataset: str, filename: str, limit: int):
    dataset_registered(dataset)
    fpath = abs_dataset_path(filename)
    ds = get_dataset(dataset, fpath)
    records = to_direct_records(ds)
    output_filename = Path(filename).name
    PlanningB_gen_code(
        records,
        dataset_name=f"{dataset}_planning_B",
        limit=limit,
        output_filename=output_filename,
    )


def run_one_shot(dataset: str, filename: str, limit: int, k: int):
    dataset_registered(dataset)
    fpath = abs_dataset_path(filename)
    ds = get_dataset(dataset, fpath)
    records = to_direct_records(ds)
    output_filename = Path(filename).name
    OneShot_gen_code(records,dataset_name=f"{dataset}_one_shot",
        limit=limit,output_filename=output_filename,k=k)

def run_cot(dataset: str, filename: str, limit: int):
    dataset_registered(dataset)
    fpath = abs_dataset_path(filename)
    ds = get_dataset(dataset, fpath)
    records = to_direct_records(ds)
    output_filename = Path(filename).name
    cot_gen_code(records, dataset_name=f"{dataset}_CoT", limit=limit, output_filename=output_filename)

def run_planning_C(dataset: str, filename: str, limit: int):
    dataset_registered(dataset)
    fpath = abs_dataset_path(filename)
    ds = get_dataset(dataset, fpath)
    records = to_direct_records(ds)
    output_filename = Path(filename).name
    planningC_gen_code(records, dataset_name=f"{dataset}_planning_C", limit=limit, output_filename=output_filename)

def run_planning_cot(dataset: str, filename: str, limit: int):
    dataset_registered(dataset)
    fpath = abs_dataset_path(filename)
    ds = get_dataset(dataset, fpath)
    records = to_direct_records(ds)
    output_filename = Path(filename).name
    coding(records, dataset_name=f"{dataset}_planning_cot", limit=limit, output_filename=output_filename)




def main():
    parser = argparse.ArgumentParser(description="Secure CodeGen CLI (Bandit-only)")
    parser.add_argument(
        "--mode",
        choices=["direct", "planning_A", "naive_secure", "rci", "self_refine", "planning_B", "one_shot","cot","planning_C","planning_cot"],
        required=True,
    )
    parser.add_argument("--dataset", choices=["securityeval", "llmseceval", "sallm"], required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=0)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    if args.mode == "direct":
        run_direct(args.dataset, args.file, args.limit)
    elif args.mode == "planning_A":
        run_planning_A(args.dataset, args.file, args.limit, args.iterations)
    elif args.mode == "naive_secure":
        run_nive_secure(args.dataset, args.file, args.limit)
    elif args.mode == "rci":
        run_rci(args.dataset, args.file, args.limit, args.iterations)
    elif args.mode == "self_refine":
        run_self_refine(args.dataset, args.file, args.limit, args.iterations)
    elif args.mode == "planning_B":
        run_planning_B(args.dataset, args.file, args.limit)
    elif args.mode == "one_shot":
        run_one_shot(args.dataset, args.file, args.limit, args.k)
    elif args.mode == "cot":
        run_cot(args.dataset, args.file, args.limit)
    elif args.mode == "planning_C":
        run_planning_C(args.dataset, args.file, args.limit)
    elif args.mode == "planning_cot":
        run_planning_cot(args.dataset, args.file, args.limit)


if __name__ == "__main__":
    main()


# python main.py --mode direct --dataset securityeval --file SecurityEval.jsonl
#  python main.py --mode direct --dataset sallm --file SALLM.jsonl

