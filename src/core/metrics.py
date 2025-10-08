import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Any, Iterable

NumberFrac = Tuple[int, int, float]

def safe_percent(num: int, den: int) -> float:
    return (num / den * 100.0) if den else 0.0

def first_present(d: Dict[str, Any], keys: Iterable[str]):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None

class MetricsCalculator:
    CODE_KEYS = (
    "final_code",   
    "code" 
)
    def __init__(self, jsonl_path: str):
        self.path = Path(jsonl_path)
        if not self.path.exists():
            raise FileNotFoundError(f"JSONL file not found: {self.path}")
        self.records: List[Dict[str, Any]] = []
        self.read_jsonl(self.path)

        self.per_item: List[Dict[str, Any]] = []
        for rec in self.records:
            secure, issues = self.extract_secure_issues(rec)
            loc = self.extract_loc(rec)
            self.per_item.append({
                "secure": bool(secure),
                "issue_count": int(len(issues)),
                "loc": int(loc),
            })

        self.n_items = len(self.per_item)
        self.total_issues = sum(x["issue_count"] for x in self.per_item)
        self.total_locs = sum(x["loc"] for x in self.per_item)
        self.secure_items = sum(1 for x in self.per_item if x["secure"])
        self.items_with_issue = sum(1 for x in self.per_item if x["issue_count"] > 0)


    def accuracy(self) -> NumberFrac:
        return (self.secure_items, self.n_items, safe_percent(self.secure_items, self.n_items))

    def vulnerability_rate(self) -> NumberFrac:
        return (self.total_issues, self.n_items, safe_percent(self.total_issues, self.n_items))

    def vulnerability_at_1(self) -> NumberFrac:
        return (self.items_with_issue, self.n_items, safe_percent(self.items_with_issue, self.n_items))

    def density(self) -> NumberFrac:
        return (self.total_issues, self.total_locs, safe_percent(self.total_issues, self.total_locs))

    def summary(self) -> Dict[str, Any]:
        acc = self.accuracy()
        vr = self.vulnerability_rate()
        v1 = self.vulnerability_at_1()
        den = self.density()

        return {
            "dataset": self.path.stem,
            "items": self.n_items,
            "secure_items": self.secure_items,
            "items_with_issue": self.items_with_issue,
            "total_issues": self.total_issues,
            "total_loc": self.total_locs,
            "accuracy_frac": f"{acc[0]}/{acc[1]}",
            "accuracy_percent": round(acc[2], 2),
            "vuln_rate_frac": f"{vr[0]}/{vr[1]}",
            "vuln_rate_percent": round(vr[2], 2),
            "vuln@1_frac": f"{v1[0]}/{v1[1]}",
            "vuln@1_percent": round(v1[2], 2),
            "density_frac": f"'{den[0]}/{den[1]}'",
            "density_percent": round(den[2], 4),
        }


    def read_jsonl(self, path: Path):
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {i} of {path}: {e}") from e
                self.records.append(obj)

    def extract_secure_issues(self, rec: Dict[str, Any]) -> Tuple[bool, List[Any]]:
        secure = rec.get("secure")
        issues = rec.get("issues")

        scan = rec.get("bandit_result") or rec.get("scan_result") or rec.get("security_scan")
        if isinstance(scan, dict):
            if secure is None:
                secure = scan.get("secure")
            if issues is None:
                issues = scan.get("issues")

        if issues is None:
            issues = []
        elif isinstance(issues, int):
            issues = [None] * max(0, issues)
        elif isinstance(issues, list):
            pass
        else:
            if isinstance(issues, dict) and isinstance(issues.get("count"), int):
                issues = [None] * max(0, int(issues["count"]))
            else:
                issues = []

        if secure is None:
            secure = (len(issues) == 0)

        return bool(secure), list(issues)

    def extract_loc(self, rec: Dict[str, Any]) -> int:
        loc = first_present(rec, ("loc", "line_count", "num_loc", "n_loc"))
        if isinstance(loc, int):
            return max(0, loc)

        ci = rec.get("code_info")
        if isinstance(ci, dict):
            loc = first_present(ci, ("loc", "line_count", "n_loc"))
            if isinstance(loc, int):
                return max(0, loc)
            code_blob = first_present(ci, self.CODE_KEYS)
            if isinstance(code_blob, str):
                return self.count_nonempty_lines(code_blob)

        code_blob = first_present(rec, self.CODE_KEYS)
        if isinstance(code_blob, str):
            return self.count_nonempty_lines(code_blob)

        return 0

    @staticmethod
    def count_nonempty_lines(s: str) -> int:
        return sum(1 for ln in s.splitlines() if ln.strip())
    
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Compute security metrics from JSONL results.")
    ap.add_argument("files", nargs="+", help="One or more JSONL result files.")
    ap.add_argument("--csv", help="Path to save results as CSV (optional).")
    args = ap.parse_args()

    all_results: List[Dict[str, Any]] = []
    for f in args.files:
        mc = MetricsCalculator(f)
        summary = mc.summary()
        all_results.append(summary)

    for res in all_results:
        print(res)

    if args.csv and all_results:
        fieldnames = list(all_results[0].keys())
        with open(args.csv, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults saved to {args.csv}")

        
