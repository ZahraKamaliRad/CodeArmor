from pathlib import Path
from typing import Iterable, Dict, Any, List
import orjson
import os, csv

def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("rb") as f:
        for line in f:
            if not line.strip():
                continue
            yield orjson.loads(line)

def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for row in rows:
            f.write(orjson.dumps(row))
            f.write(b"\n")

def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def csv_log(run_out_dir: str, task_id: str, iter_idx: int, language: str,
            issues: List[Dict[str, Any]]) -> None:
  
    os.makedirs(run_out_dir, exist_ok=True)
    path = os.path.join(run_out_dir, "Effect_Of_Refinment.csv")
    exists = os.path.exists(path)

    headers = ["task_id","iter","status","language","total",
               "rule_id","cwe","line","severity","issue"]

    issues = issues or []
    total = len(issues)
    status = "Secure" if total == 0 else "Vulnerable"

    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        if not exists:
            w.writeheader()

        if total == 0:
            w.writerow({
                "task_id": task_id, "iter": iter_idx, "status": status,
                "language": language or "", "total": 0,
                "rule_id": "", "cwe": "", "line": "", "severity": "", "issue": ""
            })
            return

        first = issues[0]
        w.writerow({
            "task_id": task_id,
            "iter": iter_idx,
            "status": status,
            "language": language or "",
            "total": total,
            "rule_id": first.get("rule_id") or "",
            "cwe": first.get("cwe") if first.get("cwe") is not None else "",
            "line": int(first.get("line") or 0),
            "severity": (first.get("severity") or "").upper(),
            "issue": (first.get("message") or "")[:250],
        })

        for it in issues[1:]:
            w.writerow({
                "task_id": task_id,
                "iter": iter_idx,
                "status": "",
                "language": "",
                "total": "",
                "rule_id": it.get("rule_id") or "",
                "cwe": it.get("cwe") if it.get("cwe") is not None else "",
                "line": int(it.get("line") or 0),
                "severity": (it.get("severity") or "").upper(),
                "issue": (it.get("message") or "")[:250],
            })
