import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


CODE_KEYS = ("final_code", "code")


def safe_div(num: float, den: float) -> float:
    return (num / den) if den else 0.0


def first_present(d: Dict[str, Any], keys: Iterable[str]):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def count_nonempty_lines(s: str) -> int:
    return sum(1 for ln in s.splitlines() if ln.strip())


def _issues_to_list(issues: Any) -> List[Any]:
    if issues is None:
        return []
    if isinstance(issues, int):
        return [None] * max(0, issues)
    if isinstance(issues, list):
        return issues
    if isinstance(issues, dict) and isinstance(issues.get("count"), int):
        return [None] * max(0, int(issues["count"]))
    return []


def secure_issues_from_block(block: Any) -> Tuple[bool, List[Any]]:
    if not isinstance(block, dict):
        return True, []
    secure = block.get("secure")
    issues = _issues_to_list(block.get("issues"))
    if secure is None:
        secure = (len(issues) == 0)
    return bool(secure), issues


def secure_issues_from_legacy(rec: Dict[str, Any]) -> Tuple[bool, List[Any]]:
    secure = rec.get("secure")
    issues = rec.get("issues")
    scan = rec.get("scan_result") or rec.get("security_scan")
    if isinstance(scan, dict):
        if secure is None:
            secure = scan.get("secure")
        if issues is None:
            issues = scan.get("issues")
    issues_list = _issues_to_list(issues)
    if secure is None:
        secure = (len(issues_list) == 0)
    return bool(secure), issues_list


def secure_issues_from_tool(rec: Dict[str, Any], tool_key: str) -> Tuple[bool, List[Any]]:
    block = rec.get(tool_key)
    if isinstance(block, dict):
        return secure_issues_from_block(block)

    if tool_key == "bandit_result":
        legacy_block = rec.get("bandit_result")
        if isinstance(legacy_block, dict):
            return secure_issues_from_block(legacy_block)

    if tool_key == "semgrep_result":
        legacy_block = rec.get("semgrep_result")
        if isinstance(legacy_block, dict):
            return secure_issues_from_block(legacy_block)

    return secure_issues_from_legacy(rec)


def loc_from(rec: Dict[str, Any]) -> int:
    loc = first_present(rec, ("loc", "line_count", "num_loc", "n_loc"))
    if isinstance(loc, int):
        return max(0, loc)

    ci = rec.get("code_info")
    if isinstance(ci, dict):
        loc = first_present(ci, ("loc", "line_count", "n_loc"))
        if isinstance(loc, int):
            return max(0, loc)
        code_blob = first_present(ci, CODE_KEYS)
        if isinstance(code_blob, str):
            return count_nonempty_lines(code_blob)

    code_blob = first_present(rec, CODE_KEYS)
    if isinstance(code_blob, str):
        return count_nonempty_lines(code_blob)

    return 0


def read_records(jsonl_path: str) -> List[Dict[str, Any]]:
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    recs: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def compute_metrics_for_tool(records: List[Dict[str, Any]], tool_key: str) -> Dict[str, Any]:
    per_item: List[Dict[str, Any]] = []
    for rec in records:
        secure, issues = secure_issues_from_tool(rec, tool_key)
        loc = loc_from(rec)
        per_item.append({"secure": bool(secure), "issue_count": int(len(issues)), "loc": int(loc)})

    N = len(per_item)
    V = sum(x["issue_count"] for x in per_item)
    L = sum(x["loc"] for x in per_item)

    rate_val = round(safe_div(V, N), 6)
    dens_val = round(safe_div(V, L), 6)

    return {
        "items": N,
        "total_issues": V,
        "total_loc": L,
        "vuln_rate_frac": f"{V}/{N}",
        "vuln_rate_value": rate_val,
        "density_frac": f"{V}/{L}",
        "density_value": dens_val,
    }


def compute_metrics(jsonl_path: str) -> Dict[str, Any]:
    records = read_records(jsonl_path)
    path = Path(jsonl_path)

    bandit = compute_metrics_for_tool(records, "bandit_result")
    semgrep = compute_metrics_for_tool(records, "semgrep_result")

    return {
        "dataset": path.stem,
        "bandit": bandit,
        "semgrep": semgrep,
    }


def format_metrics_text(jsonl_path: str, run_info: dict | None = None) -> str:
    r = compute_metrics(jsonl_path)

    lines: List[str] = [""]

    if run_info:
        lines.append("=== Run Info ===")
        ds = run_info.get("dataset")
        if ds is not None:
            lines.append(f"dataset:            {ds}")

        model = run_info.get("model")
        if model is not None:
            lines.append(f"model:              {model}")

        retrieval_strategy = run_info.get("retrieval_strategy")
        if retrieval_strategy is not None:
            lines.append(f"retrieval_strategy: {retrieval_strategy}")
        prompt_tokens = run_info.get("prompt_tokens")
        completion_tokens = run_info.get("completion_tokens")
        total_tokens = run_info.get("total_tokens")
        if prompt_tokens is not None or completion_tokens is not None or total_tokens is not None:
            lines.append("")
            lines.append("=== Token Usage ===")
            if prompt_tokens is not None:
                lines.append(f"prompt_tokens:      {prompt_tokens}")
            if completion_tokens is not None:
                lines.append(f"completion_tokens:  {completion_tokens}")
            if total_tokens is not None:
                lines.append(f"total_tokens:       {total_tokens}")

        runtime_seconds = run_info.get("runtime_seconds")
        if runtime_seconds is not None:
            lines.append("")
            lines.append("=== Runtime ===")
            lines.append(f"runtime_seconds:    {float(runtime_seconds):.2f}")

        lines.append("")

    lines.append("=== Security Metrics (Rate & Density) ===")
    lines.append("")

    for label, key in (("Bandit", "bandit"), ("Semgrep", "semgrep")):
        m = r[key]
        lines += [
            f"--- {label} ---",
            f"items (N):         {m['items']}",
            f"total issues (V):  {m['total_issues']}",
            f"total LOC (L):     {m['total_loc']}",
            "",
            f"Vulnerability Rate = V/N = {m['vuln_rate_frac']} = {m['vuln_rate_value']}",
            f"Vulnerability Density = V/L = {m['density_frac']} = {m['density_value']}",
            "",
        ]

    return "\n".join(lines)


def default_txt_path_for(jsonl_path: str) -> Path:
    p = Path(jsonl_path)
    return p.parent / "rate" / "vuln_density" / f"{p.stem}_metrics.txt"


def save_metrics_result(jsonl_path: str, out_txt_path: str | None = None, run_info: dict | None = None) -> Path:
    text = format_metrics_text(jsonl_path, run_info=run_info)
    out_path = Path(out_txt_path) if out_txt_path else default_txt_path_for(jsonl_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return out_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Save rate/density per tool (Bandit & Semgrep) to a text file.")
    ap.add_argument("files", nargs="+", help="One or more JSONL result files.")
    ap.add_argument("--outdir", help="Optional directory for outputs; default is next to each JSONL under rate/vuln_density.")
    args = ap.parse_args()

    for f in args.files:
        if args.outdir:
            p = Path(args.outdir) / f"{Path(f).stem}_metrics.txt"
            save_metrics_result(f, str(p))
        else:
            save_metrics_result(f)
