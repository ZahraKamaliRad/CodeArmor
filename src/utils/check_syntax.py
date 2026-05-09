import ast
import json


def is_syntax_valid(code: str) -> bool:
    if not code or not code.strip():
        return False
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def filter_jsonl(file_path: str):
    removed = 0
    kept = 0

    with open(file_path, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    valid_records = []

    for line in lines:
        record = json.loads(line)
        code = record.get("code", "")

        if is_syntax_valid(code):
            valid_records.append(record)
            kept += 1
        else:
            removed += 1

    with open(file_path, "w", encoding="utf-8") as outfile:
        for record in valid_records:
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[Filter] Removed {removed} invalid snippets.")
    print(f"[Filter] {kept} valid snippets written to {file_path}")
