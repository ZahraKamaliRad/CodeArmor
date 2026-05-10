import ast
import json
import os


def is_syntax_valid(code: str) -> bool:
    if not code or not code.strip():
        print("[Syntax INVALID] Empty code snippet.")
        return False

    try:
        ast.parse(code)
        return True
    except SyntaxError as e:
        print(f"[Syntax INVALID] line={e.lineno} col={e.offset} | {e.msg}")
        return False


def filter_jsonl(file_path: str):
    removed = 0
    kept = 0

    with open(file_path, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    valid_records = []
    invalid_records = []

    for line in lines:
        record = json.loads(line)
        code = record.get("code", "")

        if is_syntax_valid(code):
            valid_records.append(record)
            kept += 1
        else:
            invalid_records.append(record)
            removed += 1

    with open(file_path, "w", encoding="utf-8") as outfile:
        for record in valid_records:
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

    if invalid_records:
        dir_name = os.path.dirname(file_path)
        invalid_path = os.path.join(dir_name, "invalid.jsonl")

        with open(invalid_path, "w", encoding="utf-8") as invalid_file:
            for record in invalid_records:
                invalid_file.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"[Filter] Invalid snippets written to {invalid_path}")

    print(f"[Filter] Removed {removed} invalid snippets.")
    print(f"[Filter] {kept} valid snippets written to {file_path}")
