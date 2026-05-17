import subprocess
import tempfile
from pathlib import Path


def evaluate_record(record, timeout=5):

    code = record["code"]

    metadata = record.get("metadata", {})

    test_code = metadata.get("test")

    entry_point = metadata.get("entry_point")

    if not test_code or not entry_point:

        return {
            "task_id": record.get("task"),
            "passed": False,
            "error": "Missing test or entry_point"
        }

    full_program = f"""
{code}

{test_code}

check({entry_point})
"""

    try:

        with tempfile.TemporaryDirectory() as tmpdir:

            file_path = Path(tmpdir) / "main.py"

            file_path.write_text(full_program)

            result = subprocess.run(
                ["python", str(file_path)],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            passed = result.returncode == 0

            return {
                "task_id": record.get("task"),
                "passed": passed,
                "stdout": result.stdout,
                "stderr": result.stderr
            }

    except subprocess.TimeoutExpired:

        return {
            "task_id": record.get("task"),
            "passed": False,
            "error": "Timeout"
        }

    except Exception as e:

        return {
            "task_id": record.get("task"),
            "passed": False,
            "error": str(e)
        }